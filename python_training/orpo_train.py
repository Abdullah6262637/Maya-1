import os
import sys
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

# Append project path to imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import MayaModel, MayaConfig
from muon import Muon
from db_logger import AsyncMetricLogger

class ORPODataset(Dataset):
    """
    Dataset class that loads preference pairs (prompt, chosen, rejected)
    and tokenizes them.
    """
    def __init__(self, jsonl_path, tokenizer_path, max_seq_len=512):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        # Ensure special tokens exist
        self.pad_id = self.tokenizer.token_to_id("<pad>")
        if self.pad_id is None:
            self.pad_id = self.tokenizer.token_to_id("</s>")  # fallback
            
        self.records = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))
                    
    def __len__(self):
        return len(self.records)
        
    def __getitem__(self, idx):
        item = self.records[idx]
        
        # Tokenize components
        prompt_ids = self.tokenizer.encode(item["prompt"]).ids
        chosen_ids = self.tokenizer.encode(item["chosen"]).ids
        rejected_ids = self.tokenizer.encode(item["rejected"]).ids
        
        # For multi-turn golden dialogues, prompt is already formatted in tokens.
        # But for simple prompts, format it by adding tags if not already present.
        if not item["prompt"].startswith("<|"):
            # Format simple prompt-response pair
            prompt_ids = self.tokenizer.encode(f"<|kullanici|>{item['prompt']}<|son|><|asistan|>").ids
            chosen_ids = self.tokenizer.encode(f"{item['chosen']}<|son|>").ids
            rejected_ids = self.tokenizer.encode(f"{item['rejected']}<|son|>").ids
            
        return {
            "prompt_ids": prompt_ids,
            "chosen_ids": chosen_ids,
            "rejected_ids": rejected_ids
        }

def collate_fn(batch, pad_id, max_seq_len):
    """
    Collate function to dynamically pad prompt+chosen and prompt+rejected sequences
    for batch execution.
    """
    batch_prompt_chosen = []
    batch_prompt_rejected = []
    batch_sft_mask = []
    prompt_lens = []
    
    for item in batch:
        p_ids = item["prompt_ids"]
        c_ids = item["chosen_ids"]
        r_ids = item["rejected_ids"]
        
        # Chosen sequence: prompt + chosen
        sw = p_ids + c_ids
        # Rejected sequence: prompt + rejected
        sl = p_ids + r_ids
        
        # Truncate if exceeds max length
        if len(sw) > max_seq_len:
            sw = sw[:max_seq_len]
        if len(sl) > max_seq_len:
            sl = sl[:max_seq_len]
            
        # SFT Mask: 0 for prompt tokens, 1 for chosen tokens
        sft_mask = [0] * len(p_ids) + [1] * len(c_ids)
        if len(sft_mask) > max_seq_len:
            sft_mask = sft_mask[:max_seq_len]
            
        batch_prompt_chosen.append(torch.tensor(sw, dtype=torch.long))
        batch_prompt_rejected.append(torch.tensor(sl, dtype=torch.long))
        batch_sft_mask.append(torch.tensor(sft_mask, dtype=torch.float32))
        prompt_lens.append(len(p_ids))
        
    # Pad sequences dynamically
    x_chosen = nn.utils.rnn.pad_sequence(batch_prompt_chosen, batch_first=True, padding_value=pad_id)
    x_rejected = nn.utils.rnn.pad_sequence(batch_prompt_rejected, batch_first=True, padding_value=pad_id)
    sft_loss_mask = nn.utils.rnn.pad_sequence(batch_sft_mask, batch_first=True, padding_value=0.0)
    
    return x_chosen, x_rejected, sft_loss_mask, prompt_lens

def compute_logps(logits, targets, mask):
    """
    Computes length-normalized log probabilities of target tokens.
    """
    # targets are shifted by 1 token relative to logits (logits predict next token)
    # Logits shape: [B, S, V]
    # Targets shape: [B, S]
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    target_probs = torch.gather(log_probs, dim=-1, index=targets[:, 1:].unsqueeze(-1)).squeeze(-1)
    
    # Apply mask for chosen/rejected tokens only
    masked_target_probs = target_probs * mask[:, 1:]
    
    # Calculate sum and length-normalized average logprob
    sum_logps = masked_target_probs.sum(dim=-1)
    lengths = mask[:, 1:].sum(dim=-1).clamp(min=1.0)
    avg_logps = sum_logps / lengths
    
    return avg_logps

def main():
    parser = argparse.ArgumentParser(description="Maya-1 ORPO Alignment Training")
    parser.add_argument("--data_path", type=str, required=True, help="Path to curated preference JSONL data")
    parser.add_argument("--tokenizer_path", type=str, default="shared/turkish_bpe_50k.json", help="Path to tokenizer file")
    parser.add_argument("--resume_from", type=str, required=True, help="Path to pretraining base model checkpoint")
    parser.add_argument("--checkpoint_dir", type=str, default="shared/checkpoints/sft_1b", help="Directory to save aligned checkpoints")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for MuAdamW")
    parser.add_argument("--muon_lr", type=float, default=2e-3, help="Learning rate for Muon")
    parser.add_argument("--beta", type=float, default=0.1, help="ORPO beta weight (odds-ratio penalty)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of SFT/ORPO alignment epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per training step")
    parser.add_argument("--seq_len", type=int, default=512, help="Max sequence length")
    parser.add_argument("--use_mup", action="store_true", help="Enable muP optimization matching pretraining")
    parser.add_argument("--mup_base_hidden", type=int, default=256, help="muP base model hidden dimension")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running ORPO training on device: {device}")
    
    # Initialize tokenizer to fetch pad token and vocab size
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    pad_id = tokenizer.token_to_id("<pad>")
    if pad_id is None:
        pad_id = tokenizer.token_to_id("</s>")
        
    # Load base model configuration and weights
    print(f"Loading checkpoint model parameters from: {args.resume_from}")
    checkpoint = torch.load(args.resume_from, map_location="cpu")
    
    # Deduce hidden layers from model state dict
    state_dict = checkpoint["model_state_dict"]
    layer_indices = []
    for k in state_dict.keys():
        parts = k.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    layer_indices.append(int(parts[i+1]))
                except ValueError:
                    pass
    num_layers = max(layer_indices) + 1 if layer_indices else 2
    
    # Construct base model config (Matching actual 1.1B dimension or checkpoint values)
    # Defaulting to 1.1B dimensions, but configurable
    config = MayaConfig(
        vocab_size=vocab_size,
        hidden_size=2048 if "layers.0.attention.wq.weight" in state_dict or "_orig_mod.layers.0.attention.wq.weight" in state_dict else 512, # auto-adjust for testing
        num_hidden_layers=num_layers,
        use_mtp=False # MTP is disabled or omitted during normal chat SFT/ORPO alignment
    )
    
    # If checkpoint state dict has smaller hidden size or intermediate size, extract it
    for key in state_dict.keys():
        if "attention.wq.weight" in key:
            config.hidden_size = state_dict[key].shape[1]
        if "feed_forward.w1.weight" in key:
            config.intermediate_size = state_dict[key].shape[0]
            
    print(f"Loaded Config: Vocab Size: {config.vocab_size} | Hidden Size: {config.hidden_size} | Intermediate Size: {config.intermediate_size} | Layers: {config.num_hidden_layers}")
    
    model = MayaModel(config)
    
    # muP Configuration
    if args.use_mup:
        import mup
        print("Applying Maximal Update Parametrization (muP) alignment shapes...")
        # Scale intermediate_size proportionally for mup consistency
        base_intermediate_size = max(1, int(config.intermediate_size * args.mup_base_hidden / config.hidden_size))
        base_config = MayaConfig(
            vocab_size=vocab_size,
            hidden_size=args.mup_base_hidden,
            num_hidden_layers=config.num_hidden_layers,
            intermediate_size=base_intermediate_size,
            use_mtp=False
        )
        base_model = MayaModel(base_config)
        mup.set_base_shapes(model, base_model)
        
    # Load model weights
    # Strip compile prefix if not compiling during SFT
    first_key = list(state_dict.keys())[0]
    if first_key.startswith("_orig_mod."):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    
    # Handle vocab size mismatch (added special chat tokens)
    checkpoint_vocab_size = state_dict["tok_embeddings.weight"].shape[0]
    if checkpoint_vocab_size != config.vocab_size:
        print(f"Vocab size mismatch: Checkpoint has {checkpoint_vocab_size}, Model expects {config.vocab_size}.")
        print("Resizing embedding weights and copying existing parameters...")
        
        # Pop embedding keys to prevent size mismatch failure in load_state_dict
        old_embeddings = state_dict.pop("tok_embeddings.weight")
        state_dict.pop("output.weight", None)
        
        # Load rest of model state dict
        model.load_state_dict(state_dict, strict=False)
        
        # Manually copy and pad embedding weight
        with torch.no_grad():
            model.tok_embeddings.weight[:checkpoint_vocab_size].copy_(old_embeddings)
            # Initialize new special tokens with mean pretraining embedding
            mean_emb = old_embeddings.mean(dim=0)
            model.tok_embeddings.weight[checkpoint_vocab_size:].copy_(
                mean_emb.unsqueeze(0).repeat(config.vocab_size - checkpoint_vocab_size, 1)
            )
            print("Embedding weights resized and initialized successfully.")
    else:
        model.load_state_dict(state_dict, strict=False)
        
    model.to(device)
    
    # Setup Optimizers
    muon_params = []
    adamw_params = []
    
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and "tok_embeddings" not in name and "output" not in name:
            muon_params.append(p)
        else:
            adamw_params.append(p)
            
    print(f"ORPO Optimizers -> Muon Parameters: {len(muon_params)} | AdamW Parameters: {len(adamw_params)}")
    
    optimizers = []
    if len(muon_params) > 0:
        optimizers.append(Muon(muon_params, lr=args.muon_lr))
    if len(adamw_params) > 0:
        if args.use_mup:
            import mup
            optimizers.append(mup.MuAdamW(adamw_params, lr=args.lr))
        else:
            optimizers.append(torch.optim.AdamW(adamw_params, lr=args.lr))
            
    # Load dataset
    dataset = ORPODataset(args.data_path, args.tokenizer_path, max_seq_len=args.seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id, args.seq_len)
    )
    
    logger = AsyncMetricLogger(node_id="orpo-node")
    
    print(f"Starting ORPO training loop for {args.epochs} epoch(s)...")
    step = 0
    t0 = time.time()
    
    for epoch in range(args.epochs):
        model.train()
        for x_chosen, x_rejected, sft_loss_mask, prompt_lens in dataloader:
            x_chosen = x_chosen.to(device)
            x_rejected = x_rejected.to(device)
            sft_loss_mask = sft_loss_mask.to(device)
            
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
                
            # 1. Forward pass on chosen response
            # targets for SFT is same as x_chosen shifted (autoregressive modeling)
            logits_chosen, _ = model(x_chosen)
            
            # Compute SFT cross-entropy loss
            # Shift inputs and logits for target alignment
            shift_logits = logits_chosen[:, :-1, :].contiguous()
            shift_targets = x_chosen[:, 1:].contiguous()
            shift_mask = sft_loss_mask[:, 1:].contiguous()
            
            loss_sft = F.cross_entropy(
                shift_logits.view(-1, vocab_size),
                shift_targets.view(-1),
                reduction="none"
            )
            # Mask out non-chosen tokens (prompts) from loss calculation
            loss_sft = (loss_sft * shift_mask.view(-1)).sum() / shift_mask.sum().clamp(min=1.0)
            
            # Compute log probabilities of chosen output
            avg_logps_chosen = compute_logps(logits_chosen, x_chosen, sft_loss_mask)
            
            # 2. Forward pass on rejected response
            logits_rejected, _ = model(x_rejected)
            
            # SFT Mask for rejected (0 for prompt, 1 for rejected)
            # The prompt length is identical, so we construct mask for rejected
            # prompt token count is extracted directly from prompt_lens
            rejected_sft_mask = torch.zeros_like(x_rejected, dtype=torch.float32)
            for i, p_len in enumerate(prompt_lens):
                if i < len(rejected_sft_mask):
                    prompt_len = min(p_len, x_rejected.shape[1])
                    rejected_sft_mask[i, prompt_len:] = 1.0
                    
            avg_logps_rejected = compute_logps(logits_rejected, x_rejected, rejected_sft_mask)
            
            # 3. ORPO Odds Ratio computation
            # log odds = log P(y|x) - log(1 - P(y|x))
            log_odds_chosen = avg_logps_chosen - torch.log1p(-torch.exp(avg_logps_chosen).clamp(max=0.999))
            log_odds_rejected = avg_logps_rejected - torch.log1p(-torch.exp(avg_logps_rejected).clamp(max=0.999))
            
            # Odds ratio loss
            log_odds_ratio = log_odds_chosen - log_odds_rejected
            loss_or = -F.logsigmoid(log_odds_ratio).mean()
            
            # Combined ORPO loss
            loss = loss_sft + args.beta * loss_or
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            for opt in optimizers:
                opt.step()
                
            loss_val = loss.item()
            sft_val = loss_sft.item()
            or_val = loss_or.item()
            
            logger.log(step, "orpo_loss", loss_val)
            logger.log(step, "sft_loss", sft_val)
            logger.log(step, "or_loss", or_val)
            
            if step % 1 == 0:
                print(f"Step {step} | Total Loss: {loss_val:.4f} | SFT Loss: {sft_val:.4f} | OR Loss: {or_val:.4f} | Chosen LogP: {avg_logps_chosen.mean().item():.4f}")
                
            step += 1
            
    # Save aligned checkpoint
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    aligned_path = os.path.join(args.checkpoint_dir, "ckpt_aligned.pt")
    print(f"Saving aligned checkpoint to {aligned_path}...")
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "config": config,
    }, aligned_path)
    
    total_time = time.time() - t0
    print(f"ORPO Alignment completed successfully in {total_time:.2f} seconds!")
    logger.close()

if __name__ == "__main__":
    main()
