# Supervised Fine-Tuning (SFT) Instruction-Tuning Script
# Implements mask-based loss computation (labels=-100 for instructions) to align language models.

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

# Load local imports
sys.path.append(os.path.dirname(__file__))
from model import MayaModel, MayaConfig
from db_logger import AsyncMetricLogger
from lua_config_parser import parse_lua_file

class SFTDataset(Dataset):
    def __init__(self, data_path: str, tokenizer_path: str, max_seq_len: int = 128):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_seq_len = max_seq_len
        
        # Load instruction JSON
        with open(data_path, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)
            
        self.samples = []
        self.prepare_samples()

    def prepare_samples(self):
        bos_id = self.tokenizer.token_to_id("<s>")
        if bos_id is None:
            bos_id = 0
        eos_id = self.tokenizer.token_to_id("</s>")
        if eos_id is None:
            eos_id = 2
        
        for item in self.raw_data:
            inst_tokens = self.tokenizer.encode(item["instruction"]).ids
            out_tokens = self.tokenizer.encode(item["output"]).ids
            
            # Combine sequence: <s> + Instruction + Response + </s>
            input_ids = [bos_id] + inst_tokens + out_tokens + [eos_id]
            
            # Mask instruction: set target label to -100 (ignored in PyTorch CrossEntropyLoss)
            # Only response tokens and EOS will contribute to the gradient calculation
            labels = [-100] * (len(inst_tokens) + 1) + out_tokens + [eos_id]
            
            # Truncate to max sequence length if necessary
            if len(input_ids) > self.max_seq_len:
                input_ids = input_ids[:self.max_seq_len]
                labels = labels[:self.max_seq_len]
                
            # Dynamic padding to max_seq_len
            pad_len = self.max_seq_len - len(input_ids)
            input_ids += [0] * pad_len
            labels += [-100] * pad_len
            
            self.samples.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def run_sft():
    print("Initializing Supervised Fine-Tuning (SFT)...")
    
    # 1. Setup paths
    base_dir = os.path.dirname(__file__)
    data_path = os.path.abspath(os.path.join(base_dir, "..", "shared", "sft_data.json"))
    tokenizer_path = os.path.abspath(os.path.join(base_dir, "..", "shared", "mock_tokenizer.json"))
    checkpoint_dir = os.path.abspath(os.path.join(base_dir, "..", "shared", "checkpoints"))
    
    import argparse
    parser = argparse.ArgumentParser(description="Maya-1 SFT Fine-Tuning")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to base checkpoint")
    args, _ = parser.parse_known_args()
    
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        if os.path.exists(checkpoint_dir):
            files = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.startswith("ckpt_step_") and f.endswith(".pt")]
            if files:
                def get_step_num(fpath):
                    try:
                        return int(os.path.basename(fpath).replace("ckpt_step_", "").replace(".pt", ""))
                    except ValueError:
                        return -1
                checkpoint_path = max(files, key=get_step_num)
        
        if not checkpoint_path:
            checkpoint_path = os.path.abspath(os.path.join(checkpoint_dir, "ckpt_step_49.pt"))
            
    output_path = os.path.abspath(os.path.join(checkpoint_dir, "sft_checkpoint.pt"))
    
    # 2. Load Lua Configuration for model size defaults
    lua_cfg = parse_lua_file(os.path.join(base_dir, "config.lua"))
    m_cfg = lua_cfg.get("model", {})
    num_hidden_layers = m_cfg.get("num_hidden_layers", 6)
    num_attention_heads = m_cfg.get("num_attention_heads", 8)
    num_key_value_heads = m_cfg.get("num_key_value_heads", 2)
    intermediate_size = m_cfg.get("intermediate_size", 512)
    max_seq_len = m_cfg.get("seq_len", 128)

    # 3. Instantiate dataset and loader
    dataset = SFTDataset(data_path, tokenizer_path, max_seq_len=max_seq_len)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # 4. Load pre-trained base model weights
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading base checkpoint: {checkpoint_path} on {device.upper()}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    
    vocab_size = clean_state_dict["tok_embeddings.weight"].shape[0]
    hidden_size = clean_state_dict["tok_embeddings.weight"].shape[1]
    
    config = MayaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=intermediate_size,
        max_seq_len=max_seq_len
    )
    
    model = MayaModel(config)
    model.load_state_dict(clean_state_dict)
    model.to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    logger = AsyncMetricLogger(node_id="sft-node")
    
    # 4. Fine-Tuning Epoch Loop
    epochs = 15
    print(f"Starting SFT training loop for {epochs} epochs...")
    
    step = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits, _ = model(input_ids)
            
            # Shift tokens for next-token prediction loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate loss only on responses
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)
            )
            
            loss.backward()
            optimizer.step()
            
            loss_val = loss.item()
            epoch_loss += loss_val
            
            # Log SFT metrics
            logger.log(step, "sft_loss", loss_val)
            step += 1
            
        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {epoch_loss/len(dataloader):.4f}")
        
    # 5. Save SFT Fine-Tuned Checkpoint
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Saving SFT fine-tuned checkpoint to: {output_path}...")
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": epoch_loss / len(dataloader)
    }, output_path)
    
    logger.close()
    print("[SUCCESS] SFT Fine-Tuning completed successfully!")

if __name__ == "__main__":
    run_sft()
