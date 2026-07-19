# Autoregressive Inference Engine and Metin Generation Pipeline
# Implements next-token generation with Temperature, Top-k, and Top-p (Nucleus) sampling.

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

# Load local imports
sys.path.append(os.path.dirname(__file__))
from model import MayaModel, MayaConfig
from lua_config_parser import parse_lua_file

def sample_top_k_top_p(logits, temperature=1.0, top_k=50, top_p=0.9):
    """
    Applies temperature scaling, top-k, and top-p (nucleus) filtering to logits.
    """
    # 1. Temperature scaling
    logits = logits / max(temperature, 1e-5)
    
    # 2. Top-K filtering
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float('Inf')
        
    # 3. Top-P (Nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep the first token that exceeds top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float('Inf')
        
    # 4. Multinomial sampling
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token.item()

@torch.inference_mode()
def generate(
    model: MayaModel, 
    tokenizer: Tokenizer, 
    prompt: str, 
    max_new_tokens: int = 50, 
    temperature: float = 1.0, 
    top_k: int = 50, 
    top_p: float = 0.9,
    stream: bool = True,
    stop_token_ids = None
):
    device = next(model.parameters()).device
    
    # 1. Encode prompt to token IDs
    encoded = tokenizer.encode(prompt)
    token_ids = encoded.ids
    
    if len(token_ids) == 0:
        bos_id = tokenizer.token_to_id("<s>")
        if bos_id is None:
            bos_id = 0
        token_ids = [bos_id]
        
    # Build initial input tensor
    input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
    
    generated_text = ""
    prompt_text_for_stream = tokenizer.decode(token_ids)
    
    # Autoregressive token generation loop
    for _ in range(max_new_tokens):
        # Slice to max_seq_len to prevent out-of-bounds relative position frequencies
        seq_len = input_tensor.size(1)
        if seq_len > model.config.max_seq_len:
            input_tensor = input_tensor[:, -model.config.max_seq_len:]
            
        # Forward pass
        logits, _ = model(input_tensor)
        
        # Get logits of the last token in sequence
        next_token_logits = logits[0, -1, :]
        
        # Sample next token ID
        next_token_id = sample_top_k_top_p(
            next_token_logits, 
            temperature=temperature, 
            top_k=top_k, 
            top_p=top_p
        )
        
        # Stop on End of Sequence / Chat End tokens
        eos_id = tokenizer.token_to_id("</s>")
        if eos_id is None:
            eos_id = 2
        if next_token_id == eos_id or (stop_token_ids and next_token_id in stop_token_ids):
            break
            
        # Append token to input tensor
        next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
        input_tensor = torch.cat([input_tensor, next_token_tensor], dim=1)
        
        # Cumulative decoding for streaming with proper whitespace
        token_ids.append(next_token_id)
        current_text = tokenizer.decode(token_ids)
        decoded_token = current_text[len(prompt_text_for_stream):]
        prompt_text_for_stream = current_text
        
        generated_text += decoded_token
        
        if stream:
            sys.stdout.write(decoded_token)
            sys.stdout.flush()
            
    if stream:
        sys.stdout.write("\n")
        
    return generated_text

def main():
    parser = argparse.ArgumentParser(description="Maya-1 1B Autoregressive Inference")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt text")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint .pt file")
    parser.add_argument("--tokenizer", type=str, default="shared/mock_tokenizer.json", help="Path to BPE tokenizer JSON")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-K threshold")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P threshold")
    parser.add_argument("--chat", action="store_true", help="Format prompt and stop tokens for Chat/SFT models")
    
    args = parser.parse_args()
    
    # 1. Load tokenizer
    if not os.path.exists(args.tokenizer):
        raise FileNotFoundError(f"Tokenizer not found at {args.tokenizer}")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    
    # 2. Check if it is a directory (HF model directory)
    if os.path.isdir(args.checkpoint) and not args.checkpoint.endswith(".pt"):
        from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
        print("Loading Maya-1 1.5B Model...")
        hf_tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True
        )
        hf_model.eval()
        
        # Build prompt using ChatML template
        messages = [{"role": "user", "content": args.prompt}]
        text = hf_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = hf_tokenizer([text], return_tensors="pt").to(hf_model.device)
        streamer = TextStreamer(hf_tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        print(f"\nPrompt: {args.prompt}")
        print("Generation: ", end="")
        hf_model.generate(
            **model_inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            streamer=streamer,
            do_sample=True if args.temperature > 0.0 else False
        )
        return

    # 3. Re-create model structure from checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}")
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading checkpoint weights from: {args.checkpoint} on {device.upper()}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    
    # Clean state dict keys if checkpoint has '_orig_mod.' prefix from torch.compile
    clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    
    # Extract dimensions dynamically from checkpoint shapes to avoid configuration mismatches
    embed_weight = clean_state_dict["tok_embeddings.weight"]
    vocab_size = embed_weight.shape[0]
    hidden_size = embed_weight.shape[1]
    
    # Deduce num_hidden_layers
    layer_indices = []
    for k in clean_state_dict.keys():
        parts = k.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    layer_indices.append(int(parts[i+1]))
                except ValueError:
                    pass
    num_hidden_layers = max(layer_indices) + 1 if layer_indices else 6
    
    # Deduce intermediate_size
    intermediate_size = 5632
    for k in clean_state_dict.keys():
        if "feed_forward.w1.weight" in k:
            intermediate_size = clean_state_dict[k].shape[0]
            break
            
    # Deduce GQA dimensions
    num_attention_heads = 32
    num_key_value_heads = 8
    if "layers.0.attention.wq.weight" in clean_state_dict and "layers.0.attention.wk.weight" in clean_state_dict:
        wq_out = clean_state_dict["layers.0.attention.wq.weight"].shape[0]
        wk_out = clean_state_dict["layers.0.attention.wk.weight"].shape[0]
        head_dim = hidden_size // 8 if hidden_size < 2048 else 64
        num_attention_heads = max(1, wq_out // head_dim)
        num_key_value_heads = max(1, wk_out // head_dim)

    # Load Lua configuration dynamically for max_seq_len fallback
    base_dir = os.path.dirname(__file__)
    try:
        lua_cfg = parse_lua_file(os.path.join(base_dir, "config.lua"))
        m_cfg = lua_cfg.get("model", {})
        max_seq_len = m_cfg.get("seq_len", 512)
    except Exception:
        max_seq_len = 512

    # Reconstruct MayaConfig
    # Detect if checkpoint has MTP weights
    use_mtp = any(k.startswith("mtp_head") for k in clean_state_dict.keys())

    config = MayaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=intermediate_size,
        max_seq_len=max_seq_len,
        use_mtp=use_mtp
    )
    
    model = MayaModel(config)
    
    # Apply muP base shapes for scaling factor initialization
    is_mup_model = hasattr(model.output, "width_mult") or any("feed_forward.w1" in name for name, _ in model.named_modules())
    if is_mup_model:
        try:
            import mup
            mup.set_base_shapes(model, None)
        except Exception as e:
            print(f"Warning: Could not set muP base shapes: {e}")
            
    model.load_state_dict(clean_state_dict)
    model.to(device)
    model.eval()
    
    prompt = args.prompt
    stop_token_ids = []
    if args.chat:
        # Format using the standard Maya Chat template
        prompt = f"<|sistem|>Sen Maya, yardımsever bir Türkçe asistansın.<|son|><|kullanici|>{args.prompt}<|son|><|asistan|>"
        son_id = tokenizer.token_to_id("<|son|>")
        if son_id is not None:
            stop_token_ids.append(son_id)
            
    print(f"\nPrompt: {prompt}")
    print("Generation: ", end="")
    generate(
        model, 
        tokenizer, 
        prompt, 
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        stream=True,
        stop_token_ids=stop_token_ids
    )

if __name__ == "__main__":
    main()
