# Persistent Python Inference Server for Maya-1 Models
# Loads the model once at startup and serves requests instantly via HTTP.

import os
import sys
import json
import torch
import torch.nn.functional as F
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
from threading import Thread
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# Add current directory to path for local imports
sys.path.append(os.path.dirname(__file__))
from model import MayaModel, MayaConfig
from lua_config_parser import parse_lua_file

# Global cache to keep the active model in RAM
loaded_checkpoint = None
loaded_model = None
loaded_tokenizer = None

def sample_top_k_top_p(logits, temperature=1.0, top_k=50, top_p=0.9):
    logits = logits / max(temperature, 1e-5)
    
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float('Inf')
        
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float('Inf')
        
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token.item()

def load_model(checkpoint_path):
    global loaded_checkpoint, loaded_model, loaded_tokenizer
    
    # Return immediately if requested model is already loaded in RAM
    if loaded_checkpoint == checkpoint_path:
        return loaded_model, loaded_tokenizer
        
    print(f"[SERVER] Loading new checkpoint: {checkpoint_path}...")
    
    # 1. HuggingFace / Transformers model (e.g. sft_1b)
    if os.path.isdir(checkpoint_path) or "sft_1b" in checkpoint_path:
        print("[SERVER] Detected HF Transformers model directory.")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.float32,
            device_map=None,
            low_cpu_mem_usage=True
        )
        model.eval()
    else:
        # 2. Custom Maya PyTorch model (.pt checkpoints)
        print("[SERVER] Detected custom PyTorch checkpoint.")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["model_state_dict"]
        clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        
        embed_weight = clean_state_dict["tok_embeddings.weight"]
        vocab_size = embed_weight.shape[0]
        hidden_size = embed_weight.shape[1]
        
        base_dir = os.path.dirname(__file__)
        lua_cfg = parse_lua_file(os.path.join(base_dir, "config.lua"))
        m_cfg = lua_cfg.get("model", {})
        num_hidden_layers = m_cfg.get("num_hidden_layers", 6)
        num_attention_heads = m_cfg.get("num_attention_heads", 8)
        num_key_value_heads = m_cfg.get("num_key_value_heads", 2)
        intermediate_size = m_cfg.get("intermediate_size", 512)
        max_seq_len = m_cfg.get("seq_len", 128)

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
        model.load_state_dict(clean_state_dict)
        model.to(device)
        model.eval()
        
        # Load tokenizer
        tokenizer_path = os.path.join(os.path.dirname(base_dir), "shared", "mock_tokenizer.json")
        tokenizer = Tokenizer.from_file(tokenizer_path)

    loaded_checkpoint = checkpoint_path
    loaded_model = model
    loaded_tokenizer = tokenizer
    print("[SERVER] Model loaded into RAM successfully.")
    return model, tokenizer

class ModelServerHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress logging every request to keep stdout clean
        return

    def do_POST(self):
        if self.path != "/generate":
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        payload = json.loads(post_data.decode('utf-8'))

        prompt = payload.get("prompt", "")
        checkpoint_path = payload.get("checkpoint", "")
        temperature = float(payload.get("temperature", 0.7))
        top_p = float(payload.get("top_p", 0.9))

        try:
            model, tokenizer = load_model(checkpoint_path)
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        # Run model inference and stream responses
        if isinstance(tokenizer, Tokenizer):
            # Custom Maya Model Generation & Streaming
            device = "cuda" if torch.cuda.is_available() else "cpu"
            encoded = tokenizer.encode(prompt)
            token_ids = encoded.ids
            if len(token_ids) == 0:
                bos_id = tokenizer.token_to_id("<s>") or 0
                token_ids = [bos_id]
                
            input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
            prompt_text_for_stream = tokenizer.decode(token_ids)
            
            with torch.inference_mode():
                for _ in range(50):
                    seq_len = input_tensor.size(1)
                    if seq_len > model.config.max_seq_len:
                        input_tensor = input_tensor[:, -model.config.max_seq_len:]
                        
                    logits, _ = model(input_tensor)
                    next_token_logits = logits[0, -1, :]
                    
                    next_token_id = sample_top_k_top_p(
                        next_token_logits, 
                        temperature=temperature, 
                        top_p=top_p
                    )
                    
                    eos_id = tokenizer.token_to_id("</s>") or 2
                    if next_token_id == eos_id:
                        break
                        
                    next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
                    input_tensor = torch.cat([input_tensor, next_token_tensor], dim=1)
                    
                    token_ids.append(next_token_id)
                    current_text = tokenizer.decode(token_ids)
                    decoded_token = current_text[len(prompt_text_for_stream):]
                    prompt_text_for_stream = current_text
                    
                    if decoded_token:
                        data = decoded_token.encode('utf-8')
                        self.wfile.write(f"{len(data):x}\r\n".encode('utf-8'))
                        self.wfile.write(data)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                # Send terminal chunk
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
        else:
            # HuggingFace Model Generation & Streaming
            model_device = next(model.parameters()).device
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model_device)
            
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs = dict(
                **model_inputs,
                max_new_tokens=50,
                temperature=temperature,
                top_p=top_p,
                streamer=streamer,
                do_sample=True if temperature > 0.0 else False
            )
            
            # Start generation in thread so we can consume stream concurrently
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()
            
            for new_text in streamer:
                if new_text:
                    data = new_text.encode('utf-8')
                    self.wfile.write(f"{len(data):x}\r\n".encode('utf-8'))
                    self.wfile.write(data)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
            
            # Send terminal chunk
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            
            thread.join()

def run_server():
    server_address = ('', 4002)
    httpd = HTTPServer(server_address, ModelServerHandler)
    print("[SERVER] Python Model Inference Server running on http://localhost:4002")
    
    # Preload sft_1b to warm up the cache
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_ckpt = os.path.join(base_dir, "shared", "checkpoints", "sft_1b")
    if os.path.exists(default_ckpt):
        try:
            load_model(default_ckpt)
        except Exception as e:
            print(f"[SERVER] Failed to preload default model: {e}")
            
    httpd.serve_forever()

if __name__ == "__main__":
    run_server()
