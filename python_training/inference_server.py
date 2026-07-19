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
    if os.path.isdir(checkpoint_path) and not checkpoint_path.endswith(".pt"):
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
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint["model_state_dict"]
        clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        
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
        
        # Load tokenizer
        tokenizer_path = os.path.join(os.path.dirname(base_dir), "shared", "turkish_bpe_50k.json")
        if not os.path.exists(tokenizer_path):
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
        if self.path not in ["/generate", "/v1/chat/completions"]:
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        payload = json.loads(post_data.decode('utf-8'))

        checkpoint_path = payload.get("checkpoint", "")
        temperature = float(payload.get("temperature", 0.7))
        top_p = float(payload.get("top_p", 0.9))
        stream = payload.get("stream", True)
        messages = payload.get("messages", [])
        prompt = payload.get("prompt", "")

        try:
            model, tokenizer = load_model(checkpoint_path)
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
            return

        is_openai_chat = self.path == "/v1/chat/completions"
        is_chat_mode = is_openai_chat or len(messages) > 0

        # Chat Formatting logic
        if is_chat_mode:
            if messages:
                formatted_prompt = ""
                for m in messages:
                    role = m["role"]
                    content = m["content"]
                    if role == "system":
                        formatted_prompt += f"<|sistem|>{content}<|son|>"
                    elif role == "user":
                        formatted_prompt += f"<|kullanici|>{content}<|son|>"
                    elif role == "assistant":
                        formatted_prompt += f"<|asistan|>{content}<|son|>"
                formatted_prompt += "<|asistan|>"
                prompt = formatted_prompt
            elif not prompt:
                prompt = "<|sistem|>Sen Maya, yardımsever bir Türkçe asistansın.<|son|><|kullanici|>Selam<|son|><|asistan|>"

        # Send initial headers
        self.send_response(200)
        if is_openai_chat and stream:
            self.send_header("Content-Type", "text/event-stream")
        elif is_openai_chat:
            self.send_header("Content-Type", "application/json")
        else:
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
            
            # Chat Stop ID configuration
            stop_ids = [tokenizer.token_to_id("</s>") or 2]
            son_id = tokenizer.token_to_id("<|son|>")
            if son_id is not None:
                stop_ids.append(son_id)
                
            generated_text = ""
            
            with torch.inference_mode():
                for _ in range(150):
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
                    
                    if next_token_id in stop_ids:
                        break
                        
                    next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
                    input_tensor = torch.cat([input_tensor, next_token_tensor], dim=1)
                    
                    token_ids.append(next_token_id)
                    current_text = tokenizer.decode(token_ids)
                    decoded_token = current_text[len(prompt_text_for_stream):]
                    prompt_text_for_stream = current_text
                    
                    if decoded_token:
                        generated_text += decoded_token
                        if stream:
                            if is_openai_chat:
                                sse_payload = f"data: {json.dumps({'choices': [{'delta': {'content': decoded_token}, 'finish_reason': None}]})}\n\n"
                                data = sse_payload.encode('utf-8')
                            else:
                                data = decoded_token.encode('utf-8')
                            self.wfile.write(f"{len(data):x}\r\n".encode('utf-8'))
                            self.wfile.write(data)
                            self.wfile.write(b"\r\n")
                            self.wfile.flush()
                
                # If non-streaming, output whole response now
                if not stream:
                    if is_openai_chat:
                        response_body = json.dumps({
                            "choices": [{
                                "message": {"role": "assistant", "content": generated_text},
                                "finish_reason": "stop"
                            }]
                        })
                    else:
                        response_body = generated_text
                    data = response_body.encode('utf-8')
                    self.wfile.write(f"{len(data):x}\r\n".encode('utf-8'))
                    self.wfile.write(data)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                else:
                    if is_openai_chat:
                        data = "data: [DONE]\n\n".encode('utf-8')
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
            if not messages:
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
                max_new_tokens=150,
                temperature=temperature,
                top_p=top_p,
                streamer=streamer,
                do_sample=True if temperature > 0.0 else False
            )
            
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()
            
            generated_text = ""
            for new_text in streamer:
                if new_text:
                    generated_text += new_text
                    if stream:
                        if is_openai_chat:
                            sse_payload = f"data: {json.dumps({'choices': [{'delta': {'content': new_text}, 'finish_reason': None}]})}\n\n"
                            data = sse_payload.encode('utf-8')
                        else:
                            data = new_text.encode('utf-8')
                        self.wfile.write(f"{len(data):x}\r\n".encode('utf-8'))
                        self.wfile.write(data)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
            
            if not stream:
                if is_openai_chat:
                    response_body = json.dumps({
                        "choices": [{
                            "message": {"role": "assistant", "content": generated_text},
                            "finish_reason": "stop"
                        }]
                    })
                else:
                    response_body = generated_text
                data = response_body.encode('utf-8')
                self.wfile.write(f"{len(data):x}\r\n".encode('utf-8'))
                self.wfile.write(data)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            else:
                if is_openai_chat:
                    data = "data: [DONE]\n\n".encode('utf-8')
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
    default_ckpt = os.path.join(base_dir, "shared", "checkpoints", "sft_1b", "ckpt_aligned.pt")
    if os.path.exists(default_ckpt):
        try:
            load_model(default_ckpt)
        except Exception as e:
            print(f"[SERVER] Failed to preload default model: {e}")
            
    httpd.serve_forever()

if __name__ == "__main__":
    run_server()
