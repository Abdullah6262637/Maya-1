# PyTorch Training Profiler and Performance Instrumentation Module
# Instruments model training and exports Chrome trace files for execution analysis.

import os
import sys
import torch
from torch.utils.data import DataLoader

# Align directories
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "python_training"))
from model import LlamaModel, LlamaConfig
from dataset import ShardedMmapDataset

def profile_model():
    print("Initializing performance profiling suite...")
    
    # 1. Prepare output log directory
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared", "prof_logs"))
    os.makedirs(log_dir, exist_ok=True)
    
    # 2. Config model parameters (small scale for quick profiling trace)
    config = LlamaConfig(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        max_seq_len=64
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LlamaModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    # 3. Load dummy binary dataset
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared", "mock_data.bin"))
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Mock data not found at {data_path}. Run scripts/prepare_mock_data.py first.")
        
    dataset = ShardedMmapDataset(data_path, config.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    # Define profiling activities (CPU + CUDA if GPU is present)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        
    print(f"Profiling training loop on {device.upper()}...")
    
    # 4. Instrument with torch.profiler
    trace_file_path = os.path.join(log_dir, "trace.json")
    
    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=1,      # Wait steps (no profile)
            warmup=1,    # Warmup steps (JIT compilation warmup)
            active=1,    # Active profiling collection steps
            repeat=1
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        
        # Run 5 training steps
        for step, (input_ids, labels) in enumerate(dataloader):
            if step >= 5:
                break
                
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits, loss = model(input_ids, targets=labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Advance profiler step
            prof.step()
            print(f"Completed step {step} in profiler...")
            
    print(f"[SUCCESS] Profiler complete. Logs written to directory: {log_dir}")
    
    # Export raw Chrome tracing json as fallback for direct inspection
    try:
        prof.export_chrome_trace(trace_file_path)
        print(f"Exported raw Chrome tracing log to: {trace_file_path}")
    except Exception as e:
        print(f"Warning: Could not export raw Chrome trace: {e}")

if __name__ == "__main__":
    profile_model()
