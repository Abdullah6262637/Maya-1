import os
import torch
import torch.nn as nn
import time

# Determine source paths for compilation
module_dir = os.path.dirname(__file__)
cpp_source = os.path.join(module_dir, "rms_norm.cpp")
cuda_source = os.path.join(module_dir, "rms_norm.cu")

HAS_CUDA_EXT = False
rms_norm_cuda = None

# JIT Compile CUDA extension if compiler tools and GPU are available
if torch.cuda.is_available():
    try:
        from torch.utils.cpp_extension import load
        print("Compiling custom CUDA C++ RMSNorm kernel extension...")
        
        rms_norm_cuda = load(
            name="rms_norm_cuda",
            sources=[cpp_source, cuda_source],
            verbose=True,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"]
        )
        HAS_CUDA_EXT = True
        print("[SUCCESS] Custom CUDA RMSNorm compiled and loaded successfully!")
    except Exception as e:
        print(f"[INFO] Could not compile CUDA extension (Error: {e}). Fallback enabled.")
        HAS_CUDA_EXT = False
else:
    print("[INFO] CUDA is not available. Skipping JIT compilation. CPU/native fallback enabled.")

# ----------------------------------------------------------------
# Dual-path PyTorch RMSNorm Module
# ----------------------------------------------------------------
class LlamaRMSNormCUDA(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Launch custom CUDA kernel if parameters match, else run CPU fallback
        if HAS_CUDA_EXT and hidden_states.is_cuda and hidden_states.dtype == torch.float32:
            # Flatten high dimensions (keeping hidden dim -1) to match 2D kernel expectation
            orig_shape = hidden_states.shape
            flat_states = hidden_states.view(-1, orig_shape[-1])
            
            output = rms_norm_cuda.forward(flat_states, self.weight, self.variance_epsilon)
            return output.view_as(hidden_states)
        else:
            # Safe PyTorch native CPU fallback
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            normalized = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            return (normalized * self.weight).to(input_dtype)

# ----------------------------------------------------------------
# Validation & Benchmarking
# ----------------------------------------------------------------
def test_extension():
    # Setup test parameters
    hidden_size = 512
    norm = LlamaRMSNormCUDA(hidden_size)
    
    # Generate mock inputs
    x = torch.randn(10, hidden_size)
    
    # 1. Verification of CPU fallback path
    out_cpu = norm(x)
    
    # Standard formula verification
    var = x.pow(2).mean(-1, keepdim=True)
    expected = x * torch.rsqrt(var + 1e-6)
    
    cpu_err = torch.max(torch.abs(out_cpu - expected))
    print(f"CPU Fallback Max Error: {cpu_err.item():.8f}")
    assert cpu_err.item() < 1e-5, "CPU Fallback mismatch!"
    
    # 2. Verification of GPU CUDA path if available
    if HAS_CUDA_EXT:
        x_cuda = x.cuda()
        norm_cuda = norm.cuda()
        
        out_cuda = norm_cuda(x_cuda)
        
        # Cross check CPU vs GPU outputs
        cuda_err = torch.max(torch.abs(out_cuda.cpu() - out_cpu))
        print(f"CUDA JIT Kernel vs CPU Max Error: {cuda_err.item():.8f}")
        assert cuda_err.item() < 1e-5, "CUDA Kernel mismatch!"
        print("[SUCCESS] CUDA RMSNorm matches reference PyTorch calculations.")

if __name__ == "__main__":
    test_extension()
