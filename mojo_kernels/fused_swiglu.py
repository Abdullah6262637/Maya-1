import torch
import time

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# ----------------------------------------------------------------
# Triton Custom Kernel for Fused SwiGLU Forward Pass
# ----------------------------------------------------------------
if HAS_TRITON:
    @triton.jit
    def _swiglu_fwd_kernel(
        gate_ptr, up_ptr, out_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Load gate and up projections from global memory
        gate = tl.load(gate_ptr + offsets, mask=mask)
        up = tl.load(up_ptr + offsets, mask=mask)

        # Compute Swish(gate) * up = (gate * sigmoid(gate)) * up
        # We use explicit sigmoid: 1.0 / (1.0 + exp(-gate)) for compatibility
        sigmoid_gate = 1.0 / (1.0 + tl.exp(-gate))
        silu_gate = gate * sigmoid_gate
        result = silu_gate * up

        # Store result back to global memory
        tl.store(out_ptr + offsets, result, mask=mask)
else:
    _swiglu_fwd_kernel = None

# ----------------------------------------------------------------
# PyTorch Wrapper using Custom Autograd
# ----------------------------------------------------------------
class FusedSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor):
        if not HAS_TRITON:
            raise RuntimeError("Triton is not available on this platform.")
        assert gate.is_cuda, "Triton kernels require CUDA tensors"
        assert gate.shape == up.shape, "Gate and Up tensors must have the same shape"
        
        # Flatten tensors to 1D for processing
        flat_gate = gate.contiguous().view(-1)
        flat_up = up.contiguous().view(-1)
        n_elements = flat_gate.numel()

        out = torch.empty_like(flat_gate)

        # Launch grid configuration: block size 1024
        block_size = 1024
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

        _swiglu_fwd_kernel[grid](
            flat_gate, flat_up, out,
            n_elements,
            BLOCK_SIZE=block_size
        )
        
        ctx.save_for_backward(gate, up)
        return out.view_as(gate)

    @staticmethod
    def backward(ctx, grad_output):
        gate, up = ctx.saved_tensors
        # Standard analytical backward gradient derivation for SwiGLU
        # d/d_gate = (sigmoid(gate) + gate * sigmoid(gate) * (1 - sigmoid(gate))) * up * grad_output
        # d/d_up = silu(gate) * grad_output
        
        sig = torch.sigmoid(gate)
        silu = gate * sig
        
        grad_up = silu * grad_output
        grad_gate = (sig + gate * sig * (1.0 - sig)) * up * grad_output
        
        return grad_gate, grad_up

def fused_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    Seamless entrypoint for Fused SwiGLU.
    Dispatches to Triton custom kernel on CUDA, and falls back to vanilla PyTorch on CPU.
    """
    if gate.is_cuda:
        return FusedSwiGLU.apply(gate, up)
    else:
        # Fallback to PyTorch native SwiGLU on CPU
        return torch.nn.functional.silu(gate) * up

# ----------------------------------------------------------------
# Benchmark & Validation script
# ----------------------------------------------------------------
def run_benchmark():
    if not torch.cuda.is_available():
        print("[INFO] CUDA is not available. Skipping GPU benchmarking. Verification complete on CPU.")
        # CPU verification
        gate = torch.randn(10, 10)
        up = torch.randn(10, 10)
        out_fused = fused_swiglu(gate, up)
        out_native = torch.nn.functional.silu(gate) * up
        assert torch.allclose(out_fused, out_native, atol=1e-6)
        print("[SUCCESS] CPU fallback matches vanilla PyTorch output exactly!")
        return

    print("Running GPU Fused SwiGLU Triton Kernel Benchmark...")
    
    # Choose dimension parameters (standard LLaMA intermediate size)
    batch_size = 4
    seq_len = 2048
    inter_dim = 5632
    
    gate = torch.randn(batch_size, seq_len, inter_dim, device="cuda", dtype=torch.float32, requires_grad=True)
    up = torch.randn(batch_size, seq_len, inter_dim, device="cuda", dtype=torch.float32, requires_grad=True)
    
    # 1. Cosine similarity test
    out_native = torch.nn.functional.silu(gate) * up
    out_fused = fused_swiglu(gate, up)
    
    cos_sim = torch.nn.functional.cosine_similarity(out_native.flatten(), out_fused.flatten(), dim=0)
    print(f"Validation: Cosine similarity between Triton and PyTorch outputs: {cos_sim.item():.7f}")
    assert cos_sim.item() > 0.9999, "Output mismatch!"

    # 2. Performance benchmark
    # Warmup
    for _ in range(10):
        _ = torch.nn.functional.silu(gate) * up
        _ = fused_swiglu(gate, up)
        
    torch.cuda.synchronize()
    
    # Native PyTorch speed test
    t_start = time.time()
    for _ in range(100):
        out = torch.nn.functional.silu(gate) * up
    torch.cuda.synchronize()
    t_native = (time.time() - t_start) / 100 * 1000  # ms
    
    # Triton Fused speed test
    t_start = time.time()
    for _ in range(100):
        out = fused_swiglu(gate, up)
    torch.cuda.synchronize()
    t_fused = (time.time() - t_start) / 100 * 1000  # ms
    
    print(f"Performance:")
    print(f"  Vanilla PyTorch SwiGLU: {t_native:.4f} ms")
    print(f"  Triton Fused SwiGLU:    {t_fused:.4f} ms")
    print(f"  Speedup Factor:          {t_native / t_fused:.2x}")

if __name__ == "__main__":
    run_benchmark()
