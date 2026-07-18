# Julia Optimizer Prototyping and Benchmarking Module
# This script benchmarks a vectorized, custom AdamW optimizer step.
# It measures float updates of weights, gradients, first and second moments.

using Random
using Dates

# ----------------------------------------------------------------
# Struct representing the state of the model parameters
# ----------------------------------------------------------------
struct OptimizerState
    params::Vector{Float32}
    grads::Vector{Float32}
    exp_avg::Vector{Float32}  # First moment vector (m)
    exp_avg_sq::Vector{Float32} # Second moment vector (v)
end

function init_state(size::Int)
    # Generate random parameters and gradients for simulation
    params = randn(Float32, size)
    grads = randn(Float32, size) * 0.1f0
    exp_avg = zeros(Float32, size)
    exp_avg_sq = zeros(Float32, size)
    return OptimizerState(params, grads, exp_avg, exp_avg_sq)
end

# ----------------------------------------------------------------
# AdamW Vectorized Step function
# ----------------------------------------------------------------
function adamw_step!(state::OptimizerState, step::Int; lr::Float32=3e-4f0, beta1::Float32=0.9f0, beta2::Float32=0.999f0, eps::Float32=1e-8f0, weight_decay::Float32=0.01f0)
    # Cache bias correction factors
    bias_correction1 = 1.0f0 - beta1^step
    bias_correction2 = 1.0f0 - beta2^step
    step_size = lr / bias_correction1
    
    # Vectorized loop (parallelized/SIMD optimized by Julia JIT compiler)
    @inbounds @simd for i in 1:length(state.params)
        g = state.grads[i]
        w = state.params[i]
        
        # 1. Update biased first moment estimate
        m = beta1 * state.exp_avg[i] + (1.0f0 - beta1) * g
        state.exp_avg[i] = m
        
        # 2. Update biased second raw moment estimate
        v = beta2 * state.exp_avg_sq[i] + (1.0f0 - beta2) * g^2
        state.exp_avg_sq[i] = v
        
        # 3. Compute denominator with bias-corrected second moment
        denom = sqrt(v) / sqrt(bias_correction2) + eps
        
        # 4. Apply weight decay and update parameter
        w_decayed = w - lr * weight_decay * w
        state.params[i] = w_decayed - step_size * m / denom
    end
end

function main()
    println("Julia Optimization Benchmarking Suite (JIT Juntima 2026)")
    
    # Standard parameter size for simulation: 10,000,000 parameters
    num_params = 10_000_000
    println("Initializing optimizer state for $num_params parameters (~40MB arrays)...")
    
    state = init_state(num_params)
    
    # Warmup step to let Julia JIT-compile the function
    println("Warmup compilation step...")
    adamw_step!(state, 1)
    
    # Run 100 benchmark iterations
    println("Running 100 iterations of vectorized AdamW updates...")
    t_start = now()
    
    for step in 2:101
        adamw_step!(state, step)
    end
    
    t_end = now()
    duration_ms = (t_end - t_start).value
    
    println("Performance results:")
    println("  Total time for 100 steps: ", duration_ms, " ms")
    println("  Average time per step: ", duration_ms / 100.0, " ms")
    println("  Throughput: ", (num_params * 100.0) / (duration_ms / 1000.0) / 1e6, " million param updates / second")
    println("Success! Optimizer benchmark completed.")
end

main()
