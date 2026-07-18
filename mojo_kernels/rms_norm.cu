#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ----------------------------------------------------------------
// CUDA Kernel for Row-wise RMSNorm Forward Pass
// ----------------------------------------------------------------
__global__ void rms_norm_fwd_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    float eps,
    int N
) {
    // Each block processes exactly one row (hidden dimension N)
    int row_idx = blockIdx.x;
    int col_idx = threadIdx.x;
    
    // Dynamic shared memory allocated at block launch
    extern __shared__ float s_mem[];
    
    // 1. Thread-local sum of squares
    float local_sum = 0.0f;
    for (int i = col_idx; i < N; i += blockDim.x) {
        float val = input[row_idx * N + i];
        local_sum += val * val;
    }
    
    s_mem[col_idx] = local_sum;
    __syncthreads();
    
    // 2. Block-level reduction (tree reduction in shared memory)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (col_idx < stride) {
            s_mem[col_idx] += s_mem[col_idx + stride];
        }
        __syncthreads();
    }
    
    // 3. Compute mean square and inverse root mean square (RMS)
    float mean_square = s_mem[0] / N;
    float inv_rms = rsqrtf(mean_square + eps);
    
    // 4. Write back normalized and scaled values to global memory
    for (int i = col_idx; i < N; i += blockDim.x) {
        int idx = row_idx * N + i;
        output[idx] = input[idx] * inv_rms * weight[i];
    }
}

// ----------------------------------------------------------------
// C++ Entrypoint called by PyTorch Bindings
// ----------------------------------------------------------------
void rms_norm_cuda_forward(
    const float* input,
    const float* weight,
    float* output,
    float eps,
    int rows,
    int cols,
    cudaStream_t stream
) {
    // Define block and grid dimensions
    // Max 1024 threads per block, aligned to standard GPU block sizing
    int threads = 256;
    if (cols < threads) {
        threads = cols;
    }
    
    int shared_mem_size = threads * sizeof(float);
    
    // Launch kernel: 1 block per row
    rms_norm_fwd_kernel<<<rows, threads, shared_mem_size, stream>>>(
        input, weight, output, eps, cols
    );
}
