#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

// Forward declaration of the low-level CUDA launch entrypoint
void rms_norm_cuda_forward(
    const float* input,
    const float* weight,
    float* output,
    float eps,
    int rows,
    int cols,
    cudaStream_t stream
);

// ----------------------------------------------------------------
// PyTorch Binding Entrypoint
// ----------------------------------------------------------------
torch::Tensor rms_norm_forward(
    torch::Tensor input,
    torch::Tensor weight,
    double eps
) {
    // Assert tensor attributes
    TORCH_CHECK(input.is_cuda(), "Input tensor must be stored on CUDA device");
    TORCH_CHECK(weight.is_cuda(), "Weight scale tensor must be stored on CUDA device");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be Float32 precision");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "Weight must be Float32 precision");

    auto output = torch::empty_like(input);

    // Calculate dimensions
    int cols = input.size(-1);
    int rows = input.numel() / cols;

    // Retrieve raw pointers
    const float* input_ptr = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    // Retrieve the active PyTorch CUDA stream to align execution asynchronously
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // Call CUDA kernel
    rms_norm_cuda_forward(
        input_ptr,
        weight_ptr,
        output_ptr,
        static_cast<float>(eps),
        rows,
        cols,
        stream
    );

    return output;
}

// Bind C++ function to Python module using PyBind11
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rms_norm_forward, "Fused RMSNorm forward pass (CUDA)");
}
