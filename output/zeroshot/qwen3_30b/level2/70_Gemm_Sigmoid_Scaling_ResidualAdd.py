import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gemm_sigmoid_scale_residual_add_kernel(
    x_ptr,  # Pointer to input
    w_ptr,  # Pointer to weights
    out_ptr,  # Pointer to output
    bias_ptr,  # Pointer to bias (if any)
    batch_size,  # Number of batches
    input_size,  # Input dimension
    hidden_size,  # Output dimension
    scaling_factor,  # Scaling factor for sigmoid output
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Block indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Block offsets
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N

    # Create offsets for M and N dimensions
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)

    # Mask for valid indices
    mask_m = offs_m < batch_size
    mask_n = offs_n < hidden_size

    # Initialize accumulator for the matrix multiplication
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K (input_size) in chunks
    for k in range(0, input_size, BLOCK_SIZE_K):
        # Load input data: (BLOCK_SIZE_M, BLOCK_SIZE_K)
        k_offset = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = k_offset < input_size

        x = tl.load(
            x_ptr + (offs_m[:, None] * input_size + k_offset[None, :]),
            mask=(mask_m[:, None] & mask_k[None, :]),
            other=0.0,
        )

        # Load weights: (BLOCK_SIZE_K, BLOCK_SIZE_N)
        w = tl.load(
            w_ptr + (k_offset[:, None] * hidden_size + offs_n[None, :]),
            mask=(mask_k[:, None] & mask_n[None, :]),
            other=0.0,
        )

        # Perform matrix multiplication (M x K) * (K x N) -> (M x N)
        accumulator += tl.dot(x, w)

    # Apply bias if provided
    if bias_ptr is not None:
        bias = tl.load(
            bias_ptr + offs_n,
            mask=mask_n,
            other=0.0,
        )
        accumulator += bias[None, :]

    # Sigmoid activation: x = 1 / (1 + exp(-x))
    # Use online softmax-like trick: use stable sigmoid via 1 / (1 + exp(-x))
    # Scale input to reduce magnitude and improve stability
    x_for_sigmoid = accumulator

    # Stable sigmoid: 1 / (1 + exp(-x))
    # Avoid overflow: use exp(-x) for positive x, exp(x) for negative x
    # Scale to mitigate overflow
    x_for_sigmoid = x_for_sigmoid * 0.5  # Scales the input, helps stability

    # Stable sigmoid approximation using fused exp
    # exp(-|x|) to avoid overflow
    x_abs = tl.abs(x_for_sigmoid)
    # exp(-x_abs) is safe
    exp_neg_xabs = tl.exp(-x_abs)
    # sign = -1 if x < 0, else 1
    sign = 1.0 - 2.0 * (x_for_sigmoid < 0)
    # sigmoid = 0.5 * (1 + sign * (1 - exp(-2*|x|)))
    # But better: 1 / (1 + exp(-x)) = 0.5 + 0.5 * tanh(x/2) → not direct
    # Instead: use 1 / (1 + exp(-x))
    # But to avoid overflow: use exp(-x) for x > 0, exp(x) for x < 0
    # We can write: sigmoid(x) = 1 / (1 + exp(-x)) = exp(x) / (1 + exp(x)) for x < 0
    # But use: clamp x to [-20, 20] to avoid overflow
    x_clamped = tl.clip(x_for_sigmoid, -20.0, 20.0)
    exp_neg_x_clamped = tl.exp(-x_clamped)
    sigmoid = 1.0 / (1.0 + exp_neg_x_clamped)

    # Scale sigmoid output
    sigmoid_scaled = sigmoid * scaling_factor

    # Residual add: output = sigmoid_scaled + input_x (original gemm result)
    # We have original gemm result in accumulator
    # So we add accumulator to scaled sigmoid
    # But note: residual is original gemm output
    # So output = accumulator + sigmoid_scaled
    output = accumulator + sigmoid_scaled

    # Store output
    tl.store(
        out_ptr + (offs_m[:, None] * hidden_size + offs_n[None, :]),
        output,
        mask=(mask_m[:, None] & mask_n[None, :]),
    )


def triton_gemm_sigmoid_scale_residual_add(x, w, bias, scaling_factor):
    """
    Custom Triton kernel wrapper for Gemm + Sigmoid + Scaling + Residual Add.

    Args:
        x: Input tensor of shape (batch_size, input_size)
        w: Weight tensor of shape (input_size, hidden_size)
        bias: Optional bias tensor of shape (hidden_size,)
        scaling_factor: float scaling factor

    Returns:
        Output tensor of shape (batch_size, hidden_size)
    """
    # Ensure contiguous and CUDA
    x = x.contiguous()
    w = w.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    # Get dimensions
    batch_size, input_size = x.shape
    _, hidden_size = w.shape

    # Initialize output tensor
    out = torch.empty(batch_size, hidden_size, device=x.device, dtype=x.dtype)

    # Heuristic block sizes based on A100 and performance best practices
    # Use BLOCK_SIZE_M = 128, BLOCK_SIZE_N = 128, BLOCK_SIZE_K = 64 for better shared memory use
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64

    # Grid configuration: (M_blocks, N_blocks)
    grid = lambda meta: (
        triton.cdiv(batch_size, meta["BLOCK_SIZE_M"]),
        triton.cdiv(hidden_size, meta["BLOCK_SIZE_N"]),
    )

    # Launch kernel
    gemm_sigmoid_scale_residual_add_kernel[grid](
        x,
        w,
        out,
        bias,
        batch_size,
        input_size,
        hidden_size,
        scaling_factor,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        # Register the weight and bias as parameters
        self.register_buffer("weight", torch.randn(input_size, hidden_size, dtype=torch.bfloat16))
        self.register_buffer("bias", torch.zeros(hidden_size, dtype=torch.bfloat16))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        # Use Triton kernel for all operations: Gemm + Sigmoid + Scale + Residual Add
        # We assume input is fp16 or bf16 to leverage Tensor Cores
        # Ensure inputs are float32 or bf16
        x = x.to(torch.bfloat16)
        out = triton_gemm_sigmoid_scale_residual_add(
            x,
            self.weight,
            self.bias,
            self.scaling_factor,
        )
        return out