import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_gelu_kernel(
    input_ptr,  # Pointer to input tensor (batch, input_size)
    weight_ptr,  # Pointer to weight tensor (input_size, output_size)
    output_ptr,  # Pointer to output tensor (batch, output_size)
    bias_ptr,  # Pointer to bias tensor (output_size,) - optional, can be null
    batch_size,  # Number of samples in the batch
    input_size,  # Input dimension
    output_size,  # Output dimension
    divisor,  # Scalar divisor for scaling
    BLOCK_SIZE_M: tl.constexpr,  # Tile size for M dimension (batch)
    BLOCK_SIZE_N: tl.constexpr,  # Tile size for N dimension (output_size)
    BLOCK_SIZE_K: tl.constexpr,  # Tile size for K dimension (input_size)
    GROUP_SIZE_M: tl.constexpr,  # Number of blocks per group for M dimension
):
    # Each program handles a block of the output matrix
    pid = tl.program_id(0)
    pid_m = pid // (output_size // BLOCK_SIZE_N)
    pid_n = pid % (output_size // BLOCK_SIZE_N)

    # Offset for the block of output matrix
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N

    # Create offsets for this block
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Create masks to avoid out-of-bounds access
    mask_m = offs_m < batch_size
    mask_n = offs_n < output_size
    mask = mask_m[:, None] & mask_n[None, :]

    # Load input data (batch, input_size) - load by tile
    input_ptrs = input_ptr + (offs_m[:, None] * input_size + offs_k[None, :])
    input = tl.load(input_ptrs, mask=(mask_m[:, None]) & (offs_k[None, :] < input_size), other=0.0)

    # Load weight data (input_size, output_size) - load by tile
    weight_ptrs = weight_ptr + (offs_k[:, None] * output_size + offs_n[None, :])
    weight = tl.load(weight_ptrs, mask=(offs_k[:, None] < input_size) & (mask_n[None, :]), other=0.0)

    # Perform matrix multiplication: output = input @ weight
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    accumulator += tl.dot(input, weight)
    # Optional: load bias if provided
    if bias_ptr is not None:
        bias_ptrs = bias_ptr + offs_n
        bias = tl.load(bias_ptrs, mask=mask_n, other=0.0)
        accumulator += bias[None, :]

    # Scale by divisor
    accumulator /= divisor

    # Apply GELU activation
    # Use approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    x = accumulator
    x_squared = x * x
    x_cubed = x_squared * x
    # Precompute constants
    sqrt_2_over_pi = 0.7978845608028654
    coeff = 0.044715
    # Compute GELU
    tanh_input = sqrt_2_over_pi * (x + coeff * x_cubed)
    tanh_output = tl.tanh(tanh_input)
    gelu_output = 0.5 * x * (1.0 + tanh_output)

    # Store the result
    output_ptrs = output_ptr + (offs_m[:, None] * output_size + offs_n[None, :])
    tl.store(output_ptrs, gelu_output, mask=mask)


def triton_matmul_gelu(input, weight, bias, divisor):
    # Ensure inputs are on GPU and contiguous
    assert input.is_cuda and weight.is_cuda, "Inputs must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    batch_size, input_size = input.shape
    _, output_size = weight.shape

    # Define block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64
    GROUP_SIZE_M = 8

    # Determine grid size
    grid_m = triton.cdiv(batch_size, BLOCK_SIZE_M)
    grid_n = triton.cdiv(output_size, BLOCK_SIZE_N)
    grid = (grid_m * grid_n,)

    # Launch the kernel
    matmul_gelu_kernel[grid](
        input,
        weight,
        output,
        bias,
        batch_size,
        input_size,
        output_size,
        divisor,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    return output


class ModelNew(nn.Module):
    def __init__(self, input_size, output_size, divisor):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size, bias=True)
        self.divisor = divisor
        # Initialize weights
        nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))
        if self.linear.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.linear.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.linear.bias, -bound, bound)

    def forward(self, x):
        # Use custom Triton kernel for fused matmul, divide, and GELU
        output = triton_matmul_gelu(x, self.linear.weight, self.linear.bias, self.divisor)
        return output