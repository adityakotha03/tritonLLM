import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mish_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute Mish: x * tanh(ln(1 + exp(x)))
    x_exp = tl.exp(x)
    x_exp_plus_1 = x_exp + 1.0
    ln_x_exp_plus_1 = tl.math.log(x_exp_plus_1)
    tanh_ln = tl.math.tanh(ln_x_exp_plus_1)
    out = x * tanh_ln

    # Store output
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def matmul_kernel(
    a_ptr,  # Pointer to first matrix (M x K)
    b_ptr,  # Pointer to second matrix (K x N)
    out_ptr,  # Pointer to output matrix (M x N)
    M, K, N,  # Dimensions
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of output
    pid = tl.program_id(0)
    num_blocks = (M + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_m = pid % num_blocks
    block_n = pid // num_blocks

    # Compute the block's row and column indices
    m_start = block_m * BLOCK_SIZE
    n_start = block_n * BLOCK_SIZE
    m_end = tl.minimum(m_start + BLOCK_SIZE, M)
    n_end = tl.minimum(n_start + BLOCK_SIZE, N)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    # Load and compute
    for k in range(0, K, BLOCK_SIZE):
        a = tl.load(a_ptr + (m_start + tl.arange(0, BLOCK_SIZE)) * K + k, mask=(m_start + tl.arange(0, BLOCK_SIZE)) < M, other=0.0)
        b = tl.load(b_ptr + k * N + (n_start + tl.arange(0, BLOCK_SIZE)), mask=(n_start + tl.arange(0, BLOCK_SIZE)) < N, other=0.0)
        acc += tl.dot(a, b)

    # Write back
    tl.store(out_ptr + (m_start + tl.arange(0, BLOCK_SIZE)) * N + n_start + tl.arange(0, BLOCK_SIZE), acc, mask=(m_start + tl.arange(0, BLOCK_SIZE)) < M & (n_start + tl.arange(0, BLOCK_SIZE)) < N)


def triton_mish(x: torch.Tensor):
    """
    Custom Triton implementation of the Mish activation function.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    mish_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    """
    Custom Triton implementation of matrix multiplication.
    """
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    a = a.contiguous()
    b = b.contiguous()

    # Prepare output tensor
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "Matrix dimensions must match for multiplication."
    out = torch.empty((M, N), dtype=a.dtype, device=a.device)

    # Determine block size
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((M + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    matmul_kernel[grid](a, b, out, M, K, N, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.linear(x)
        x = triton_mish(x)
        x = triton_mish(x)
        return x