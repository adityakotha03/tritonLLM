import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def gemm_kernel(
    a_ptr,  # pointer to matrix A
    b_ptr,  # pointer to matrix B
    c_ptr,  # pointer to matrix C
    n, m, k,  # dimensions of A (n x k), B (k x m), C (n x m)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a block of data
    pid = tl.program_id(0)
    # Compute the block's row and column indices
    row = pid // m
    col = pid % m
    # Compute the block's starting index in A and B
    a_start = row * k
    b_start = col * k
    # Create offsets for the block
    offsets = tl.arange(0, BLOCK_SIZE)
    # Compute the block's row and column indices in A and B
    a_offsets = a_start + offsets
    b_offsets = b_start + offsets
    # Load A and B
    a = tl.load(a_ptr + a_offsets, mask=a_offsets < n * k, other=0.0)
    b = tl.load(b_ptr + b_offsets, mask=b_offsets < k * m, other=0.0)
    # Compute the dot product
    c = tl.dot(a, b)
    # Store the result
    tl.store(c_ptr + row * m + col, c)

def triton_gemm(a, b, n, m, k):
    """
    Perform GEMM using Triton kernel.
    """
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    a = a.contiguous()
    b = b.contiguous()
    c = torch.empty((n, m), device=a.device, dtype=a.dtype)
    # Determine the number of blocks needed
    num_blocks = (m + BLOCK_SIZE - 1) // BLOCK_SIZE
    # Launch the Triton kernel
    gemm_kernel[ num_blocks ](a, b, c, n, m, k, BLOCK_SIZE=BLOCK_SIZE)
    return c

@triton.jit
def sigmoid_kernel(
    x_ptr,  # pointer to input
    y_ptr,  # pointer to output
    n_elements,  # number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(y_ptr + offsets, y, mask=mask)

def triton_sigmoid(x):
    """
    Apply sigmoid using Triton kernel.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    y = torch.empty_like(x)
    n_elements = x.numel()
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    sigmoid_kernel[ num_blocks ](x, y, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return y

@triton.jit
def logsumexp_kernel(
    x_ptr,  # pointer to input
    y_ptr,  # pointer to output
    n_elements,  # number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
    max_x = tl.max(x, axis=0)
    exp_x = tl.exp(x - max_x)
    sum_exp_x = tl.sum(exp_x, axis=0)
    y = max_x + tl.math.log(sum_exp_x)
    tl.store(y_ptr + pid, y, mask=mask)

def triton_logsumexp(x):
    """
    Apply logsumexp along the last dimension using Triton kernel.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    # Ensure x is 2D
    if x.dim() == 1:
        x = x.unsqueeze(1)
    n, m = x.shape
    y = torch.empty(n, device=x.device, dtype=x.dtype)
    num_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    logsumexp_kernel[ num_blocks ](x, y, n * m, BLOCK_SIZE=BLOCK_SIZE)
    return y

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ModelNew, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Custom Gemm for linear1
        x = triton_gemm(self.linear1.weight, x, hidden_size, batch_size, input_size)
        x = triton_sigmoid(x)
        # Custom Gemm for linear2
        x = triton_gemm(self.linear2.weight, x, output_size, batch_size, hidden_size)
        # Custom LogSumExp
        x = triton_logsumexp(x)
        return x