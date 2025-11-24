import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_tanh_kernel(
    a_ptr,  # pointer to first input matrix
    b_ptr,  # pointer to second input matrix
    out_ptr,  # pointer to output matrix
    m: tl.constexpr,  # number of rows in a and out
    n: tl.constexpr,  # number of columns in b and out
    k: tl.constexpr,  # number of columns in a and rows in b
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a block of data
    pid = tl.program_id(0)
    # Compute the block's row and column indices
    block_row = pid // (n // BLOCK_SIZE)
    block_col = pid % (n // BLOCK_SIZE)
    # Compute the starting row and column in the matrix
    row_start = block_row * BLOCK_SIZE
    col_start = block_col * BLOCK_SIZE
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    # Load a matrix block
    a = tl.load(a_ptr + row_start + tl.arange(0, BLOCK_SIZE)[:, None] * k + tl.arange(0, k), mask=(row_start + tl.arange(0, BLOCK_SIZE))[:, None] * k + tl.arange(0, k) < m * k, other=0.0)
    # Load b matrix block
    b = tl.load(b_ptr + col_start + tl.arange(0, BLOCK_SIZE)[None, :] * k + tl.arange(0, k), mask=(col_start + tl.arange(0, BLOCK_SIZE))[None, :] * k + tl.arange(0, k) < n * k, other=0.0)
    # Compute the matrix multiplication
    for i in range(k // BLOCK_SIZE):
        a_block = tl.load(a_ptr + row_start + tl.arange(0, BLOCK_SIZE)[:, None] * k + tl.arange(0, i * BLOCK_SIZE) + i * BLOCK_SIZE, mask=(row_start + tl.arange(0, BLOCK_SIZE))[:, None] * k + tl.arange(0, i * BLOCK_SIZE) + i * BLOCK_SIZE < m * k, other=0.0)
        b_block = tl.load(b_ptr + col_start + tl.arange(0, i * BLOCK_SIZE) + i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :], mask=(col_start + tl.arange(0, i * BLOCK_SIZE) + i * BLOCK_SIZE)[None, :] * k + tl.arange(0, BLOCK_SIZE) < n * k, other=0.0)
        acc += tl.dot(a_block, b_block)
    # Apply tanh
    acc = tl.tanh(acc)
    # Store the result
    tl.store(out_ptr + row_start + tl.arange(0, BLOCK_SIZE)[:, None] * n + col_start + tl.arange(0, BLOCK_SIZE)[None, :], acc, mask=(row_start + tl.arange(0, BLOCK_SIZE))[:, None] * n + (col_start + tl.arange(0, BLOCK_SIZE))[None, :] < m * n)


def triton_matmul_tanh(a: torch.Tensor, b: torch.Tensor, m: int, n: int, k: int):
    """
    Triton kernel for matrix multiplication followed by tanh.
    """
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    a = a.contiguous()
    b = b.contiguous()
    # Output tensor
    out = torch.empty((m, n), dtype=torch.float32, device=a.device)
    # Determine block size
    BLOCK_SIZE = 128
    # Number of blocks
    num_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    # Launch kernel
    grid = (num_blocks,)
    matmul_tanh_kernel[grid](a, b, out, m, n, k, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # Define the RNN cell components
        self.i2h_weights = torch.nn.Parameter(torch.randn(hidden_size + input_size, hidden_size, device='cuda'))
        self.i2h_bias = torch.nn.Parameter(torch.randn(hidden_size, device='cuda'))
        self.h2o_weights = torch.nn.Parameter(torch.randn(hidden_size, output_size, device='cuda'))
        self.h2o_bias = torch.nn.Parameter(torch.randn(output_size, device='cuda'))
        # Initialize hidden state
        self.register_buffer('hidden', torch.randn(1, hidden_size, device='cuda'))

    def forward(self, x: torch.Tensor, initial_hidden=None) -> torch.Tensor:
        # Ensure x is on the correct device
        x = x.to(self.hidden.device)
        # Use initial hidden state if provided
        if initial_hidden is not None:
            self.hidden.copy_(initial_hidden)
        # Concatenate input and hidden state
        combined = torch.cat((x, self.hidden), dim=1)
        # Compute hidden state using custom Triton kernel
        # Reshape combined to (batch_size * input_size, 1)
        combined = combined.view(-1, 1)
        # Compute i2h
        i2h = torch.matmul(combined, self.i2h_weights) + self.i2h_bias
        # Reshape i2h to (batch_size, hidden_size)
        i2h = i2h.view(x.size(0), self.hidden_size)
        # Apply tanh using Triton kernel
        self.hidden = triton_matmul_tanh(i2h, torch.ones_like(i2h), self.hidden_size, self.hidden_size, self.hidden_size)
        # Compute output
        output = torch.matmul(self.hidden, self.h2o_weights) + self.h2o_bias
        return output