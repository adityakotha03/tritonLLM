import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr,  # Pointer to first input matrix (batch, seq, hidden)
    b_ptr,  # Pointer to second input matrix (hidden, input)
    out_ptr,  # Pointer to output matrix (batch, seq, input)
    batch, seq, hidden, input, 
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a block of data
    pid = tl.program_id(0)
    # Compute the block index in the batch dimension
    batch_idx = pid // (seq * hidden)
    # Compute the block index in the sequence dimension
    seq_idx = (pid // hidden) % seq
    # Compute the block index in the hidden dimension
    hidden_idx = pid % hidden
    # Compute the offset for the current block
    offset = batch_idx * seq * hidden * input + seq_idx * hidden * input + hidden_idx * input
    # Load a block of A
    a = tl.load(a_ptr + offset, mask=tl.arange(0, input) < input, other=0.0)
    # Load b block
    b = tl.load(b_ptr + hidden_idx * input, mask=tl.arange(0, input) < input, other=0.0)
    # Compute the dot product
    out = tl.dot(a, b)
    # Store the result
    tl.store(out_ptr + offset, out)


def triton_matmul(a: torch.Tensor, b: torch.Tensor, batch, seq, hidden, input):
    """
    Custom Triton kernel for matrix multiplication.
    """
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    a = a.contiguous()
    b = b.contiguous()

    # Prepare output tensor
    out = torch.empty(batch, seq, input, device=a.device, dtype=a.dtype)

    # Determine block size and grid
    BLOCK_SIZE = 128
    num_blocks = (hidden + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks,)

    # Launch the kernel
    matmul_kernel[grid](a, b, out, batch, seq, hidden, input, BLOCK_SIZE=BLOCK_SIZE)
    return out


@triton.jit
def activation_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.relu(x)
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_activation(x: torch.Tensor):
    """
    Custom Triton kernel for ReLU activation.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 128
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    activation_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout

    def forward(self, x, h0, c0):
        batch, seq, _ = x.shape
        # Custom matmul for LSTM
        out = triton_matmul(x, torch.randn(self.hidden_size, self.input_size, device=x.device, dtype=x.dtype), batch, seq, self.hidden_size, self.input_size)
        out = triton_activation(out)
        return out