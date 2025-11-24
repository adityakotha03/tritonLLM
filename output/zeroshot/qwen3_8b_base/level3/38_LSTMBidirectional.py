import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    n, m, k,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a block of data
    pid = tl.program_id(0)
    num_block = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_row = pid % num_block
    block_col = pid // num_block

    # Compute the block row and column indices
    row_start = block_row * BLOCK_SIZE
    col_start = block_col * BLOCK_SIZE

    # Compute the offset for a and b
    a_offsets = tl.arange(0, BLOCK_SIZE) + row_start * m
    b_offsets = tl.arange(0, BLOCK_SIZE) + col_start * k
    a_offsets = a_offsets[:, None] + tl.arange(0, BLOCK_SIZE)[None, :]
    b_offsets = b_offsets[None, :] + tl.arange(0, BLOCK_SIZE)[:, None]

    # Load a and b
    a = tl.load(a_ptr + a_offsets, mask=(a_offsets < n * m))
    b = tl.load(b_ptr + b_offsets, mask=(b_offsets < k * m))

    # Compute the dot product
    c = tl.dot(a, b)

    # Compute the offset for c
    c_offsets = row_start * m + tl.arange(0, BLOCK_SIZE)
    c_offsets = c_offsets[:, None] + tl.arange(0, BLOCK_SIZE)[None, :]

    # Store the result
    tl.store(c_ptr + c_offsets, c, mask=(c_offsets < n * m))


def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    """
    Perform matrix multiplication using a Triton kernel.
    """
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    a = a.contiguous()
    b = b.contiguous()

    # Ensure the dimensions are correct
    n, k = a.shape
    m, _ = b.shape
    c = torch.empty((n, m), dtype=a.dtype, device=a.device)

    # Determine the block size
    BLOCK_SIZE = 128

    # Determine the number of blocks
    num_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the kernel
    grid = lambda meta: (num_blocks,)
    matmul_kernel[grid](a, b, c, n, m, k, BLOCK_SIZE=BLOCK_SIZE)
    return c


@triton.jit
def relu_kernel(
    x_ptr, out_ptr,
    n, m,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a block of data
    pid = tl.program_id(0)
    num_block = (n * m + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_idx = pid % num_block
    block_start = block_idx * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load x
    x = tl.load(x_ptr + offsets, mask=offsets < n * m, other=0.0)

    # Apply ReLU
    out = tl.maximum(x, 0.0)

    # Store result
    tl.store(out_ptr + offsets, out, mask=offsets < n * m)


def triton_relu(x: torch.Tensor):
    """
    Apply ReLU using a Triton kernel.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n, m = x.shape
    BLOCK_SIZE = 128
    num_blocks = (n * m + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = lambda meta: (num_blocks,)
    relu_kernel[grid](x, out, n, m, BLOCK_SIZE=BLOCK_SIZE)
    return out


@triton.jit
def softmax_kernel(
    x_ptr, out_ptr,
    n, m,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a block of data
    pid = tl.program_id(0)
    num_block = (n * m + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_idx = pid % num_block
    block_start = block_idx * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load x
    x = tl.load(x_ptr + offsets, mask=offsets < n * m, other=0.0)

    # Compute max and exp
    max_val = tl.max(x)
    x -= max_val
    exp_x = tl.exp(x)

    # Compute sum
    sum_exp = tl.sum(exp_x)

    # Compute softmax
    out = exp_x / sum_exp

    # Store result
    tl.store(out_ptr + offsets, out, mask=offsets < n * m)


def triton_softmax(x: torch.Tensor):
    """
    Apply softmax using a Triton kernel.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n, m = x.shape
    BLOCK_SIZE = 128
    num_blocks = (n * m + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = lambda meta: (num_blocks,)
    softmax_kernel[grid](x, out, n, m, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout

        # Define the LSTM layers
        self.lstm_layers = nn.ModuleList()
        for i in range(num_layers):
            self.lstm_layers.append(nn.LSTMCell(input_size if i == 0 else hidden_size * 2, hidden_size))
        
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x, h0, c0):
        # h0 and c0 are of shape (num_layers * 2, batch_size, hidden_size)
        batch_size = x.size(1)
        seq_len = x.size(2)
        output = []

        # Process each time step
        for t in range(seq_len):
            x_t = x[:, t, :]
            h_prev, c_prev = h0[:, t, :], c0[:, t, :]

            # Process each layer
            for i in range(self.num_layers):
                h_prev, c_prev = self.lstm_layers[i](x_t, (h_prev, c_prev))
                if self.dropout > 0 and i < self.num_layers - 1:
                    h_prev = F.dropout(h_prev, self.dropout, training=self.training)

            output.append(h_prev)

        # Stack the outputs
        output = torch.stack(output, dim=1)
        out = self.fc(output[:, -1, :])
        return out