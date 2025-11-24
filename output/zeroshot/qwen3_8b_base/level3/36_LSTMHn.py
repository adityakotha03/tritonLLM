import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr,  # pointer to matrix A
    b_ptr,  # pointer to matrix B
    c_ptr,  # pointer to matrix C (output)
    n_cols: tl.constexpr,
    n_rows: tl.constexpr,
    k: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a block of the output matrix
    pid = tl.program_id(0)
    num_blocks = tl.num_programs(0)
    block_col = pid % num_blocks
    block_row = pid // num_blocks

    # Compute the block offsets
    row_start = block_row * BLOCK_SIZE
    col_start = block_col * BLOCK_SIZE

    # Initialize the accumulator
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    # Iterate over the k dimension
    for i in range(0, k, BLOCK_SIZE):
        # Load the a matrix block
        a = tl.load(a_ptr + i * n_cols + row_start, mask=(i + BLOCK_SIZE) < k, other=0.0)
        # Load the b matrix block
        b = tl.load(b_ptr + col_start + i, mask=(i + BLOCK_SIZE) < k, other=0.0)
        # Compute the dot product
        acc += tl.dot(a, b)
    
    # Store the result
    tl.store(c_ptr + row_start + col_start, acc, mask=(row_start + BLOCK_SIZE) < n_rows and (col_start + BLOCK_SIZE) < n_cols)


@triton.jit
def relu_kernel(
    x_ptr,  # pointer to input
    out_ptr,  # pointer to output
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Apply ReLU
    out = tl.maximum(x, 0.0)
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    """
    Custom Triton kernel for matrix multiplication.
    """
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    a = a.contiguous()
    b = b.contiguous()

    # Ensure dimensions are correct
    assert a.dim() == 2 and b.dim() == 2, "Only 2D matrices are supported."
    m, k = a.shape
    n, _ = b.shape

    # Prepare output
    c = torch.empty((m, n), dtype=a.dtype, device=a.device)

    # Determine block size (adjust based on hardware and performance)
    BLOCK_SIZE = 128

    # Determine number of blocks
    num_blocks = (m + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks,)

    # Launch the kernel
    matmul_kernel[grid](a, b, c, n, m, k, BLOCK_SIZE=BLOCK_SIZE)
    return c


def triton_relu(x: torch.Tensor):
    """
    Custom Triton kernel for ReLU activation.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()

    # Prepare output
    out = torch.empty_like(x)

    # Determine block size
    BLOCK_SIZE = 128

    # Determine number of blocks
    num_blocks = (x.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks,)

    # Launch the kernel
    relu_kernel[grid](x, out, x.numel(), BLOCK_SIZE=BLOCK_SIZE)
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
        # Convert to CUDA tensors if not already
        x = x.cuda()
        h0 = h0.cuda()
        c0 = c0.cuda()

        # Custom matmul for LSTM input to hidden
        # x: (batch_size, sequence_length, input_size)
        # h0: (num_layers, batch_size, hidden_size)
        # c0: (num_layers, batch_size, hidden_size)

        # Reshape x to (batch_size * sequence_length, input_size)
        batch_size = x.size(1)
        seq_len = x.size(2)
        x = x.view(-1, self.input_size)

        # Initialize hidden and cell states
        h = h0
        c = c0

        # Process each time step
        for t in range(seq_len):
            # Extract current input
            x_t = x[t * batch_size:(t + 1) * batch_size]

            # Compute input gate
            i = triton_matmul(x_t, self.input_weights) + triton_matmul(h[0], self.hidden_weights_i) + self.bias_i
            i = triton_relu(i)

            # Compute forget gate
            f = triton_matmul(x_t, self.input_weights) + triton_matmul(h[0], self.hidden_weights_f) + self.bias_f
            f = triton_relu(f)

            # Compute cell input
            c_in = triton_matmul(x_t, self.input_weights) + triton_matmul(h[0], self.hidden_weights_c) + self.bias_c
            c_in = triton_relu(c_in)

            # Compute cell state
            c[0] = f * c[0] + i * c_in

            # Compute output gate
            o = triton_matmul(x_t, self.input_weights) + triton_matmul(h[0], self.hidden_weights_o) + self.bias_o
            o = triton_relu(o)

            # Update hidden state
            h[0] = o

        # Output the final hidden state
        return h[0]