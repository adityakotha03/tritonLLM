import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def lstm_cell_kernel(
    x_ptr, h_ptr, c_ptr,
    W_ih_ptr, W_hh_ptr,
    b_ih_ptr, b_hh_ptr,
    h_out_ptr, c_out_ptr,
    batch_size, seq_len, hidden_size,
    BLOCK_SIZE: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr
):
    # Block indices
    block_idx = tl.program_id(0)
    seq_idx = block_idx // batch_size
    batch_idx = block_idx % batch_size

    # Compute offsets
    seq_offset = seq_idx * hidden_size
    batch_offset = batch_idx * hidden_size
    h_offset = batch_offset + seq_offset
    c_offset = batch_offset + seq_offset

    # Load input data
    x = tl.load(x_ptr + seq_offset + batch_offset, mask=tl.arange(0, HIDDEN_SIZE) < hidden_size, other=0.0)
    h = tl.load(h_ptr + h_offset, mask=tl.arange(0, HIDDEN_SIZE) < hidden_size, other=0.0)
    c = tl.load(c_ptr + c_offset, mask=tl.arange(0, HIDDEN_SIZE) < hidden_size, other=0.0)

    # Load weights
    W_ih = tl.load(W_ih_ptr + tl.arange(0, HIDDEN_SIZE)[:, None] * hidden_size + tl.arange(0, hidden_size)[None, :], mask=tl.arange(0, HIDDEN_SIZE)[:, None] < hidden_size, other=0.0)
    W_hh = tl.load(W_hh_ptr + tl.arange(0, HIDDEN_SIZE)[:, None] * hidden_size + tl.arange(0, hidden_size)[None, :], mask=tl.arange(0, HIDDEN_SIZE)[:, None] < hidden_size, other=0.0)
    b_ih = tl.load(b_ih_ptr + tl.arange(0, HIDDEN_SIZE), mask=tl.arange(0, HIDDEN_SIZE) < hidden_size, other=0.0)
    b_hh = tl.load(b_hh_ptr + tl.arange(0, HIDDEN_SIZE), mask=tl.arange(0, HIDDEN_SIZE) < hidden_size, other=0.0)

    # Compute gates
    i_h = tl.dot(h, W_hh.T) + b_hh
    i_x = tl.dot(x, W_ih.T) + b_ih
    i_sum = i_h + i_x

    # Sigmoid and tanh
    i = tl.sigmoid(i_sum[:, :hidden_size])
    f = tl.sigmoid(i_sum[:, hidden_size:2*hidden_size])
    g = tl.tanh(i_sum[:, 2*hidden_size:3*hidden_size])
    o = tl.sigmoid(i_sum[:, 3*hidden_size:4*hidden_size])

    # Update cell state and hidden state
    c = f * c + i * g
    h = o * tl.tanh(c)

    # Store outputs
    tl.store(h_out_ptr + h_offset, h, mask=tl.arange(0, HIDDEN_SIZE) < hidden_size)
    tl.store(c_out_ptr + c_offset, c, mask=tl.arange(0, HIDDEN_SIZE) < hidden_size)


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    m, n, k,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    # Load A and B
    a = tl.load(a_ptr + offs_m[:, None] * k + offs_k[None, :], mask=offs_m[:, None] < m, other=0.0)
    b = tl.load(b_ptr + offs_k[:, None] * n + offs_n[None, :], mask=offs_k[:, None] < k, other=0.0)

    # Perform matrix multiplication
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc += tl.dot(a, b)
    c = acc

    # Store result
    tl.store(c_ptr + offs_m[:, None] * n + offs_n[None, :], c, mask=offs_m[:, None] < m, other=0.0)


@triton.jit
def gelu_kernel(
    x_ptr, y_ptr,
    n_elements, BLOCK_SIZE: tl.constexpr
):
    # Compute offset
    offset = tl.program_id(0) * BLOCK_SIZE
    # Create indices
    indices = offset + tl.arange(0, BLOCK_SIZE)
    # Mask to prevent out-of-bounds access
    mask = indices < n_elements
    # Load input
    x = tl.load(x_ptr + indices, mask=mask, other=0.0)
    # Apply GELU
    y = 0.5 * x * (1 + tl.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))
    # Store output
    tl.store(y_ptr + indices, y, mask=mask)


@triton.jit
def linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    batch_size, seq_len, input_size, hidden_size,
    BLOCK_SIZE: tl.constexpr
):
    # Block index
    block_idx = tl.program_id(0)
    batch_idx = block_idx // seq_len
    seq_idx = block_idx % seq_len

    # Compute offset
    offset = (batch_idx * seq_len + seq_idx) * input_size
    x = tl.load(x_ptr + offset + tl.arange(0, input_size), mask=tl.arange(0, input_size) < input_size, other=0.0)
    w = tl.load(w_ptr + tl.arange(0, input_size)[:, None] * hidden_size + tl.arange(0, hidden_size)[None, :], mask=tl.arange(0, input_size)[:, None] < input_size, other=0.0)
    b = tl.load(b_ptr + tl.arange(0, hidden_size), mask=tl.arange(0, hidden_size) < hidden_size, other=0.0)

    out = tl.dot(x, w) + b
    offset_out = (batch_idx * seq_len + seq_idx) * hidden_size
    tl.store(out_ptr + offset_out + tl.arange(0, hidden_size), out, mask=tl.arange(0, hidden_size) < hidden_size)


def triton_lstm_cell(x, h, c, W_ih, W_hh, b_ih, b_hh, hidden_size):
    batch_size, seq_len, _ = x.shape
    BLOCK_SIZE = 256
    HIDDEN_SIZE = hidden_size

    # Output tensors
    h_out = torch.empty_like(h)
    c_out = torch.empty_like(c)

    # Grid
    grid = lambda meta: (batch_size * seq_len,)

    # Launch kernel
    lstm_cell_kernel[grid](
        x, h, c,
        W_ih, W_hh, b_ih, b_hh,
        h_out, c_out,
        batch_size, seq_len, hidden_size,
        BLOCK_SIZE=BLOCK_SIZE, HIDDEN_SIZE=HIDDEN_SIZE
    )

    return h_out, c_out


def triton_matmul(a, b):
    m, k = a.shape
    k, n = b.shape
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64

    # Output
    c = torch.empty(m, n, device=a.device, dtype=a.dtype)

    # Grid
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_SIZE_M"]), triton.cdiv(n, meta["BLOCK_SIZE_N"]), triton.cdiv(k, meta["BLOCK_SIZE_K"]))

    # Launch kernel
    matmul_kernel[grid](a, b, c, m, n, k, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K)

    return c


def triton_gelu(x):
    n_elements = x.numel()
    BLOCK_SIZE = 128

    # Output
    y = torch.empty_like(x)

    # Grid
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # Launch kernel
    gelu_kernel[grid](x, y, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return y


def triton_linear(x, w, b):
    batch_size, seq_len, input_size = x.shape
    hidden_size = w.shape[1]

    # Output
    out = torch.empty(batch_size, seq_len, hidden_size, device=x.device, dtype=x.dtype)

    # Grid
    grid = lambda meta: (batch_size * seq_len,)

    # Launch kernel
    linear_kernel[grid](x, w, b, out, batch_size, seq_len, input_size, hidden_size, BLOCK_SIZE=256)

    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # We will fuse the LSTM layer by combining matmul + activation + cell update
        # We use Triton kernels to speed up the computation

        # Create weights
        self.W_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.b_ih = nn.Parameter(torch.randn(4 * hidden_size))
        self.b_hh = nn.Parameter(torch.randn(4 * hidden_size))

        # Linear layer for output
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0=None, c0=None):
        batch_size, seq_len, _ = x.shape
        device = x.device

        if h0 is None:
            h0 = torch.randn(self.num_layers, batch_size, self.hidden_size, device=device)
        if c0 is None:
            c0 = torch.randn(self.num_layers, batch_size, self.hidden_size, device=device)

        # We will process one layer at a time
        h = h0
        c = c0

        # Process each LSTM layer
        for layer_idx in range(self.num_layers):
            # Get layer-specific weights
            W_ih_layer = self.W_ih
            W_hh_layer = self.W_hh
            b_ih_layer = self.b_ih
            b_hh_layer = self.b_hh

            # Get input for current layer
            layer_x = x if layer_idx == 0 else h[layer_idx - 1]

            # Process each sequence step
            h_out = torch.empty_like(h[layer_idx])
            c_out = torch.empty_like(c[layer_idx])

            # Fuse: matmul + GELU + LSTM cell
            for seq_idx in range(seq_len):
                # Extract current timestep
                x_t = layer_x[:, seq_idx, :].contiguous()
                h_t = h[layer_idx, :, :].contiguous()
                c_t = c[layer_idx, :, :].contiguous()

                # Compute gates using fused matmul and activation
                # We use Triton kernels for all operations
                i_h = triton_matmul(h_t, W_hh_layer.T)
                i_x = triton_matmul(x_t, W_ih_layer.T)
                i_sum = i_h + i_x
                i_sum = i_sum + b_ih_layer[None, :] + b_hh_layer[None, :]
                i_sum = triton_gelu(i_sum)  # Optional: could be simplified in Triton

                # Split gates
                i = i_sum[:, :self.hidden_size]
                f = i_sum[:, self.hidden_size:2*self.hidden_size]
                g = i_sum[:, 2*self.hidden_size:3*self.hidden_size]
                o = i_sum[:, 3*self.hidden_size:4*self.hidden_size]

                # Update cell state and hidden state
                c_new = f * c_t + i * g
                h_new = o * torch.tanh(c_new)

                # Store
                h_out[:, seq_idx, :] = h_new
                c_out[:, seq_idx, :] = c_new

            # Update h and c for next layer
            h = h_out
            c = c_out

        # Final linear layer
        out = h[:, -1, :]
        out = triton_linear(out, self.fc.weight, self.fc.bias)

        return out