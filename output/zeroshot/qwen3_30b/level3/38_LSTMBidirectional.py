import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def lstm_cell_kernel(
    x_ptr, h_ptr, c_ptr,
    w_ih_ptr, w_hh_ptr,
    bias_ptr,
    h_out_ptr, c_out_ptr,
    seq_len, hidden_size, input_size,
    batch_size, num_layers, bidirectional,
    BLOCK_SIZE: tl.constexpr,
):
    # Thread block index
    pid = tl.program_id(0)
    block_size = BLOCK_SIZE
    num_blocks = tl.cdiv(seq_len * batch_size, block_size)

    # Calculate offset for this block
    offset = pid * block_size
    seq_idx = offset // batch_size
    batch_idx = offset % batch_size

    # Bounds check
    if seq_idx >= seq_len or batch_idx >= batch_size:
        return

    # Calculate offsets for current (seq, batch)
    x_offset = (seq_idx * batch_size + batch_idx) * input_size
    h_offset = (seq_idx * batch_size + batch_idx) * hidden_size
    c_offset = (seq_idx * batch_size + batch_idx) * hidden_size

    # Load input and previous hidden/cell states
    x = tl.load(x_ptr + x_offset + tl.arange(0, input_size), mask=tl.arange(0, input_size) < input_size)
    h = tl.load(h_ptr + h_offset + tl.arange(0, hidden_size), mask=tl.arange(0, hidden_size) < hidden_size)
    c = tl.load(c_ptr + c_offset + tl.arange(0, hidden_size), mask=tl.arange(0, hidden_size) < hidden_size)

    # Linear projection: input gate, forget gate, cell gate, output gate
    # Weights: (4*hidden_size, input_size + hidden_size)
    # Bias: (4*hidden_size,)
    w_ih = tl.load(w_ih_ptr + tl.arange(0, 4*hidden_size)[:, None] * input_size + tl.arange(0, input_size)[None, :], mask=tl.arange(0, 4*hidden_size)[:, None] < 4*hidden_size)
    w_hh = tl.load(w_hh_ptr + tl.arange(0, 4*hidden_size)[:, None] * hidden_size + tl.arange(0, hidden_size)[None, :], mask=tl.arange(0, 4*hidden_size)[:, None] < 4*hidden_size)
    bias = tl.load(bias_ptr + tl.arange(0, 4*hidden_size), mask=tl.arange(0, 4*hidden_size) < 4*hidden_size)

    # Concatenate x and h
    xh = tl.concatenate((x, h), axis=0)
    # Dot product
    gates = tl.dot(w_ih, xh) + tl.dot(w_hh, h) + bias
    gates = tl.reshape(gates, (4, hidden_size))

    # Apply activation functions
    i = tl.sigmoid(gates[0])
    f = tl.sigmoid(gates[1])
    g = tl.tanh(gates[2])
    o = tl.sigmoid(gates[3])

    # Update cell state
    c_next = f * c + i * g
    # Update hidden state
    h_next = o * tl.tanh(c_next)

    # Store outputs
    tl.store(h_out_ptr + h_offset + tl.arange(0, hidden_size), h_next, mask=tl.arange(0, hidden_size) < hidden_size)
    tl.store(c_out_ptr + c_offset + tl.arange(0, hidden_size), c_next, mask=tl.arange(0, hidden_size) < hidden_size)


@triton.jit
def lstm_layer_kernel(
    x_ptr, h0_ptr, c0_ptr,
    w_ih_ptr, w_hh_ptr, bias_ptr,
    output_ptr,
    seq_len, hidden_size, input_size,
    batch_size, num_layers, bidirectional,
    BLOCK_SIZE: tl.constexpr,
):
    # Block size per thread block
    block_size = BLOCK_SIZE
    num_blocks = tl.cdiv(seq_len * batch_size, block_size)

    # Grid size
    grid = lambda meta: (num_blocks,)

    # Launch the cell kernel
    lstm_cell_kernel[grid](
        x_ptr, h0_ptr, c0_ptr,
        w_ih_ptr, w_hh_ptr, bias_ptr,
        output_ptr, output_ptr + batch_size * seq_len * hidden_size,
        seq_len, hidden_size, input_size,
        batch_size, num_layers, bidirectional,
        BLOCK_SIZE=block_size
    )


@triton.jit
def linear_kernel(
    x_ptr, w_ptr, b_ptr,
    out_ptr,
    n_rows, n_cols, n_features,
    BLOCK_SIZE: tl.constexpr,
):
    # Thread block index
    pid = tl.program_id(0)
    block_size = BLOCK_SIZE
    num_blocks = tl.cdiv(n_rows * n_cols, block_size)

    # Calculate offset for this block
    offset = pid * block_size
    row = offset // n_cols
    col = offset % n_cols

    # Bounds check
    if row >= n_rows or col >= n_cols:
        return

    # Load input and weights
    x = tl.load(x_ptr + row * n_features + tl.arange(0, n_features), mask=tl.arange(0, n_features) < n_features)
    w = tl.load(w_ptr + col * n_features + tl.arange(0, n_features), mask=tl.arange(0, n_features) < n_features)
    b = tl.load(b_ptr + col, mask=col < n_cols)

    # Compute dot product
    out = tl.dot(x, w) + b

    # Store output
    tl.store(out_ptr + row * n_cols + col, out)


@triton.jit
def fusion_kernel(
    x_ptr, w_ih_ptr, w_hh_ptr, bias_ptr,
    w_fc_ptr, b_fc_ptr,
    output_ptr,
    seq_len, hidden_size, input_size, output_size,
    batch_size, num_layers, bidirectional,
    BLOCK_SIZE: tl.constexpr,
):
    # Total elements
    total_elements = seq_len * batch_size

    # Grid for LSTM cell
    grid = lambda meta: ((total_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Shared memory for intermediate states
    shared = tl.shared_memory(shape=(2 * batch_size * hidden_size), dtype=tl.float32)
    h_ptr = shared
    c_ptr = shared + batch_size * hidden_size

    # Initialize hidden and cell states
    h = tl.load(h0_ptr + tl.arange(0, hidden_size)[None, :] + tl.arange(0, batch_size)[:, None] * hidden_size)
    c = tl.load(c0_ptr + tl.arange(0, hidden_size)[None, :] + tl.arange(0, batch_size)[:, None] * hidden_size)
    tl.store(h_ptr, h, mask=tl.arange(0, batch_size * hidden_size) < batch_size * hidden_size)
    tl.store(c_ptr, c, mask=tl.arange(0, batch_size * hidden_size) < batch_size * hidden_size)

    # Process each sequence step
    for step in range(seq_len):
        # Load x
        x = tl.load(x_ptr + (step * batch_size) * input_size + tl.arange(0, input_size)[None, :] + tl.arange(0, batch_size)[:, None] * input_size)
        
        # Load weights
        w_ih = tl.load(w_ih_ptr + tl.arange(0, 4*hidden_size)[:, None] * input_size + tl.arange(0, input_size)[None, :], mask=tl.arange(0, 4*hidden_size)[:, None] < 4*hidden_size)
        w_hh = tl.load(w_hh_ptr + tl.arange(0, 4*hidden_size)[:, None] * hidden_size + tl.arange(0, hidden_size)[None, :], mask=tl.arange(0, 4*hidden_size)[:, None] < 4*hidden_size)
        bias = tl.load(bias_ptr + tl.arange(0, 4*hidden_size), mask=tl.arange(0, 4*hidden_size) < 4*hidden_size)

        # Load h and c from shared memory
        h = tl.load(h_ptr + tl.arange(0, batch_size * hidden_size), mask=tl.arange(0, batch_size * hidden_size) < batch_size * hidden_size)
        h = tl.reshape(h, (batch_size, hidden_size))
        c = tl.load(c_ptr + tl.arange(0, batch_size * hidden_size), mask=tl.arange(0, batch_size * hidden_size) < batch_size * hidden_size)
        c = tl.reshape(c, (batch_size, hidden_size))

        # Compute gates
        xh = tl.concatenate((x, h), axis=1)
        gates = tl.dot(xh, tl.transpose(tl.concatenate((w_ih, w_hh)))) + bias
        gates = tl.reshape(gates, (batch_size, 4, hidden_size))

        i = tl.sigmoid(gates[:, 0])
        f = tl.sigmoid(gates[:, 1])
        g = tl.tanh(gates[:, 2])
        o = tl.sigmoid(gates[:, 3])

        c_next = f * c + i * g
        h_next = o * tl.tanh(c_next)

        # Store back to shared memory
        tl.store(h_ptr, tl.reshape(h_next, (batch_size * hidden_size)), mask=tl.arange(0, batch_size * hidden_size) < batch_size * hidden_size)
        tl.store(c_ptr, tl.reshape(c_next, (batch_size * hidden_size)), mask=tl.arange(0, batch_size * hidden_size) < batch_size * hidden_size)

    # After processing all steps, get last hidden state
    h_last = tl.load(h_ptr + tl.arange(0, batch_size * hidden_size), mask=tl.arange(0, batch_size * hidden_size) < batch_size * hidden_size)
    h_last = tl.reshape(h_last, (batch_size, hidden_size))

    # Apply final linear layer
    w_fc = tl.load(w_fc_ptr + tl.arange(0, output_size)[:, None] * hidden_size + tl.arange(0, hidden_size)[None, :], mask=tl.arange(0, output_size)[:, None] < output_size)
    b_fc = tl.load(b_fc_ptr + tl.arange(0, output_size), mask=tl.arange(0, output_size) < output_size)

    out = tl.dot(h_last, tl.transpose(w_fc)) + b_fc

    # Store final output
    tl.store(output_ptr, out)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout

        # Create LSTM cell weights
        self.w_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.w_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.randn(4 * hidden_size))

        # Create final linear layer
        self.w_fc = nn.Parameter(torch.randn(output_size, hidden_size * 2))
        self.b_fc = nn.Parameter(torch.randn(output_size))

        # Register as buffers for compatibility
        self.register_buffer('w_ih', self.w_ih)
        self.register_buffer('w_hh', self.w_hh)
        self.register_buffer('bias', self.bias)
        self.register_buffer('w_fc', self.w_fc)
        self.register_buffer('b_fc', self.b_fc)

    def forward(self, x, h0, c0):
        batch_size, seq_len, _ = x.shape

        # Fuse LSTM and linear layer into one Triton kernel
        output = torch.empty(batch_size, self.output_size, device=x.device, dtype=x.dtype)

        # Use autotuned block size
        BLOCK_SIZE = 128
        grid = lambda meta: ((seq_len * batch_size + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

        # Launch fusion kernel
        fusion_kernel[grid](
            x, self.w_ih, self.w_hh, self.bias,
            self.w_fc, self.b_fc,
            output,
            seq_len, self.hidden_size, self.input_size, self.output_size,
            batch_size, self.num_layers, True,
            BLOCK_SIZE=BLOCK_SIZE
        )

        return output