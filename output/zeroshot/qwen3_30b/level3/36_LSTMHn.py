import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Triton kernel for fused LSTM cell computation
@triton.jit
def lstm_cell_kernel(
    x_ptr, h_ptr, c_ptr,
    w_ih_ptr, w_hh_ptr,
    out_h_ptr, out_c_ptr,
    batch_size, seq_len, hidden_size,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    # Thread block index
    pid = tl.program_id(0)
    block_id = pid % (seq_len * batch_size)
    layer_id = pid // (seq_len * batch_size)

    # Calculate offsets for this thread block
    seq_idx = block_id // batch_size
    batch_idx = block_id % batch_size

    # Each block processes one element of the sequence
    # For each time step and batch, we process one hidden unit
    offset_h = batch_idx * hidden_size + seq_idx * hidden_size * batch_size
    offset_c = batch_idx * hidden_size + seq_idx * hidden_size * batch_size

    # Load input and hidden state
    x = tl.load(x_ptr + offset_h, mask=tl.arange(0, hidden_size) < hidden_size, other=0.0)
    h = tl.load(h_ptr + offset_h, mask=tl.arange(0, hidden_size) < hidden_size, other=0.0)
    c = tl.load(c_ptr + offset_c, mask=tl.arange(0, hidden_size) < hidden_size, other=0.0)

    # Get input weights and hidden weights
    w_ih = tl.load(w_ih_ptr + layer_id * hidden_size * 4 * hidden_size + tl.arange(0, hidden_size) * 4 * hidden_size + tl.arange(0, 4 * hidden_size))
    w_hh = tl.load(w_hh_ptr + layer_id * hidden_size * 4 * hidden_size + tl.arange(0, hidden_size) * 4 * hidden_size + tl.arange(0, 4 * hidden_size))

    # Reshape weights for computation
    w_ih_i = w_ih[:hidden_size]
    w_ih_f = w_ih[hidden_size:2*hidden_size]
    w_ih_c = w_ih[2*hidden_size:3*hidden_size]
    w_ih_o = w_ih[3*hidden_size:]

    w_hh_i = w_hh[:hidden_size]
    w_hh_f = w_hh[hidden_size:2*hidden_size]
    w_hh_c = w_hh[2*hidden_size:3*hidden_size]
    w_hh_o = w_hh[3*hidden_size:]

    # Compute gates
    i = tl.sigmoid(tl.dot(x, w_ih_i) + tl.dot(h, w_hh_i))
    f = tl.sigmoid(tl.dot(x, w_ih_f) + tl.dot(h, w_hh_f))
    c_tilde = tl.tanh(tl.dot(x, w_ih_c) + tl.dot(h, w_hh_c))
    o = tl.sigmoid(tl.dot(x, w_ih_o) + tl.dot(h, w_hh_o))

    # Update cell state
    c_new = f * c + i * c_tilde

    # Update hidden state
    h_new = o * tl.tanh(c_new)

    # Store output
    tl.store(out_h_ptr + offset_h, h_new, mask=tl.arange(0, hidden_size) < hidden_size)
    tl.store(out_c_ptr + offset_c, c_new, mask=tl.arange(0, hidden_size) < hidden_size)


# Triton kernel for full LSTM layer (batched and sequence-length optimized)
@triton.jit
def lstm_layer_kernel(
    x_ptr, h_ptr, c_ptr,
    w_ih_ptr, w_hh_ptr,
    out_h_ptr, out_c_ptr,
    batch_size, seq_len, hidden_size,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    pid = tl.program_id(0)
    block_id = pid % (seq_len * batch_size)
    layer_id = pid // (seq_len * batch_size)

    # Get current time step and batch
    seq_idx = block_id // batch_size
    batch_idx = block_id % batch_size

    # Shared memory for input and output
    shmem = tl.load(tl.pointer_type(tl.float16), (0, 0))
    # Load input and hidden state
    x = tl.load(x_ptr + batch_idx * seq_len * hidden_size + seq_idx * hidden_size, mask=tl.arange(0, hidden_size) < hidden_size, other=0.0)
    h = tl.load(h_ptr + layer_id * batch_size * hidden_size + batch_idx * hidden_size, mask=tl.arange(0, hidden_size) < hidden_size, other=0.0)
    c = tl.load(c_ptr + layer_id * batch_size * hidden_size + batch_idx * hidden_size, mask=tl.arange(0, hidden_size) < hidden_size, other=0.0)

    # Load weight matrices
    w_ih = tl.load(w_ih_ptr + layer_id * 4 * hidden_size * hidden_size, mask=tl.arange(0, 4 * hidden_size * hidden_size) < 4 * hidden_size * hidden_size, other=0.0)
    w_hh = tl.load(w_hh_ptr + layer_id * 4 * hidden_size * hidden_size, mask=tl.arange(0, 4 * hidden_size * hidden_size) < 4 * hidden_size * hidden_size, other=0.0)

    # Reshape to separate gates
    w_ih_i = w_ih[:hidden_size]
    w_ih_f = w_ih[hidden_size:2*hidden_size]
    w_ih_c = w_ih[2*hidden_size:3*hidden_size]
    w_ih_o = w_ih[3*hidden_size:]

    w_hh_i = w_hh[:hidden_size]
    w_hh_f = w_hh[hidden_size:2*hidden_size]
    w_hh_c = w_hh[2*hidden_size:3*hidden_size]
    w_hh_o = w_hh[3*hidden_size:]

    # Compute gates
    i = tl.sigmoid(tl.dot(x, w_ih_i) + tl.dot(h, w_hh_i))
    f = tl.sigmoid(tl.dot(x, w_ih_f) + tl.dot(h, w_hh_f))
    c_tilde = tl.tanh(tl.dot(x, w_ih_c) + tl.dot(h, w_hh_c))
    o = tl.sigmoid(tl.dot(x, w_ih_o) + tl.dot(h, w_hh_o))

    # Update cell and hidden states
    c_new = f * c + i * c_tilde
    h_new = o * tl.tanh(c_new)

    # Write back
    tl.store(out_h_ptr + layer_id * batch_size * hidden_size + batch_idx * hidden_size, h_new, mask=tl.arange(0, hidden_size) < hidden_size)
    tl.store(out_c_ptr + layer_id * batch_size * hidden_size + batch_idx * hidden_size, c_new, mask=tl.arange(0, hidden_size) < hidden_size)


# Triton kernel for fused linear layer
@triton.jit
def linear_kernel(
    x_ptr, w_ptr, out_ptr,
    batch_size, seq_len, hidden_size, output_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_id = pid % (batch_size * seq_len)
    seq_idx = block_id // batch_size
    batch_idx = block_id % batch_size

    # Output block
    offset = batch_idx * seq_len * output_size + seq_idx * output_size

    # Load input
    x = tl.load(x_ptr + batch_idx * seq_len * hidden_size + seq_idx * hidden_size, mask=tl.arange(0, hidden_size) < hidden_size, other=0.0)

    # Process output in blocks
    for i in range(0, output_size, BLOCK_SIZE):
        # Output indices
        out_offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = out_offsets < output_size

        # Load weights
        w = tl.load(w_ptr + i * hidden_size, mask=tl.arange(0, hidden_size) < hidden_size, other=0.0)

        # Compute dot product
        out = tl.dot(x, w)

        # Store output
        tl.store(out_ptr + offset + i, out, mask=mask)


# Triton kernel for fused LSTM forward with multiple layers
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=4, num_warps=4),
    ],
    key=['hidden_size', 'seq_len'],
)
@triton.jit
def lstm_forward_kernel(
    x_ptr, h0_ptr, c0_ptr,
    w_ih_ptr, w_hh_ptr,
    out_h_ptr, out_c_ptr,
    batch_size, seq_len, hidden_size, num_layers,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    # Grid setup
    num_blocks = batch_size * seq_len * num_layers
    pid = tl.program_id(0)

    # Decompose block ID
    layer_id = pid // (batch_size * seq_len)
    block_id = pid % (batch_size * seq_len)
    seq_idx = block_id // batch_size
    batch_idx = block_id % batch_size

    # Get input and initial states
    x = tl.load(x_ptr + batch_idx * seq_len * hidden_size + seq_idx * hidden_size, mask=tl.arange(0, hidden_size) < hidden_size, other=0.0)
    h = tl.load(h0_ptr + layer_id * batch_size * hidden_size + batch_idx * hidden_size, mask=tl.arange(0, hidden_size) < hidden_size, other=0.0)
    c = tl.load(c0_ptr + layer_id * batch_size * hidden_size + batch_idx * hidden_size, mask=tl.arange(0, hidden_size) < hidden_size, other=0.0)

    # Load weights
    w_ih = tl.load(w_ih_ptr + layer_id * 4 * hidden_size * hidden_size + tl.arange(0, 4 * hidden_size * hidden_size), mask=tl.arange(0, 4 * hidden_size * hidden_size) < 4 * hidden_size * hidden_size, other=0.0)
    w_hh = tl.load(w_hh_ptr + layer_id * 4 * hidden_size * hidden_size + tl.arange(0, 4 * hidden_size * hidden_size), mask=tl.arange(0, 4 * hidden_size * hidden_size) < 4 * hidden_size * hidden_size, other=0.0)

    # Split weights
    w_ih_i = w_ih[:hidden_size]
    w_ih_f = w_ih[hidden_size:2*hidden_size]
    w_ih_c = w_ih[2*hidden_size:3*hidden_size]
    w_ih_o = w_ih[3*hidden_size:]

    w_hh_i = w_hh[:hidden_size]
    w_hh_f = w_hh[hidden_size:2*hidden_size]
    w_hh_c = w_hh[2*hidden_size:3*hidden_size]
    w_hh_o = w_hh[3*hidden_size:]

    # Compute gates
    i = tl.sigmoid(tl.dot(x, w_ih_i) + tl.dot(h, w_hh_i))
    f = tl.sigmoid(tl.dot(x, w_ih_f) + tl.dot(h, w_hh_f))
    c_tilde = tl.tanh(tl.dot(x, w_ih_c) + tl.dot(h, w_hh_c))
    o = tl.sigmoid(tl.dot(x, w_ih_o) + tl.dot(h, w_hh_o))

    # Update cell and hidden states
    c_new = f * c + i * c_tilde
    h_new = o * tl.tanh(c_new)

    # Store output
    tl.store(out_h_ptr + layer_id * batch_size * hidden_size + batch_idx * hidden_size, h_new, mask=tl.arange(0, hidden_size) < hidden_size)
    tl.store(out_c_ptr + layer_id * batch_size * hidden_size + batch_idx * hidden_size, c_new, mask=tl.arange(0, hidden_size) < hidden_size)


# Wrapper for Triton-based LSTM forward
def triton_lstm_forward(x, h0, c0, w_ih, w_hh, num_layers, hidden_size, batch_size, seq_len):
    # Ensure inputs are in FP16 or BF16 for tensor core utilization
    x = x.to(torch.float16)
    h0 = h0.to(torch.float16)
    c0 = c0.to(torch.float16)
    w_ih = w_ih.to(torch.float16)
    w_hh = w_hh.to(torch.float16)

    # Allocate output tensors
    h_out = torch.zeros_like(h0)
    c_out = torch.zeros_like(c0)

    # Grid setup
    num_blocks = batch_size * seq_len * num_layers
    grid = lambda meta: (num_blocks,)

    # Launch kernel
    lstm_forward_kernel[grid](
        x, h0, c0,
        w_ih, w_hh,
        h_out, c_out,
        batch_size, seq_len, hidden_size, num_layers,
        BLOCK_SIZE=128,
        BLOCK_SIZE_H=128,
    )

    return h_out, c_out


# Wrapper for Triton-based linear layer
def triton_linear_forward(x, w, output_size, batch_size, seq_len):
    x = x.to(torch.float16)
    w = w.to(torch.float16)

    # Output tensor
    out = torch.empty(batch_size, seq_len, output_size, dtype=torch.float16, device=x.device)

    # Grid
    num_blocks = batch_size * seq_len
    grid = lambda meta: (num_blocks,)

    # Launch
    linear_kernel[grid](
        x, w, out,
        batch_size, seq_len, w.size(1), output_size,
        BLOCK_SIZE=128
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # Initialize weights as nn.Parameter
        self.w_ih_list = nn.ParameterList()
        self.w_hh_list = nn.ParameterList()

        for _ in range(num_layers):
            # Weight matrices: 4 gates, each gate has hidden_size x input_size
            w_ih = torch.randn(4 * hidden_size, input_size, dtype=torch.float16, requires_grad=True)
            w_hh = torch.randn(4 * hidden_size, hidden_size, dtype=torch.float16, requires_grad=True)
            self.w_ih_list.append(w_ih)
            self.w_hh_list.append(w_hh)

        # Output layer
        self.fc_weight = nn.Parameter(torch.randn(output_size, hidden_size, dtype=torch.float16, requires_grad=True))

    def forward(self, x, h0, c0):
        batch_size, seq_len, _ = x.shape

        # Initialize output tensors
        h_out = h0.clone()
        c_out = c0.clone()

        # Process each layer
        for layer in range(self.num_layers):
            w_ih = self.w_ih_list[layer]
            w_hh = self.w_hh_list[layer]
            h_out, c_out = triton_lstm_forward(
                x, h0, c0, w_ih, w_hh, self.num_layers,
                self.hidden_size, batch_size, seq_len
            )

            # Update h0 and c0 for next layer
            h0 = h_out
            c0 = c_out

        # Last layer outputs
        out_last = h_out[:, -1, :]  # (batch_size, hidden_size)

        # Final linear layer
        out = triton_linear_forward(out_last, self.fc_weight, self.output_size, batch_size, 1)

        return out