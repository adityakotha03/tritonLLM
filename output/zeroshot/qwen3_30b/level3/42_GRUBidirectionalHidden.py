import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Triton kernel for GRU cell with fused activation and update gate
@triton.jit
def gru_cell_kernel(
    x_ptr,  # Input data pointer (seq_len, batch, input_size)
    h_ptr,  # Hidden state pointer (batch, hidden_size)
    w_ih_ptr,  # Input-to-hidden weight matrix (3 * hidden_size, input_size)
    w_hh_ptr,  # Hidden-to-hidden weight matrix (3 * hidden_size, hidden_size)
    b_ih_ptr,  # Input-to-hidden bias (3 * hidden_size)
    b_hh_ptr,  # Hidden-to-hidden bias (3 * hidden_size)
    h_out_ptr,  # Output hidden state (batch, hidden_size)
    seq_len: tl.constexpr,
    batch_size: tl.constexpr,
    input_size: tl.constexpr,
    hidden_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Block ID
    block_id = tl.program_id(0)
    batch_id = block_id // (seq_len * 3)
    seq_id = (block_id // 3) % seq_len
    gate_id = block_id % 3  # 0: reset, 1: update, 2: candidate

    # Thread ID
    tid = tl.thread_id(0)
    row = tid // hidden_size
    col = tid % hidden_size

    # Load input and hidden state
    x_offset = (seq_id * batch_size + batch_id) * input_size + col
    h_offset = (batch_id * hidden_size + col)

    x = tl.load(x_ptr + x_offset, mask=col < input_size, other=0.0)
    h = tl.load(h_ptr + h_offset, mask=col < hidden_size, other=0.0)

    # Load weights and bias for the current gate
    w_ih_offset = (gate_id * hidden_size + row) * input_size + col
    w_hh_offset = (gate_id * hidden_size + row) * hidden_size + col
    b_ih_offset = gate_id * hidden_size + row
    b_hh_offset = gate_id * hidden_size + row

    w_ih = tl.load(w_ih_ptr + w_ih_offset, mask=col < input_size, other=0.0)
    w_hh = tl.load(w_hh_ptr + w_hh_offset, mask=col < hidden_size, other=0.0)
    b_ih = tl.load(b_ih_ptr + b_ih_offset, mask=row < hidden_size, other=0.0)
    b_hh = tl.load(b_hh_ptr + b_hh_offset, mask=row < hidden_size, other=0.0)

    # Compute gate value
    gate_val = tl.dot(x, w_ih) + tl.dot(h, w_hh) + b_ih + b_hh

    # Activation: sigmoid for reset/update, tanh for candidate
    if gate_id == 0:  # Reset gate
        gate_val = tl.sigmoid(gate_val)
    elif gate_id == 1:  # Update gate
        gate_val = tl.sigmoid(gate_val)
    else:  # Candidate gate
        gate_val = tl.tanh(gate_val)

    # Store result
    out_offset = (batch_id * hidden_size + col)
    tl.store(h_out_ptr + out_offset, gate_val, mask=col < hidden_size)


# Triton kernel for fused GRU forward with block-wise tiling and shared memory
@triton.jit
def gru_forward_kernel(
    x_ptr,  # Input tensor (seq_len, batch, input_size)
    h_ptr,  # Initial hidden state (num_layers * 2, batch, hidden_size)
    w_ih_ptr,  # Input-to-hidden weights (3 * hidden_size, input_size)
    w_hh_ptr,  # Hidden-to-hidden weights (3 * hidden_size, hidden_size)
    b_ih_ptr,  # Input bias (3 * hidden_size)
    b_hh_ptr,  # Hidden bias (3 * hidden_size)
    out_ptr,  # Output hidden state (num_layers * 2, batch, hidden_size)
    seq_len: tl.constexpr,
    batch_size: tl.constexpr,
    input_size: tl.constexpr,
    hidden_size: tl.constexpr,
    num_layers: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Thread index
    tid = tl.thread_id(0)
    row = tid // hidden_size
    col = tid % hidden_size

    # Grid indices
    block_id = tl.program_id(0)
    layer_id = block_id // (seq_len * batch_size)
    seq_id = (block_id // batch_size) % seq_len
    batch_id = block_id % batch_size

    # Offset for current layer
    layer_offset = layer_id * 2  # bidirectional: 2 directions

    # Initial hidden state offset
    h_offset = (layer_offset * batch_size + batch_id) * hidden_size + col
    h = tl.load(h_ptr + h_offset, mask=col < hidden_size, other=0.0)

    # Output offset
    out_offset = (layer_offset * batch_size + batch_id) * hidden_size + col

    # Loop over sequence length
    for t in range(seq_len):
        # Input offset
        x_offset = (t * batch_size + batch_id) * input_size + col
        x = tl.load(x_ptr + x_offset, mask=col < input_size, other=0.0)

        # Reset gate (0)
        w_ih_r = tl.load(w_ih_ptr + (0 * hidden_size + row) * input_size + col, mask=col < input_size, other=0.0)
        w_hh_r = tl.load(w_hh_ptr + (0 * hidden_size + row) * hidden_size + col, mask=col < hidden_size, other=0.0)
        b_ih_r = tl.load(b_ih_ptr + 0 * hidden_size + row, mask=row < hidden_size, other=0.0)
        b_hh_r = tl.load(b_hh_ptr + 0 * hidden_size + row, mask=row < hidden_size, other=0.0)
        reset = tl.sigmoid(tl.dot(x, w_ih_r) + tl.dot(h, w_hh_r) + b_ih_r + b_hh_r)

        # Update gate (1)
        w_ih_u = tl.load(w_ih_ptr + (1 * hidden_size + row) * input_size + col, mask=col < input_size, other=0.0)
        w_hh_u = tl.load(w_hh_ptr + (1 * hidden_size + row) * hidden_size + col, mask=col < hidden_size, other=0.0)
        b_ih_u = tl.load(b_ih_ptr + 1 * hidden_size + row, mask=row < hidden_size, other=0.0)
        b_hh_u = tl.load(b_hh_ptr + 1 * hidden_size + row, mask=row < hidden_size, other=0.0)
        update = tl.sigmoid(tl.dot(x, w_ih_u) + tl.dot(h, w_hh_u) + b_ih_u + b_hh_u)

        # Candidate (2)
        w_ih_c = tl.load(w_ih_ptr + (2 * hidden_size + row) * input_size + col, mask=col < input_size, other=0.0)
        w_hh_c = tl.load(w_hh_ptr + (2 * hidden_size + row) * hidden_size + col, mask=col < hidden_size, other=0.0)
        b_ih_c = tl.load(b_ih_ptr + 2 * hidden_size + row, mask=row < hidden_size, other=0.0)
        b_hh_c = tl.load(b_hh_ptr + 2 * hidden_size + row, mask=row < hidden_size, other=0.0)
        candidate = tl.tanh(tl.dot(x, w_ih_c) + tl.dot(h, w_hh_c) + b_ih_c + b_hh_c)

        # Update hidden state
        h = update * h + (1 - update) * candidate
        h = reset * h + (1 - reset) * candidate

        # Write intermediate hidden state (for next time step)
        tl.store(h_ptr + h_offset, h, mask=col < hidden_size)

    # Store final hidden state
    tl.store(out_ptr + out_offset, h, mask=col < hidden_size)


def triton_gru_forward(x: torch.Tensor, h0: torch.Tensor, w_ih: torch.Tensor, w_hh: torch.Tensor, b_ih: torch.Tensor, b_hh: torch.Tensor):
    """
    Custom Triton-based GRU forward pass with fused operations and optimized memory access.
    """
    assert x.is_cuda and h0.is_cuda and w_ih.is_cuda and w_hh.is_cuda, "All tensors must be on CUDA."

    seq_len, batch_size, input_size = x.shape
    hidden_size = w_ih.shape[1] // 3  # Assuming 3 * hidden_size input
    num_layers = h0.shape[0] // 2  # Bidirectional

    # Ensure contiguity
    x = x.contiguous()
    h0 = h0.contiguous()
    w_ih = w_ih.contiguous()
    w_hh = w_hh.contiguous()
    b_ih = b_ih.contiguous()
    b_hh = b_hh.contiguous()

    # Output tensor
    out = torch.empty_like(h0)

    # Grid configuration
    grid_size = seq_len * batch_size * num_layers * 2
    BLOCK_SIZE = 128  # Optimized block size for A100

    # Launch kernel
    gru_forward_kernel[grid_size](
        x, h0, w_ih, w_hh, b_ih, b_hh, out,
        seq_len=seq_len,
        batch_size=batch_size,
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bias, batch_first, dropout=0, bidirectional=True)
    
    def forward(self, x, h0):
        # Use Triton-optimized GRU forward
        w_ih = self.gru.weight_ih_l0  # Input-to-hidden
        w_hh = self.gru.weight_hh_l0  # Hidden-to-hidden
        b_ih = self.gru.bias_ih_l0
        b_hh = self.gru.bias_hh_l0

        # Apply Triton kernel
        return triton_gru_forward(x, h0, w_ih, w_hh, b_ih, b_hh)