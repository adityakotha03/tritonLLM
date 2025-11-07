import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def lstm_cell_kernel(
    x_ptr, h_ptr, c_ptr, 
    W_ih_ptr, W_hh_ptr,
    bias_ptr,
    h_out_ptr, c_out_ptr,
    batch_size: tl.int32,
    seq_len: tl.int32,
    input_size: tl.int32,
    hidden_size: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    # Thread index
    pid = tl.program_id(0)
    tid = tl.program_id(1)
    block_size = BLOCK_SIZE

    # Grid: batch * seq_len * num_layers
    total_elements = batch_size * seq_len * hidden_size
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    # Shared memory for storing input and hidden state
    # We will split computation per cell
    for i in range(0, seq_len):
        # Load input
        x_offset = i * batch_size * input_size + tid * input_size
        x = tl.load(x_ptr + x_offset, mask=mask, other=0.0)

        # Load hidden state
        h_offset = i * batch_size * hidden_size + tid * hidden_size
        h = tl.load(h_ptr + h_offset, mask=mask, other=0.0)

        # Load cell state
        c_offset = i * batch_size * hidden_size + tid * hidden_size
        c = tl.load(c_ptr + c_offset, mask=mask, other=0.0)

        # Compute gates: input, forget, output, candidate
        # Linear layers (W_ih @ x + W_hh @ h + bias)
        gates = tl.zeros((4 * hidden_size,), dtype=tl.float32)

        # Compute input gate (i)
        for j in range(0, input_size):
            w_ih_i = tl.load(W_ih_ptr + (j * hidden_size * 4 + 0 * hidden_size) + tid * hidden_size)
            w_hh_i = tl.load(W_hh_ptr + (j * hidden_size * 4 + 0 * hidden_size) + tid * hidden_size)
            gates[0 * hidden_size + j] += w_ih_i * x[j] + w_hh_i * h[j]

        # Compute forget gate (f)
        for j in range(0, input_size):
            w_ih_f = tl.load(W_ih_ptr + (j * hidden_size * 4 + 1 * hidden_size) + tid * hidden_size)
            w_hh_f = tl.load(W_hh_ptr + (j * hidden_size * 4 + 1 * hidden_size) + tid * hidden_size)
            gates[1 * hidden_size + j] += w_ih_f * x[j] + w_hh_f * h[j]

        # Compute output gate (o)
        for j in range(0, input_size):
            w_ih_o = tl.load(W_ih_ptr + (j * hidden_size * 4 + 2 * hidden_size) + tid * hidden_size)
            w_hh_o = tl.load(W_hh_ptr + (j * hidden_size * 4 + 2 * hidden_size) + tid * hidden_size)
            gates[2 * hidden_size + j] += w_ih_o * x[j] + w_hh_o * h[j]

        # Compute candidate cell state (g)
        for j in range(0, input_size):
            w_ih_g = tl.load(W_ih_ptr + (j * hidden_size * 4 + 3 * hidden_size) + tid * hidden_size)
            w_hh_g = tl.load(W_hh_ptr + (j * hidden_size * 4 + 3 * hidden_size) + tid * hidden_size)
            gates[3 * hidden_size + j] += w_ih_g * x[j] + w_hh_g * h[j]

        # Add bias
        for j in range(4 * hidden_size):
            bias_val = tl.load(bias_ptr + j, mask=(j < 4 * hidden_size), other=0.0)
            gates[j] += bias_val

        # Apply sigmoid and tanh
        i_gate = tl.sigmoid(gates[0 * hidden_size:1 * hidden_size])
        f_gate = tl.sigmoid(gates[1 * hidden_size:2 * hidden_size])
        o_gate = tl.sigmoid(gates[2 * hidden_size:3 * hidden_size])
        g = tl.tanh(gates[3 * hidden_size:4 * hidden_size])

        # Update cell state
        c_next = f_gate * c + i_gate * g

        # Update hidden state
        h_next = o_gate * tl.tanh(c_next)

        # Store output
        h_out_offset = i * batch_size * hidden_size + tid * hidden_size
        c_out_offset = i * batch_size * hidden_size + tid * hidden_size
        tl.store(h_out_ptr + h_out_offset, h_next, mask=mask)
        tl.store(c_out_ptr + c_out_offset, c_next, mask=mask)


@triton.jit
def linear_kernel(
    x_ptr, w_ptr, out_ptr,
    batch_size: tl.int32,
    seq_len: tl.int32,
    input_size: tl.int32,
    output_size: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Assume row-major: (batch, seq, input_size)
    # Only last time step
    idx = offsets % (batch_size * seq_len)
    batch_idx = idx // seq_len
    seq_idx = idx % seq_len

    # Only last sequence index
    if seq_idx != (seq_len - 1):
        return

    # Compute output
    for i in range(output_size):
        acc = tl.zeros((1,), dtype=tl.float32)
        for j in range(input_size):
            x_val = tl.load(x_ptr + (batch_idx * seq_len * input_size + (seq_len - 1) * input_size + j), mask=(j < input_size))
            w_val = tl.load(w_ptr + (i * input_size + j), mask=(j < input_size))
            acc += x_val * w_val
        out_offset = batch_idx * output_size + i
        tl.store(out_ptr + out_offset, acc, mask=(i < output_size))


def triton_lstm_forward(x, h0, c0, W_ih, W_hh, bias):
    batch_size, seq_len, input_size = x.shape
    hidden_size = W_ih.shape[1] // 4
    num_layers = h0.shape[0]

    # Ensure contiguous tensors
    x = x.contiguous()
    h0 = h0.contiguous()
    c0 = c0.contiguous()
    W_ih = W_ih.contiguous()
    W_hh = W_hh.contiguous()
    bias = bias.contiguous()

    # Initialize output states
    h_out = torch.zeros_like(h0)
    c_out = torch.zeros_like(c0)

    # Define grid and block size
    BLOCK_SIZE = 256
    grid = lambda meta: (batch_size * seq_len * num_layers + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"]

    # Launch LSTM cell kernel
    lstm_cell_kernel[grid](
        x, h0, c0,
        W_ih, W_hh,
        bias,
        h_out, c_out,
        batch_size, seq_len, input_size, hidden_size, BLOCK_SIZE=BLOCK_SIZE
    )

    # Return final cell state
    return h_out[-1], c_out[-1]


def triton_linear_forward(x, w, b):
    batch_size, seq_len, input_size = x.shape
    output_size = w.shape[0]
    x_last = x[:, -1, :]  # Shape: (batch_size, input_size)
    out = torch.zeros(batch_size, output_size, device=x.device, dtype=x.dtype)

    # Use Triton kernel
    BLOCK_SIZE = 256
    grid = lambda meta: (batch_size + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"]

    linear_kernel[grid](
        x_last, w, out,
        batch_size, seq_len, input_size, output_size, BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=False)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, h0, c0):
        # Get weights
        W_ih = self.lstm.weight_ih_l0
        W_hh = self.lstm.weight_hh_l0
        bias = self.lstm.bias_ih_l0 + self.lstm.bias_hh_l0

        # Use Triton kernels for LSTM and linear layers
        h_out, c_out = triton_lstm_forward(x, h0, c0, W_ih, W_hh, bias)
        out = triton_linear_forward(x, self.fc.weight, self.fc.bias)
        return c_out[-1]