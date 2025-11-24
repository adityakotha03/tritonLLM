import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def lstm_cell_kernel(
    x_ptr,          # input: (batch, seq, input_size)
    h_prev_ptr,     # h_prev: (batch, hidden_size)
    c_prev_ptr,     # c_prev: (batch, hidden_size)
    w_ih_ptr,       # weight input to hidden: (4 * hidden_size, input_size)
    w_hh_ptr,       # weight hidden to hidden: (4 * hidden_size, hidden_size)
    b_ih_ptr,       # bias input to hidden: (4 * hidden_size,)
    b_hh_ptr,       # bias hidden to hidden: (4 * hidden_size,)
    out_ptr,        # output: (batch, hidden_size)
    c_out_ptr,      # output cell: (batch, hidden_size)
    batch_size: tl.constexpr,
    hidden_size: tl.constexpr,
    input_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of batch elements
    batch_idx = tl.program_id(0)
    if batch_idx >= batch_size:
        return

    # Load previous hidden and cell states
    h_prev = tl.load(h_prev_ptr + batch_idx * hidden_size, mask=(batch_idx < batch_size), other=0.0)
    c_prev = tl.load(c_prev_ptr + batch_idx * hidden_size, mask=(batch_idx < batch_size), other=0.0)

    # Load input data for this batch
    x = tl.load(x_ptr + batch_idx * input_size, mask=(batch_idx < batch_size), other=0.0)

    # Load weights and biases
    w_ih = tl.load(w_ih_ptr + (batch_idx * input_size) * 4 * hidden_size, mask=(batch_idx < batch_size), other=0.0)
    w_hh = tl.load(w_hh_ptr + (batch_idx * hidden_size) * 4 * hidden_size, mask=(batch_idx < batch_size), other=0.0)
    b_ih = tl.load(b_ih_ptr + batch_idx * 4 * hidden_size, mask=(batch_idx < batch_size), other=0.0)
    b_hh = tl.load(b_hh_ptr + batch_idx * 4 * hidden_size, mask=(batch_idx < batch_size), other=0.0)

    # Unpack weights and biases
    # Weights are stored in 4 * hidden_size: (i, f, g, o)
    # Use vectorized operations for each gate
    w_ih = w_ih.reshape(4 * hidden_size, input_size)
    w_hh = w_hh.reshape(4 * hidden_size, hidden_size)
    b_ih = b_ih.reshape(4 * hidden_size)

    # Compute gates
    # Input gate: i = sigmoid(W_ih * x + W_hh * h + b_ih)
    # Forget gate: f = sigmoid(W_ih * x + W_hh * h + b_ih)
    # Cell gate: g = tanh(W_ih * x + W_hh * h + b_ih)
    # Output gate: o = sigmoid(W_ih * x + W_hh * h + b_ih)

    # Compute inputs to gates
    ixt = tl.dot(x, w_ih)  # (batch, input_size) @ (4*hidden_size, input_size) -> (batch, 4*hidden_size)
    hth = tl.dot(h_prev, w_hh)  # (batch, hidden_size) @ (4*hidden_size, hidden_size) -> (batch, 4*hidden_size)

    # Add biases
    gate_inputs = ixt + hth + b_ih  # (batch, 4*hidden_size)

    # Apply sigmoid and tanh
    i = tl.sigmoid(gate_inputs[:, 0:hidden_size])
    f = tl.sigmoid(gate_inputs[:, hidden_size:2*hidden_size])
    g = tl.tanh(gate_inputs[:, 2*hidden_size:3*hidden_size])
    o = tl.sigmoid(gate_inputs[:, 3*hidden_size:4*hidden_size])

    # Compute new cell state
    c_new = f * c_prev + i * g

    # Compute new hidden state
    h_new = o * tl.tanh(c_new)

    # Store outputs
    tl.store(out_ptr + batch_idx * hidden_size, h_new, mask=(batch_idx < batch_size))
    tl.store(c_out_ptr + batch_idx * hidden_size, c_new, mask=(batch_idx < batch_size))


@triton.jit
def linear_kernel(
    x_ptr,          # input: (batch, seq, hidden_size)
    w_ptr,          # weight: (hidden_size, output_size)
    b_ptr,          # bias: (output_size,)
    out_ptr,        # output: (batch, output_size)
    batch_size: tl.constexpr,
    hidden_size: tl.constexpr,
    output_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of batch elements
    batch_idx = tl.program_id(0)
    if batch_idx >= batch_size:
        return

    # Load input and weights
    x = tl.load(x_ptr + batch_idx * hidden_size, mask=(batch_idx < batch_size), other=0.0)
    w = tl.load(w_ptr + (batch_idx * hidden_size) * output_size, mask=(batch_idx < batch_size), other=0.0)
    b = tl.load(b_ptr, mask=(batch_idx < batch_size), other=0.0)

    # Reshape weights: (hidden_size, output_size)
    w = w.reshape(hidden_size, output_size)

    # Compute output
    out = tl.dot(x, w) + b

    # Store output
    tl.store(out_ptr + batch_idx * output_size, out, mask=(batch_idx < batch_size))


def triton_lstm_cell(x: torch.Tensor, h0: torch.Tensor, c0: torch.Tensor, 
                     w_ih: torch.Tensor, w_hh: torch.Tensor, b_ih: torch.Tensor, b_hh: torch.Tensor):
    """
    Custom LSTM cell implementation using Triton kernel.
    """
    assert x.is_cuda and h0.is_cuda and c0.is_cuda, "All tensors must be on CUDA."
    assert w_ih.is_cuda and w_hh.is_cuda and b_ih.is_cuda and b_hh.is_cuda, "Weights and biases must be on CUDA."

    # Ensure tensors are contiguous
    x = x.contiguous()
    h0 = h0.contiguous()
    c0 = c0.contiguous()
    w_ih = w_ih.contiguous()
    w_hh = w_hh.contiguous()
    b_ih = b_ih.contiguous()
    b_hh = b_hh.contiguous()

    # Prepare output tensors
    h_out = torch.empty_like(h0)
    c_out = torch.empty_like(c0)

    # Define block size
    BLOCK_SIZE = 128

    # Grid size
    grid = lambda meta: ((h0.shape[0] + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch kernel
    lstm_cell_kernel[grid](
        x, h0, c0, w_ih, w_hh, b_ih, b_hh,
        h_out, c_out,
        h0.shape[0], h0.shape[1], x.shape[2],
        BLOCK_SIZE=BLOCK_SIZE
    )

    return h_out, c_out


def triton_linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
    """
    Custom linear layer using Triton kernel.
    """
    assert x.is_cuda and w.is_cuda and b.is_cuda, "All tensors must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()
    b = b.contiguous()

    # Prepare output
    out = torch.empty(x.shape[0], w.shape[1], device=x.device)

    # Define block size
    BLOCK_SIZE = 128

    # Grid size
    grid = lambda meta: ((x.shape[0] + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch kernel
    linear_kernel[grid](
        x, w, b, out,
        x.shape[0], x.shape[1], w.shape[1],
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        """
        Initialize the LSTM model with custom Triton kernels for LSTM cell and linear layer.
        """
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout

        # Initialize weights and biases
        self.w_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.w_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.b_ih = nn.Parameter(torch.randn(4 * hidden_size))
        self.b_hh = nn.Parameter(torch.randn(4 * hidden_size))

        # Final linear layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0, c0):
        """
        Forward pass through the LSTM model with custom Triton kernels.

        :param x: The input tensor, shape (batch_size, sequence_length, input_size)
        :param h0: Initial hidden state, shape (num_layers, batch_size, hidden_size)
        :param c0: Initial cell state, shape (num_layers, batch_size, hidden_size)
        :return: The output tensor, shape (batch_size, output_size)
        """
        batch_size = x.shape[0]
        seq_length = x.shape[1]
        hidden_size = self.hidden_size

        # Reshape h0 and c0 to (num_layers, batch, hidden_size)
        h0 = h0.reshape(self.num_layers, batch_size, hidden_size)
        c0 = c0.reshape(self.num_layers, batch_size, hidden_size)

        # Process each time step
        out = torch.empty((batch_size, seq_length, hidden_size), device=x.device)
        h_t = h0.clone()
        c_t = c0.clone()

        # Forward through each time step
        for t in range(seq_length):
            x_t = x[:, t, :]  # (batch, input_size)

            # Apply LSTM cell using Triton kernel
            h_t, c_t = triton_lstm_cell(
                x_t.unsqueeze(1),  # (batch, 1, input_size)
                h_t[:, 0, :],     # (batch, hidden_size)
                c_t[:, 0, :],     # (batch, hidden_size)
                self.w_ih,        # (4*hidden_size, input_size)
                self.w_hh,        # (4*hidden_size, hidden_size)
                self.b_ih,        # (4*hidden_size)
                self.b_hh         # (4*hidden_size)
            )

        # Extract last time step hidden state
        final_h = h_t[:, -1, :]  # (batch, hidden_size)

        # Apply final linear layer
        out_final = self.fc(final_h)  # (batch, output_size)

        return out_final