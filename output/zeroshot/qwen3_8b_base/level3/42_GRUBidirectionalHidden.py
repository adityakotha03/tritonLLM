import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.num_directions = 2  # Since bidirectional is True by default in the original model

        # Define the custom GRU kernel
        self.gru_kernel = gru_kernel

    def forward(self, x, h0):
        if not self.batch_first:
            x = x.transpose(0, 1)
        batch_size = x.size(1)
        seq_len = x.size(0)
        num_directions = self.num_directions
        num_layers = self.num_layers
        hidden_size = self.hidden_size

        # Ensure h0 is in the correct format
        if h0 is None:
            h0 = torch.zeros(num_layers * num_directions, batch_size, hidden_size, device=x.device)
        else:
            h0 = h0.contiguous()

        # Initialize output
        output = torch.zeros(seq_len, batch_size, num_directions * hidden_size, device=x.device)

        # Process each layer and direction
        for layer in range(num_layers):
            for direction in range(num_directions):
                # Get the current hidden state
                h_prev = h0[layer * num_directions + direction]
                # Process the sequence
                h_current = self.gru_kernel(x, h_prev, hidden_size, self.input_size, self.bias)
                output[:, :, layer * num_directions + direction * hidden_size : (layer + 1) * num_directions + direction * hidden_size] = h_current
                h0[layer * num_directions + direction] = h_current

        if not self.batch_first:
            output = output.transpose(0, 1)

        return output, h0

@triton.jit
def gru_kernel(
    x_ptr,  # Pointer to input tensor
    h_prev_ptr,  # Pointer to previous hidden state
    hidden_size: tl.constexpr,
    input_size: tl.constexpr,
    bias: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < hidden_size

    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    h_prev = tl.load(h_prev_ptr + offsets, mask=mask, other=0.0)

    # Compute the GRU update
    z = tl.sigmoid(x[0:hidden_size] @ tl.reshape(h_prev, (hidden_size, 1)) + bias)
    r = tl.sigmoid(x[hidden_size:2*hidden_size] @ tl.reshape(h_prev, (hidden_size, 1)) + bias)
    h_tilde = tl.tanh(x[2*hidden_size:3*hidden_size] @ tl.reshape(r * h_prev, (hidden_size, 1)) + bias)
    h_current = (1 - z) * h_prev + z * h_tilde

    # Store the result
    tl.store(x_ptr + offsets, h_current, mask=mask)

def gru_kernel_launcher(x, h_prev, hidden_size, input_size, bias):
    # Ensure inputs are contiguous
    x = x.contiguous()
    h_prev = h_prev.contiguous()

    # Prepare output tensor
    output = torch.empty_like(h_prev)

    # Number of elements in the tensor
    n_elements = hidden_size
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    gru_kernel[grid](x, h_prev, hidden_size, input_size, bias, BLOCK_SIZE=BLOCK_SIZE)
    return output