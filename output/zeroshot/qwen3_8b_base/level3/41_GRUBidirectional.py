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
        self.num_directions = 2  # Since bidirectional is True

        # Define Triton kernels for GRU operations
        self._init_gru_kernels()

    def _init_gru_kernels(self):
        # Define kernel for GRU cell
        @triton.jit
        def gru_cell_kernel(
            input_ptr,  # Pointer to input tensor
            weight_ih_ptr,  # Pointer to input-hidden weights
            weight_hh_ptr,  # Pointer to hidden-hidden weights
            bias_ih_ptr,  # Pointer to input bias
            bias_hh_ptr,  # Pointer to hidden bias
            output_ptr,  # Pointer to output tensor
            h_prev_ptr,  # Pointer to previous hidden state
            seq_len,  # Sequence length
            batch_size,  # Batch size
            hidden_size,  # Hidden size
            num_directions,  # Number of directions
            BLOCK_SIZE: tl.constexpr,
            DTYPE: tl.constexpr
        ):
            # Each program processes a block of data
            pid = tl.program_id(0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)

            # Mask to avoid out-of-bounds
            mask = offsets < seq_len * batch_size * hidden_size

            # Load input
            input_val = tl.load(input_ptr + offsets, mask=mask, other=0.0)
            h_prev = tl.load(h_prev_ptr + offsets, mask=mask, other=0.0)

            # Compute gates
            # Gate 1: reset gate
            r = tl.sigmoid(input_val + weight_ih_ptr[0] * h_prev + bias_ih_ptr[0])
            # Gate 2: update gate
            z = tl.sigmoid(input_val + weight_ih_ptr[1] * h_prev + bias_ih_ptr[1])
            # Gate 3: new hidden state
            h_tilda = tl.tanh(input_val + weight_ih_ptr[2] * (r * h_prev) + bias_ih_ptr[2])
            # Final hidden state
            h_new = (1 - z) * h_prev + z * h_tilda

            # Store output
            tl.store(output_ptr + offsets, h_new, mask=mask)

        # Define wrapper function for GRU cell
        def gru_cell(input_tensor, weight_ih, weight_hh, bias_ih, bias_hh, h_prev):
            # Ensure input is contiguous
            input_tensor = input_tensor.contiguous()
            h_prev = h_prev.contiguous()
            output = torch.empty_like(input_tensor)
            seq_len = input_tensor.size(0)
            batch_size = input_tensor.size(1)
            hidden_size = input_tensor.size(2)
            num_directions = self.num_directions
            DTYPE = input_tensor.dtype

            # Determine block size
            BLOCK_SIZE = 128
            grid = lambda meta: ((seq_len * batch_size * hidden_size + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

            # Launch kernel
            gru_cell_kernel[grid](input_tensor, weight_ih, weight_hh, bias_ih, bias_hh, output, h_prev, seq_len, batch_size, hidden_size, num_directions, BLOCK_SIZE=BLOCK_SIZE, DTYPE=DTYPE)
            return output

        # Define kernel for GRU layer
        @triton.jit
        def gru_layer_kernel(
            input_ptr,  # Pointer to input tensor
            weight_ih_ptr,  # Pointer to input-hidden weights
            weight_hh_ptr,  # Pointer to hidden-hidden weights
            bias_ih_ptr,  # Pointer to input bias
            bias_hh_ptr,  # Pointer to hidden bias
            output_ptr,  # Pointer to output tensor
            h_prev_ptr,  # Pointer to previous hidden state
            seq_len,  # Sequence length
            batch_size,  # Batch size
            hidden_size,  # Hidden size
            num_layers,  # Number of layers
            num_directions,  # Number of directions
            BLOCK_SIZE: tl.constexpr,
            DTYPE: tl.constexpr
        ):
            # Each program processes a block of data
            pid = tl.program_id(0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)

            # Mask to avoid out-of-bounds
            mask = offsets < seq_len * batch_size * hidden_size

            # Load input
            input_val = tl.load(input_ptr + offsets, mask=mask, other=0.0)
            h_prev = tl.load(h_prev_ptr + offsets, mask=mask, other=0.0)

            # Compute gates
            # Gate 1: reset gate
            r = tl.sigmoid(input_val + weight_ih_ptr[0] * h_prev + bias_ih_ptr[0])
            # Gate 2: update gate
            z = tl.sigmoid(input_val + weight_ih_ptr[1] * h_prev + bias_ih_ptr[1])
            # Gate 3: new hidden state
            h_tilda = tl.tanh(input_val + weight_ih_ptr[2] * (r * h_prev) + bias_ih_ptr[2])
            # Final hidden state
            h_new = (1 - z) * h_prev + z * h_tilda

            # Store output
            tl.store(output_ptr + offsets, h_new, mask=mask)

        # Define wrapper function for GRU layer
        def gru_layer(input_tensor, weight_ih, weight_hh, bias_ih, bias_hh, h_prev):
            # Ensure input is contiguous
            input_tensor = input_tensor.contiguous()
            h_prev = h_prev.contiguous()
            output = torch.empty_like(input_tensor)
            seq_len = input_tensor.size(0)
            batch_size = input_tensor.size(1)
            hidden_size = input_tensor.size(2)
            num_directions = self.num_directions
            DTYPE = input_tensor.dtype

            # Determine block size
            BLOCK_SIZE = 128
            grid = lambda meta: ((seq_len * batch_size * hidden_size + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

            # Launch kernel
            gru_layer_kernel[grid](input_tensor, weight_ih, weight_hh, bias_ih, bias_hh, output, h_prev, seq_len, batch_size, hidden_size, num_layers, num_directions, BLOCK_SIZE=BLOCK_SIZE, DTYPE=DTYPE)
            return output

        # Set the GRU functions
        self.gru_cell = gru_cell
        self.gru_layer = gru_layer

    def forward(self, x, h0):
        # Ensure input is contiguous
        x = x.contiguous()
        h0 = h0.contiguous()

        # Handle batch_first
        if not self.batch_first:
            x = x.transpose(0, 1)
            h0 = h0.transpose(0, 1)

        # Initialize output
        output = torch.empty_like(x)

        # Initialize hidden states
        h_prev = h0

        # Process each sequence step
        for t in range(x.size(0)):
            # Extract current input
            x_t = x[t]

            # Apply GRU cell
            h_prev = self.gru_cell(x_t, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh, h_prev)

        # Transpose back if batch_first
        if not self.batch_first:
            output = output.transpose(0, 1)
            h_prev = h_prev.transpose(0, 1)

        return output, h_prev