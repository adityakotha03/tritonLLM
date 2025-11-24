import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gru_cell_kernel(
    input_ptr,  # Pointer to input tensor
    hidden_ptr,  # Pointer to hidden state
    output_ptr,  # Pointer to output tensor
    weight_ih_ptr,  # Pointer to input-hidden weights
    weight_hh_ptr,  # Pointer to hidden-hidden weights
    bias_ih_ptr,  # Pointer to input bias
    bias_hh_ptr,  # Pointer to hidden bias
    seq_len_ptr,  # Pointer to sequence length
    batch_size_ptr,  # Pointer to batch size
    input_size_ptr,  # Pointer to input size
    hidden_size_ptr,  # Pointer to hidden size
    num_layers_ptr,  # Pointer to number of layers
    num_blocks,  # Number of blocks
    BLOCK_SIZE: tl.constexpr,
):
    # Get block index
    block_idx = tl.program_id(0)
    # Get thread index within block
    thread_idx = tl.program_id(1)
    # Compute the offset for this thread
    offset = thread_idx + tl.program_id(2) * tl.num_programs(1)
    # Get the current sequence index
    seq_idx = tl.load(seq_len_ptr + offset)
    # Get the current batch index
    batch_idx = tl.load(batch_size_ptr + offset)
    # Get the input size and hidden size
    input_size = tl.load(input_size_ptr)
    hidden_size = tl.load(hidden_size_ptr)
    # Get the number of layers
    num_layers = tl.load(num_layers_ptr)
    # Compute the offset within the input tensor
    input_offset = seq_idx * batch_size * input_size + batch_idx * input_size
    # Compute the offset within the hidden tensor
    hidden_offset = seq_idx * batch_size * hidden_size + batch_idx * hidden_size
    # Compute the offset within the output tensor
    output_offset = seq_idx * batch_size * hidden_size + batch_idx * hidden_size
    # Load input
    x = tl.load(input_ptr + input_offset, mask=offset < seq_len_ptr, other=0.0)
    # Load hidden
    h = tl.load(hidden_ptr + hidden_offset, mask=offset < seq_len_ptr, other=0.0)
    # Compute gates
    # Input to hidden weights
    w_ih = tl.load(weight_ih_ptr + offset * hidden_size * input_size, mask=offset < num_layers_ptr, other=0.0)
    # Hidden to hidden weights
    w_hh = tl.load(weight_hh_ptr + offset * hidden_size * hidden_size, mask=offset < num_layers_ptr, other=0.0)
    # Input bias
    b_ih = tl.load(bias_ih_ptr + offset * hidden_size, mask=offset < num_layers_ptr, other=0.0)
    # Hidden bias
    b_hh = tl.load(bias_hh_ptr + offset * hidden_size, mask=offset < num_layers_ptr, other=0.0)
    # Compute gates
    # Reset gate: r = sigmoid(W_ir * x + W_hr * h + b_r)
    r = tl.sigmoid(tl.dot(x, w_ih) + tl.dot(h, w_hh) + b_ih)
    # Update gate: z = sigmoid(W_iz * x + W_hz * h + b_z)
    z = tl.sigmoid(tl.dot(x, w_ih) + tl.dot(h, w_hh) + b_hh)
    # New hidden state: h' = tanh(W_ih * x + W_hh * (r .* h) + b_h)
    h_tilde = tl.tanh(tl.dot(x, w_ih) + tl.dot(r * h, w_hh) + b_ih)
    # Final hidden state: h = (1 - z) .* h + z .* h_tilde
    h_new = (1 - z) * h + z * h_tilde
    # Store output
    tl.store(output_ptr + output_offset, h_new, mask=offset < seq_len_ptr)


def triton_gru_cell(input, hidden, weight_ih, weight_hh, bias_ih, bias_hh, seq_len, batch_size, input_size, hidden_size, num_layers):
    """
    This function wraps the Triton kernel call for the GRU cell.
    """
    # Ensure inputs are on the GPU
    assert input.is_cuda and hidden.is_cuda and weight_ih.is_cuda and weight_hh.is_cuda and bias_ih.is_cuda and bias_hh.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    hidden = hidden.contiguous()
    weight_ih = weight_ih.contiguous()
    weight_hh = weight_hh.contiguous()
    bias_ih = bias_ih.contiguous()
    bias_hh = bias_hh.contiguous()

    # Prepare output tensor
    output = torch.empty_like(hidden)

    # Number of elements in the tensor
    n_elements = seq_len * batch_size * hidden_size
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    gru_cell_kernel[triton.next_power_of_two(num_blocks), triton.next_power_of_two(BLOCK_SIZE), triton.next_power_of_two(seq_len)](input, hidden, output, weight_ih, weight_hh, bias_ih, bias_hh, seq_len, batch_size, input_size, hidden_size, num_layers, num_blocks, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        """
        :param input_size: The number of expected features in the input x
        :param hidden_size: The number of features in the hidden state h
        :param num_layers: Number of recurrent layers (default: 1)
        :param bias: If False, then the layer does not use bias weights b_ih and b_hh (default: True)
        :param batch_first: If True, then the input and output tensors are provided as (batch, seq, feature) (default: False)
        """
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first

        # Initialize weights and biases
        self.weight_ih = nn.Parameter(torch.randn(num_layers, hidden_size, input_size, dtype=torch.float16))
        self.weight_hh = nn.Parameter(torch.randn(num_layers, hidden_size, hidden_size, dtype=torch.float16))
        self.bias_ih = nn.Parameter(torch.randn(num_layers, hidden_size, dtype=torch.float16)) if bias else None
        self.bias_hh = nn.Parameter(torch.randn(num_layers, hidden_size, dtype=torch.float16)) if bias else None

    def forward(self, x, h0):
        """
        :param x: The input tensor, shape (seq_len, batch_size, input_size) if batch_first=False, otherwise (batch_size, seq_len, input_size)
        :param h0: The initial hidden state for the input sequence, shape (num_layers, batch_size, hidden_size) (default: None)
        :return: output, h_n
            - output: The output features (h_t) from the last layer of the GRU, for each t, shape (seq_len, batch_size, num_directions * hidden_size) if batch_first=False, otherwise (batch_size, seq_len, num_directions * hidden_size)
            - h_n: The hidden state for t = seq_len, shape (num_layers, batch_size, hidden_size)
        """
        if not self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, input_size = x.size()
        hidden_size = self.hidden_size
        num_layers = self.num_layers

        # Ensure input and hidden sizes match
        assert input_size == self.input_size, "Input size mismatch"
        assert hidden_size == self.hidden_size, "Hidden size mismatch"
        assert num_layers == self.num_layers, "Number of layers mismatch"

        # Initialize hidden state if not provided
        if h0 is None:
            h0 = torch.zeros(num_layers, batch_size, hidden_size, device=x.device, dtype=x.dtype)

        # Process each layer
        h_n = h0
        output = []
        for layer in range(num_layers):
            # Extract weights and biases for this layer
            w_ih = self.weight_ih[layer]
            w_hh = self.weight_hh[layer]
            b_ih = self.bias_ih[layer] if self.bias else None
            b_hh = self.bias_hh[layer] if self.bias else None

            # Process each sequence step
            for seq_idx in range(seq_len):
                # Extract input and hidden for this sequence
                x_t = x[seq_idx]
                h_t = h_n[layer]

                # Compute GRU cell
                h_t = triton_gru_cell(x_t, h_t, w_ih, w_hh, b_ih, b_hh, seq_len, batch_size, input_size, hidden_size, 1)

                # Store output
                output.append(h_t)

            # Update hidden state for next layer
            h_n[layer] = h_t

        # Stack outputs
        output = torch.stack(output, dim=0)

        # Transpose back if batch_first
        if not self.batch_first:
            output = output.transpose(0, 1)

        return output, h_n