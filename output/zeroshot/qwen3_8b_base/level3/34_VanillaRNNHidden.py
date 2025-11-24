import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def rnn_cell_kernel(
    x_ptr,  # Pointer to input tensor (seq_len, batch_size, input_size)
    h_prev_ptr,  # Pointer to previous hidden state (batch_size, hidden_size)
    h_out_ptr,  # Pointer to output hidden state (batch_size, hidden_size)
    out_ptr,  # Pointer to output tensor (seq_len, batch_size, output_size)
    seq_len,  # Sequence length
    batch_size,  # Batch size
    input_size,  # Input size
    hidden_size,  # Hidden size
    output_size,  # Output size
    i2h_weights_ptr,  # Pointer to i2h weights (input_size + hidden_size, hidden_size)
    h2o_weights_ptr,  # Pointer to h2o weights (hidden_size, output_size)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of data of size BLOCK_SIZE
    pid = tl.program_id(0)
    block_idx = pid // batch_size
    block_offset = pid % batch_size
    block_start = block_idx * batch_size + block_offset

    # Compute the offset for the current input and hidden state
    x_offset = block_idx * batch_size * input_size + block_offset * input_size
    h_prev_offset = block_offset * hidden_size
    h_out_offset = h_prev_offset
    out_offset = block_idx * batch_size * output_size + block_offset * output_size

    # Load input and previous hidden state
    x = tl.load(x_ptr + x_offset, mask=tl.arange(0, input_size) < input_size, other=0.0)
    h_prev = tl.load(h_prev_ptr + h_prev_offset, mask=tl.arange(0, hidden_size) < hidden_size, other=0.0)

    # Concatenate input and hidden state
    combined = tl.concatenate([x, h_prev], dim=0)

    # Load i2h weights
    i2h_weights = tl.load(i2h_weights_ptr + tl.arange(0, input_size + hidden_size) * hidden_size + tl.arange(0, hidden_size), mask=tl.arange(0, input_size + hidden_size) < (input_size + hidden_size) and tl.arange(0, hidden_size) < hidden_size, other=0.0)

    # Compute hidden state
    hidden = tl.dot(combined, i2h_weights)
    hidden = tl.tanh(hidden)

    # Load h2o weights
    h2o_weights = tl.load(h2o_weights_ptr + tl.arange(0, hidden_size) * output_size + tl.arange(0, output_size), mask=tl.arange(0, hidden_size) < hidden_size and tl.arange(0, output_size) < output_size, other=0.0)

    # Compute output
    output = tl.dot(hidden, h2o_weights)

    # Store output hidden state and output
    tl.store(h_out_ptr + h_out_offset, hidden, mask=tl.arange(0, hidden_size) < hidden_size)
    tl.store(out_ptr + out_offset, output, mask=tl.arange(0, output_size) < output_size)


def triton_rnn_cell(x: torch.Tensor, h_prev: torch.Tensor, i2h_weights: torch.Tensor, h2o_weights: torch.Tensor):
    """
    This function wraps the Triton kernel call for the RNN cell.
    """
    assert x.is_cuda and h_prev.is_cuda and i2h_weights.is_cuda and h2o_weights.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    h_prev = h_prev.contiguous()
    i2h_weights = i2h_weights.contiguous()
    h2o_weights = h2o_weights.contiguous()

    # Prepare output tensors
    h_out = torch.empty_like(h_prev)
    out = torch.empty(x.size(0), x.size(1), h2o_weights.size(1), device=x.device)

    # Calculate grid size
    num_blocks = (x.size(0) * x.size(1) + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks,)

    # Launch the Triton kernel
    rnn_cell_kernel[grid](
        x, h_prev, h_out, out,
        x.size(0), x.size(1), x.size(2),
        h_prev.size(1), h2o_weights.size(1),
        i2h_weights.size(0), i2h_weights.size(1),
        h2o_weights.size(0), h2o_weights.size(1),
        BLOCK_SIZE=128
    )
    return h_out, out


class ModelNew(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Define the RNN cell components (input to hidden, hidden to output)
        self.i2h_weights = nn.Parameter(torch.randn(input_size + hidden_size, hidden_size))
        self.h2o_weights = nn.Parameter(torch.randn(hidden_size, output_size))

    def forward(self, x: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the RNN with custom Triton kernel.
        """
        seq_len, batch_size, _ = x.size()
        hidden = h0.to(x.device)
        outputs = torch.empty(seq_len, batch_size, self.output_size, device=x.device)

        for t in range(seq_len):
            h_prev = hidden
            h_prev = h_prev.contiguous()
            x_t = x[t].contiguous()
            i2h_weights = self.i2h_weights.contiguous()
            h2o_weights = self.h2o_weights.contiguous()

            h_out, out = triton_rnn_cell(x_t, h_prev, i2h_weights, h2o_weights)
            outputs[t] = out
            hidden = h_out

        return outputs