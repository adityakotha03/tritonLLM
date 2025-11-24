import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def rnn_i2h_kernel(
    x_ptr,         # Input tensor (batch_size, input_size)
    h_ptr,         # Previous hidden state (batch_size, hidden_size)
    i2h_weight_ptr,  # Weight matrix for input-to-hidden (input_size + hidden_size, hidden_size)
    i2h_bias_ptr,    # Bias vector for input-to-hidden (hidden_size)
    h_out_ptr,       # Output hidden state (batch_size, hidden_size)
    batch_size: tl.constexpr,
    input_size: tl.constexpr,
    hidden_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of batch elements
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size

    # Load input and hidden state
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    h = tl.load(h_ptr + offsets, mask=mask, other=0.0)

    # Concatenate input and hidden state
    combined = tl.cat((x, h), dim=1)

    # Load weights and bias
    weight = tl.load(i2h_weight_ptr + (tl.arange(0, input_size + hidden_size) * hidden_size), mask=tl.arange(0, input_size + hidden_size) < (input_size + hidden_size), other=0.0)
    bias = tl.load(i2h_bias_ptr, mask=tl.arange(0, hidden_size) < hidden_size, other=0.0)

    # Matrix multiplication: (batch_size, input_size + hidden_size) @ (input_size + hidden_size, hidden_size)
    # We compute it in a block-wise fashion using fused GEMM pattern
    # We assume weights are stored in row-major format: (in_dim, out_dim)
    # We compute output element by element
    out = tl.zeros((BLOCK_SIZE, hidden_size), dtype=tl.float32)
    for i in range(hidden_size):
        # Compute dot product for each output dimension
        w_i = tl.load(i2h_weight_ptr + (i * (input_size + hidden_size)), mask=tl.arange(0, input_size + hidden_size) < (input_size + hidden_size), other=0.0)
        # Compute dot product with combined input
        dot = tl.dot(combined, w_i)
        out = out + dot
    # Add bias
    out = out + bias

    # Apply Tanh activation
    out = tl.tanh(out)

    # Store output
    tl.store(h_out_ptr + offsets, out, mask=mask)


@triton.jit
def rnn_h2o_kernel(
    h_ptr,         # Hidden state (batch_size, hidden_size)
    h2o_weight_ptr,  # Weight matrix for hidden-to-output (hidden_size, output_size)
    h2o_bias_ptr,    # Bias vector for hidden-to-output (output_size)
    output_ptr,      # Output tensor (batch_size, output_size)
    batch_size: tl.constexpr,
    hidden_size: tl.constexpr,
    output_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size

    # Load hidden state
    h = tl.load(h_ptr + offsets, mask=mask, other=0.0)

    # Load weights and bias
    weight = tl.load(h2o_weight_ptr + (tl.arange(0, hidden_size) * output_size), mask=tl.arange(0, hidden_size) < hidden_size, other=0.0)
    bias = tl.load(h2o_bias_ptr, mask=tl.arange(0, output_size) < output_size, other=0.0)

    # Matrix multiplication: (batch_size, hidden_size) @ (hidden_size, output_size)
    out = tl.zeros((BLOCK_SIZE, output_size), dtype=tl.float32)
    for i in range(output_size):
        w_i = tl.load(h2o_weight_ptr + (i * hidden_size), mask=tl.arange(0, hidden_size) < hidden_size, other=0.0)
        dot = tl.dot(h, w_i)
        out = out + dot
    out = out + bias

    # Store output
    tl.store(output_ptr + offsets, out, mask=mask)


def triton_rnn_i2h(x: torch.Tensor, h: torch.Tensor, i2h_weight: torch.Tensor, i2h_bias: torch.Tensor):
    """
    Custom Triton kernel for input-to-hidden transformation in RNN.
    """
    assert x.is_cuda and h.is_cuda and i2h_weight.is_cuda and i2h_bias.is_cuda, "All tensors must be on CUDA."
    x = x.contiguous()
    h = h.contiguous()
    i2h_weight = i2h_weight.contiguous()
    i2h_bias = i2h_bias.contiguous()

    batch_size = x.shape[0]
    input_size = x.shape[1]
    hidden_size = h.shape[1]

    # Output tensor for hidden state
    h_out = torch.empty_like(h)

    # Grid size
    grid = lambda meta: ((batch_size + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch kernel
    rnn_i2h_kernel[grid](
        x_ptr=x.data_ptr(),
        h_ptr=h.data_ptr(),
        i2h_weight_ptr=i2h_weight.data_ptr(),
        i2h_bias_ptr=i2h_bias.data_ptr(),
        h_out_ptr=h_out.data_ptr(),
        batch_size=batch_size,
        input_size=input_size,
        hidden_size=hidden_size,
        BLOCK_SIZE=256,
    )
    return h_out


def triton_rnn_h2o(h: torch.Tensor, h2o_weight: torch.Tensor, h2o_bias: torch.Tensor):
    """
    Custom Triton kernel for hidden-to-output transformation in RNN.
    """
    assert h.is_cuda and h2o_weight.is_cuda and h2o_bias.is_cuda, "All tensors must be on CUDA."
    h = h.contiguous()
    h2o_weight = h2o_weight.contiguous()
    h2o_bias = h2o_bias.contiguous()

    batch_size = h.shape[0]
    hidden_size = h.shape[1]
    output_size = h2o_weight.shape[1]

    output = torch.empty((batch_size, output_size), device=h.device, dtype=h.dtype)

    grid = lambda meta: ((batch_size + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    rnn_h2o_kernel[grid](
        h_ptr=h.data_ptr(),
        h2o_weight_ptr=h2o_weight.data_ptr(),
        h2o_bias_ptr=h2o_bias.data_ptr(),
        output_ptr=output.data_ptr(),
        batch_size=batch_size,
        hidden_size=hidden_size,
        output_size=output_size,
        BLOCK_SIZE=256,
    )
    return output


class ModelNew(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initialize the optimized RNN model using custom Triton kernels.
        
        :param input_size: The number of input features (int).
        :param hidden_size: The size of the hidden state (int).
        :param output_size: The number of output features (int).
        """
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases
        self.i2h_weight = nn.Parameter(torch.randn((input_size + hidden_size, hidden_size), dtype=torch.float32))
        self.i2h_bias = nn.Parameter(torch.randn(hidden_size, dtype=torch.float32))
        self.h2o_weight = nn.Parameter(torch.randn((hidden_size, output_size), dtype=torch.float32))
        self.h2o_bias = nn.Parameter(torch.randn(output_size, dtype=torch.float32))

        # Initial hidden state (will be set during forward pass)
        self.hidden = torch.randn((1, hidden_size), dtype=torch.float32)

    def forward(self, x: torch.Tensor, initial_hidden=None) -> torch.Tensor:
        """
        Forward pass of the optimized RNN model using custom Triton kernels.
        
        :param x: Input tensor of shape (batch_size, input_size).
        :param initial_hidden: Initial hidden state of shape (batch_size, hidden_size).
        :return: Output tensor of shape (batch_size, output_size).
        """
        batch_size = x.shape[0]

        # Initialize hidden state if needed
        if initial_hidden is not None:
            self.hidden.copy_(initial_hidden)
        self.hidden = self.hidden.to(x.device)

        # Expand hidden state to match batch size
        h = self.hidden.expand(batch_size, -1)

        # Compute new hidden state using Triton kernel
        h_new = triton_rnn_i2h(x, h, self.i2h_weight, self.i2h_bias)

        # Compute output using Triton kernel
        output = triton_rnn_h2o(h_new, self.h2o_weight, self.h2o_bias)

        return output