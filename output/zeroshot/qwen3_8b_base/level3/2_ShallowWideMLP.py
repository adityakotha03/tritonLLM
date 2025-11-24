import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def linear_relu_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    bias_ptr,  # Pointer to bias tensor
    output_ptr,  # Pointer to output tensor
    input_size: tl.constexpr,
    output_size: tl.constexpr,
    num_blocks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID
    pid = tl.program_id(0)
    # Compute the block start index
    block_start = pid * BLOCK_SIZE
    # Compute the block end index
    block_end = block_start + BLOCK_SIZE
    # Compute the offset in the input and output
    offsets = tl.arange(0, BLOCK_SIZE)
    # Mask to avoid out-of-bounds access
    mask = offsets < output_size

    # Load input
    input = tl.load(input_ptr + block_start + offsets, mask=mask, other=0.0)
    # Compute linear transformation
    weights = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    # Compute output
    output = input * weights + bias
    # Apply ReLU
    output = tl.maximum(output, 0.0)
    # Store output
    tl.store(output_ptr + block_start + offsets, output, mask=mask)


def triton_linear_relu(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, output_size: int):
    """
    Custom Triton kernel for linear + ReLU operation.
    """
    assert input.is_cuda and weight.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    output = torch.empty(output_size, device=input.device, dtype=input.dtype)

    # Determine block size and grid size
    BLOCK_SIZE = 1024  # Tunable parameter for block size
    num_blocks = (output_size + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    linear_relu_kernel[ num_blocks ](input, weight, bias, output, input_size=input.size(0), output_size=output_size, num_blocks=num_blocks, BLOCK_SIZE=BLOCK_SIZE)
    return output


@triton.jit
def final_linear_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    bias_ptr,  # Pointer to bias tensor
    output_ptr,  # Pointer to output tensor
    input_size: tl.constexpr,
    output_size: tl.constexpr,
    num_blocks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID
    pid = tl.program_id(0)
    # Compute the block start index
    block_start = pid * BLOCK_SIZE
    # Compute the block end index
    block_end = block_start + BLOCK_SIZE
    # Compute the offset in the input and output
    offsets = tl.arange(0, BLOCK_SIZE)
    # Mask to avoid out-of-bounds access
    mask = offsets < output_size

    # Load input
    input = tl.load(input_ptr + block_start + offsets, mask=mask, other=0.0)
    # Compute linear transformation
    weights = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    # Compute output
    output = input * weights + bias
    # Store output
    tl.store(output_ptr + block_start + offsets, output, mask=mask)


def triton_final_linear(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, output_size: int):
    """
    Custom Triton kernel for final linear operation.
    """
    assert input.is_cuda and weight.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    output = torch.empty(output_size, device=input.device, dtype=input.dtype)

    # Determine block size and grid size
    BLOCK_SIZE = 1024  # Tunable parameter for block size
    num_blocks = (output_size + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    final_linear_kernel[ num_blocks ](input, weight, bias, output, input_size=input.size(0), output_size=output_size, num_blocks=num_blocks, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        """
        :param input_size: The number of input features
        :param hidden_layer_sizes: A list of ints containing the sizes of each hidden layer
        :param output_size: The number of output features
        """
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_sizes = hidden_layer_sizes

        # Weights and biases for the hidden layers
        self.hidden_weights = []
        self.hidden_biases = []
        for i, hidden_size in enumerate(hidden_layer_sizes):
            # Weights: (input_size, hidden_size)
            weight = torch.nn.Parameter(torch.randn(hidden_size, input_size).cuda())
            self.register_parameter(f'hidden_weight_{i}', weight)
            # Biases: (hidden_size,)
            bias = torch.nn.Parameter(torch.randn(hidden_size).cuda())
            self.register_parameter(f'hidden_bias_{i}', bias)
            self.hidden_weights.append(weight)
            self.hidden_biases.append(bias)
            input_size = hidden_size

        # Final layer weights and biases
        final_weight = torch.nn.Parameter(torch.randn(output_size, input_size).cuda())
        self.register_parameter('final_weight', final_weight)
        final_bias = torch.nn.Parameter(torch.randn(output_size).cuda())
        self.register_parameter('final_bias', final_bias)

    def forward(self, x):
        """
        :param x: The input tensor, shape (batch_size, input_size)
        :return: The output tensor, shape (batch_size, output_size)
        """
        batch_size = x.size(0)
        # Process each hidden layer
        for i, hidden_size in enumerate(self.hidden_layer_sizes):
            # Get weights and biases
            weight = self.hidden_weights[i]
            bias = self.hidden_biases[i]
            # Reshape input to (batch_size, input_size) for matrix multiplication
            input_flat = x.view(-1, self.input_size)
            # Compute linear + ReLU using Triton kernel
            output_flat = triton_linear_relu(input_flat, weight.t(), bias, hidden_size)
            # Reshape back to (batch_size, hidden_size)
            x = output_flat.view(batch_size, hidden_size)
        # Final layer
        final_weight = self.final_weight
        final_bias = self.final_bias
        # Reshape input to (batch_size, input_size) for matrix multiplication
        input_flat = x.view(-1, self.input_size)
        # Compute final linear using Triton kernel
        output_flat = triton_final_linear(input_flat, final_weight.t(), final_bias, self.output_size)
        # Reshape back to (batch_size, output_size)
        return output_flat.view(batch_size, self.output_size)