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
    n_elements,  # Total number of elements in input/output
    input_size,  # Size of input features
    output_size,  # Size of output features
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    input = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # Compute matrix multiplication
    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(input_size):
        weight = tl.load(weight_ptr + i * output_size + offsets, mask=mask, other=0.0)
        accumulator += input * weight
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
        accumulator += bias

    # Apply ReLU
    accumulator = tl.maximum(accumulator, 0.0)

    # Store output
    tl.store(output_ptr + offsets, accumulator, mask=mask)


def triton_linear_relu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, output_size: int):
    """
    This function wraps the Triton kernel call for linear + ReLU.
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    # Prepare output tensor
    output = torch.empty((x.shape[0], output_size), dtype=x.dtype, device=x.device)

    # Number of elements in the tensor
    n_elements = x.shape[0] * output_size
    BLOCK_SIZE = 1024  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    linear_relu_kernel[grid](x, weight, bias, output, n_elements, x.shape[1], output_size, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        """
        :param input_size: The number of input features
        :param layer_sizes: A list of ints containing the sizes of each hidden layer
        :param output_size: The number of output features
        """
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.layer_sizes = layer_sizes

        self.layers = nn.ModuleList()
        current_input_size = input_size

        for layer_size in layer_sizes:
            self.layers.append(nn.Linear(current_input_size, layer_size))
            self.layers.append(nn.Linear(layer_size, layer_size))  # Replace ReLU with linear + custom kernel
            current_input_size = layer_size

        self.final_layer = nn.Linear(current_input_size, output_size)

    def forward(self, x):
        """
        :param x: The input tensor, shape (batch_size, input_size)
        :return: The output tensor, shape (batch_size, output_size)
        """
        for i in range(len(self.layers)):
            if i % 2 == 0:
                x = self.layers[i](x)
            else:
                weight = self.layers[i].weight
                bias = self.layers[i].bias
                x = triton_linear_relu(x, weight, bias, self.layer_sizes[i // 2])
        x = self.final_layer(x)
        return x