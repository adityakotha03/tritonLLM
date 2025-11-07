import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def linear_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    batch_size,
    input_size,
    output_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Each block processes one row of the output (one batch element)
    # Use the first axis for batch and second for output size
    pid_batch = tl.program_id(0)
    pid_output = tl.program_id(1)

    # Handle cases where output_size is not divisible by BLOCK_SIZE
    block_start = pid_output * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_size

    # Compute the start index for the current batch in the input
    batch_start = pid_batch * input_size
    x_offsets = batch_start + tl.arange(0, input_size)

    # Load the input batch row
    x = tl.load(x_ptr + x_offsets, mask=tl.arange(0, input_size) < input_size, other=0.0)

    # Load the corresponding weight row (transposed, so each output dim has its own weights)
    w_row_start = pid_output * input_size
    w = tl.load(w_ptr + w_row_start + tl.arange(0, input_size), mask=tl.arange(0, input_size) < input_size, other=0.0)

    # Perform matrix-vector multiplication: x @ w.T
    # Use dot product across input_size dimension
    acc = tl.zeros(BLOCK_SIZE, dtype=tl.float32)
    for i in range(0, input_size, BLOCK_SIZE):
        i_offsets = i + tl.arange(0, BLOCK_SIZE)
        x_load = tl.load(x_ptr + x_offsets + i, mask=(i_offsets < input_size), other=0.0)
        w_load = tl.load(w_ptr + w_row_start + i_offsets, mask=(i_offsets < input_size), other=0.0)
        acc += x_load * w_load

    # Apply bias if present
    bias = tl.load(b_ptr + offsets, mask=mask, other=0.0)

    # Write output
    out = acc + bias
    tl.store(out_ptr + (pid_batch * output_size + offsets), out, mask=mask)


@triton.jit
def relu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.maximum(x, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def matmul_relu_fused_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    batch_size,
    input_size,
    output_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_output = tl.program_id(1)

    # Work on one output dimension block at a time
    block_start = pid_output * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_size

    # Load input batch row
    batch_start = pid_batch * input_size
    x_offsets = batch_start + tl.arange(0, input_size)
    x = tl.load(x_ptr + x_offsets, mask=tl.arange(0, input_size) < input_size, other=0.0)

    # Load weight row
    w_row_start = pid_output * input_size
    w = tl.load(w_ptr + w_row_start + tl.arange(0, input_size), mask=tl.arange(0, input_size) < input_size, other=0.0)

    # Compute dot product in blocks
    acc = tl.zeros(BLOCK_SIZE, dtype=tl.float32)
    for i in range(0, input_size, BLOCK_SIZE):
        i_offsets = i + tl.arange(0, BLOCK_SIZE)
        x_load = tl.load(x_ptr + x_offsets + i, mask=(i_offsets < input_size), other=0.0)
        w_load = tl.load(w_ptr + w_row_start + i_offsets, mask=(i_offsets < input_size), other=0.0)
        acc += x_load * w_load

    # Apply bias and ReLU in one step
    bias = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    out = tl.maximum(acc + bias, 0.0)
    tl.store(out_ptr + (pid_batch * output_size + offsets), out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['input_size', 'output_size'],
)
def triton_linear(x, w, b, output_size, BLOCK_SIZE=128):
    # Ensure inputs are contiguous on GPU
    x = x.contiguous()
    w = w.contiguous()
    b = b.contiguous()

    batch_size, input_size = x.shape
    out = torch.empty(batch_size, output_size, device=x.device, dtype=x.dtype)

    grid = lambda meta: (batch_size, (meta['output_size'] + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'])

    # Launch kernel
    linear_kernel[grid](x, w, b, out, batch_size, input_size, output_size, BLOCK_SIZE=BLOCK_SIZE)
    return out


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['n_elements'],
)
def triton_relu(x, n_elements, BLOCK_SIZE=128):
    out = torch.empty_like(x)
    grid = lambda meta: ((meta['n_elements'] + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    relu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['input_size', 'output_size'],
)
def triton_linear_relu(x, w, b, output_size, BLOCK_SIZE=128):
    # Ensure inputs are contiguous
    x = x.contiguous()
    w = w.contiguous()
    b = b.contiguous()

    batch_size, input_size = x.shape
    out = torch.empty(batch_size, output_size, device=x.device, dtype=x.dtype)

    grid = lambda meta: (batch_size, (meta['output_size'] + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'])

    matmul_relu_fused_kernel[grid](x, w, b, out, batch_size, input_size, output_size, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        super().__init__()
        
        layers = []
        current_input_size = input_size
        
        for hidden_size in hidden_layer_sizes:
            # Replace nn.Linear + nn.ReLU with fused Triton kernel
            linear = nn.Linear(current_input_size, hidden_size, bias=True)
            layers.append(linear)
            current_input_size = hidden_size

        # Last layer doesn't have ReLU
        last_linear = nn.Linear(current_input_size, output_size, bias=True)
        layers.append(last_linear)

        self.network = nn.Sequential(*layers)
        
        # Store input sizes for fusion
        self.input_sizes = [input_size] + hidden_layer_sizes
        self.output_sizes = hidden_layer_sizes + [output_size]

    def forward(self, x):
        # Process each layer
        for i, layer in enumerate(self.network):
            # If it's a linear layer with ReLU (not the last one), fuse linear and ReLU
            if i < len(self.network) - 1:
                # Get the weights and bias
                weight = layer.weight
                bias = layer.bias
                output_size = self.output_sizes[i]
                
                # Use Triton fused linear + ReLU
                x = triton_linear_relu(x, weight, bias, output_size)
            else:
                # Final linear layer only
                weight = layer.weight
                bias = layer.bias
                output_size = self.output_sizes[i]
                x = triton_linear(x, weight, bias, output_size)
        
        return x