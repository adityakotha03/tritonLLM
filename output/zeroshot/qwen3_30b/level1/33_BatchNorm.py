import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def bn_forward_kernel(
    x_ptr,
    mean_ptr,
    inv_std_ptr,
    out_ptr,
    batch_size,
    num_features,
    height,
    width,
    x_stride0,
    x_stride1,
    x_stride2,
    x_stride3,
    out_stride0,
    out_stride1,
    out_stride2,
    out_stride3,
    BLOCK_SIZE: tl.constexpr,
):
    # Block ID
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE

    # Calculate the total number of elements per feature
    total_elements = height * width
    total_elements_per_feature = total_elements * batch_size

    # Process a block of elements for each feature
    for feature_idx in range(num_features):
        # Calculate the starting index for this feature across all elements
        feature_start = feature_idx * total_elements_per_feature

        # Compute the offset for this block within the feature
        start_idx = block_start + feature_start
        offsets = start_idx + tl.arange(0, BLOCK_SIZE)

        # Create mask to handle boundary conditions
        mask = offsets < (start_idx + total_elements_per_feature)

        # Load input data
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

        # Load mean and inverse standard deviation for this feature
        mean = tl.load(mean_ptr + feature_idx)
        inv_std = tl.load(inv_std_ptr + feature_idx)

        # Normalize
        x = (x - mean) * inv_std

        # Store output
        tl.store(out_ptr + offsets, x, mask=mask)


@triton.jit
def bn_backward_kernel(
    x_ptr,
    grad_out_ptr,
    mean_ptr,
    inv_std_ptr,
    grad_input_ptr,
    batch_size,
    num_features,
    height,
    width,
    x_stride0,
    x_stride1,
    x_stride2,
    x_stride3,
    grad_out_stride0,
    grad_out_stride1,
    grad_out_stride2,
    grad_out_stride3,
    grad_input_stride0,
    grad_input_stride1,
    grad_input_stride2,
    grad_input_stride3,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE

    total_elements = height * width
    total_elements_per_feature = total_elements * batch_size

    for feature_idx in range(num_features):
        feature_start = feature_idx * total_elements_per_feature
        start_idx = block_start + feature_start
        offsets = start_idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < (start_idx + total_elements_per_feature)

        # Load input and gradient output
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0)

        # Load statistics
        mean = tl.load(mean_ptr + feature_idx)
        inv_std = tl.load(inv_std_ptr + feature_idx)

        # Compute mean and variance for gradient
        x_centered = x - mean
        grad_scale = inv_std * grad_out

        # Compute gradient w.r.t. input
        grad_input = grad_scale

        # Store gradient
        tl.store(grad_input_ptr + offsets, grad_input, mask=mask)


def triton_batch_norm_forward(x: torch.Tensor, mean: torch.Tensor, inv_std: torch.Tensor):
    # Ensure input is contiguous on GPU
    assert x.is_cuda and mean.is_cuda and inv_std.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    mean = mean.contiguous()
    inv_std = inv_std.contiguous()

    # Prepare output
    out = torch.empty_like(x)

    # Get dimensions
    batch_size, num_features, height, width = x.shape
    x_stride0, x_stride1, x_stride2, x_stride3 = x.stride()
    out_stride0, out_stride1, out_stride2, out_stride3 = out.stride()

    # Determine number of elements
    total_elements = height * width
    total_elements_per_feature = total_elements * batch_size
    n_elements = total_elements_per_feature * num_features

    # Choose block size
    BLOCK_SIZE = 128
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch kernel
    bn_forward_kernel[
        grid_size
    ](
        x,
        mean,
        inv_std,
        out,
        batch_size,
        num_features,
        height,
        width,
        x_stride0,
        x_stride1,
        x_stride2,
        x_stride3,
        out_stride0,
        out_stride1,
        out_stride2,
        out_stride3,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def triton_batch_norm_backward(x: torch.Tensor, grad_out: torch.Tensor, mean: torch.Tensor, inv_std: torch.Tensor):
    # Ensure inputs are contiguous
    assert x.is_cuda and grad_out.is_cuda and mean.is_cuda and inv_std.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    grad_out = grad_out.contiguous()
    mean = mean.contiguous()
    inv_std = inv_std.contiguous()

    # Prepare output
    grad_input = torch.empty_like(x)

    # Get dimensions
    batch_size, num_features, height, width = x.shape
    x_stride0, x_stride1, x_stride2, x_stride3 = x.stride()
    grad_out_stride0, grad_out_stride1, grad_out_stride2, grad_out_stride3 = grad_out.stride()
    grad_input_stride0, grad_input_stride1, grad_input_stride2, grad_input_stride3 = grad_input.stride()

    # Total number of elements
    total_elements = height * width
    total_elements_per_feature = total_elements * batch_size
    n_elements = total_elements_per_feature * num_features

    # Block size
    BLOCK_SIZE = 128
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch kernel
    bn_backward_kernel[
        grid_size
    ](
        x,
        grad_out,
        mean,
        inv_std,
        grad_input,
        batch_size,
        num_features,
        height,
        width,
        x_stride0,
        x_stride1,
        x_stride2,
        x_stride3,
        grad_out_stride0,
        grad_out_stride1,
        grad_out_stride2,
        grad_out_stride3,
        grad_input_stride0,
        grad_input_stride1,
        grad_input_stride2,
        grad_input_stride3,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return grad_input


class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        # Register buffers for running mean and var (just for consistency)
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

        # Parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

        # Store input shape info
        self.num_features = num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # We are replacing BatchNorm2d with Triton-based computation
        # Compute mean and var along batch, height, and width
        # Using Triton for efficient reduction
        with torch.no_grad():
            mean = torch.mean(x, dim=[0, 2, 3], keepdim=True)
            var = torch.var(x, dim=[0, 2, 3], keepdim=True, unbiased=False)
            # Compute inverse standard deviation
            inv_std = torch.rsqrt(var + 1e-5)

        # Scale and shift
        x = (x - mean) * inv_std
        x = x * self.weight.view(1, self.num_features, 1, 1) + self.bias.view(1, self.num_features, 1, 1)

        # Store running statistics
        self.running_mean = 0.9 * self.running_mean + 0.1 * mean.squeeze()
        self.running_var = 0.9 * self.running_var + 0.1 * var.squeeze()
        self.num_batches_tracked += 1

        # Use Triton kernel for forward
        return triton_batch_norm_forward(x, mean.squeeze(), inv_std.squeeze())

    def backward(self, x: torch.Tensor, grad_out: torch.Tensor) -> torch.Tensor:
        # Use Triton for backward pass
        with torch.no_grad():
            mean = torch.mean(x, dim=[0, 2, 3], keepdim=True)
            var = torch.var(x, dim=[0, 2, 3], keepdim=True, unbiased=False)
            inv_std = torch.rsqrt(var + 1e-5)

        # Reshape for compatibility
        mean = mean.squeeze()
        inv_std = inv_std.squeeze()

        # Compute gradient w.r.t. input
        grad_input = triton_batch_norm_backward(x, grad_out, mean, inv_std)

        # Scale gradient by weight
        grad_input *= self.weight.view(1, self.num_features, 1, 1)

        return grad_input