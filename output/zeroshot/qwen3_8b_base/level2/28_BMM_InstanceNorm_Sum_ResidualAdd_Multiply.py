import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def bmm_kernel(
    x_ptr,  # Pointer to input x
    y_ptr,  # Pointer to input y
    out_ptr,  # Pointer to output
    n_batch,  # Number of batches
    n_features,  # Number of features
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    pid = tl.program_id(0)
    # Compute the batch index
    batch_idx = pid
    # Compute the offset for the current batch
    offset = batch_idx * n_features
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_features
    # Load input values
    x = tl.load(x_ptr + offset + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offset + offsets, mask=mask, other=0.0)
    # Perform the elementwise multiplication
    out = x * y
    # Store the result
    tl.store(out_ptr + offset + offsets, out, mask=mask)


@triton.jit
def instance_norm_kernel(
    x_ptr,  # Pointer to input x
    mean_ptr,  # Pointer to mean
    var_ptr,  # Pointer to variance
    gamma_ptr,  # Pointer to gamma
    beta_ptr,  # Pointer to beta
    out_ptr,  # Pointer to output
    n_elements,  # Total number of elements in input/output
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    pid = tl.program_id(0)
    # Compute the offset for the current block
    offset = pid * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = offset + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Load mean and variance
    mean = tl.load(mean_ptr, mask=mask, other=0.0)
    var = tl.load(var_ptr, mask=mask, other=0.0)
    gamma = tl.load(gamma_ptr + offsets, mask=mask, other=0.0)
    beta = tl.load(beta_ptr + offsets, mask=mask, other=0.0)
    # Compute normalization
    x_hat = (x - mean) / tl.sqrt(var + eps)
    # Apply gamma and beta
    out = gamma * x_hat + beta
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def add_mul_kernel(
    x_ptr,  # Pointer to input x
    y_ptr,  # Pointer to input y
    out_ptr,  # Pointer to output
    n_elements,  # Total number of elements in input/output
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    pid = tl.program_id(0)
    # Compute the offset for the current block
    offset = pid * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = offset + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    # Perform the elementwise addition and multiplication
    out = (x + y) * y
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_bmm(x: torch.Tensor, y: torch.Tensor):
    """
    This function wraps the Triton kernel call for batch matrix multiplication.
    """
    assert x.is_cuda and y.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    y = y.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    n_batch = x.shape[0]
    n_features = x.shape[1]

    # Determine the number of blocks needed
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    bmm_kernel[grid](x, y, out, n_batch, n_features, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_instance_norm(x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor):
    """
    This function wraps the Triton kernel call for instance normalization.
    """
    assert x.is_cuda and mean.is_cuda and var.is_cuda and gamma.is_cuda and beta.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    mean = mean.contiguous()
    var = var.contiguous()
    gamma = gamma.contiguous()
    beta = beta.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()

    # Determine the number of blocks needed
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    instance_norm_kernel[grid](x, mean, var, gamma, beta, out, n_elements, eps=1e-5, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_add_mul(x: torch.Tensor, y: torch.Tensor):
    """
    This function wraps the Triton kernel call for addition and multiplication.
    """
    assert x.is_cuda and y.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    y = y.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()

    # Determine the number of blocks needed
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    add_mul_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.momentum = momentum

        # Register buffers for instance normalization
        self.register_buffer('running_mean', torch.zeros(out_features))
        self.register_buffer('running_var', torch.ones(out_features))
        self.gamma = nn.Parameter(torch.ones(out_features))
        self.beta = nn.Parameter(torch.zeros(out_features))

    def forward(self, x, y):
        # Batch matrix multiplication with Triton
        x = triton_bmm(x, y)
        # Instance normalization with Triton
        x = triton_instance_norm(x, self.running_mean, self.running_var, self.gamma, self.beta)
        # Addition and multiplication with Triton
        x = triton_add_mul(x, y)
        return x