import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def batchnorm2d_kernel(
    x_ptr, 
    weight_ptr, 
    bias_ptr, 
    mean_ptr, 
    var_ptr, 
    out_ptr,
    batch_size, 
    num_features, 
    height, 
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Block size is per thread block
    # Each block processes BLOCK_SIZE elements
    block_id = tl.program_id(0)
    # Create a linear index for each element
    offset = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Total number of elements
    total_elements = batch_size * num_features * height * width
    # Mask for boundary
    mask = offset < total_elements

    # Calculate the flattened index for x
    # Convert 4D index to 1D index
    idx = offset
    # Compute batch, channel, h, w indices
    w = idx % width
    h = (idx // width) % height
    c = (idx // (width * height)) % num_features
    b = idx // (width * height * num_features)

    # Load x from global memory
    x = tl.load(x_ptr + idx, mask=mask, other=0.0)

    # Load mean, var, weight, bias
    # mean and var are per-channel
    mean = tl.load(mean_ptr + c, mask=c < num_features, other=0.0)
    var = tl.load(var_ptr + c, mask=c < num_features, other=0.0)
    weight = tl.load(weight_ptr + c, mask=c < num_features, other=0.0)
    bias = tl.load(bias_ptr + c, mask=c < num_features, other=0.0)

    # Normalize x
    x_norm = (x - mean) / tl.sqrt(var + 1e-5)
    # Scale and shift
    x_out = x_norm * weight + bias

    # Store result
    tl.store(out_ptr + idx, x_out, mask=mask)


@triton.jit
def conv2d_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    kernel_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of elements
    # We assume kernel_size is 1 for simplicity (given in problem)
    # For 1x1 conv, each output element is a dot product across input channels

    # Get block index
    block_id = tl.program_id(0)
    # Create offset
    offset = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total_elements = batch_size * out_channels * height * width

    # Mask for boundary
    mask = offset < total_elements

    # Compute batch, out_c, h, w indices for output
    w = offset % width
    h = (offset // width) % height
    out_c = (offset // (width * height)) % out_channels
    b = offset // (width * height * out_channels)

    # Compute input indices (b, in_c, h, w)
    in_c = tl.arange(0, in_channels)

    # Load input
    x = tl.load(x_ptr + b * in_channels * height * width + in_c * height * width + h * width + w, mask=mask, other=0.0)

    # Load weights
    w = tl.load(w_ptr + out_c * in_channels + in_c, mask=tl.arange(0, in_channels) < in_channels, other=0.0)

    # Compute dot product
    # Reduce across channels
    out = tl.dot(x, w)

    # Store output
    tl.store(out_ptr + offset, out, mask=mask)


@triton.jit
def avgpool2d_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    num_channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Average pooling over 2x2 windows
    block_id = tl.program_id(0)
    offset = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total_elements = batch_size * num_channels * (height // 2) * (width // 2)
    mask = offset < total_elements

    # Compute output indices (b, c, h, w)
    w = offset % (width // 2)
    h = (offset // (width // 2)) % (height // 2)
    c = (offset // (width // 2 * height // 2)) % num_channels
    b = offset // (width // 2 * height // 2 * num_channels)

    # Compute input indices for the 2x2 window
    h_in = h * 2
    w_in = w * 2

    # Load 4 values: (h_in, w_in), (h_in, w_in+1), (h_in+1, w_in), (h_in+1, w_in+1)
    # Use vectorized load with mask
    idx1 = b * num_channels * height * width + c * height * width + h_in * width + w_in
    idx2 = idx1 + 1
    idx3 = idx1 + width
    idx4 = idx1 + width + 1

    # Load 4 values
    val1 = tl.load(x_ptr + idx1, mask=mask, other=0.0)
    val2 = tl.load(x_ptr + idx2, mask=mask, other=0.0)
    val3 = tl.load(x_ptr + idx3, mask=mask, other=0.0)
    val4 = tl.load(x_ptr + idx4, mask=mask, other=0.0)

    # Average
    avg = (val1 + val2 + val3 + val4) / 4.0

    # Store
    tl.store(out_ptr + offset, avg, mask=mask)


def triton_batchnorm2d(x, weight, bias, running_mean, running_var):
    assert x.is_cuda, "x must be on GPU"
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    running_mean = running_mean.contiguous()
    running_var = running_var.contiguous()
    
    out = torch.empty_like(x)

    total_elements = x.numel()
    BLOCK_SIZE = 128

    grid = lambda meta: (triton.cdiv(total_elements, meta["BLOCK_SIZE"]),)

    batch_size, num_features, height, width = x.shape
    batchnorm2d_kernel[grid](
        x_ptr=x, 
        weight_ptr=weight, 
        bias_ptr=bias, 
        mean_ptr=running_mean, 
        var_ptr=running_var, 
        out_ptr=out,
        batch_size=batch_size, 
        num_features=num_features, 
        height=height, 
        width=width,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out


def triton_conv2d(x, weight):
    assert x.is_cuda and weight.is_cuda, "inputs must be on GPU"
    x = x.contiguous()
    weight = weight.contiguous()

    out = torch.empty(x.shape[0], weight.shape[0], x.shape[2], x.shape[3])

    total_elements = out.numel()
    BLOCK_SIZE = 128

    grid = lambda meta: (triton.cdiv(total_elements, meta["BLOCK_SIZE"]),)

    batch_size, in_channels, height, width = x.shape
    out_channels = weight.shape[0]

    conv2d_kernel[grid](
        x_ptr=x,
        w_ptr=weight,
        out_ptr=out,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        height=height,
        width=width,
        kernel_size=1,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out


def triton_avgpool2d(x):
    assert x.is_cuda, "x must be on GPU"
    x = x.contiguous()

    out = torch.empty(x.shape[0], x.shape[1], x.shape[2] // 2, x.shape[3] // 2)

    total_elements = out.numel()
    BLOCK_SIZE = 128

    grid = lambda meta: (triton.cdiv(total_elements, meta["BLOCK_SIZE"]),)

    batch_size, num_channels, height, width = x.shape

    avgpool2d_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        batch_size=batch_size,
        num_channels=num_channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        super().__init__()
        self.num_input_features = num_input_features
        self.num_output_features = num_output_features

        # Initialize parameters
        self.weight = nn.Parameter(torch.randn(num_output_features, num_input_features, 1, 1))
        self.bias = nn.Parameter(torch.randn(num_output_features))
        self.running_mean = nn.Parameter(torch.zeros(num_input_features))
        self.running_var = nn.Parameter(torch.ones(num_input_features))

        # Ensure parameters are on GPU
        self.weight = self.weight.cuda()
        self.bias = self.bias.cuda()
        self.running_mean = self.running_mean.cuda()
        self.running_var = self.running_var.cuda()

    def forward(self, x):
        # Fuse batchnorm, conv, and avgpool in a single forward pass
        # Step 1: BatchNorm2d
        x = triton_batchnorm2d(x, self.weight, self.bias, self.running_mean, self.running_var)

        # Step 2: Conv2d with 1x1 kernel
        x = triton_conv2d(x, self.weight)

        # Step 3: AvgPool2d with kernel_size=2, stride=2
        x = triton_avgpool2d(x)

        return x