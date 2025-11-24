import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr,  # pointer to input tensor (batch, channels, H, W)
    weight_ptr,  # pointer to weight tensor (out_channels, in_channels, k, k)
    bias_ptr,    # pointer to bias tensor (out_channels)
    output_ptr,  # pointer to output tensor (batch, out_channels, H_out, W_out)
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    k: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the block and thread indices
    batch_id = tl.program_id(0)
    out_h = tl.program_id(1)
    out_w = tl.program_id(2)

    # Define the output spatial dimensions
    H_out = (H + 2 * padding - k) // stride + 1
    W_out = (W + 2 * padding - k) // stride + 1

    # Define the offset for the current block
    base_h = out_h * stride
    base_w = out_w * stride

    # Load the input and weight data
    # We process one output feature map at a time
    # Each thread handles a small region of the output
    # We use a 2D block of size BLOCK_SIZE x BLOCK_SIZE

    # For each output channel
    for oc in range(out_channels):
        # Load bias if exists
        bias_val = 0.0
        if bias_ptr is not None:
            bias_val = tl.load(bias_ptr + oc, mask=tl.ones(1), other=0.0)

        # Initialize output accumulator
        out_acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

        # Loop over the input spatial dimensions
        for ih in range(H):
            for iw in range(W):
                # Compute the input position
                input_idx = batch_id * in_channels * H * W + ih * W + iw
                # Load input value
                input_val = tl.load(input_ptr + input_idx, mask=tl.ones(1), other=0.0)

                # Compute the output position
                oh = (ih + padding) // stride
                ow = (iw + padding) // stride

                # Check if this input contributes to the current output position
                if oh >= H_out or ow >= W_out:
                    continue

                # Compute the weight index
                for ic in range(in_channels):
                    # Compute the weight index
                    w_idx = oc * in_channels * k * k + ic * k * k + (ih - padding) * k + (iw - padding)
                    weight_val = tl.load(weight_ptr + w_idx, mask=tl.ones(1), other=0.0)
                    out_acc += input_val * weight_val

        # Store the output
        output_idx = batch_id * out_channels * H_out * W_out + oc * H_out * W_out + out_h * W_out + out_w
        tl.store(output_ptr + output_idx, out_acc, mask=tl.ones(1))


@triton.jit
def relu6_kernel(
    x_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.where(x > 0, tl.minimum(x, 6.0), 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def batch_norm2d_kernel(
    x_ptr,
    gamma_ptr,
    beta_ptr,
    moving_mean_ptr,
    moving_var_ptr,
    eps: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_id = tl.program_id(0)
    channel_id = tl.program_id(1)

    # Load gamma and beta
    gamma_val = tl.load(gamma_ptr + channel_id, mask=tl.ones(1), other=1.0)
    beta_val = tl.load(beta_ptr + channel_id, mask=tl.ones(1), other=0.0)

    # Accumulate per-channel mean and variance
    mean_val = 0.0
    var_val = 0.0
    count = 0

    # Process each spatial location
    for h in range(H):
        for w in range(W):
            # Load input
            input_val = tl.load(x_ptr + batch_id * C * H * W + channel_id * H * W + h * W + w, mask=tl.ones(1), other=0.0)
            mean_val += input_val
            var_val += input_val * input_val
            count += 1

    mean_val /= count
    var_val = var_val / count - mean_val * mean_val
    var_val = tl.where(var_val < eps, eps, var_val)

    # Compute normalization
    inv_std = 1.0 / tl.sqrt(var_val + eps)
    norm_val = (x_ptr + channel_id * H * W) - mean_val
    norm_val = norm_val * inv_std

    # Apply scale and shift
    out_val = gamma_val * norm_val + beta_val

    # Store output
    tl.store(out_ptr + batch_id * C * H * W + channel_id * H * W, out_val, mask=tl.ones(1))


def triton_conv2d(
    input_tensor,
    weight_tensor,
    bias_tensor=None,
    stride=1,
    padding=0,
    kernel_size=3,
):
    """
    Custom Triton kernel for 2D convolution.
    """
    assert input_tensor.is_cuda and weight_tensor.is_cuda, "Tensors must be on CUDA."
    input_tensor = input_tensor.contiguous()
    weight_tensor = weight_tensor.contiguous()

    batch_size, in_channels, H, W = input_tensor.shape
    out_channels, _, k, k = weight_tensor.shape

    # Output dimensions
    H_out = (H + 2 * padding - k) // stride + 1
    W_out = (W + 2 * padding - k) // stride + 1

    # Output tensor
    output = torch.empty(batch_size, out_channels, H_out, W_out, device=input_tensor.device, dtype=input_tensor.dtype)

    # Grid definition
    grid = lambda meta: (
        (batch_size,),
        (H_out,),
        (W_out,),
    )

    # Launch kernel
    conv2d_kernel[grid](
        input_tensor.data_ptr(),
        weight_tensor.data_ptr(),
        bias_tensor.data_ptr() if bias_tensor is not None else None,
        output.data_ptr(),
        batch_size,
        in_channels,
        out_channels,
        H,
        W,
        k,
        stride,
        padding,
        BLOCK_SIZE=128,
    )
    return output


def triton_relu6(x: torch.Tensor):
    """
    Custom Triton kernel for ReLU6 activation.
    """
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    relu6_kernel[lambda meta: ((x.numel() + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)](
        x.data_ptr(),
        out.data_ptr(),
        x.numel(),
        BLOCK_SIZE=128,
    )
    return out


def triton_batch_norm2d(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    moving_mean: torch.Tensor,
    moving_var: torch.Tensor,
    eps: float = 1e-5,
):
    """
    Custom Triton kernel for batch normalization.
    """
    assert x.is_cuda and gamma.is_cuda and beta.is_cuda, "All tensors must be on CUDA."
    x = x.contiguous()
    gamma = gamma.contiguous()
    beta = beta.contiguous()

    batch_size, channels, H, W = x.shape
    output = torch.empty_like(x)

    grid = lambda meta: (
        (batch_size,),
        (channels,),
    )

    batch_norm2d_kernel[grid](
        x.data_ptr(),
        gamma.data_ptr(),
        beta.data_ptr(),
        moving_mean.data_ptr(),
        moving_var.data_ptr(),
        eps,
        batch_size,
        channels,
        H,
        W,
        BLOCK_SIZE=128,
    )
    return output


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # MBConv blocks
        self.blocks = nn.Sequential(
            # MBConv1 (32, 16, 1, 1)
            MBConv(32, 16, kernel_size=3, stride=1, expand_ratio=1),
            # MBConv6 (16, 24, 2, 6)
            MBConv(16, 24, kernel_size=3, stride=2, expand_ratio=6),
            # MBConv6 (24, 24, 1, 6)
            MBConv(24, 24, kernel_size=3, stride=1, expand_ratio=6),
            # MBConv6 (24, 40, 2, 6)
            MBConv(24, 40, kernel_size=5, stride=2, expand_ratio=6),
            # MBConv6 (40, 40, 1, 6)
            MBConv(40, 40, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (40, 80, 2, 6)
            MBConv(40, 80, kernel_size=3, stride=2, expand_ratio=6),
            # MBConv6 (80, 80, 1, 6)
            MBConv(80, 80, kernel_size=3, stride=1, expand_ratio=6),
            # MBConv6 (80, 112, 1, 6)
            MBConv(80, 112, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (112, 112, 1, 6)
            MBConv(112, 112, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (112, 192, 2, 6)
            MBConv(112, 192, kernel_size=5, stride=2, expand_ratio=6),
            # MBConv6 (192, 192, 1, 6)
            MBConv(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (192, 192, 1, 6)
            MBConv(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (192, 320, 1, 6)
            MBConv(192, 320, kernel_size=3, stride=1, expand_ratio=6)
        )
        
        # Final convolutional layer
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        
        # Fully connected layer
        self.fc = nn.Linear(1280, num_classes)
    
    def forward(self, x):
        """
        Forward pass of the EfficientNetB0 model.

        :param x: The input tensor, shape (batch_size, 3, 224, 224)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.blocks(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super(MBConv, self).__init__()
        
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio
        
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            )
        
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )
        
        self.project_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        identity = x
        
        if hasattr(self, 'expand_conv'):
            x = self.expand_conv(x)
        
        x = self.depthwise_conv(x)
        x = self.project_conv(x)
        
        if self.use_residual:
            x += identity
        
        return x