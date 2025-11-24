import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr,  # pointer to input tensor (batch, channels, H, W)
    weight_ptr,  # pointer to weight tensor (out_channels, in_channels, k, k)
    bias_ptr,  # pointer to bias tensor (out_channels)
    output_ptr,  # pointer to output tensor (batch, out_channels, H_out, W_out)
    batch_size: tl.constexpr,
    channels: tl.constexpr,
    in_height: tl.constexpr,
    in_width: tl.constexpr,
    out_channels: tl.constexpr,
    k: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Define block indices
    block_id = tl.program_id(0)
    block_start_h = block_id // (in_height // BLOCK_SIZE) * BLOCK_SIZE
    block_start_w = (block_id % (in_height // BLOCK_SIZE)) * BLOCK_SIZE

    # Compute output dimensions
    out_h = (in_height + 2 * pad_h - k) // stride_h + 1
    out_w = (in_width + 2 * pad_w - k) // stride_w + 1

    # Compute the range of output positions this block handles
    h_start = block_start_h
    h_end = min(h_start + BLOCK_SIZE, out_h)
    w_start = block_start_w
    w_end = min(w_start + BLOCK_SIZE, out_w)

    # Loop over output positions
    for h in range(h_start, h_end):
        for w in range(w_start, w_end):
            # Compute input coordinates
            h_in = h * stride_h - pad_h
            w_in = w * stride_w - pad_w

            # Check bounds for input
            h_in_valid = (h_in >= 0) & (h_in < in_height)
            w_in_valid = (w_in >= 0) & (w_in < in_width)
            valid = h_in_valid & w_in_valid

            # Compute the output index
            out_idx = h * out_w + w
            out_offset = out_idx * channels * batch_size
            out_val = 0.0

            # Compute the kernel weights and input values
            for i in range(channels):
                # Compute input coordinates with padding
                h_in_padded = h_in + pad_h
                w_in_padded = w_in + pad_w
                h_in_padded = h_in_padded - pad_h
                w_in_padded = w_in_padded - pad_w

                # Compute the kernel indices
                k_h = tl.arange(0, k)
                k_w = tl.arange(0, k)

                # Load input values
                input_h = h_in_padded + k_h
                input_w = w_in_padded + k_w
                input_offset = (h_in_padded + k_h) * in_width + (w_in_padded + k_w)
                input_val = tl.load(input_ptr + (0 * in_height * in_width + input_offset), mask=valid, other=0.0)

                # Load weights
                weight_offset = i * k * k + k_h * k + k_w
                weight_val = tl.load(weight_ptr + weight_offset, mask=valid, other=0.0)

                # Accumulate dot product
                out_val += input_val * weight_val

            # Add bias
            if bias_ptr is not None:
                bias_idx = out_idx
                bias_val = tl.load(bias_ptr + bias_idx, mask=valid, other=0.0)
                out_val += bias_val

            # Store output
            tl.store(output_ptr + out_offset, out_val, mask=valid)


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
    x = tl.where(x > 0, x, 0.0)
    x = tl.where(x > 6.0, 6.0, x)
    out = x
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    m: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of size BLOCK_SIZE
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < m

    # Load A matrix (m x k)
    a = tl.load(a_ptr + offsets[:, None] * k + tl.arange(0, k)[None, :], mask=mask[:, None], other=0.0)

    # Load B matrix (k x n)
    b = tl.load(b_ptr + tl.arange(0, k)[:, None] * n + tl.arange(0, n)[None, :], mask=tl.arange(0, k)[:, None] < k, other=0.0)

    # Compute dot product
    out = tl.dot(a, b)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def adaptive_avg_pool2d_kernel(
    x_ptr,
    out_ptr,
    h_in: tl.constexpr,
    w_in: tl.constexpr,
    h_out: tl.constexpr,
    w_out: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Block index
    block_id = tl.program_id(0)
    block_h = block_id // (h_out // BLOCK_SIZE)
    block_w = block_id % (h_out // BLOCK_SIZE)

    # Compute the output position
    h_out_idx = block_h * BLOCK_SIZE
    w_out_idx = block_w * BLOCK_SIZE

    # Load input values
    h_in_start = h_out_idx * (h_in // h_out)
    h_in_end = min(h_in_start + BLOCK_SIZE, h_in)
    w_in_start = w_out_idx * (w_in // w_out)
    w_in_end = min(w_in_start + BLOCK_SIZE, w_in)

    # Accumulate values
    sum_val = 0.0
    count = 0
    for h in range(h_in_start, h_in_end):
        for w in range(w_in_start, w_in_end):
            val = tl.load(x_ptr + h * w_in + w, mask=(h < h_in) & (w < w_in), other=0.0)
            sum_val += val
            count += 1

    # Store average
    avg_val = sum_val / count
    tl.store(out_ptr + h_out_idx * w_out + w_out_idx, avg_val)


def triton_conv2d(
    input_tensor,
    weight_tensor,
    bias_tensor=None,
    stride_h=1,
    stride_w=1,
    pad_h=0,
    pad_w=0,
    out_channels=None,
    kernel_size=3,
):
    assert input_tensor.is_cuda and weight_tensor.is_cuda, "Tensors must be on CUDA."
    input_tensor = input_tensor.contiguous()
    weight_tensor = weight_tensor.contiguous()

    batch_size, channels, in_h, in_w = input_tensor.shape
    out_h = (in_h + 2 * pad_h - kernel_size) // stride_h + 1
    out_w = (in_w + 2 * pad_w - kernel_size) // stride_w + 1
    out_channels = out_channels or weight_tensor.shape[0]

    output = torch.empty((batch_size, out_channels, out_h, out_w), device=input_tensor.device, dtype=input_tensor.dtype)

    # Grid size
    grid = lambda meta: ((out_h * out_w + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch kernel
    conv2d_kernel[grid](
        input_tensor.data_ptr(),
        weight_tensor.data_ptr(),
        bias_tensor.data_ptr() if bias_tensor is not None else None,
        output.data_ptr(),
        batch_size,
        channels,
        in_h,
        in_w,
        out_channels,
        kernel_size,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        128,
    )
    return output


def triton_relu6(x: torch.Tensor):
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    relu6_kernel[lambda meta: ((x.numel() + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)](
        x.data_ptr(),
        out.data_ptr(),
        x.numel(),
        128,
    )
    return out


def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    a = a.contiguous()
    b = b.contiguous()

    m, k = a.shape
    k, n = b.shape
    out = torch.empty((m, n), device=a.device, dtype=a.dtype)

    grid = lambda meta: ((m + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    matmul_kernel[grid](
        a.data_ptr(),
        b.data_ptr(),
        out.data_ptr(),
        m,
        n,
        k,
        128,
    )
    return out


def triton_adaptive_avg_pool2d(x: torch.Tensor, output_size=(1, 1)):
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()
    batch_size, channels, h, w = x.shape
    out_h, out_w = output_size

    out = torch.empty((batch_size, channels, out_h, out_w), device=x.device, dtype=x.dtype)

    grid = lambda meta: ((out_h * out_w + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    adaptive_avg_pool2d_kernel[grid](
        x.data_ptr(),
        out.data_ptr(),
        h,
        w,
        out_h,
        out_w,
        128,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # MBConv blocks
        self.mbconv1 = self._make_mbconv_block(32, 16, 1, 1)
        self.mbconv2 = self._make_mbconv_block(16, 24, 2, 6)
        self.mbconv3 = self._make_mbconv_block(24, 40, 2, 6)
        self.mbconv4 = self._make_mbconv_block(40, 80, 2, 6)
        self.mbconv5 = self._make_mbconv_block(80, 112, 1, 6)
        self.mbconv6 = self._make_mbconv_block(112, 192, 2, 6)
        self.mbconv7 = self._make_mbconv_block(192, 320, 1, 6)
        
        # Final convolutional layer
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        
        # Fully connected layer
        self.fc = nn.Linear(1280, num_classes)
    
    def _make_mbconv_block(self, in_channels, out_channels, stride, expand_ratio):
        hidden_dim = round(in_channels * expand_ratio)
        return nn.Sequential(
            triton_conv2d(
                input_tensor=torch.zeros(1, in_channels, 1, 1),
                weight_tensor=torch.zeros(hidden_dim, in_channels, 1, 1),
                stride_h=1,
                stride_w=1,
                pad_h=0,
                pad_w=0,
            ),
            nn.BatchNorm2d(hidden_dim),
            triton_relu6,
            triton_conv2d(
                input_tensor=torch.zeros(1, hidden_dim, 1, 1),
                weight_tensor=torch.zeros(hidden_dim, hidden_dim, 3, 3),
                stride_h=stride,
                stride_w=stride,
                pad_h=1,
                pad_w=1,
            ),
            nn.BatchNorm2d(hidden_dim),
            triton_relu6,
            triton_conv2d(
                input_tensor=torch.zeros(1, hidden_dim, 1, 1),
                weight_tensor=torch.zeros(out_channels, hidden_dim, 1, 1),
                stride_h=1,
                stride_w=1,
                pad_h=0,
                pad_w=0,
            ),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        x = self.mbconv4(x)
        x = self.mbconv5(x)
        x = self.mbconv6(x)
        x = self.mbconv7(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = triton_adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x