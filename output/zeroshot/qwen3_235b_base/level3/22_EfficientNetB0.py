import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _dw_conv_kernel(
    x_ptr, w_ptr, y_ptr,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wn, stride_wc, stride_wh, stride_ww,
    stride_yn, stride_yc, stride_yh, stride_yw,
    out_h, out_w,
    in_c,
    kernel_h, kernel_w,
    dilation_h, dilation_w,
    padding_h, padding_w,
    stride_h, stride_w,
    eps: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Program IDs
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    # Calculate offsets
    offset_c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    mask_c = offset_c < in_c

    offset_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offset_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)

    # Input spatial indices
    h_idx = offset_h * stride_h - padding_h
    w_idx = offset_w * stride_w - padding_w

    # Weight indices
    kh_idx = tl.arange(0, kernel_h)[:, None]
    kw_idx = tl.arange(0, kernel_w)[None, :]

    # Input pointers
    x_ptrs = x_ptr + pid_n * stride_xn + \
             offset_c[:, None, None] * stride_xc + \
             (h_idx[None, :] + kh_idx * dilation_h) * stride_xh + \
             (w_idx[None, :] + kw_idx * dilation_w) * stride_xw
    w_ptrs = w_ptr + offset_c[:, None, None] * stride_wc + \
             kh_idx * stride_wh + kw_idx * stride_ww

    # Output pointers
    y_ptrs = y_ptr + pid_n * stride_yn + \
             offset_c[:, None, None] * stride_yc + \
             offset_h[None, :] * stride_yh + \
             offset_w[None, :] * stride_yw

    # Initialize output
    acc = tl.zeros((BLOCK_SIZE_C, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)

    # Convolution
    for _kh in range(kernel_h):
        for _kw in range(kernel_w):
            mask_x = (offset_c[:, None, None] < in_c) & \
                     (h_idx[None, :] >= 0) & (h_idx[None, :] < out_h) & \
                     (w_idx[None, :] >= 0) & (w_idx[None, :] < out_w)
            x = tl.load(x_ptrs + _kh * dilation_h * stride_xh + _kw * dilation_w * stride_xw,
                        mask=mask_x, other=0.0)
            w = tl.load(w_ptrs + _kh * stride_wh + _kw * stride_ww,
                        mask=mask_c[:, None, None], other=0.0)
            acc += x.to(tl.float32) * w.to(tl.float32)

    # Store output
    mask_y = (offset_c[:, None, None] < in_c) & \
             (offset_h[None, :] < out_h) & (offset_w[None, :] < out_w)
    tl.store(y_ptrs, acc, mask=mask_y)


def triton_depthwise_conv2d(x, weight, stride=1, padding=0, dilation=1, groups=1):
    B, C, H, W = x.shape
    K, _, kH, kW = weight.shape
    assert K == C and groups == C, "Must be depthwise convolution"

    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    out_h = (H + 2 * padding[0] - dilation[0] * (kH - 1) - 1) // stride[0] + 1
    out_w = (W + 2 * padding[1] - dilation[1] * (kW - 1) - 1) // stride[1] + 1

    out = torch.empty((B, C, out_h, out_w), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (B, triton.cdiv(C, meta['BLOCK_SIZE_C']),
                triton.cdiv(out_h, meta['BLOCK_SIZE_H']),
                triton.cdiv(out_w, meta['BLOCK_SIZE_W']))

    _dw_conv_kernel[grid](
        x, weight, out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        out_h, out_w,
        C,
        kH, kW,
        dilation[0], dilation[1],
        padding[0], padding[1],
        stride[0], stride[1],
        eps=1e-5,
        BLOCK_SIZE_C=16,
        BLOCK_SIZE_H=16,
        BLOCK_SIZE_W=16,
    )
    return out


@triton.jit
def _add_relu_kernel(
    x_ptr, y_ptr,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    N, C, H, W,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    offset_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offset_c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    offset_h = tl.arange(0, BLOCK_SIZE_H)
    offset_w = tl.arange(0, 16)

    mask_n = offset_n < N
    mask_c = offset_c < C

    x_ptrs = x_ptr + \
        offset_n[:, None, None, None] * stride_xn + \
        offset_c[None, :, None, None] * stride_xc + \
        offset_h[None, None, :, None] * stride_xh + \
        offset_w[None, None, None, :] * stride_xw
    y_ptrs = y_ptr + \
        offset_n[:, None, None, None] * stride_yn + \
        offset_c[None, :, None, None] * stride_yc + \
        offset_h[None, None, :, None] * stride_yh + \
        offset_w[None, None, None, :] * stride_yw

    for h in range(0, H, BLOCK_SIZE_H):
        curr_h = h + offset_h
        mask_hw = mask_n[:, None, None, None] & mask_c[None, :, None, None] & \
                  (curr_h[None, None, :, None] < H) & (offset_w[None, None, None, :] < W)
        x = tl.load(x_ptrs + h * stride_xh, mask=mask_hw, other=0.0)
        y = tl.load(y_ptrs + h * stride_yh, mask=mask_hw, other=0.0)
        out = tl.maximum(x + y, 0.0)
        tl.store(y_ptrs + h * stride_yh, out, mask=mask_hw)


def triton_add_relu(x, y):
    assert x.shape == y.shape
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(x.shape[0], meta['BLOCK_SIZE_N']),
                         triton.cdiv(x.shape[1], meta['BLOCK_SIZE_C']))
    _add_relu_kernel[grid](
        x, y,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        *x.shape,
        BLOCK_SIZE_N=4,
        BLOCK_SIZE_C=16,
        BLOCK_SIZE_H=16,
    )
    return out


class TritonBatchNorm2d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        return self.bn(x)


class MBConvTriton(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super(MBConvTriton, self).__init__()
        
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio
        
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                TritonBatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            )
        
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=hidden_dim, bias=False),
            TritonBatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )
        
        self.project_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            TritonBatchNorm2d(out_channels)
        )
        
        # Store configs for Triton kernel usage
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2

    def forward(self, x):
        identity = x
        
        if hasattr(self, 'expand_conv'):
            x = self.expand_conv(x)
        
        # Replace depthwise conv with Triton kernel
        w = self.depthwise_conv[0].weight
        x = triton_depthwise_conv2d(x, w, stride=self.stride, padding=self.padding, groups=self.hidden_dim)
        x = self.depthwise_conv[1](x)  # BatchNorm
        x = F.relu6(x)  # Activation
        
        x = self.project_conv(x)
        
        if self.use_residual:
            x = triton_add_relu(x, identity)
        
        return x


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = TritonBatchNorm2d(32)
        
        # MBConv blocks with Triton kernels
        self.blocks = nn.Sequential(
            MBConvTriton(32, 16, kernel_size=3, stride=1, expand_ratio=1),
            MBConvTriton(16, 24, kernel_size=3, stride=2, expand_ratio=6),
            MBConvTriton(24, 24, kernel_size=3, stride=1, expand_ratio=6),
            MBConvTriton(24, 40, kernel_size=5, stride=2, expand_ratio=6),
            MBConvTriton(40, 40, kernel_size=5, stride=1, expand_ratio=6),
            MBConvTriton(40, 80, kernel_size=3, stride=2, expand_ratio=6),
            MBConvTriton(80, 80, kernel_size=3, stride=1, expand_ratio=6),
            MBConvTriton(80, 112, kernel_size=5, stride=1, expand_ratio=6),
            MBConvTriton(112, 112, kernel_size=5, stride=1, expand_ratio=6),
            MBConvTriton(112, 192, kernel_size=5, stride=2, expand_ratio=6),
            MBConvTriton(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            MBConvTriton(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            MBConvTriton(192, 320, kernel_size=3, stride=1, expand_ratio=6)
        )
        
        # Final convolutional layer
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = TritonBatchNorm2d(1280)
        
        # Fully connected layer
        self.fc = nn.Linear(1280, num_classes)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.blocks(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x