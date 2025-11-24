import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# --------------------------------------------------------------------
# Triton kernels
# --------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8),
    ],
    key=['M', 'N', 'K', 'stride_in', 'stride_out'],
)
@triton.jit
def _conv2d_kernel(
    X_ptr,          # input [C_in, H_in, W_in]
    W_ptr,          # weights [C_out, C_in, kH, kW]
    Y_ptr,          # output [C_out, H_out, W_out]
    stride_h, stride_w,
    pad_h, pad_w,
    H_in, W_in,
    C_in, C_out,
    kH, kW,
    stride_XH, stride_XW,
    stride_WC, stride_WK, stride_WO,
    stride_YH, stride_YW,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Matrix multiplication based convolution.
    Y[i, m, n] = sum_k X[k, m*stride_h : m*stride_h + kH, n*stride_w : n*stride_w + kW] * W[i, k, :, :]
    """
    # Thread block indices
    i = tl.program_id(0)
    m = tl.program_id(1)
    n = tl.program_id(2)

    # Compute the output coordinates
    row = i * BLOCK_M + tl.arange(0, BLOCK_M)
    col = n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Allocate registers for the accumulators
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over the channel dimension in tiles
    for k in range(0, C_in, BLOCK_K):
        # Load input patch
        in_row = m * stride_h + row
        in_col = n * stride_w + col
        mask = (in_row < H_in) & (in_col < W_in) & (row < BLOCK_M) & (col < BLOCK_N)

        # Gather the input patch of size kH x kW
        # Load all kH*kW positions into a temporary array
        input_tile = tl.zeros((BLOCK_M, BLOCK_N, kH, kW), dtype=tl.float32)
        for kh in range(kH):
            for kw in range(kW):
                r = in_row + kh
                c = in_col + kw
                valid = (r < H_in) & (c < W_in)
                offsets = X_ptr + k * stride_XC + r * stride_XH + c * stride_XW
                val = tl.load(offsets, mask=valid, other=0.0)
                input_tile[:, :, kh, kw] = val

        # Load weights
        # Weight tile for current k slice
        weight_tile = tl.load(
            W_ptr + i * stride_WO + k * stride_WC,
            mask=mask,
            other=0.0
        )  # shape (BLOCK_M, BLOCK_N, kH, kW)

        # Compute partial dot product
        acc += tl.dot(input_tile, weight_tile)

    # Store result
    out_offsets = Y_ptr + i * stride_YC + m * stride_YH + n * stride_YW
    tl.store(out_offsets, acc, mask=mask)

def triton_conv2d(
    X: torch.Tensor,
    W: torch.Tensor,
    stride: int = 1,
    padding: int = 0,
):
    """
    X: (N, C_in, H_in, W_in)
    W: (C_out, C_in, kH, kW)
    """
    N, C_in, H_in, W_in = X.shape
    C_out, _, kH, kW = W.shape
    stride_h = stride_w = stride
    pad_h = pad_w = padding
    H_out = (H_in + 2 * pad_h - kH) // stride_h + 1
    W_out = (W_in + 2 * pad_w - kW) // stride_w + 1

    Y = torch.empty((N, C_out, H_out, W_out), device=X.device, dtype=X.dtype)

    # Padding
    if padding > 0:
        X = F.pad(X, (padding, padding, padding, padding))

    # Strides
    stride_XC = X.stride(1)
    stride_XH = X.stride(2)
    stride_XW = X.stride(3)

    stride_WC = W.stride(1)
    stride_WK = W.stride(2)
    stride_WO = W.stride(0)

    stride_YC = Y.stride(1)
    stride_YH = Y.stride(2)
    stride_YW = Y.stride(3)

    grid = lambda META: (
        (N, (C_out + META['BLOCK_M'] - 1) // META['BLOCK_M']),
        (H_out + META['BLOCK_N'] - 1) // META['BLOCK_N'],
        1,
    )

    _conv2d_kernel[grid](
        X_ptr=X.contiguous().data_ptr(),
        W_ptr=W.contiguous().data_ptr(),
        Y_ptr=Y.data_ptr(),
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        H_in=H_in,
        W_in=W_in,
        C_in=C_in,
        C_out=C_out,
        kH=kH,
        kW=kW,
        stride_XH=stride_XH,
        stride_XW=stride_XW,
        stride_XC=stride_XC,
        stride_WC=stride_WC,
        stride_WK=stride_WK,
        stride_WO=stride_WO,
        stride_YH=stride_YH,
        stride_YW=stride_YW,
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=16,
    )
    return Y

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 256}, num_warps=4),
        triton.Config({'BLOCK_D': 512}, num_warps=8),
    ],
    key=['N', 'M'],
)
@triton.jit
def _linear_kernel(
    X_ptr, Y_ptr, W_ptr, B_ptr,
    stride_XN, stride_XD,
    stride_WN, stride_WD,
    stride_YN, stride_YD,
    N, D_in, D_out,
    BLOCK_D: tl.constexpr,
):
    i = tl.program_id(0)
    offset_x = X_ptr + i * stride_XN
    offset_y = Y_ptr + i * stride_YN
    acc = tl.zeros((BLOCK_D), dtype=tl.float32)

    for d_in in range(0, D_in, BLOCK_D):
        x = tl.load(offset_x + d_in * stride_XD, mask=(d_in + tl.arange(0, BLOCK_D) < D_in), other=0.0)
        w = tl.load(W_ptr + d_in * stride_WD, mask=(d_in + tl.arange(0, BLOCK_D) < D_in), other=0.0)
        acc += x * w

    bias = tl.load(B_ptr, mask=(i < D_out), other=0.0)
    acc += bias
    tl.store(offset_y, acc)

def triton_linear(X: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    """
    X: (N, D_in)
    weight: (D_out, D_in)
    bias: (D_out)
    """
    N, D_in = X.shape
    D_out = weight.shape[0]
    Y = torch.empty((N, D_out), device=X.device, dtype=X.dtype)

    stride_XN = X.stride(0)
    stride_XD = X.stride(1)

    stride_WN = weight.stride(0)
    stride_WD = weight.stride(1)

    stride_YN = Y.stride(0)
    stride_YD = Y.stride(1)

    grid = lambda META: ((N + META['BLOCK_D'] - 1) // META['BLOCK_D'],)
    _linear_kernel[grid](
        X_ptr=X.contiguous().data_ptr(),
        Y_ptr=Y.contiguous().data_ptr(),
        W_ptr=weight.contiguous().data_ptr(),
        B_ptr=bias.contiguous().data_ptr(),
        stride_XN=stride_XN,
        stride_XD=stride_XD,
        stride_WN=stride_WN,
        stride_WD=stride_WD,
        stride_YN=stride_YN,
        stride_YD=stride_YD,
        N=N,
        D_in=D_in,
        D_out=D_out,
        BLOCK_D=256,
    )
    return Y

# --------------------------------------------------------------------
# Optimized DenseNet
# --------------------------------------------------------------------
class DenseBlockNew(nn.Module):
    def __init__(self, num_layers, num_input_features, growth_rate):
        super().__init__()
        self.layers = nn.ModuleList()
        self.growth_rate = growth_rate
        self.num_input_features = num_input_features
        for i in range(num_layers):
            layer = nn.Sequential(
                nn.BatchNorm2d(num_input_features + i * growth_rate),
                nn.ReLU(inplace=True),
                # convolution replaced by Triton
                nn.Identity(),  # placeholder for conv
            )
            self.layers.append(layer)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            # BatchNorm + ReLU
            bn_relu = layer[0](torch.cat(features, 1))
            # Conv
            conv_in_channels = bn_relu.shape[1]
            conv_weight = layer[2].weight if hasattr(layer[2], 'weight') else None
            if conv_weight is None:
                # create weight if not set
                conv_weight = nn.Parameter(
                    torch.randn(self.growth_rate, conv_in_channels, 3, 3, device=x.device)
                )
                layer[2] = conv_weight
            new_feature = triton_conv2d(bn_relu, layer[2], stride=1, padding=1)
            features.append(new_feature)
            x = torch.cat(features, 1)
        return x

class TransitionLayerNew(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv_weight = nn.Parameter(
            torch.randn(num_output_features, num_input_features, 1, 1, device='cuda')
        )
        self.pool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = triton_conv2d(x, self.conv_weight, stride=1, padding=0)
        x = self.pool(x)
        return x

class ModelNew(nn.Module):
    def __init__(self, growth_rate=32, num_classes=1000):
        super().__init__()

        # Initial conv
        self.init_conv_weight = nn.Parameter(
            torch.randn(64, 3, 7, 7, device='cuda')
        )
        self.init_bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(3, 2, 1)

        # Dense blocks
        num_features = 64
        block_layers = [6, 12, 24, 16]
        self.dense_blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()

        for i, num_layers in enumerate(block_layers):
            block = DenseBlockNew(num_layers, num_features, growth_rate)
            self.dense_blocks.append(block)
            num_features += num_layers * growth_rate
            if i != len(block_layers) - 1:
                trans = TransitionLayerNew(num_features, num_features // 2)
                self.transitions.append(trans)
                num_features //= 2

        self.final_bn = nn.BatchNorm2d(num_features)
        self.classifier_weight = nn.Parameter(
            torch.randn(num_classes, num_features, device='cuda')
        )
        self.classifier_bias = nn.Parameter(
            torch.randn(num_classes, device='cuda')
        )

    def forward(self, x):
        # Initial conv + BN + ReLU + pool
        x = triton_conv2d(x, self.init_conv_weight, stride=2, padding=3)
        x = self.init_bn(x)
        x = self.relu(x)
        x = self.pool(x)

        # Dense blocks + transitions
        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transitions[i](x)

        # Final BN + ReLU
        x = self.final_bn(x)
        x = self.relu(x)

        # Global average pool
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)

        # Linear classifier
        logits = triton_linear(x, self.classifier_weight.t(), self.classifier_bias)
        return logits