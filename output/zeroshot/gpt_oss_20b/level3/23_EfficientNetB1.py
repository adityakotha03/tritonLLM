import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# --------------------------------------------------
# Triton kernel: fused Conv2d (3x3 depthwise) + BatchNorm + ReLU6
# Assumes groups=hidden_dim (depthwise), stride=1 or 2, padding=1
# --------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8),
    ],
    key=["N", "C", "H", "W", "K", "stride"],
)
@triton.jit
def _conv_bn_relu6_fused_kernel(
    X_ptr,        # [N, C, H, W]
    K_ptr,        # [C, 1, 3, 3] depthwise kernel
    gamma_ptr,    # [C]
    beta_ptr,     # [C]
    mean_ptr,     # [C]
    var_ptr,      # [C]
    out_ptr,      # [N, C, H_out, W_out]
    N, C, H, W,
    H_out, W_out,
    stride,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    # Each program processes a contiguous slice of output elements
    total = N * C * H_out * W_out
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total

    # Decode the linear offset into 4D indices
    n = offsets // (C * H_out * W_out)
    rem = offsets % (C * H_out * W_out)
    c = rem // (H_out * W_out)
    rem = rem % (H_out * W_out)
    h_out = rem // W_out
    w_out = rem % W_out

    # Compute input coordinates
    h_in = h_out * stride
    w_in = w_out * stride

    # Load input patch (3x3)
    vals = []
    for kh in range(3):
        for kw in range(3):
            h = h_in + kh
            w = w_in + kw
            valid = (h < H) & (w < W)
            idx = ((n * C + c) * H + h) * W + w
            val = tl.load(X_ptr + idx, mask=valid, other=0.0)
            vals.append(val)
    patch = tl.stack(vals, dim=0)  # shape [9]

    # Load kernel
    k_idx = (c * 3 + 0) * 3 + 0
    kernel = tl.load(K_ptr + k_idx)  # broadcasted to 9

    # Convolution
    conv = tl.sum(patch * kernel)

    # BatchNorm
    gamma = tl.load(gamma_ptr + c)
    beta = tl.load(beta_ptr + c)
    mean = tl.load(mean_ptr + c)
    var = tl.load(var_ptr + c)
    norm = (conv - mean) * tl.rsqrt(var + eps)
    bn = gamma * norm + beta

    # ReLU6
    relu6 = tl.min(tl.max(bn, 0.0), 6.0)

    # Store
    out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out
    tl.store(out_ptr + out_idx, relu6, mask=mask)

def conv_bn_relu6_fused(X, kernel, gamma, beta, mean, var, stride, eps=1e-5):
    N, C, H, W = X.shape
    K = kernel.shape[0]
    H_out = (H + 2 * 1 - 3) // stride + 1
    W_out = (W + 2 * 1 - 3) // stride + 1

    out = torch.empty((N, C, H_out, W_out), device=X.device, dtype=X.dtype)

    grid = lambda meta: ((N * C * H_out * W_out + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    _conv_bn_relu6_fused_kernel[grid](
        X, kernel, gamma, beta, mean, var, out,
        N, C, H, W, H_out, W_out, stride, eps,
        BLOCK_SIZE=128,
    )
    return out

# --------------------------------------------------
# EfficientNetB1 with fused depthwise blocks
# --------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # Pre‑compute depthwise kernels and BN params for fused blocks
        self.mbconv1 = self._make_fused_mbconv(32, 16, 1, 1)
        self.mbconv2 = self._make_fused_mbconv(16, 24, 2, 6)
        self.mbconv3 = self._make_fused_mbconv(24, 40, 2, 6)
        self.mbconv4 = self._make_fused_mbconv(40, 80, 2, 6)
        self.mbconv5 = self._make_fused_mbconv(80, 112, 1, 6)
        self.mbconv6 = self._make_fused_mbconv(112, 192, 2, 6)
        self.mbconv7 = self._make_fused_mbconv(192, 320, 1, 6)

        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)

        self.fc = nn.Linear(1280, num_classes)

    def _make_fused_mbconv(self, in_ch, out_ch, stride, exp):
        hidden_ch = round(in_ch * exp)
        # 1x1 expand
        pw1 = nn.Conv2d(in_ch, hidden_ch, kernel_size=1, bias=False)
        bn1 = nn.BatchNorm2d(hidden_ch)
        # depthwise 3x3
        dw = nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, stride=stride,
                       padding=1, groups=hidden_ch, bias=False)
        bn2 = nn.BatchNorm2d(hidden_ch)
        # 1x1 project
        pw2 = nn.Conv2d(hidden_ch, out_ch, kernel_size=1, bias=False)
        bn3 = nn.BatchNorm2d(out_ch)

        # Store kernels & BN params for Triton fusion
        return nn.ModuleDict({
            "pw1": pw1, "bn1": bn1,
            "dw": dw,   "bn2": bn2,
            "pw2": pw2, "bn3": bn3,
        })

    def forward(self, x):
        # Stem
        x = F.relu(self.bn1(self.conv1(x)))

        # MBConv blocks with fused depthwise
        for blk in [self.mbconv1, self.mbconv2, self.mbconv3,
                    self.mbconv4, self.mbconv5, self.mbconv6,
                    self.mbconv7]:
            # 1x1 expand
            x = F.relu(blk["bn1"](blk["pw1"](x)))
            # depthwise 3x3 fused with BN + ReLU6
            dw = blk["dw"]
            bn = blk["bn2"]
            # Prepare tensors for Triton
            kernel = dw.weight.squeeze(1).squeeze(1)  # [C, 3, 3] -> [C, 3, 3]
            gamma = bn.weight
            beta = bn.bias
            mean = bn.running_mean
            var = bn.running_var
            stride = dw.stride[0]
            x = conv_bn_relu6_fused(x, kernel, gamma, beta, mean, var, stride)
            # 1x1 project
            x = blk["bn3"](blk["pw2"](x))

        # Final conv
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x