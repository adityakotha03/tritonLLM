import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# ------------------------------------------------------------
# 1.  Triton kernel : fused BatchNorm + ReLU + Conv2d (3x3)
# ------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32},
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32},
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 512, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32},
                      num_warps=16),
    ],
    key=['M', 'N', 'K', 'P', 'Q', 'R'],
)
@triton.jit
def conv_bn_relu_fused_kernel(
    input_ptr,           # [B, C_in, H_in, W_in]
    weight_ptr,          # [C_out, C_in, 3, 3]
    gamma_ptr,           # BatchNorm gamma [C_out]
    beta_ptr,            # BatchNorm beta [C_out]
    mean_ptr,            # BatchNorm running mean [C_out]
    var_ptr,             # BatchNorm running var [C_out]
    output_ptr,          # [B, C_out, H_out, W_out]
    H_in: tl.constexpr,
    W_in: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    stride: tl.constexpr,
    pad: tl.constexpr,
    eps: tl.constexpr,
    M: tl.constexpr,   # C_in * 3 * 3
    N: tl.constexpr,   # C_out
    K: tl.constexpr,   # batch * H_out * W_out
    P: tl.constexpr,   # stride
    Q: tl.constexpr,   # pad
    R: tl.constexpr,   # 3
):
    pid_m = tl.program_id(axis=0)  # M (rows)
    pid_n = tl.program_id(axis=1)  # N (cols)
    pid_k = tl.program_id(axis=2)  # K (tiles)

    # grid dims
    M_block = M
    N_block = N
    K_block = K

    m_idx = pid_m
    n_idx = pid_n
    k_idx = pid_k

    # Load the relevant input patches into registers (im2col)
    # Each thread processes one output channel (n_idx) and one position (k_idx)
    # Compute spatial coordinates
    b = k_idx // (H_out * W_out)
    hw = k_idx % (H_out * W_out)
    h_out = hw // W_out
    w_out = hw % W_out
    h_in = h_out * stride - pad
    w_in = w_out * stride - pad

    # Compute the linearized index for input channel and spatial offset
    acc = 0.0
    for ic in range(tl.arange(0, M).to(tl.int32)):
        # Compute input position for this receptive field
        i_h = h_in + (ic // (M // 3))
        i_w = w_in + (ic % (M // 3))
        # Boundary check
        cond = (i_h >= 0) & (i_h < H_in) & (i_w >= 0) & (i_w < W_in)
        idx = b * C_in * H_in * W_in + ic * H_in * W_in + i_h * W_in + i_w
        val = tl.load(input_ptr + idx, mask=cond, other=0.0)
        # Weight lookup
        w_idx = ic * N + n_idx
        w_val = tl.load(weight_ptr + w_idx)
        acc += val * w_val

    # BatchNorm
    mean = tl.load(mean_ptr + n_idx)
    var = tl.load(var_ptr + n_idx)
    gamma = tl.load(gamma_ptr + n_idx)
    beta = tl.load(beta_ptr + n_idx)
    inv_std = 1.0 / tl.sqrt(var + eps)
    bn = gamma * (acc - mean) * inv_std + beta

    # ReLU
    out = tl.where(bn > 0, bn, 0.0)

    # Store
    out_idx = b * N * H_out * W_out + n_idx * H_out * W_out + h_out * W_out + w_out
    tl.store(output_ptr + out_idx, out)

# ------------------------------------------------------------
# 2.  Wrapper function
# ------------------------------------------------------------
def conv_bn_relu_fused(input, weight, gamma, beta, mean, var, stride=1, padding=1, eps=1e-5):
    B, Cin, H, W = input.shape
    Cout, _, K, _ = weight.shape
    H_out = (H + 2 * padding - K) // stride + 1
    W_out = (W + 2 * padding - K) // stride + 1

    output = torch.empty((B, Cout, H_out, W_out), device=input.device, dtype=input.dtype)

    # Compute constants for kernel launch
    M = Cin * K * K
    N = Cout
    K_grid = B * H_out * W_out

    grid = lambda meta: (
        (M + meta['BLOCK_SIZE_M'] - 1) // meta['BLOCK_SIZE_M'],
        (N + meta['BLOCK_SIZE_N'] - 1) // meta['BLOCK_SIZE_N'],
        (K_grid + meta['BLOCK_SIZE_K'] - 1) // meta['BLOCK_SIZE_K'],
    )

    conv_bn_relu_fused_kernel[grid](
        input, weight, gamma, beta, mean, var, output,
        H, W, H_out, W_out, stride, padding, eps,
        M, N, K_grid, stride, padding, K,
    )
    return output

# ------------------------------------------------------------
# 3.  Optimised Model
# ------------------------------------------------------------
class ConvBNReLULayer(nn.Module):
    """
    Equivalent to the original _make_layer but uses a fused Triton kernel.
    """
    def __init__(self, in_features, growth_rate):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_features)
        self.conv = nn.Conv2d(in_features, growth_rate,
                              kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout(0.0)

    def forward(self, x):
        # Apply BN + ReLU
        bn = self.bn(x)
        relu = F.relu(bn, inplace=True)

        # Prepare parameters for Triton kernel
        weight = self.conv.weight
        gamma = self.conv.weight  # unused, placeholder
        beta = torch.zeros_like(weight)
        mean = self.bn.running_mean
        var = self.bn.running_var

        # Use fused kernel
        out = conv_bn_relu_fused(relu, weight, gamma, beta, mean, var)
        out = self.dropout(out)
        return out

class ModelNew(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        super(ModelNew, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(
                ConvBNReLULayer(num_input_features + i * growth_rate, growth_rate)
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(x)
            features.append(new_feature)
            x = torch.cat(features, 1)
        return x