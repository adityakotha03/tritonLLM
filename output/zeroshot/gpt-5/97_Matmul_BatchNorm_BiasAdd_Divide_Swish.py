import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _bias_div_swish_kernel(
    x_ptr,
    out_ptr,
    bias_ptr,
    n_elements,
    inv_divide_value: tl.constexpr,  # compile-time constant if desired
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    bias_val = tl.load(bias_ptr)  # scalar
    z = (x + bias_val) * inv_divide_value
    # sigmoid(z) = 1 / (1 + exp(-z))
    sig = 1.0 / (1.0 + tl.exp(-z))
    out = z * sig
    tl.store(out_ptr + offs, out, mask=mask)


def triton_bias_div_swish(x: torch.Tensor, bias: torch.Tensor, divide_value: float):
    if not x.is_cuda:
        z = (x + bias) / divide_value
        return z * torch.sigmoid(z)
    assert bias.numel() == 1, "bias must be a scalar tensor with shape (1,)"
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    inv_div = 1.0 / float(divide_value)
    _bias_div_swish_kernel[grid](
        x,
        out,
        bias,
        n_elements,
        inv_divide_value=inv_div,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


@triton.jit
def _bn_infer_bias_div_swish_kernel(
    x_ptr,          # float* [B, C]
    w_ptr,          # float* [C]   (gamma)
    b_ptr,          # float* [C]   (beta)
    mean_ptr,       # float* [C]
    var_ptr,        # float* [C]
    out_ptr,        # float* [B, C]
    B, C,           # int
    stride_x_m, stride_x_n,  # strides for x: row-major expected (C, 1)
    eps,            # float
    bias_scalar,    # float
    inv_div,        # float
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)  # row id
    pid_n = tl.program_id(1)  # col block id

    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = cols < C

    # Load per-channel params
    gamma = tl.load(w_ptr + cols, mask=mask, other=1.0)
    beta = tl.load(b_ptr + cols, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + cols, mask=mask, other=0.0)
    var = tl.load(var_ptr + cols, mask=mask, other=0.0)

    inv_std = 1.0 / tl.sqrt(var + eps)

    x_row_ptr = x_ptr + pid_m * stride_x_m + cols * stride_x_n
    x = tl.load(x_row_ptr, mask=mask, other=0.0)

    y = (x - mean) * inv_std
    y = y * gamma + beta
    y = (y + bias_scalar) * inv_div
    sig = 1.0 / (1.0 + tl.exp(-y))
    out = y * sig

    out_row_ptr = out_ptr + pid_m * stride_x_m + cols * stride_x_n
    tl.store(out_row_ptr, out, mask=mask)


def triton_bn_infer_bias_div_swish(x: torch.Tensor, bn: nn.BatchNorm1d, bias: torch.Tensor, divide_value: float):
    if not x.is_cuda:
        # Fallback to PyTorch for CPU
        w = bn.weight if bn.weight is not None else torch.ones(x.size(1), dtype=x.dtype, device=x.device)
        b = bn.bias if bn.bias is not None else torch.zeros(x.size(1), dtype=x.dtype, device=x.device)
        mean = bn.running_mean
        var = bn.running_var
        y = (x - mean) / torch.sqrt(var + bn.eps)
        y = y * w + b
        y = (y + bias) / divide_value
        return y * torch.sigmoid(y)

    assert bias.numel() == 1, "bias must be a scalar tensor with shape (1,)"
    B, C = x.shape
    x = x.contiguous()
    out = torch.empty_like(x)

    # Prepare BN params with defaults if None
    w = bn.weight if bn.weight is not None else torch.ones(C, dtype=x.dtype, device=x.device)
    b = bn.bias if bn.bias is not None else torch.zeros(C, dtype=x.dtype, device=x.device)
    mean = bn.running_mean
    var = bn.running_var

    # Ensure contiguity
    w = w.contiguous()
    b = b.contiguous()
    mean = mean.contiguous()
    var = var.contiguous()

    # Strides for row-major [B, C]
    stride_x_m = x.stride(0)
    stride_x_n = x.stride(1)

    BLOCK_N = 256
    grid = (B, (C + BLOCK_N - 1) // BLOCK_N)
    _bn_infer_bias_div_swish_kernel[grid](
        x, w, b, mean, var, out,
        B, C,
        stride_x_m, stride_x_n,
        bn.eps,
        float(bias.item()),
        1.0 / float(divide_value),
        BLOCK_N=BLOCK_N,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that keeps the matmul and BatchNorm as PyTorch ops in training,
    and uses fused Triton kernels for elementwise ops in training and fully fused
    BN+post-ops in inference.
    """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.divide_value = divide_value

    def forward(self, x):
        x = self.matmul(x)
        if self.training:
            # Use native BN for training (updates running stats), then fused elementwise ops
            x = self.bn(x)
            x = triton_bias_div_swish(x, self.bias, self.divide_value)
        else:
            # Fully fuse BN inference + bias + division + swish into one Triton kernel
            x = triton_bn_infer_bias_div_swish(x, self.bn, self.bias, self.divide_value)
        return x


batch_size = 1024
in_features = 8192
out_features = 8192
bn_eps = 1e-5
bn_momentum = 0.1
bias_shape = (1,)
divide_value = 1.0


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, bn_eps, bn_momentum, bias_shape, divide_value]