import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ---------------- Triton kernels ------------------------------------------

# Fused BatchNorm + ReLU
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 32}, num_warps=8),
    ],
    key=['M', 'K'],
)
@triton.jit
def batchnorm_relu_kernel(
    X_ptr,          # [N, C, H, W] flattened
    gamma_ptr,      # [C]
    beta_ptr,       # [C]
    mean_ptr,       # [C]
    var_ptr,        # [C]
    eps,            # float
    out_ptr,        # output [N, C, H, W]
    N, C, H, W, K,  # K = C*H*W
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row = pid_m * BLOCK_SIZE_M
    col = pid_n * BLOCK_SIZE_N

    for n in range(row, min(row + BLOCK_SIZE_M, N)):
        for c in range(col, min(col + BLOCK_SIZE_N, C)):
            for h in range(H):
                for w in range(W):
                    idx = ((n * C + c) * H + h) * W + w
                    x = tl.load(X_ptr + idx)
                    m = tl.load(mean_ptr + c)
                    v = tl.load(var_ptr + c)
                    g = tl.load(gamma_ptr + c)
                    b = tl.load(beta_ptr + c)
                    bn = (x - m) * tl.math.rsqrt(v + eps)
                    y = g * bn + b
                    y = tl.max(y, 0.0)  # ReLU
                    tl.store(out_ptr + idx, y)


def triton_batchnorm_relu(x, gamma, beta, mean, var, eps=1e-5):
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    M = N * C
    K = C * H * W

    grid = lambda meta: ( (M + meta['BLOCK_SIZE_M'] - 1) // meta['BLOCK_SIZE_M'],
                          (C + meta['BLOCK_SIZE_N'] - 1) // meta['BLOCK_SIZE_N'] )
    batchnorm_relu_kernel[grid](
        x.contiguous().view(-1),
        gamma,
        beta,
        mean,
        var,
        eps,
        out.contiguous().view(-1),
        M, C, H, W, K
    )
    return out


# Triton linear (matrix multiply + bias)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    A_ptr,  # [M, K]
    B_ptr,  # [K, N]
    C_ptr,  # [M, N]
    bias_ptr,  # [N]
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row = pid_m * BLOCK_SIZE_M
    col = pid_n * BLOCK_SIZE_N

    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        A_block = tl.load(A_ptr + (row + tl.arange(0, BLOCK_SIZE_M))[:, None] * K
                          + (k + tl.arange(0, BLOCK_SIZE_K))[None, :], 
                         mask=(row + tl.arange(0, BLOCK_SIZE_M))[:, None] < M, 
                         other=0.0)
        B_block = tl.load(B_ptr + (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * N
                          + (col + tl.arange(0, BLOCK_SIZE_N))[None, :], 
                         mask=(col + tl.arange(0, BLOCK_SIZE_N))[None, :] < N, 
                         other=0.0)
        acc += tl.dot(A_block, B_block)

    if row + tl.arange(0, BLOCK_SIZE_M) < M and col + tl.arange(0, BLOCK_SIZE_N) < N:
        bias = tl.load(bias_ptr + col + tl.arange(0, BLOCK_SIZE_N))
        acc += bias[None, :]
        tl.store(C_ptr + (row + tl.arange(0, BLOCK_SIZE_M))[:, None] * N
                 + (col + tl.arange(0, BLOCK_SIZE_N))[None, :], acc)


def triton_linear(x, weight, bias):
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    grid = lambda meta: ( (M + meta['BLOCK_SIZE_M'] - 1) // meta['BLOCK_SIZE_M'],
                          (N + meta['BLOCK_SIZE_N'] - 1) // meta['BLOCK_SIZE_N'] )
    linear_kernel[grid](
        x.contiguous(),
        weight.t().contiguous(),
        out.contiguous(),
        bias,
        M, N, K
    )
    return out


# ---------------- Custom Modules ------------------------------------------

class FusedBatchNormReLU(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.training = True

    def forward(self, x):
        if self.training:
            batch_mean = x.mean([0, 2, 3])
            batch_var = x.var([0, 2, 3], unbiased=False)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            mean = batch_mean
            var = batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        gamma = self.gamma if self.affine else torch.ones_like(mean)
        beta = self.beta if self.affine else torch.zeros_like(mean)

        return triton_batchnorm_relu(x, gamma, beta, mean, var, self.eps)


class TritonLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return triton_linear(x, self.weight, self.bias)


# ---------------- Optimized Architecture -----------------------------------

class DenseBlock(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(self._make_layer(num_input_features + i * growth_rate, growth_rate))
        self.layers = nn.ModuleList(layers)

    def _make_layer(self, in_features: int, growth_rate: int):
        return nn.Sequential(
            FusedBatchNormReLU(in_features),
            nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.Dropout(0.0),
        )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(x)
            features.append(new_feature)
            x = torch.cat(features, 1)
        return x


class TransitionLayer(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        super(TransitionLayer, self).__init__()
        self.transition = nn.Sequential(
            FusedBatchNormReLU(num_input_features),
            nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.transition(x)


class ModelNew(nn.Module):
    def __init__(self, growth_rate: int = 32, num_classes: int = 1000):
        super(ModelNew, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            FusedBatchNormReLU(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        num_features = 64
        block_layers = [6, 12, 48, 32]

        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()

        for i, num_layers in enumerate(block_layers):
            block = DenseBlock(num_layers=num_layers, num_input_features=num_features, growth_rate=growth_rate)
            self.dense_blocks.append(block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_layers) - 1:
                transition = TransitionLayer(num_input_features=num_features, num_output_features=num_features // 2)
                self.transition_layers.append(transition)
                num_features = num_features // 2

        self.final_bn = FusedBatchNormReLU(num_features)
        self.classifier = TritonLinear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)

        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transition_layers[i](x)

        x = self.final_bn(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.classifier(x)
        return x