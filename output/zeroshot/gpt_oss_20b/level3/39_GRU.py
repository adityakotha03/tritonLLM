import torch
import torch.nn as nn
import triton
import triton.language as tl

# ---------- Triton kernels for the core GRU operations ----------

# Matrix multiplication kernel (matmul of shape [M, K] @ [K, N] -> [M, N])
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 8}, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 16}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16}, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_f32_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptr + (row[:, None] * stride_am) + (k + tl.arange(0, BLOCK_SIZE_K)[None, :] * stride_ak),
                    mask=row[:, None] < M,
                    other=0.0)
        b = tl.load(b_ptr + (k + tl.arange(0, BLOCK_SIZE_K)[None, :] * stride_bk) + (col[None, :] * stride_bn),
                    mask=col[None, :] < N,
                    other=0.0)
        acc += tl.dot(a, b)

    tl.store(c_ptr + (row[:, None] * stride_cm) + (col[None, :] * stride_cn),
             acc,
             mask=(row[:, None] < M) & (col[None, :] < N))


def matmul_f32(A, B):
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)
    grid = ( (M + 127) // 128, (N + 127) // 128 )
    _matmul_f32_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am=1, stride_ak=M,
        stride_bk=1, stride_bn=K,
        stride_cm=1, stride_cn=N,
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=8
    )
    return C


# Sigmoid kernel
@triton.jit
def _sigmoid_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(y_ptr + offsets, y, mask=mask)


def sigmoid_t(x):
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 256
    grid = lambda meta: ((n + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _sigmoid_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out


# Tanh kernel
@triton.jit
def _tanh_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = (tl.exp(2.0 * x) - 1.0) / (tl.exp(2.0 * x) + 1.0)
    tl.store(y_ptr + offsets, y, mask=mask)


def tanh_t(x):
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 256
    grid = lambda meta: ((n + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _tanh_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out


# ---------- Custom GRU implementation using Triton kernels ----------

class TritonGRUCell(nn.Module):
    """
    One-step GRU cell implemented with Triton kernels.
    Assumes all inputs are in FP32.
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # weights: (hidden_size, input_size + hidden_size)
        self.weight_ih = nn.Parameter(torch.randn(hidden_size, input_size + hidden_size, dtype=torch.float32))
        self.weight_hh = nn.Parameter(torch.randn(hidden_size, hidden_size, dtype=torch.float32))
        # biases
        self.bias_ih = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))
        self.bias_hh = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))

    def forward(self, x_t, h_prev):
        # Concatenate x_t and h_prev
        concat = torch.cat([x_t, h_prev], dim=1)  # shape (B, input+hidden)
        # Pre-activations
        pre = matmul_f32(concat, self.weight_ih.T) + self.bias_ih
        hidden = matmul_f32(h_prev, self.weight_hh.T) + self.bias_hh

        # Gates
        z = sigmoid_t(pre[:, :self.weight_ih.size(0)//3] + hidden[:, :self.weight_hh.size(0)//3])
        r = sigmoid_t(pre[:, self.weight_ih.size(0)//3:2*self.weight_ih.size(0)//3] + hidden[:, self.weight_hh.size(0)//3:2*self.weight_hh.size(0)//3])
        n = tanh_t(pre[:, 2*self.weight_ih.size(0)//3:] + r * hidden[:, 2*self.weight_hh.size(0)//3:])

        h_new = (1 - z) * n + z * h_prev
        return h_new


class TritonGRU(nn.Module):
    """
    Multi-layer unidirectional GRU implemented with Triton kernels.
    """
    def __init__(self, input_size, hidden_size, num_layers, bias=True):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.cells = nn.ModuleList()
        for l in range(num_layers):
            in_size = input_size if l == 0 else hidden_size
            self.cells.append(TritonGRUCell(in_size, hidden_size))

    def forward(self, x, h0):
        # x: (seq_len, batch, input)
        seq_len, batch, _ = x.shape
        outputs = []
        h_n = []
        h = h0
        for t in range(seq_len):
            x_t = x[t]
            layer_h = []
            for l, cell in enumerate(self.cells):
                h_prev = h[l]
                h_new = cell(x_t, h_prev)
                x_t = h_new
                layer_h.append(h_new)
            h = torch.stack(layer_h, dim=0)
            outputs.append(x_t)
        output = torch.stack(outputs, dim=0)
        return output, h


# ---------- ModelNew ----------

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        super(ModelNew, self).__init__()
        self.gru = TritonGRU(input_size, hidden_size, num_layers, bias)

    def forward(self, x, h0):
        # Assuming x is (seq_len, batch, input)
        return self.gru(x, h0)[0]