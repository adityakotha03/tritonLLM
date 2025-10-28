import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=4),
    ],
    key=["n_cols"],
)
@triton.jit
def softmax_rowwise_kernel(
    X_ptr,
    Y_ptr,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    X_row_ptr = X_ptr + row_id * stride_xm
    Y_row_ptr = Y_ptr + row_id * stride_ym

    offs = tl.arange(0, BLOCK_SIZE)

    # Pass 1: online reduction to compute row-wise max and sum of exp
    m = tl.full((), -float("inf"), tl.float32)
    l = tl.zeros((), tl.float32)

    col = 0
    while col < n_cols:
        idx = col + offs
        mask = idx < n_cols
        x = tl.load(X_row_ptr + idx * stride_xn, mask=mask, other=-float("inf"))
        x_fp32 = x.to(tl.float32)

        x_max = tl.max(x_fp32, axis=0)
        m_new = tl.maximum(m, x_max)
        exp_scale = tl.exp(m - m_new)
        l = l * exp_scale + tl.sum(tl.exp(x_fp32 - m_new), axis=0)
        m = m_new
        col += BLOCK_SIZE

    # Pass 2: write normalized probabilities
    col = 0
    inv_l = 1.0 / l
    while col < n_cols:
        idx = col + offs
        mask = idx < n_cols
        x = tl.load(X_row_ptr + idx * stride_xn, mask=mask, other=-float("inf"))
        x_fp32 = x.to(tl.float32)
        y = tl.exp(x_fp32 - m) * inv_l
        tl.store(Y_row_ptr + idx * stride_yn, y, mask=mask)
        col += BLOCK_SIZE


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input must be on CUDA for Triton softmax."
    assert x.dim() == 2, "Expected 2D tensor (batch_size, num_features)."
    x = x.contiguous()
    M, N = x.shape
    out = torch.empty_like(x)
    grid = lambda meta: (M,)
    softmax_rowwise_kernel[grid](
        x,
        out,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        N,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a row-wise Softmax using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            return triton_softmax(x)
        # Fallback to PyTorch on CPU or unsupported cases
        return torch.softmax(x, dim=1)


# Keep the same API for generating inputs
batch_size = 4096
dim = 393216


def get_inputs():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.rand(batch_size, dim, device=device)
    return [x]


def get_init_inputs():
    return []