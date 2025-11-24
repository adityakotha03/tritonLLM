import torch
import torch.nn as nn
import triton
import triton.language as tl


# ---------- Triton linear layer -----------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_K": 128, "BLOCK_SIZE_N": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_K": 128, "BLOCK_SIZE_N": 256}, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 512, "BLOCK_SIZE_K": 256, "BLOCK_SIZE_N": 512}, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_fused_kernel(
    a_ptr,   # (M, K)
    b_ptr,   # (N, K)
    c_ptr,   # (M, N)
    stride_a_m: tl.constexpr,
    stride_a_k: tl.constexpr,
    stride_b_n: tl.constexpr,
    stride_b_k: tl.constexpr,
    stride_c_m: tl.constexpr,
    stride_c_n: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N

    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

    for k_start in range(0, K, BLOCK_SIZE_K):
        # Load blocks of A and B
        a_block = tl.load(
            a_ptr + (m_start + tl.arange(0, BLOCK_SIZE_M))[:, None] * stride_a_m +
                     (k_start + tl.arange(0, BLOCK_SIZE_K))[None, :] * stride_a_k,
            mask=(m_start + tl.arange(0, BLOCK_SIZE_M))[:, None] < M &
                 (k_start + tl.arange(0, BLOCK_SIZE_K))[None, :] < K,
            other=0.0,
        )
        b_block = tl.load(
            b_ptr + (n_start + tl.arange(0, BLOCK_SIZE_N))[:, None] * stride_b_n +
                     (k_start + tl.arange(0, BLOCK_SIZE_K))[None, :] * stride_b_k,
            mask=(n_start + tl.arange(0, BLOCK_SIZE_N))[:, None] < N &
                 (k_start + tl.arange(0, BLOCK_SIZE_K))[None, :] < K,
            other=0.0,
        )
        acc += tl.dot(a_block, b_block.T)

    # Write the result
    tl.store(
        c_ptr + (m_start + tl.arange(0, BLOCK_SIZE_M))[:, None] * stride_c_m +
                (n_start + tl.arange(0, BLOCK_SIZE_N))[None, :] * stride_c_n,
        acc,
        mask=(m_start + tl.arange(0, BLOCK_SIZE_M))[:, None] < M &
             (n_start + tl.arange(0, BLOCK_SIZE_N))[None, :] < N,
    )


def triton_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    """
    x: (batch, in_features)
    weight: (out_features, in_features)
    bias: (out_features,)
    """
    assert x.is_cuda and weight.is_cuda, "Inputs must be on CUDA."
    batch, in_features = x.shape
    out_features = weight.shape[0]
    out = torch.empty(batch, out_features, device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        (batch + meta["BLOCK_SIZE_M"] - 1) // meta["BLOCK_SIZE_M"],
        (out_features + meta["BLOCK_SIZE_N"] - 1) // meta["BLOCK_SIZE_N"],
    )

    _matmul_fused_kernel[grid](
        x,
        weight,
        out,
        stride_a_m=1,
        stride_a_k=in_features,
        stride_b_n=1,
        stride_b_k=weight.shape[1],
        stride_c_m=out_features,
        stride_c_n=1,
        M=batch,
        N=out_features,
        K=in_features,
    )

    if bias is not None:
        out += bias
    return out


# ---------- ModelNew -------------------------------------------------------

class ModelNew(nn.Module):
    def __init__(self, num_classes, embed_dim=512, num_heads=8, num_layers=6,
                 mlp_ratio=4.0, patch_size=4, in_channels=3, image_size=32):
        super(ModelNew, self).__init__()

        self.patch_size = patch_size
        self.image_size = image_size
        self.embed_dim = embed_dim

        self.conv1 = nn.Conv2d(in_channels, embed_dim,
                               kernel_size=patch_size, stride=patch_size)

        num_patches = (image_size // patch_size) ** 2
        self.linear_proj_weight = nn.Parameter(
            torch.randn(embed_dim, embed_dim * num_patches, dtype=torch.float32))
        self.linear_proj_bias = nn.Parameter(torch.zeros(embed_dim, dtype=torch.float32))

        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=0.0,
                batch_first=True
            ) for _ in range(num_layers)
        ])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.fc_out = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.conv1(x)                    # (B, embed_dim, H', W')
        x = x.flatten(start_dim=1)           # (B, embed_dim * num_patches)

        # Triton linear projection
        x = triton_linear(x, self.linear_proj_weight, self.linear_proj_bias)  # (B, embed_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x.unsqueeze(1)), dim=1)  # (B, 2, embed_dim)

        for layer in self.transformer_layers:
            x = layer(x)

        return self.fc_out(x[:, 0])          # (B, num_classes)