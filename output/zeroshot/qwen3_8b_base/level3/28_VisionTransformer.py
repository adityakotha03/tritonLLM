import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def patch_to_embedding_kernel(
    img_ptr,  # Pointer to input image tensor
    out_ptr,  # Pointer to output embedding tensor
    batch_size: tl.constexpr,
    num_patches: tl.constexpr,
    patch_dim: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_patches * batch_size

    # Compute the index in the input image
    idx = tl.load(img_ptr + offsets, mask=mask, other=0.0)
    idx = idx.to(tl.int32)

    # Compute the linear index in the output
    out_idx = tl.arange(0, BLOCK_SIZE)
    out_idx = out_idx + block_start

    # Compute the input indices
    batch_idx = (idx // (patch_dim)) % batch_size
    patch_idx = (idx // (patch_dim)) // batch_size
    channel_idx = (idx % (patch_dim)) // (patch_size * patch_size)
    row_idx = (idx % (patch_dim)) // (patch_size)
    col_idx = (idx % (patch_dim)) % patch_size

    # Compute the input tensor indices
    input_idx = batch_idx * (patch_size * patch_size * channels) + \
                patch_idx * (patch_size * patch_size) + \
                channel_idx * (patch_size * patch_size) + \
                row_idx * patch_size + col_idx

    # Load input values
    input_val = tl.load(img_ptr + input_idx, mask=mask, other=0.0)

    # Compute the output values
    output_val = tl.sum(input_val, axis=0)

    # Store output values
    tl.store(out_ptr + out_idx, output_val, mask=mask)


@triton.jit
def matmul_kernel(
    A_ptr,  # Pointer to matrix A
    B_ptr,  # Pointer to matrix B
    C_ptr,  # Pointer to output matrix C
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    k: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_rows

    # Compute the row index in A
    row_idx = tl.load(A_ptr + offsets, mask=mask, other=0.0).to(tl.int32)

    # Compute the column index in B
    col_idx = tl.arange(0, BLOCK_SIZE)
    col_idx = col_idx + block_start

    # Compute the dot product
    dot = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(k // BLOCK_SIZE):
        a = tl.load(A_ptr + row_idx * k + i * BLOCK_SIZE + offsets, mask=mask, other=0.0)
        b = tl.load(B_ptr + col_idx * k + i * BLOCK_SIZE, other=0.0)
        dot += tl.dot(a, b)

    # Store the result
    tl.store(C_ptr + row_idx * n_cols + col_idx, dot, mask=mask)


@triton.jit
def gelu_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = 0.5 * x * (1.0 + tl.erf(x / tl.sqrt(2.0)))
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def mlp_head_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements: tl.constexpr,
    mlp_dim: tl.constexpr,
    num_classes: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x = tl.sum(x, axis=0)
    x = x / n_elements

    x = x * mlp_dim
    x = x + mlp_dim
    x = x / num_classes

    tl.store(out_ptr + offsets, x, mask=mask)


def triton_patch_to_embedding(img: torch.Tensor, batch_size: int, num_patches: int, patch_dim: int, dim: int):
    assert img.is_cuda, "Input tensor must be on CUDA."
    img = img.contiguous()
    out = torch.empty(batch_size * num_patches, dim, device=img.device, dtype=img.dtype)

    n_elements = batch_size * num_patches
    BLOCK_SIZE = 128

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    patch_to_embedding_kernel[grid](img, out, batch_size, num_patches, patch_dim, dim, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_matmul(A: torch.Tensor, B: torch.Tensor, n_rows: int, n_cols: int, k: int):
    assert A.is_cuda and B.is_cuda, "Input tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    C = torch.empty(n_rows, n_cols, device=A.device, dtype=A.dtype)

    n_elements = n_rows * n_cols
    BLOCK_SIZE = 128

    grid = lambda meta: ((n_rows + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    matmul_kernel[grid](A, B, C, n_rows, n_cols, k, BLOCK_SIZE=BLOCK_SIZE)
    return C


def triton_gelu(x: torch.Tensor, n_elements: int):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty(n_elements, device=x.device, dtype=x.dtype)

    BLOCK_SIZE = 128

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    gelu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_mlp_head(x: torch.Tensor, n_elements: int, mlp_dim: int, num_classes: int):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty(n_elements, device=x.device, dtype=x.dtype)

    BLOCK_SIZE = 128

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    mlp_head_kernel[grid](x, out, n_elements, mlp_dim, num_classes, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dropout=0.1, emb_dropout=0.1):
        super(ModelNew, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.channels = channels
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.num_classes = num_classes
        self.dropout = dropout
        self.emb_dropout = emb_dropout

        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = channels * patch_size ** 2

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout),
            num_layers=depth
        )

        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, img):
        p = self.patch_size
        batch_size = img.shape[0]

        # Patch to embedding
        x = img.unfold(2, p, p).unfold(3, p, p).reshape(img.shape[0], -1, p*p*img.shape[1])
        x = triton_patch_to_embedding(x, batch_size, self.num_patches, self.patch_dim, self.dim)

        # Add cls token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)

        # Transformer
        x = self.transformer(x)

        # Get cls token
        x = self.to_cls_token(x[:, 0])

        # MLP head
        x = triton_mlp_head(x, x.shape[0], self.mlp_dim, self.num_classes)
        return x