import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    n, m, k,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the block index
    pid = tl.program_id(0)
    # Compute the row and column indices for this block
    row = pid // (m // BLOCK_SIZE)
    col = pid % (m // BLOCK_SIZE)
    # Compute the starting row and column for this block
    row_start = row * BLOCK_SIZE
    col_start = col * BLOCK_SIZE
    # Compute the offset for this block
    offsets = (
        tl.arange(0, BLOCK_SIZE)[:, None] * m + tl.arange(0, BLOCK_SIZE)[None, :]
    )
    # Load the A and B matrices
    a = tl.load(a_ptr + row_start + tl.arange(0, BLOCK_SIZE)[:, None] * k, mask=(tl.arange(0, BLOCK_SIZE)[:, None] * k + tl.arange(0, BLOCK_SIZE)) < (n * k), other=0.0)
    b = tl.load(b_ptr + col_start + tl.arange(0, BLOCK_SIZE)[None, :] * k, mask=(tl.arange(0, BLOCK_SIZE)[None, :] * k + tl.arange(0, BLOCK_SIZE)) < (m * k), other=0.0)
    # Compute the product
    c = tl.dot(a, b)
    # Store the result
    tl.store(c_ptr + row_start + col_start + tl.arange(0, BLOCK_SIZE)[:, None] * m + tl.arange(0, BLOCK_SIZE)[None, :], c, mask=(row_start + tl.arange(0, BLOCK_SIZE)[:, None] * m + col_start + tl.arange(0, BLOCK_SIZE)[None, :]) < (n * m), other=0.0)


def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    """
    Custom Triton implementation of matrix multiplication.
    """
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    a = a.contiguous()
    b = b.contiguous()
    n, k = a.shape
    m, _ = b.shape
    c = torch.empty((n, m), device=a.device, dtype=a.dtype)
    # Determine the block size
    BLOCK_SIZE = 128
    # Determine the number of blocks
    num_blocks = (n * m + BLOCK_SIZE - 1) // BLOCK_SIZE
    # Launch the kernel
    matmul_kernel[(num_blocks,)](a, b, c, n, m, k, BLOCK_SIZE=BLOCK_SIZE)
    return c


@triton.jit
def matmul_relu_kernel(
    a_ptr, b_ptr, c_ptr,
    n, m, k,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the block index
    pid = tl.program_id(0)
    # Compute the row and column indices for this block
    row = pid // (m // BLOCK_SIZE)
    col = pid % (m // BLOCK_SIZE)
    # Compute the starting row and column for this block
    row_start = row * BLOCK_SIZE
    col_start = col * BLOCK_SIZE
    # Compute the offset for this block
    offsets = (
        tl.arange(0, BLOCK_SIZE)[:, None] * m + tl.arange(0, BLOCK_SIZE)[None, :]
    )
    # Load the A and B matrices
    a = tl.load(a_ptr + row_start + tl.arange(0, BLOCK_SIZE)[:, None] * k, mask=(tl.arange(0, BLOCK_SIZE)[:, None] * k + tl.arange(0, BLOCK_SIZE)) < (n * k), other=0.0)
    b = tl.load(b_ptr + col_start + tl.arange(0, BLOCK_SIZE)[None, :] * k, mask=(tl.arange(0, BLOCK_SIZE)[None, :] * k + tl.arange(0, BLOCK_SIZE)) < (m * k), other=0.0)
    # Compute the product
    c = tl.dot(a, b)
    # Apply ReLU
    c = tl.maximum(c, 0.0)
    # Store the result
    tl.store(c_ptr + row_start + col_start + tl.arange(0, BLOCK_SIZE)[:, None] * m + tl.arange(0, BLOCK_SIZE)[None, :], c, mask=(row_start + tl.arange(0, BLOCK_SIZE)[:, None] * m + col_start + tl.arange(0, BLOCK_SIZE)[None, :]) < (n * m), other=0.0)


def triton_matmul_relu(a: torch.Tensor, b: torch.Tensor):
    """
    Custom Triton implementation of matrix multiplication followed by ReLU.
    """
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    a = a.contiguous()
    b = b.contiguous()
    n, k = a.shape
    m, _ = b.shape
    c = torch.empty((n, m), device=a.device, dtype=a.dtype)
    # Determine the block size
    BLOCK_SIZE = 128
    # Determine the number of blocks
    num_blocks = (n * m + BLOCK_SIZE - 1) // BLOCK_SIZE
    # Launch the kernel
    matmul_relu_kernel[(num_blocks,)](a, b, c, n, m, k, BLOCK_SIZE=BLOCK_SIZE)
    return c


class ModelNew(nn.Module):
    def __init__(self, num_classes, embed_dim=512, num_heads=8, num_layers=6, 
                 mlp_ratio=4.0, patch_size=4, in_channels=3, image_size=32):
        """
        Convolutional Vision Transformer (CViT) implementation with custom Triton kernels.
        :param num_classes: Number of output classes for classification.
        :param embed_dim: Dimensionality of the embedding space.
        :param num_heads: Number of attention heads.
        :param num_layers: Number of transformer layers.
        :param mlp_ratio: Ratio of the MLP hidden dimension to the embedding dimension.
        :param patch_size: Size of the convolutional patches.
        :param in_channels: Number of input channels (e.g., 3 for RGB images).
        :param image_size: Height/width of the square input image.
        """
        super(ModelNew, self).__init__()

        self.patch_size = patch_size
        self.image_size = image_size
        self.embed_dim = embed_dim

        self.conv1 = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2  # Total number of patches after conv
        self.linear_proj = nn.Linear(embed_dim * num_patches, embed_dim)

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

        # Custom Triton kernels for matrix multiplication and matmul + ReLU
        self.triton_matmul = triton_matmul
        self.triton_matmul_relu = triton_matmul_relu

    def forward(self, x):
        """
        Forward pass of the CViT model with custom Triton kernels.
        :param x: Input tensor of shape (B, C, H, W)
        :return: Output tensor of shape (B, num_classes)
        """
        B = x.size(0)
        x = self.conv1(x)                  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(start_dim=1)         # (B, embed_dim * num_patches)
        x = self.linear_proj(x)            # (B, embed_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x.unsqueeze(1)), dim=1)  # (B, 2, embed_dim)

        for layer in self.transformer_layers:
            x = layer(x)

        return self.fc_out(x[:, 0])        # Use [CLS] token for classification