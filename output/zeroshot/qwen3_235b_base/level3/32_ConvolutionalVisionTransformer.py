import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_conv2d_gelu_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch, height, width, in_channels, out_channels, patch_size,
    input_stride, output_stride,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_CIN: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    # Compute starting positions
    h_start = pid_h * BLOCK_H
    w_start = pid_w * BLOCK_W

    offs_b = pid_b
    offs_h = h_start + tl.arange(0, BLOCK_H)
    offs_w = w_start + tl.arange(0, BLOCK_W)
    offs_cin = tl.arange(0, BLOCK_CIN)
    offs_cout = tl.arange(0, out_channels)

    # Input and output masks
    h_mask = offs_h < height
    w_mask = offs_w < width
    input_mask = (offs_b < batch)[:, None, None, None] & h_mask[:, None, None] & w_mask[None, :, None] & (offs_cin[None, None, :] < in_channels)
    
    # Load input tiles
    input_offset = offs_b[:, None, None, None] * input_stride + \
                   offs_h[:, None, None] * patch_size * width * in_channels + \
                   offs_w[None, :, None] * patch_size * in_channels + \
                   offs_cin[None, None, :]
    input_tile = tl.load(input_ptr + input_offset, mask=input_mask, other=0.0)

    # Convolution: contract over input channels and spatial dims of kernel
    # Here kernel is of shape (out_channels, in_channels, patch_size, patch_size)
    weight_offset = offs_cout[:, None, None, None] * in_channels * patch_size * patch_size + \
                    offs_cin[None, :, None, None] * patch_size * patch_size + \
                    tl.arange(0, patch_size)[:, None] * patch_size + \
                    tl.arange(0, patch_size)[None, :]
    weight = tl.load(weight_ptr + weight_offset)

    # Perform convolution (einsum: 'bhwk,okpq->bo')
    conv_output = tl.zeros((BLOCK_H, BLOCK_W, out_channels), dtype=tl.float32)
    for ih in range(patch_size):
        for iw in range(patch_size):
            x = tl.expand_dims(input_tile[:, :, :, ih::patch_size, iw::patch_size], 3)  # broadcasting trick
            w = tl.expand_dims(weight[:, :, ih, iw], 0)  # (1, cin, out)
            conv_output += tl.sum(x * w, axis=2)  # sum over cin

    # Add bias
    bias = tl.load(bias_ptr + offs_cout)
    conv_output += bias[None, None, :]

    # GELU activation
    gelu_output = 0.5 * conv_output * (1.0 + tl.math.erf(conv_output * 0.70710678))

    # Store output
    output_mask = (offs_b < batch)[:, None, None] & (offs_h < height)[:, None] & (offs_w < width)[:, None]
    output_offset = offs_b[:, None, None] * output_stride + \
                    offs_h[:, None] * width + \
                    offs_w[None, :] + \
                    tl.arange(0, out_channels)[None, None, :]
    tl.store(output_ptr + output_offset, gelu_output, mask=output_mask)


def triton_fused_conv2d_gelu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, patch_size: int):
    B, C, H, W = x.shape
    out_channels, in_channels, kh, kw = weight.shape
    assert kh == patch_size and kw == patch_size
    OH, OW = H // patch_size, W // patch_size

    # Output tensor
    out = torch.empty((B, out_channels, OH, OW), device=x.device, dtype=x.dtype)

    # Compute strides
    input_stride = x.stride(0)
    output_stride = out.stride(0)

    # Launch kernel
    def grid(meta):
        return (B, triton.cdiv(OH, meta['BLOCK_H']), triton.cdiv(OW, meta['BLOCK_W']))

    fused_conv2d_gelu_kernel[grid](
        x, weight, bias, out,
        B, OH, OW, in_channels, out_channels, patch_size,
        input_stride, output_stride,
        BLOCK_H=16, BLOCK_W=16, BLOCK_CIN=16
    )
    return out


@triton.jit
def matmul_add_gelu_kernel(
    x_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, out_ptr,
    n_rows, n_cols, n_inner,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = n_start + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < n_rows
    mask_n = offs_n < n_cols

    # Load first matmul block
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, n_inner, BLOCK_K):
        k_offs = k + offs_k
        mask_k = k_offs < n_inner

        x = tl.load(x_ptr + offs_m[:, None] * n_inner + k_offs[None, :], mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w1 = tl.load(w1_ptr + k_offs[:, None] * n_cols + offs_n[None, :], mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(x, w1)

    # Add bias
    b1 = tl.load(b1_ptr + offs_n, mask=mask_n)
    acc += b1[None, :]

    # GELU
    gelu = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.70710678))

    # Second matmul
    acc2 = tl.zeros((BLOCK_M, n_cols), dtype=tl.float32)
    for k in range(0, n_cols, BLOCK_K):
        k_offs = k + offs_k
        mask_k = k_offs < n_cols

        x2 = tl.load(gelu + offs_m[:, None] * n_cols + k_offs[None, :], mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w2 = tl.load(w2_ptr + k_offs[:, None] * n_cols + offs_n[None, :], mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc2 += tl.dot(x2, w2)

    # Add second bias
    b2 = tl.load(b2_ptr + offs_n, mask=mask_n)
    acc2 += b2[None, :]

    # Store output
    out_offs = offs_m[:, None] * n_cols + offs_n[None, :]
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_offs, acc2, mask=out_mask)


def triton_fused_mlp_gelu(x: torch.Tensor, fc1_weight: torch.Tensor, fc1_bias: torch.Tensor,
                          fc2_weight: torch.Tensor, fc2_bias: torch.Tensor):
    n_rows, n_inner = x.shape
    n_cols = fc1_weight.shape[1]

    out = torch.empty((n_rows, n_cols), device=x.device, dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(n_rows, meta['BLOCK_M']), triton.cdiv(n_cols, meta['BLOCK_N']))

    matmul_add_gelu_kernel[grid](
        x, fc1_weight, fc1_bias, fc2_weight, fc2_bias, out,
        n_rows, n_cols, n_inner,
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32
    )
    return out


@triton.jit
def layer_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    n_rows, n_cols,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_N
    cols = row_start + tl.arange(0, BLOCK_N)
    mask = cols < n_cols

    for row in range(n_rows):
        mean = tl.sum(tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0)) / n_cols
        center = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0) - mean
        var = tl.sum(center * center) / n_cols
        inv_std = 1.0 / tl.sqrt(var + eps)

        x_norm = center * inv_std
        weight = tl.load(weight_ptr + cols, mask=mask, other=1.0)
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0)
        output = x_norm * weight + bias

        tl.store(out_ptr + row * n_cols + cols, output, mask=mask)


def triton_layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    N, D = x.shape
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(D, meta['BLOCK_N']),)
    layer_norm_kernel[grid](x, weight, bias, out, N, D, eps, BLOCK_N=1024)
    return out


class ModelNew(nn.Module):
    def __init__(self, num_classes, embed_dim=512, num_heads=8, num_layers=6, 
                 mlp_ratio=4.0, patch_size=4, in_channels=3, image_size=32):
        super(ModelNew, self).__init__()

        self.patch_size = patch_size
        self.image_size = image_size
        self.embed_dim = embed_dim

        self.conv1_weight = nn.Parameter(torch.empty(embed_dim, in_channels, patch_size, patch_size))
        self.conv1_bias = nn.Parameter(torch.zeros(embed_dim))
        nn.init.kaiming_uniform_(self.conv1_weight, nonlinearity='relu')

        num_patches = (image_size // patch_size) ** 2
        self.linear_proj = nn.Linear(embed_dim * num_patches, embed_dim)

        self.transformer_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=0.0,
                activation='gelu',
                batch_first=True
            )
            # Replace MLP and LayerNorms with fused ops later in forward
            self.transformer_layers.append(layer)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.fc_out = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        _, _, H, W = x.shape

        OH, OW = H // self.patch_size, W // self.patch_size

        # Custom fused Conv2D + GELU
        x = triton_fused_conv2d_gelu(x, self.conv1_weight, self.conv1_bias, self.patch_size)
        x = x.view(B, self.embed_dim, -1).transpose(1, 2)  # (B, num_patches, embed_dim)
        x = x.flatten(start_dim=1)  # (B, embed_dim * num_patches)
        x = self.linear_proj(x)  # (B, embed_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x.unsqueeze(1)), dim=1)

        for layer in self.transformer_layers:
            # Extract components
            norm1 = layer.norm1
            attn = layer.self_attn
            norm2 = layer.norm2
            fc1 = layer.linear1
            fc2 = layer.linear2

            # Self-attention block with pre-LN
            x1 = triton_layer_norm(x, norm1.weight, norm1.bias)
            x = x + attn(x1, x1, x1, need_weights=False)[0]

            # MLP block with fused GELU and matmuls
            x2 = triton_layer_norm(x, norm2.weight, norm2.bias)
            mlp_out = triton_fused_mlp_gelu(
                x2.view(-1, self.embed_dim),
                fc1.weight, fc1.bias,
                fc2.weight, fc2.bias
            )
            mlp_out = mlp_out.view(B, -1, self.embed_dim)
            x = x + mlp_out

        return self.fc_out(x[:, 0])