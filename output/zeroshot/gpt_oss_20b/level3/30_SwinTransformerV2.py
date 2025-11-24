import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import numpy as np
import collections
from itertools import repeat

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)

# ------------------------------------------------------------------
# Triton kernels
# ------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32}, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32}, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 512, "BLOCK_SIZE_N": 512, "BLOCK_SIZE_K": 32}, num_warps=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_fused_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(A_ptr + offs_m[:, None] * stride_am + (k + tl.arange(0, BLOCK_SIZE_K)) * stride_ak,
                    mask=mask_m[:, None] & (k + tl.arange(0, BLOCK_SIZE_K) < K),
                    other=0.0)
        b = tl.load(B_ptr + (k + tl.arange(0, BLOCK_SIZE_K)) * stride_bk + offs_n[None, :] * stride_bn,
                    mask=(k + tl.arange(0, BLOCK_SIZE_K) < K) & mask_n[None, :],
                    other=0.0)
        acc += tl.dot(a, b)

    if mask_m.any() and mask_n.any():
        tl.store(C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
                 acc, mask=mask_m[:, None] & mask_n[None, :])

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 512, "BLOCK_SIZE_N": 512}, num_warps=4),
    ],
    key=["M", "N"],
)
@triton.jit
def softmax_fused_kernel(
    A_ptr, B_ptr,
    M, N,
    stride_am, stride_an,
    stride_bm, stride_bn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    a = tl.load(A_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an,
                mask=mask_m[:, None] & mask_n[None, :],
                other=0.0)

    max_val = tl.max(a, axis=1, keepdims=True)
    a_exp = tl.exp(a - max_val)
    sum_exp = tl.sum(a_exp, axis=1, keepdims=True)
    out = a_exp / sum_exp

    if mask_m.any() and mask_n.any():
        tl.store(B_ptr + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn,
                 out, mask=mask_m[:, None] & mask_n[None, :])

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]),
                         triton.cdiv(N, meta["BLOCK_SIZE_N"]))
    matmul_fused_kernel[grid](A, B, C,
                              M, N, K,
                              A.stride(0), A.stride(1),
                              B.stride(0), B.stride(1),
                              C.stride(0), C.stride(1))
    return C

def triton_softmax(A: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if dim == -1:
        M, N = A.shape
        B = torch.empty_like(A)
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]),
                             triton.cdiv(N, meta["BLOCK_SIZE_N"]))
        softmax_fused_kernel[grid](A, B,
                                   M, N,
                                   A.stride(0), A.stride(1),
                                   B.stride(0), B.stride(1))
        return B
    else:
        return torch.nn.functional.softmax(A, dim=dim)

# ------------------------------------------------------------------
# Model components
# ------------------------------------------------------------------
class TritonLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.bias = None
        nn.init.xavier_uniform_(self.weight)
        if bias:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        out = triton_matmul(x, self.weight.t())
        if self.bias is not None:
            out = out + self.bias
        return out

class TritonGELU(nn.Module):
    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))

class TritonLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = ((x - mean) ** 2).mean(-1, keepdim=True)
        x_norm = (x - mean) * torch.rsqrt(var + self.eps)
        return x_norm * self.weight + self.bias

# ------------------------------------------------------------------
# Architecture
# ------------------------------------------------------------------
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=TritonGELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = TritonLinear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = TritonLinear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True,
                 attn_drop=0., proj_drop=0., pretrained_window_size=[0, 0]):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = TritonLinear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = TritonLinear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = triton_softmax

        # relative position bias
        rel_h = torch.arange(-(window_size[0]-1), window_size[0], dtype=torch.float32)
        rel_w = torch.arange(-(window_size[1]-1), window_size[1], dtype=torch.float32)
        rel_table = torch.stack(torch.meshgrid([rel_h, rel_w]))
        rel_table = rel_table.reshape(2, -1).permute(1, 0).unsqueeze(0)
        rel_table *= 8
        rel_table = torch.sign(rel_table) * torch.log2(torch.abs(rel_table)+1.0)/np.log2(8)
        self.register_buffer("rel_table", rel_table)

        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flat = torch.flatten(coords, 1)
        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
        relative_coords = relative_coords.permute(1,2,0)
        relative_coords[:,:,0] += window_size[0]-1
        relative_coords[:,:,1] += window_size[1]-1
        relative_coords[:,:,0] *= 2*window_size[1]-1
        self.register_buffer("rel_index", relative_coords.sum(-1))

        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False)
        )

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x)
        if self.q_bias is not None:
            qkv = qkv + torch.cat((self.q_bias, torch.zeros_like(self.v_bias), self.v_bias))
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2,-1)
        attn = attn * torch.exp(torch.clamp(self.logit_scale, max=torch.log(1/0.01)))

        rel_bias = self.cpb_mlp(self.rel_table).view(-1, self.num_heads)
        rel_bias = rel_bias[self.rel_index.view(-1)].view(self.window_size[0]*self.window_size[1],
                                                          self.window_size[0]*self.window_size[1],
                                                          -1).permute(2,0,1)
        rel_bias = 16 * torch.sigmoid(rel_bias)
        attn = attn + rel_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_//nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1,2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7,
                 shift_size=0, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 pretrained_window_size=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=to_2tuple(window_size),
                                    num_heads=num_heads, qkv_bias=qkv_bias,
                                    attn_drop=attn_drop, proj_drop=drop,
                                    pretrained_window_size=to_2tuple(pretrained_window_size))
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim*mlp_ratio),
                       act_layer=TritonGELU, drop=drop)
        self.shift_size = shift_size
        self.window_size = window_size
        self.input_resolution = input_resolution

        if shift_size > 0:
            H, W = input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
            w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, window_size)
            mask_windows = mask_windows.view(-1, window_size*window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask!=0, float(-100.0)).masked_fill(attn_mask==0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1,2))
        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size*self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1,2))
        x = x.view(B, H*W, C)
        x = self.norm1(x) + self.drop_path(x)
        x = self.norm2(self.mlp(x)) + x
        return x

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = TritonLinear(4*dim, 2*dim, bias=False)
        self.norm = norm_layer(2*dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4*C)
        x = self.reduction(x)
        x = self.norm(x)
        return x

class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None,
                 use_checkpoint=False, pretrained_window_size=0):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i%2==0) else window_size//2,
                                 mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path,list) else drop_path,
                                 norm_layer=norm_layer,
                                 pretrained_window_size=pretrained_window_size)
            for i in range(depth)
        ])
        self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer) if downsample else None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

class ModelNew(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2,2,6,2], num_heads=[3,6,12,24],
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=[0,0,0,0], **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2**(self.num_layers-1))
        self.mlp_ratio = mlp_ratio
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                      embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim*2**i_layer),
                               input_resolution=(self.patch_embed.patches_resolution[0]//(2**i_layer),
                                                 self.patch_embed.patches_resolution[1]//(2**i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer+1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers-1) else None,
                               use_checkpoint=use_checkpoint,
                               pretrained_window_size=pretrained_window_sizes[i_layer])
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes>0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.avgpool(x.transpose(1,2))
        x = torch.flatten(x,1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x