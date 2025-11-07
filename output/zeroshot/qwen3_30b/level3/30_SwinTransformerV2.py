import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import numpy as np
from itertools import repeat
from collections import OrderedDict

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)

# Triton kernels
@triton.jit
def _fused_matmul_relu_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, 
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ACTIVATION: tl.constexpr
):
    # matrix multiplication
    pid = tl.program_id(0)
    pid_m = pid // tl.num_programs(1)
    pid_n = pid % tl.num_programs(1)
    
    # offset for A and B
    block_start_m = pid_m * BLOCK_M
    block_start_n = pid_n * BLOCK_N
    offsets_m = block_start_m + tl.arange(0, BLOCK_M)
    offsets_n = block_start_n + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    
    # load A and B
    a_ptrs = a_ptr + (offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn)
    
    # create mask for valid indices
    mask_a = (offsets_m[:, None] < M) & (offsets_k[None, :] < K)
    mask_b = (offsets_k[:, None] < K) & (offsets_n[None, :] < N)
    
    # accumulate
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # activation
    if ACTIVATION == 1:
        accumulator = tl.relu(accumulator)
    
    # store result
    c_ptrs = c_ptr + (offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn)
    mask_c = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=mask_c)


@triton.jit
def _window_attention_kernel(
    q_ptr, k_ptr, v_ptr, 
    out_ptr,
    q_rows, q_cols, q_len,
    k_rows, k_cols, k_len,
    v_rows, v_cols, v_len,
    B, N, C,
    num_heads,
    logit_scale_ptr,
    relative_bias_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    use_bias: tl.constexpr,
    use_softmax: tl.constexpr
):
    # Get program ID and block index
    pid = tl.program_id(0)
    pid_head = pid % num_heads
    pid_batch = pid // num_heads
    
    # Calculate block start indices
    block_start_m = pid_batch * BLOCK_SIZE_M
    block_start_n = pid_batch * BLOCK_SIZE_N
    
    # Create offsets
    offsets_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create mask for valid indices
    mask_m = offsets_m < N
    mask_n = offsets_n < N
    mask_k = offsets_k < C
    
    # Calculate stride for each tensor
    stride_qm = q_rows
    stride_qk = q_cols
    stride_km = k_rows
    stride_kn = k_cols
    stride_vm = v_rows
    stride_vn = v_cols
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Compute QK^T
    for k in range(0, C, BLOCK_SIZE_K):
        # Load Q and K
        q_ptrs = q_ptr + (offsets_m[:, None] * stride_qm + (offsets_k[None, :] + k) * stride_qk)
        k_ptrs = k_ptr + ((offsets_k[:, None] + k) * stride_km + offsets_n[None, :] * stride_kn)
        
        q = tl.load(q_ptrs, mask=(mask_m[:, None] & mask_k[None, :]), other=0.0)
        k = tl.load(k_ptrs, mask=(mask_k[:, None] & mask_n[None, :]), other=0.0)
        
        # Compute QK^T
        qk = tl.dot(q, k, out_dtype=tl.float32)
        accumulator += qk
    
    # Apply logit scale
    logit_scale = tl.load(logit_scale_ptr + pid_head * 1)
    accumulator *= logit_scale
    
    # Add relative position bias
    if use_bias:
        # Get relative bias for this window
        bias_ptr = relative_bias_ptr + pid_batch * N * N * num_heads + pid_head * N * N
        bias_offsets = tl.arange(0, N)[:, None] * N + tl.arange(0, N)[None, :]
        bias_ptrs = bias_ptr + bias_offsets
        bias = tl.load(bias_ptrs, mask=(mask_m[:, None] & mask_n[None, :]), other=0.0)
        accumulator += bias
    
    # Apply softmax
    if use_softmax:
        # Compute max
        max_val = tl.max(accumulator, axis=1, keepdims=True)
        exp = tl.exp(accumulator - max_val)
        sum_exp = tl.sum(exp, axis=1, keepdims=True)
        attention_weights = exp / sum_exp
        
        # Apply attention to values
        v_ptrs = v_ptr + (offsets_m[:, None] * stride_vm + offsets_n[None, :] * stride_vn)
        v = tl.load(v_ptrs, mask=(mask_m[:, None] & mask_n[None, :]), other=0.0)
        
        # Compute weighted sum
        result = tl.dot(attention_weights, v)
    else:
        # If no softmax, just use the raw attention scores
        result = accumulator
    
    # Store output
    out_ptrs = out_ptr + (offsets_m[:, None] * q_rows + offsets_n[None, :] * q_cols)
    mask_out = (mask_m[:, None] & mask_n[None, :])
    tl.store(out_ptrs, result, mask=mask_out)


@triton.jit
def _fused_mlp_kernel(
    x_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr,
    y_ptr,
    B, N, C,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    pid = tl.program_id(0)
    pid_batch = pid % B
    pid_m = pid // B
    
    # Get offsets
    block_start_m = pid_m * BLOCK_SIZE_M
    offsets_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = tl.arange(0, BLOCK_SIZE_N)
    
    # Create mask
    mask_m = offsets_m < N
    mask_n = offsets_n < C
    
    # Calculate strides
    stride_xm = N
    stride_xn = 1
    stride_w1m = C
    stride_w1n = 1
    stride_w2m = C
    stride_w2n = 1
    stride_bm = C
    stride_bn = 1
    
    # Load input
    x_ptrs = x_ptr + (pid_batch * N * C + offsets_m[:, None] * stride_xm + offsets_n[None, :] * stride_xn)
    x = tl.load(x_ptrs, mask=(mask_m[:, None] & mask_n[None, :]), other=0.0)
    
    # First linear layer + GELU activation
    w1_ptrs = w1_ptr + (offsets_m[:, None] * stride_w1m + offsets_n[None, :] * stride_w1n)
    b1_ptrs = b1_ptr + offsets_n[None, :]
    w1 = tl.load(w1_ptrs, mask=mask_n[None, :], other=0.0)
    b1 = tl.load(b1_ptrs, mask=mask_n[None, :], other=0.0)
    
    # Apply linear transformation
    x1 = tl.dot(x, w1) + b1
    
    # GELU activation
    pi = 3.141592653589793
    x1 = x1 * 0.5 * (1.0 + tl.tanh((2.0 / pi) ** 0.5 * (x1 + 0.044715 * x1 ** 3)))
    
    # Second linear layer + bias
    w2_ptrs = w2_ptr + (offsets_m[:, None] * stride_w2m + offsets_n[None, :] * stride_w2n)
    b2_ptrs = b2_ptr + offsets_n[None, :]
    w2 = tl.load(w2_ptrs, mask=mask_n[None, :], other=0.0)
    b2 = tl.load(b2_ptrs, mask=mask_n[None, :], other=0.0)
    
    # Apply linear transformation
    x2 = tl.dot(x1, w2) + b2
    
    # Store output
    y_ptrs = y_ptr + (pid_batch * N * C + offsets_m[:, None] * stride_xm + offsets_n[None, :] * stride_xn)
    tl.store(y_ptrs, x2, mask=(mask_m[:, None] & mask_n[None, :]))


@triton.jit
def _patch_merge_kernel(
    x_ptr, y_ptr,
    B, H, W, C,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    pid = tl.program_id(0)
    pid_batch = pid // (H // 2 * W // 2)
    pid_h = (pid // (W // 2)) % (H // 2)
    pid_w = pid % (W // 2)
    
    # Calculate block indices
    h = pid_h * 2
    w = pid_w * 2
    
    # Create offsets
    offsets_h = tl.arange(0, 2)
    offsets_w = tl.arange(0, 2)
    offsets_c = tl.arange(0, 4 * C)
    
    # Calculate strides
    stride_xh = H * W * C
    stride_xw = W * C
    stride_xc = C
    stride_yh = (H // 2) * (W // 2) * (2 * C)
    stride_yw = (W // 2) * (2 * C)
    stride_yc = 2 * C
    
    # Load 4 patches
    x_ptrs = x_ptr + (pid_batch * H * W * C + 
                      (h + offsets_h[:, None]) * stride_xh + 
                      (w + offsets_w[None, :]) * stride_xw + 
                      offsets_c[None, :] * stride_xc)
    x = tl.load(x_ptrs, mask=((offsets_h[:, None] < 2) & (offsets_w[None, :] < 2) & (offsets_c[None, :] < 4 * C)), other=0.0)
    
    # Concatenate along channel dimension
    x = x.view(4, C)
    x = x.view(1, 4 * C)
    
    # Apply linear projection
    # For simplicity, we'll use a fixed weight matrix with stride=1
    # In practice, this would be passed as a parameter
    # For now, we'll just use a placeholder
    y_ptrs = y_ptr + (pid_batch * (H // 2) * (W // 2) * (2 * C) + 
                      pid_h * (W // 2) * (2 * C) + 
                      pid_w * (2 * C) + 
                      offsets_c[None, :] * stride_yc)
    tl.store(y_ptrs, x, mask=(offsets_c[None, :] < 2 * C))


def triton_matmul_relu(x, w1, b1, w2, b2, activation=True):
    assert x.is_cuda and w1.is_cuda and b1.is_cuda and w2.is_cuda and b2.is_cuda, "Tensors must be on CUDA"
    x = x.contiguous()
    w1 = w1.contiguous()
    b1 = b1.contiguous()
    w2 = w2.contiguous()
    b2 = b2.contiguous()
    
    B, N, C = x.shape
    C1, C2 = w1.shape
    assert C1 == C
    assert b1.shape == (C1,)
    assert w2.shape == (C1, C2)
    assert b2.shape == (C2,)
    
    out = torch.empty(B, N, C2, dtype=x.dtype, device=x.device)
    
    # Choose block size
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64
    
    # Calculate grid
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE_M"]) * B, triton.cdiv(C2, meta["BLOCK_SIZE_N"]))
    
    # Launch kernel
    _fused_matmul_relu_kernel[grid](
        x, w1, b1, w2, b2,
        out,
        B, N, C, C1, C2,
        x.stride(0), x.stride(1), x.stride(2),
        w1.stride(0), w1.stride(1),
        w2.stride(0), w2.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        ACTIVATION=1 if activation else 0
    )
    
    return out


def triton_attention(q, k, v, logit_scale, relative_bias, num_heads):
    assert q.is_cuda and k.is_cuda and v.is_cuda and logit_scale.is_cuda and relative_bias.is_cuda, "Tensors must be on CUDA"
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    logit_scale = logit_scale.contiguous()
    relative_bias = relative_bias.contiguous()
    
    B, N, C = q.shape
    assert k.shape == (B, N, C)
    assert v.shape == (B, N, C)
    assert logit_scale.shape == (num_heads, 1, 1)
    assert relative_bias.shape == (B, num_heads, N, N)
    
    out = torch.empty_like(q)
    
    # Choose block size
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64
    
    # Calculate grid
    grid = lambda meta: (B * num_heads, 1)
    
    # Launch kernel
    _window_attention_kernel[grid](
        q, k, v,
        out,
        q.shape[1], q.shape[2], q.shape[0],
        k.shape[1], k.shape[2], k.shape[0],
        v.shape[1], v.shape[2], v.shape[0],
        B, N, C, num_heads,
        logit_scale, relative_bias,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        use_bias=1,
        use_softmax=1
    )
    
    return out


def triton_mlp(x, w1, b1, w2, b2):
    assert x.is_cuda and w1.is_cuda and b1.is_cuda and w2.is_cuda and b2.is_cuda, "Tensors must be on CUDA"
    x = x.contiguous()
    w1 = w1.contiguous()
    b1 = b1.contiguous()
    w2 = w2.contiguous()
    b2 = b2.contiguous()
    
    B, N, C = x.shape
    C1, C2 = w1.shape
    assert C1 == C
    assert b1.shape == (C1,)
    assert w2.shape == (C1, C2)
    assert b2.shape == (C2,)
    
    out = torch.empty_like(x)
    
    # Choose block size
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    
    # Calculate grid
    grid = lambda meta: (B * N, 1)
    
    # Launch kernel
    _fused_mlp_kernel[grid](
        x, w1, b1, w2, b2,
        out,
        B, N, C,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    return out


def triton_patch_merge(x):
    assert x.is_cuda, "Tensor must be on CUDA"
    x = x.contiguous()
    
    B, H, W, C = x.shape
    assert H % 2 == 0 and W % 2 == 0, "H and W must be even"
    
    out = torch.empty(B, H // 2, W // 2, 2 * C, dtype=x.dtype, device=x.device)
    
    # Choose block size
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    
    # Calculate grid
    grid = lambda meta: (B * (H // 2) * (W // 2), 1)
    
    # Launch kernel
    _patch_merge_kernel[grid](
        x, out,
        B, H, W, C,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    return out


class TritonMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.act = act_layer()
        
    def forward(self, x):
        x = triton_matmul_relu(x, self.fc1.weight, self.fc1.bias, self.fc2.weight, self.fc2.bias, activation=True)
        x = self.drop(x)
        return x


class TritonWindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=[0, 0]):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Use custom Triton attention
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
        
        # Relative position bias
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h,
                            relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)
        
        self.register_buffer("relative_coords_table", relative_coords_table)

        # Get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        # For Triton attention, we need to precompute the relative bias
        self.relative_bias = nn.Parameter(torch.zeros(1, num_heads, window_size[0] * window_size[1], 
                                                    window_size[0] * window_size[1]), requires_grad=True)
        
        # Project to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False)
        )
        
        # Linear projection for QKV
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
            
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        
        # Use Triton attention
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=self.q_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # Compute relative position bias
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        self.relative_bias.data = relative_position_bias
        
        # Apply cosine attention with Triton
        attn = triton_attention(q, k, v, self.logit_scale, self.relative_bias, self.num_heads)
        
        # Apply dropout
        attn = self.attn_drop(attn)
        
        # Project back
        x = attn.view(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class TritonSwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = TritonWindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size))

        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = TritonMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x


class TritonPatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        # Use Triton patch merging
        x = triton_patch_merge(x)
        
        # Apply linear projection and normalization
        x = self.reduction(x)
        x = self.norm(x)

        return x


class TritonBasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            TritonSwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pretrained_window_size=pretrained_window_size)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = TritonPatchMerging(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class TritonPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


class ModelNew(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        pretrained_window_sizes (tuple(int)): Pretrained window sizes of each layer.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0], **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = TritonPatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = TritonBasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=TritonPatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               pretrained_window_size=pretrained_window_sizes[i_layer])
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x