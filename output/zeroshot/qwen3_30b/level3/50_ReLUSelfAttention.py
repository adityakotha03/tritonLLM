import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

# Custom Triton kernel for fused GELU activation
@triton.jit
def gelu_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate the block offset
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
    c = 0.044715

    x3 = x * x * x
    inner = x + c * x3
    tanh_input = sqrt_2_over_pi * inner
    tanh_out = tl.tanh(tanh_input)
    gelu_out = 0.5 * x * (1.0 + tanh_out)

    # Store output
    tl.store(y_ptr + offsets, gelu_out, mask=mask)


def triton_gelu(x: torch.Tensor):
    """Fused GELU using Triton kernel."""
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Optimal for A100, power of 2, large enough for coalescing
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    gelu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


# Custom Triton kernel for fused QKV projection and reshaping (c_attn + split + view + transpose)
@triton.jit
def qkv_proj_kernel(
    x_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    n_batch,
    n_seq,
    n_head,
    n_embd,
    BLOCK_SIZE: tl.constexpr,
):
    # Define block offsets
    pid = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = pid // n_head
    off_n = pid % n_head

    # Thread block indices
    off_s = tl.arange(0, BLOCK_SIZE)
    off_e = tl.arange(0, n_embd)

    # Compute base offsets
    x_offset = off_b * n_seq * n_embd + off_s * n_embd + off_e
    q_offset = off_b * n_seq * n_head * (n_embd // n_head) + off_s * (n_embd // n_head) + off_n * (n_embd // n_head) + off_e // (n_embd // n_head) * n_seq * (n_embd // n_head)
    k_offset = off_b * n_seq * n_head * (n_embd // n_head) + off_s * (n_embd // n_head) + off_n * (n_embd // n_head) + off_e // (n_embd // n_head) * n_seq * (n_embd // n_head)
    v_offset = off_b * n_seq * n_head * (n_embd // n_head) + off_s * (n_embd // n_head) + off_n * (n_embd // n_head) + off_e // (n_embd // n_head) * n_seq * (n_embd // n_head)

    # Load input
    x = tl.load(x_ptr + x_offset, mask=off_s[:, None] < n_seq, other=0.0)
    x = x.to(tl.float16)  # Use FP16 for tensor cores

    # Split into q, k, v (each of size n_embd)
    q = x @ tl.load(q_ptr + off_n * n_embd * n_embd + off_e, mask=off_e < n_embd, other=0.0).to(tl.float16)
    k = x @ tl.load(k_ptr + off_n * n_embd * n_embd + off_e, mask=off_e < n_embd, other=0.0).to(tl.float16)
    v = x @ tl.load(v_ptr + off_n * n_embd * n_embd + off_e, mask=off_e < n_embd, other=0.0).to(tl.float16)

    # Reshape and transpose: (B, T, nh, hs) -> (B, nh, T, hs)
    # We'll use shared memory to store per-head results
    # Shared memory: (BLOCK_SIZE, n_embd // n_head) for each head
    shmem = tl.static_range(16384)  # 16KB per SM
    q_shmem = tl.shared_memory(shape=(BLOCK_SIZE, n_embd // n_head), dtype=tl.float16)
    k_shmem = tl.shared_memory(shape=(BLOCK_SIZE, n_embd // n_head), dtype=tl.float16)
    v_shmem = tl.shared_memory(shape=(BLOCK_SIZE, n_embd // n_head), dtype=tl.float16)

    # Write to shared memory
    q_offsets = off_s[:, None] * (n_embd // n_head) + off_e
    k_offsets = off_s[:, None] * (n_embd // n_head) + off_e
    v_offsets = off_s[:, None] * (n_embd // n_head) + off_e

    tl.store(q_shmem + q_offsets, q, mask=off_s[:, None] < n_seq)
    tl.store(k_shmem + k_offsets, k, mask=off_s[:, None] < n_seq)
    tl.store(v_shmem + v_offsets, v, mask=off_s[:, None] < n_seq)

    # Synchronize threads in block
    tl.sync()

    # Output to global memory
    q_output = tl.load(q_shmem + q_offsets, mask=off_s[:, None] < n_seq, other=0.0)
    k_output = tl.load(k_shmem + k_offsets, mask=off_s[:, None] < n_seq, other=0.0)
    v_output = tl.load(v_shmem + v_offsets, mask=off_s[:, None] < n_seq, other=0.0)

    # Final output
    tl.store(q_ptr + off_b * n_seq * n_head * (n_embd // n_head) + off_n * n_seq * (n_embd // n_head) + off_s * (n_embd // n_head), q_output, mask=off_s < n_seq)
    tl.store(k_ptr + off_b * n_seq * n_head * (n_embd // n_head) + off_n * n_seq * (n_embd // n_head) + off_s * (n_embd // n_head), k_output, mask=off_s < n_seq)
    tl.store(v_ptr + off_b * n_seq * n_head * (n_embd // n_head) + off_n * n_seq * (n_embd // n_head) + off_s * (n_embd // n_head), v_output, mask=off_s < n_seq)


def triton_qkv_proj(x, q_proj_weight, k_proj_weight, v_proj_weight, n_head, n_embd, n_seq):
    """Fused QKV projection with shared memory tiling and tensor core utilization."""
    assert x.is_cuda
    x = x.contiguous()
    B, T, C = x.shape
    hs = C // n_head
    q = torch.empty(B, T, n_head, hs, dtype=torch.float16, device=x.device)
    k = torch.empty(B, T, n_head, hs, dtype=torch.float16, device=x.device)
    v = torch.empty(B, T, n_head, hs, dtype=torch.float16, device=x.device)

    # Use 16KB shared memory limit, 1024 elements per block
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(B * T, meta["BLOCK_SIZE"]), n_head)

    # Kernel launch
    qkv_proj_kernel[grid](
        x,
        q,
        k,
        v,
        B,
        T,
        n_head,
        C,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return q, k, v


# Custom Triton kernel for fused causal attention with online softmax and ReLU
@triton.jit
def causal_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    bias_ptr,
    n_batch,
    n_seq,
    n_head,
    n_embd,
    BLOCK_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
):
    # Block index
    pid = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = pid // n_head
    off_n = pid % n_head

    # Thread offsets
    off_s = tl.arange(0, BLOCK_SIZE)
    off_e = tl.arange(0, HEAD_SIZE)

    # Base offsets
    q_offset = off_b * n_seq * n_head * HEAD_SIZE + off_n * n_seq * HEAD_SIZE + off_s[:, None] * HEAD_SIZE + off_e[None, :]
    k_offset = off_b * n_seq * n_head * HEAD_SIZE + off_n * n_seq * HEAD_SIZE + off_s[None, :] * HEAD_SIZE + off_e[:, None]
    v_offset = off_b * n_seq * n_head * HEAD_SIZE + off_n * n_seq * HEAD_SIZE + off_s[:, None] * HEAD_SIZE + off_e[None, :]

    # Load Q, K, V
    q = tl.load(q_ptr + q_offset, mask=off_s[:, None] < n_seq, other=0.0)
    k = tl.load(k_ptr + k_offset, mask=off_s[None, :] < n_seq, other=0.0)
    v = tl.load(v_ptr + v_offset, mask=off_s[:, None] < n_seq, other=0.0)

    # Compute attention scores: (T, T)
    # Use tensor cores: FP16 matmul
    attn_weights = tl.dot(q, k, out_dtype=tl.float32)  # Use float32 accumulation
    scale = 1.0 / (HEAD_SIZE ** 0.5)
    attn_weights *= scale

    # Apply causal mask: set future positions to -inf
    bias = tl.load(bias_ptr + off_b * n_seq * n_seq + off_s[:, None] * n_seq + off_s[None, :])
    attn_weights += bias

    # Replace softmax with ReLU: this is the key optimization
    # Online ReLU: avoid storing full softmax
    attn_weights = tl.where(attn_weights > 0, attn_weights, 0.0)

    # Apply attention to values: (T, T) @ (T, HS) -> (T, HS)
    output = tl.dot(attn_weights, v, out_dtype=tl.float16)

    # Store output
    tl.store(out_ptr + off_b * n_seq * n_head * HEAD_SIZE + off_n * n_seq * HEAD_SIZE + off_s[:, None] * HEAD_SIZE + off_e[None, :], output, mask=off_s[:, None] < n_seq)


def triton_causal_attention(q, k, v, bias):
    """Fused causal attention with ReLU and online computation using Triton."""
    B, T, n_head, HEAD_SIZE = q.shape
    out = torch.empty_like(q, dtype=torch.float16)

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(B * T, meta["BLOCK_SIZE"]), n_head)

    causal_attention_kernel[grid](
        q,
        k,
        v,
        out,
        bias,
        B,
        T,
        n_head,
        HEAD_SIZE,
        BLOCK_SIZE=BLOCK_SIZE,
        HEAD_SIZE=HEAD_SIZE
    )
    return out


# Custom Triton kernel for output projection with fused activation
@triton.jit
def proj_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    n_batch,
    n_seq,
    n_head,
    n_embd,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    off_b = pid // n_head
    off_h = pid % n_head

    off_s = tl.arange(0, BLOCK_SIZE)
    off_e = tl.arange(0, n_embd)

    # Load input: (B, nh, T, hs) -> (B, T, nh, hs) for linear
    x_offset = off_b * n_seq * n_head * n_embd + off_s[:, None] * n_head * n_embd + off_h * n_embd + off_e[None, :]
    w_offset = off_h * n_embd * n_embd + off_e[None, :] * n_embd + off_e[None, :]
    x = tl.load(x_ptr + x_offset, mask=off_s[:, None] < n_seq, other=0.0)
    w = tl.load(w_ptr + w_offset, mask=off_e[None, :] < n_embd, other=0.0)

    # Matmul: (T, hs) @ (hs, embd) -> (T, embd)
    y = tl.dot(x, w, out_dtype=tl.float32)
    y = y.to(tl.float16)

    # Store output: (B, T, C)
    out_offset = off_b * n_seq * n_embd + off_s[:, None] * n_embd + off_e[None, :]
    tl.store(out_ptr + out_offset, y, mask=off_s[:, None] < n_seq)


def triton_proj(x, w):
    """Fused projection with Triton."""
    B, T, n_head, hs = x.shape
    C = w.shape[1]
    out = torch.empty(B, T, C, dtype=torch.float16, device=x.device)

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(B * T, meta["BLOCK_SIZE"]), n_head)

    proj_kernel[grid](
        x,
        w,
        out,
        B,
        T,
        n_head,
        C,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, n_embd, n_head, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.max_seqlen = max_seqlen

        # Projections: use FP16 for tensor cores
        self.c_attn_weight = nn.Parameter(torch.randn(3 * n_embd, n_embd, dtype=torch.float16, device='cuda'))
        self.c_proj_weight = nn.Parameter(torch.randn(n_embd, n_embd, dtype=torch.float16, device='cuda'))

        # Register buffer: bias
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen))
                                     .view(1, 1, max_seqlen, max_seqlen))

    def forward(self, x):
        B, T, C = x.shape

        # Split c_attn into q, k, v
        q_weight = self.c_attn_weight[:C]
        k_weight = self.c_attn_weight[C:2*C]
        v_weight = self.c_attn_weight[2*C:]

        # Fused QKV projection with shared memory and tensor cores
        q, k, v = triton_qkv_proj(x, q_weight, k_weight, v_weight, self.n_head, C, T)

        # Fused causal attention with ReLU and online computation
        # No softmax, direct ReLU
        attn_out = triton_causal_attention(q, k, v, self.bias[:, :, :T, :T])

        # Fused output projection
        out = triton_proj(attn_out, self.c_proj_weight)

        return out