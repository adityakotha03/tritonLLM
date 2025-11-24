import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import triton
import triton.language as tl


@triton.jit
def _segsum_kernel(
    x_ptr, out_ptr, stride_b, stride_h, stride_c, stride_l,
    batch_size, n_heads, n_chunks, chunk_len,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_c = tl.program_id(2)

    offset_l = tl.arange(0, BLOCK_SIZE)
    mask_l = offset_l < chunk_len

    x_block_ptr = x_ptr + pid_b * stride_b + pid_h * stride_h + pid_c * stride_c
    x_ptrs = x_block_ptr + offset_l
    x = tl.load(x_ptrs, mask=mask_l, other=0.0)

    x_cumsum = tl.cumsum(x, axis=0)
    out_block_ptr = out_ptr + pid_b * stride_b + pid_h * stride_h + pid_c * stride_c * chunk_len
    for i in range(0, chunk_len):
        row = x_cumsum - x_cumsum[i]
        out_ptrs = out_block_ptr + i * chunk_len + offset_l
        out_mask = (offset_l >= i) & (offset_l < chunk_len)
        tl.store(out_ptrs, tl.where(out_mask, row, float("-inf")), mask=out_mask)


def triton_segsum(x):
    B, H, C, L = x.shape
    out = torch.empty(B, H, C, L, L, device=x.device, dtype=x.dtype)

    def grid(meta):
        return (B, H, C)

    BLOCK_SIZE = triton.next_power_of_2(L)
    _segsum_kernel[grid](
        x, out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        B, H, C, L,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


@triton.jit
def _bmm_exp_sum_kernel(
    A_ptr, out_ptr, stride_b, stride_h, stride_c, stride_l,
    batch_size, n_heads, n_chunks, chunk_len,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_c = tl.program_id(2)

    offset_l = tl.arange(0, BLOCK_SIZE)
    mask = offset_l < chunk_len

    A_block_ptr = A_ptr + pid_b * stride_b + pid_h * stride_h + pid_c * stride_c
    A_ptrs = A_block_ptr + offset_l
    A = tl.load(A_ptrs, mask=mask, other=0.0)

    A_cumsum = tl.cumsum(A, axis=0)
    exp_A_cumsum = tl.exp(A_cumsum)

    sum_exp_A = tl.sum(exp_A_cumsum)
    out_ptr += pid_b * stride_b + pid_h * stride_h + pid_c
    tl.store(out_ptr, sum_exp_A)


def triton_exp_cumsum_sum(A):
    B, H, C, L = A.shape
    out = torch.empty(B, H, C, device=A.device, dtype=A.dtype)

    def grid(meta):
        return (B, H, C)

    BLOCK_SIZE = triton.next_power_of_2(L)
    _bmm_exp_sum_kernel[grid](
        A, out,
        A.stride(0), A.stride(1), A.stride(2), A.stride(3),
        B, H, C, L,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _fused_einsum_bcxhp_bclhn_bchpn_kernel(
    B_ptr, C_ptr, X_ptr, out_ptr,
    M, N, K,
    stride_bcxhp_b, stride_bcxhp_c, stride_bcxhp_x, stride_bcxhp_h, stride_bcxhp_p,
    stride_bclhn_b, stride_bclhn_c, stride_bclhn_l, stride_bclhn_h, stride_bclhn_n,
    stride_bchpn_b, stride_bchpn_c, stride_bchpn_h, stride_bchpn_p, stride_bchpn_n,
    stride_out_b, stride_out_c, stride_out_x, stride_out_h, stride_out_p,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_in_group = GROUP_SIZE * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    B_ptrs = B_ptr + (offs_m[:, None] // K) * stride_bcxhp_b + \
                      (offs_m[:, None] // (K * d_head)) * stride_bcxhp_c + \
                      ((offs_m[:, None] % K) // d_head) * stride_bcxhp_x + \
                      ((offs_m[:, None] % K) % d_head) * stride_bcxhp_h + \
                      offs_k[None, :] * stride_bcxhp_p
    C_ptrs = C_ptr + (offs_n[:, None] // (d_head * d_state)) * stride_bclhn_b + \
                      (offs_n[:, None] // (d_head * d_state * batch_size)) * stride_bclhn_c + \
                      ((offs_n[:, None] % (d_head * d_state)) // d_state) * stride_bclhn_l + \
                      ((offs_n[:, None] % d_head)) * stride_bclhn_h + \
                      (offs_k[None, :] % d_state) * stride_bclhn_n
    X_ptrs = X_ptr + (offs_m[:, None] // K) * stride_bchpn_b + \
                      (offs_m[:, None] // K) * stride_bchpn_c + \
                      ((offs_m[:, None] % K) // d_head) * stride_bchpn_h + \
                      ((offs_m[:, None] % K) % d_head) * stride_bchpn_p + \
                      offs_k[None, :] * stride_bchpn_n
    out_ptrs = out_ptr + pid_m * BLOCK_SIZE_M * stride_out_x + tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_out_x + \
                         pid_n * BLOCK_SIZE_N * stride_out_p + tl.arange(0, BLOCK_SIZE_N)[None, :] * stride_out_p

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        B_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        C_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        X_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)

        b = tl.load(B_ptrs, mask=B_mask, other=0.0)
        c = tl.load(C_ptrs, mask=C_mask, other=0.0)
        x = tl.load(X_ptrs, mask=X_mask, other=0.0)

        accumulator += tl.dot(b, tl.trans(c)) * x
        B_ptrs += BLOCK_SIZE_K * stride_bcxhp_p
        C_ptrs += BLOCK_SIZE_K * stride_bclhn_n
        X_ptrs += BLOCK_SIZE_K * stride_bchpn_n

    acc = accumulator.to(tl.float16)
    out_mask = (offs_m[:, None] < M) & (offs_n[:, None] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


def fused_einsum_bcxhp_bclhn_bchpn(B, C, X):
    B_size, C_size, X_size = B.numel(), C.numel(), X.numel()
    M = B_size // d_state
    N = C_size // d_state
    K = d_state

    out = torch.empty(B.shape[0], B.shape[1], B.shape[2], B.shape[3], device=B.device, dtype=B.dtype)

    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)

    _fused_einsum_bcxhp_bclhn_bchpn_kernel[grid](
        B, C, X, out,
        M, N, K,
        B.stride(0), B.stride(1), B.stride(2), B.stride(3), B.stride(4),
        C.stride(0), C.stride(1), C.stride(2), C.stride(3), C.stride(4),
        X.stride(0), X.stride(1), X.stride(2), X.stride(3), X.stride(4),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, batch_size, seq_length, n_heads, d_head, d_state, block_len=64):
        super(ModelNew, self).__init__()
        
        assert seq_length % block_len == 0, "Sequence length must be divisible by block length"
        
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.block_len = block_len
        
        self.A = nn.Parameter(torch.randn(batch_size, seq_length, n_heads))
        self.B = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        self.C = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
    
    def segsum(self, x):
        T = x.size(-1)
        x_cumsum = torch.cumsum(x, dim=-1)
        x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=0)
        x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
        return x_segsum

    def forward(self, X, initial_states=None):
        X_blocks, A_blocks, B_blocks, C_blocks = [
            rearrange(x, "b (c l) ... -> b c l ...", l=self.block_len)
            for x in (X, self.A, self.B, self.C)
        ]
        
        A_blocks = rearrange(A_blocks, "b c l h -> b h c l")
        A_cumsum = torch.cumsum(A_blocks, dim=-1)
        
        L = torch.exp(self.segsum(A_blocks))
        Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C_blocks, B_blocks, L, X_blocks)
        
        decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
        states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B_blocks, decay_states, X_blocks)
        
        if initial_states is None:
            initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)
        
        decay_chunk = torch.exp(self.segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
        new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
        states = new_states[:, :-1]
        
        state_decay_out = torch.exp(A_cumsum)
        Y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp', C_blocks, states, state_decay_out)
        
        Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
        
        return Y