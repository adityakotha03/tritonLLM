import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import triton
import triton.language as tl


@triton.jit
def segsum_kernel(
    A_ptr,
    L_ptr,
    T: tl.constexpr,
    n_heads: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < T

    # Load A: (b, h, c, l) -> (c, l)
    A = tl.load(A_ptr + offsets[:, None, None, None], mask=mask[:, None, None, None], other=0.0)
    A = tl.reshape(A, (1, 1, T))

    # Compute cumulative sum
    A_cumsum = tl.cumsum(A, axis=2)
    
    # Compute pairwise differences
    A_cumsum_i = tl.broadcast_to(A_cumsum, (1, 1, T, T))
    A_cumsum_j = tl.broadcast_to(tl.transpose(A_cumsum, (0, 1, 3, 2)), (1, 1, T, T))
    diff = A_cumsum_i - A_cumsum_j

    # Apply triangular mask: only lower triangular (i <= j)
    mask_tri = tl.arange(0, T)[:, None] <= tl.arange(0, T)[None, :]
    diff = tl.where(mask_tri, diff, -1e10)

    # Exponentiate
    L = tl.exp(diff)

    # Store result: (c, l, l)
    L = tl.reshape(L, (T, T))
    tl.store(L_ptr + offsets[:, None] * T + offsets[None, :], L, mask=mask[:, None] & mask[None, :])


@triton.jit
def matmul_3d_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < T

    # Load A: (T, H, D) -> (T, H, D)
    A = tl.load(A_ptr + offsets[:, None, None] * D * H, mask=mask[:, None, None], other=0.0)
    A = tl.reshape(A, (T, H, D))

    # Load B: (T, H, D) -> (T, H, D)
    B = tl.load(B_ptr + offsets[:, None, None] * D * H, mask=mask[:, None, None], other=0.0)
    B = tl.reshape(B, (T, H, D))

    # Compute matrix multiplication: (T, H, D) @ (T, H, D) -> (T, H, D)
    # Use tensor cores: use fp16 or bf16 for optimal performance
    C = tl.zeros((T, H, D), dtype=tl.float32)
    for i in range(0, T, BLOCK_SIZE):
        a = tl.load(A_ptr + i * D * H + offsets[:, None, None] * D * H, mask=mask[:, None, None], other=0.0)
        b = tl.load(B_ptr + i * D * H + offsets[None, :, None] * D * H, mask=mask[None, :, None], other=0.0)
        a = tl.reshape(a, (T, H, D))
        b = tl.reshape(b, (T, H, D))
        C += tl.dot(a, b, allow_tf32=True)

    C = tl.reshape(C, (T, H, D))
    tl.store(C_ptr + offsets[:, None, None] * D * H, C, mask=mask[:, None, None])


@triton.jit
def einsum_kernel(
    C_ptr,
    B_ptr,
    L_ptr,
    X_ptr,
    out_ptr,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    S: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < T

    # Load C: (b, h, c, l, s) -> (c, l, s)
    C = tl.load(C_ptr + offsets[:, None, None, None] * S * H, mask=mask[:, None, None, None], other=0.0)
    C = tl.reshape(C, (T, H, S))

    # Load B: (b, h, c, l, s) -> (c, l, s)
    B = tl.load(B_ptr + offsets[:, None, None, None] * S * H, mask=mask[:, None, None, None], other=0.0)
    B = tl.reshape(B, (T, H, S))

    # Load L: (c, l, l) -> (l, l)
    L = tl.load(L_ptr + offsets[:, None] * T + offsets[None, :], mask=mask[:, None] & mask[None, :], other=0.0)
    L = tl.reshape(L, (T, T))

    # Load X: (b, h, c, l, d) -> (c, l, d)
    X = tl.load(X_ptr + offsets[:, None, None, None] * D * H, mask=mask[:, None, None, None], other=0.0)
    X = tl.reshape(X, (T, H, D))

    # Compute einsum: (c, l, s), (c, l, s), (l, l), (c, l, d) -> (c, l, d)
    # Use tensor cores: optimize for fp16/bf16
    result = tl.zeros((T, H, D), dtype=tl.float32)

    for i in range(0, T, BLOCK_SIZE):
        c = tl.load(C_ptr + i * S * H + offsets[:, None, None] * S * H, mask=mask[:, None, None], other=0.0)
        c = tl.reshape(c, (T, H, S))

        b = tl.load(B_ptr + i * S * H + offsets[:, None, None] * S * H, mask=mask[:, None, None], other=0.0)
        b = tl.reshape(b, (T, H, S))

        x = tl.load(X_ptr + i * D * H + offsets[:, None, None] * D * H, mask=mask[:, None, None], other=0.0)
        x = tl.reshape(x, (T, H, D))

        l_block = tl.load(L_ptr + i * T + offsets[:, None], mask=mask[:, None], other=0.0)
        l_block = tl.reshape(l_block, (T, 1))

        # (T, H, S) * (T, H, S) * (T, 1) * (T, H, D) -> (T, H, D)
        result += tl.dot(tl.dot(tl.dot(c, b, allow_tf32=True), l_block, allow_tf32=True), x, allow_tf32=True)

    result = tl.reshape(result, (T, H, D))
    tl.store(out_ptr + offsets[:, None, None] * D * H, result, mask=mask[:, None, None])


@triton.jit
def decay_kernel(
    A_cumsum_ptr,
    decay_ptr,
    T: tl.constexpr,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < T

    # Load A_cumsum: (b, h, c, l) -> (c, l)
    A_cumsum = tl.load(A_cumsum_ptr + offsets[:, None, None] * H, mask=mask[:, None, None], other=0.0)
    A_cumsum = tl.reshape(A_cumsum, (T, H))

    # Compute decay: exp(A_cumsum[., -1] - A_cumsum[.])
    A_cumsum_last = A_cumsum[:, -1]
    A_cumsum_last = tl.broadcast_to(A_cumsum_last[:, None], (T, H))
    decay = tl.exp(A_cumsum_last - A_cumsum)

    # Store result
    tl.store(decay_ptr + offsets[:, None] * H, decay, mask=mask[:, None])


@triton.jit
def einsum_state_kernel(
    decay_chunk_ptr,
    states_ptr,
    out_ptr,
    T: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < T

    # Load decay_chunk: (b, h, c) -> (c, h)
    decay_chunk = tl.load(decay_chunk_ptr + offsets[:, None] * H, mask=mask[:, None], other=0.0)
    decay_chunk = tl.reshape(decay_chunk, (T, H))

    # Load states: (b, h, c, p, s) -> (c, p, s)
    states = tl.load(states_ptr + offsets[:, None, None] * S, mask=mask[:, None, None], other=0.0)
    states = tl.reshape(states, (T, S))

    # Compute einsum: (c, h) @ (c, p, s) -> (c, h, s)
    result = tl.zeros((T, H, S), dtype=tl.float32)

    for i in range(0, T, BLOCK_SIZE):
        decay = tl.load(decay_chunk_ptr + i * H + offsets[:, None] * H, mask=mask[:, None], other=0.0)
        decay = tl.reshape(decay, (T, H))

        state = tl.load(states_ptr + i * S + offsets[:, None, None] * S, mask=mask[:, None, None], other=0.0)
        state = tl.reshape(state, (T, S))

        result += tl.dot(decay, state, allow_tf32=True)

    result = tl.reshape(result, (T, H, S))
    tl.store(out_ptr + offsets[:, None, None] * S, result, mask=mask[:, None, None])


def triton_segsum(A, T, H):
    # Ensure contiguous and proper dtype
    A = A.contiguous().to(torch.float32)
    L = torch.empty_like(A, dtype=torch.float32)

    # Grid size
    grid = lambda meta: (triton.cdiv(T, meta["BLOCK_SIZE"]),)

    # Autotune
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 16}, num_stages=2, num_warps=4),
            triton.Config({"BLOCK_SIZE": 32}, num_stages=2, num_warps=4),
            triton.Config({"BLOCK_SIZE": 64}, num_stages=2, num_warps=4),
            triton.Config({"BLOCK_SIZE": 128}, num_stages=2, num_warps=4),
        ],
        key=["T", "H"],
        nearest_power_of_2=True,
    )
    segsum_kernel[grid](A, L, T, H, BLOCK_SIZE=128)

    return L


def triton_einsum(C, B, L, X, T, H, S, D):
    C = C.contiguous().to(torch.float16)
    B = B.contiguous().to(torch.float16)
    L = L.contiguous().to(torch.float16)
    X = X.contiguous().to(torch.float16)

    out = torch.empty_like(X, dtype=torch.float16)

    grid = lambda meta: (triton.cdiv(T, meta["BLOCK_SIZE"]),)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 16}, num_stages=2, num_warps=4),
            triton.Config({"BLOCK_SIZE": 32}, num_stages=2, num_warps=4),
            triton.Config({"BLOCK_SIZE": 64}, num_stages=2, num_warps=4),
            triton.Config({"BLOCK_SIZE": 128}, num_stages=2, num_warps=4),
        ],
        key=["T", "H", "D", "S"],
        nearest_power_of_2=True,
    )
    einsum_kernel[grid](C, B, L, X, out, T, H, D, S, BLOCK_SIZE=128)

    return out


def triton_decay(A_cumsum, T, H):
    A_cumsum = A_cumsum.contiguous().to(torch.float32)
    decay = torch.empty_like(A_cumsum, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(T, meta["BLOCK_SIZE"]),)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 16}, num_stages=2, num_warps=4),
            triton.Config({"BLOCK_SIZE": 32}, num_stages=2, num_warps=4),
            triton.Config({"BLOCK_SIZE": 64}, num_stages=2, num_warps=4),
            triton.Config({"BLOCK_SIZE": 128}, num_stages=2, num_warps=4),
        ],
        key=["T", "H"],
        nearest_power_of_2=True,
    )
    decay_kernel[grid](A_cumsum, decay, T, H, BLOCK_SIZE=128)

    return decay


def triton_einsum_state(decay_chunk, states, T, H, S):
    decay_chunk = decay_chunk.contiguous().to(torch.float32)
    states = states.contiguous().to(torch.float32)
    out = torch.empty_like(states, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(T, meta["BLOCK_SIZE"]),)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 16}, num_stages=2, num_warps=4),
            triton.Config({"BLOCK_SIZE": 32}, num_stages=2, num_warps=4),
            triton.Config({"BLOCK_SIZE": 64}, num_stages=2, num_warps=4),
            triton.Config({"BLOCK_SIZE": 128}, num_stages=2, num_warps=4),
        ],
        key=["T", "H", "S"],
        nearest_power_of_2=True,
    )
    einsum_state_kernel[grid](decay_chunk, states, out, T, H, S, BLOCK_SIZE=128)

    return out


class ModelNew(nn.Module):
    def __init__(self, batch_size, seq_length, n_heads, d_head, d_state, block_len=64):
        super().__init__()
        assert seq_length % block_len == 0, "Sequence length must be divisible by block length"
        
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.block_len = block_len
        
        # Initialize parameters
        self.A = nn.Parameter(torch.randn(batch_size, seq_length, n_heads))
        self.B = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        self.C = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))

    def forward(self, X, initial_states=None):
        # Rearrange into blocks
        X_blocks, A_blocks, B_blocks, C_blocks = [
            rearrange(x, "b (c l) ... -> b c l ...", l=self.block_len)
            for x in (X, self.A, self.B, self.C)
        ]
        
        A_blocks = rearrange(A_blocks, "b c l h -> b h c l")
        
        # Compute cumulative sum on A_blocks
        A_cumsum = torch.cumsum(A_blocks, dim=-1)
        
        # 1. Compute diagonal block outputs via Triton einsum
        # Expand dims to match Triton kernels: (b, h, c, l)
        A_blocks_expanded = A_blocks.unsqueeze(0)  # (1, h, c, l)
        L = triton_segsum(A_blocks_expanded, self.block_len, self.n_heads)
        L = rearrange(L, "h c l -> c l h")  # (c, l, h)
        
        C_blocks_expanded = C_blocks.unsqueeze(0)  # (1, h, c, l, s)
        B_blocks_expanded = B_blocks.unsqueeze(0)  # (1, h, c, l, s)
        X_blocks_expanded = X_blocks.unsqueeze(0)  # (1, h, c, l, d)
        
        Y_diag = triton_einsum(C_blocks_expanded, B_blocks_expanded, L, X_blocks_expanded, self.block_len, self.n_heads, self.d_state, self.d_head)

        # 2. Compute intra-chunk states
        decay_states = triton_decay(A_cumsum, self.block_len, self.n_heads)
        decay_states = rearrange(decay_states, "b h c l -> b c h l")

        states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B_blocks, decay_states, X_blocks)
        
        # 3. Compute inter-chunk recurrence
        if initial_states is None:
            initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)
        
        # Compute decay_chunk
        A_cumsum_last = A_cumsum[:, :, :, -1:]  # (b, h, c, 1)
        decay_chunk = torch.exp(self.segsum(A_cumsum_last))  # (b, h, c, c)
        decay_chunk = rearrange(decay_chunk, "b h c c -> b c h")
        
        new_states = triton_einsum_state(decay_chunk, states, self.seq_length // self.block_len, self.n_heads, self.d_state)
        
        return new_states[:, -1]