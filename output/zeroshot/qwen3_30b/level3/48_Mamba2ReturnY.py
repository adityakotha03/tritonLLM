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
    BLOCK_SIZE: tl.constexpr,
    n_heads: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < T

    # Load A for this block
    A = tl.load(A_ptr + offsets, mask=mask, other=0.0)

    # Compute cumulative sum along the sequence dim
    A_cumsum = tl.cumsum(A, axis=0)

    # Compute diagonal block L = exp(segsum(A))
    # segsum(A) = A_cumsum[:, None] - A_cumsum[None, :]
    # Use upper triangular mask to set lower triangular to -inf
    A_cumsum_outer = A_cumsum[:, None] - A_cumsum[None, :]
    mask_upper = tl.triu(tl.ones((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.int32), diagonal=0)
    A_cumsum_upper = tl.where(mask_upper, A_cumsum_outer, -float('inf'))
    L = tl.exp(A_cumsum_upper)

    # Store L back
    tl.store(L_ptr + (pid * BLOCK_SIZE * BLOCK_SIZE), L, mask=mask[:, None] & mask[None, :])


@triton.jit
def matmul_abc_kernel(
    C_ptr,  # (b, c, l, h, d_state)
    B_ptr,  # (b, c, l, h, d_state)
    L_ptr,  # (b, c, l, l)  - from segsum
    X_ptr,  # (b, c, l, h, d_head)
    Y_ptr,  # (b, c, l, h, d_head)
    T: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    d_state: tl.constexpr,
    d_head: tl.constexpr,
    n_heads: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load B, C, X slices
    B = tl.load(B_ptr + offsets[:, None, None, None, None] * B_ptr.stride(0) +
                offsets[None, :, None, None, None] * B_ptr.stride(1) +
                0 * B_ptr.stride(2) +  # fixed block index
                0 * B_ptr.stride(3) +  # fixed head index
                tl.arange(0, d_state)[None, None, None, :, None] * B_ptr.stride(4),
                mask=offsets[:, None] < T,
                other=0.0)

    C = tl.load(C_ptr + offsets[:, None, None, None, None] * C_ptr.stride(0) +
                offsets[None, :, None, None, None] * C_ptr.stride(1) +
                0 * C_ptr.stride(2) +
                0 * C_ptr.stride(3) +
                tl.arange(0, d_state)[None, None, None, :, None] * C_ptr.stride(4),
                mask=offsets[:, None] < T,
                other=0.0)

    X = tl.load(X_ptr + offsets[:, None, None, None, None] * X_ptr.stride(0) +
                offsets[None, :, None, None, None] * X_ptr.stride(1) +
                0 * X_ptr.stride(2) +
                0 * X_ptr.stride(3) +
                tl.arange(0, d_head)[None, None, None, None, :] * X_ptr.stride(4),
                mask=offsets[:, None] < T,
                other=0.0)

    L = tl.load(L_ptr + offsets[:, None, None, None] * L_ptr.stride(0) +
                offsets[None, :, None, None] * L_ptr.stride(1) +
                0 * L_ptr.stride(2) +
                0 * L_ptr.stride(3),
                mask=offsets[:, None] < T,
                other=0.0)

    # Compute Y_diag = einsum('bclhn,bcshn,bhcls,bcshp->bclhp', C, B, L, X)
    # Fuse C * B -> intermediate
    # Then (C*B) * L
    # Then (C*B*L) * X
    temp = tl.zeros((BLOCK_SIZE, BLOCK_SIZE, d_head), dtype=tl.float32)
    for i in range(d_state):
        temp += C[:, :, :, i, None] * B[:, :, :, i, None] * L[:, :, None]  # (l, l, d_head)

    # Final matmul with X: (l, l, d_head) @ (l, d_head) -> (l, d_head)
    Y = tl.dot(temp, X, allow_tf32=True)

    # Store result
    tl.store(Y_ptr + offsets[:, None, None] * Y_ptr.stride(0) +
             offsets[None, :, None] * Y_ptr.stride(1) +
             0 * Y_ptr.stride(2) +
             0 * Y_ptr.stride(3) +
             tl.arange(0, d_head)[None, None, :] * Y_ptr.stride(4),
             Y,
             mask=offsets[:, None] < T)


@triton.jit
def state_update_kernel(
    A_cumsum_ptr,
    B_ptr,
    X_ptr,
    states_ptr,
    T: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    d_state: tl.constexpr,
    d_head: tl.constexpr,
    n_heads: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load A_cumsum for current block
    A_cumsum = tl.load(A_cumsum_ptr + offsets[:, None] * A_cumsum_ptr.stride(0) +
                       0 * A_cumsum_ptr.stride(1) +
                       0 * A_cumsum_ptr.stride(2) +
                       0 * A_cumsum_ptr.stride(3),
                       mask=offsets[:, None] < T,
                       other=0.0)

    # Load B, X
    B = tl.load(B_ptr + offsets[:, None, None] * B_ptr.stride(0) +
                0 * B_ptr.stride(1) +
                0 * B_ptr.stride(2) +
                tl.arange(0, d_state)[None, None, :] * B_ptr.stride(3),
                mask=offsets[:, None] < T,
                other=0.0)

    X = tl.load(X_ptr + offsets[:, None, None] * X_ptr.stride(0) +
                0 * X_ptr.stride(1) +
                0 * X_ptr.stride(2) +
                tl.arange(0, d_head)[None, None, :] * X_ptr.stride(3),
                mask=offsets[:, None] < T,
                other=0.0)

    # decay_states = exp(A_cumsum[:, -1] - A_cumsum)
    decay = tl.exp(A_cumsum[:, -1] - A_cumsum)

    # states = einsum('bclhn,bhcl,bclhp->bchpn', B, decay, X)
    states = tl.dot(B, decay[:, None], allow_tf32=True)  # (l, d_state)
    states = tl.dot(states, X, allow_tf32=True)  # (l, d_head)

    # Store states
    tl.store(states_ptr + offsets[:, None, None] * states_ptr.stride(0) +
             0 * states_ptr.stride(1) +
             0 * states_ptr.stride(2) +
             tl.arange(0, d_head)[None, None, :] * states_ptr.stride(3),
             states,
             mask=offsets[:, None] < T)


@triton.jit
def inter_chunk_kernel(
    A_cumsum_ptr,
    states_ptr,
    out_ptr,
    T: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    d_head: tl.constexpr,
    n_heads: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load A_cumsum final
    A_last = tl.load(A_cumsum_ptr + (T - 1) * A_cumsum_ptr.stride(0) +
                     0 * A_cumsum_ptr.stride(1) +
                     0 * A_cumsum_ptr.stride(2) +
                     0 * A_cumsum_ptr.stride(3),
                     other=0.0)

    # Load decay_chunk = exp(segsum(pad(A_cumsum)))
    # Pad A_cumsum: [A_0, A_1, ..., A_T-1] -> [0, A_0, ..., A_T-1]
    # segsum becomes: [A_0, A_1 - A_0, ..., A_T-1 - A_T-2]
    # decay_chunk = exp(A_0, A_1 - A_0, ..., A_T-1 - A_T-2)
    decay_chunk = tl.exp(A_last - A_cumsum_ptr + A_cumsum_ptr)  # Use trick: pad at front
    decay_chunk = tl.exp(A_cumsum_ptr[0] - A_cumsum_ptr)  # For pad(0), first decay = A_0
    decay_chunk = tl.exp(A_cumsum_ptr[0] - A_cumsum_ptr)  # corrected: decay_chunk = exp(A_cumsum[0] - A_cumsum)

    # Actually compute decay_chunk for inter-chunk recurrence: exp(segsum(A_cumsum_pad))
    # Use segsum with padding
    A_cumsum_pad = tl.concatenate((tl.zeros((1,), dtype=tl.float32), A_cumsum), axis=0)
    decay_chunk = tl.exp(A_cumsum_pad[1:] - A_cumsum_pad[:-1])  # diff

    # Extract decay_chunk for current block
    decay_chunk = decay_chunk[pid * BLOCK_SIZE : (pid + 1) * BLOCK_SIZE]

    # states shape: (b, c, l, d_head)
    # Load states
    states = tl.load(states_ptr + offsets[:, None, None] * states_ptr.stride(0) +
                     0 * states_ptr.stride(1) +
                     0 * states_ptr.stride(2) +
                     tl.arange(0, d_head)[None, None, :] * states_ptr.stride(3),
                     mask=offsets[:, None] < T,
                     other=0.0)

    # new_states = einsum('bhzc,bchpn->bzhpn', decay_chunk, states)
    # Decay is (l,) -> (l, 1), states (l, d_head) -> (l, d_head)
    new_states = decay_chunk[:, None] * states

    # Store
    tl.store(out_ptr + offsets[:, None, None] * out_ptr.stride(0) +
             0 * out_ptr.stride(1) +
             0 * out_ptr.stride(2) +
             tl.arange(0, d_head)[None, None, :] * out_ptr.stride(3),
             new_states,
             mask=offsets[:, None] < T)


@triton.jit
def state_to_output_kernel(
    A_cumsum_ptr,
    C_ptr,
    states_ptr,
    Y_ptr,
    T: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    d_state: tl.constexpr,
    d_head: tl.constexpr,
    n_heads: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load A_cumsum
    A_cumsum = tl.load(A_cumsum_ptr + offsets[:, None] * A_cumsum_ptr.stride(0) +
                       0 * A_cumsum_ptr.stride(1) +
                       0 * A_cumsum_ptr.stride(2) +
                       0 * A_cumsum_ptr.stride(3),
                       mask=offsets[:, None] < T,
                       other=0.0)

    # Load C
    C = tl.load(C_ptr + offsets[:, None, None, None] * C_ptr.stride(0) +
                0 * C_ptr.stride(1) +
                0 * C_ptr.stride(2) +
                0 * C_ptr.stride(3) +
                tl.arange(0, d_state)[None, None, None, :] * C_ptr.stride(4),
                mask=offsets[:, None] < T,
                other=0.0)

    # Load states
    states = tl.load(states_ptr + offsets[:, None, None] * states_ptr.stride(0) +
                     0 * states_ptr.stride(1) +
                     0 * states_ptr.stride(2) +
                     tl.arange(0, d_head)[None, None, :] * states_ptr.stride(3),
                     mask=offsets[:, None] < T,
                     other=0.0)

    # state_decay_out = exp(A_cumsum)
    state_decay = tl.exp(A_cumsum)

    # Y_off = einsum('bclhn,bchpn,bhcl->bclhp', C, states, state_decay)
    # Fuse: C * states -> (l, d_state) * (l, d_head) -> (l, d_head)
    # Then (l, d_head) * decay -> (l, d_head)
    Y = tl.dot(C, states, allow_tf32=True)
    Y = Y * state_decay[:, None]

    tl.store(Y_ptr + offsets[:, None, None] * Y_ptr.stride(0) +
             0 * Y_ptr.stride(1) +
             0 * Y_ptr.stride(2) +
             tl.arange(0, d_head)[None, None, :] * Y_ptr.stride(3),
             Y,
             mask=offsets[:, None] < T)


def triton_segsum(A, T, BLOCK_SIZE=128):
    """Compute segsum(A) = exp(A_cumsum[:, None] - A_cumsum[None, :]) with upper triangular mask."""
    assert A.dim() == 3, "A must be (b, c, l)"
    B, C, L = A.shape
    L = L
    output = torch.empty(B, C, L, L, device=A.device, dtype=torch.float32)

    # Grid
    grid = lambda meta: (B * C,)

    segsum_kernel[grid](
        A,
        output,
        T=L,
        BLOCK_SIZE=BLOCK_SIZE,
        n_heads=1,
    )
    return output


def triton_matmul_abc(C, B, L, X, T, BLOCK_SIZE=128, d_state=16, d_head=64):
    """Fused matmul: Y_diag = einsum('bclhn,bcshn,bhcls,bcshp->bclhp')"""
    B, C, L, H, D = X.shape
    output = torch.empty(B, C, L, H, D, device=X.device, dtype=torch.float32)

    grid = lambda meta: (B * C,)

    matmul_abc_kernel[grid](
        C,
        B,
        L,
        X,
        output,
        T=L,
        BLOCK_SIZE=BLOCK_SIZE,
        d_state=d_state,
        d_head=d_head,
        n_heads=H,
    )
    return output


def triton_state_update(A_cumsum, B, X, T, BLOCK_SIZE=128, d_state=16, d_head=64):
    """Compute intra-chunk states: states = einsum('bclhn,bhcl,bclhp->bchpn')"""
    B, C, L, H = X.shape
    output = torch.empty(B, C, L, H, device=X.device, dtype=torch.float32)

    grid = lambda meta: (B * C,)

    state_update_kernel[grid](
        A_cumsum,
        B,
        X,
        output,
        T=L,
        BLOCK_SIZE=BLOCK_SIZE,
        d_state=d_state,
        d_head=d_head,
        n_heads=H,
    )
    return output


def triton_inter_chunk(A_cumsum, states, T, BLOCK_SIZE=128, d_head=64):
    """Compute inter-chunk: new_states = einsum('bhzc,bchpn->bzhpn')"""
    B, C, L, H = states.shape
    output = torch.empty(B, C, L, H, device=states.device, dtype=torch.float32)

    grid = lambda meta: (B * C,)

    inter_chunk_kernel[grid](
        A_cumsum,
        states,
        output,
        T=L,
        BLOCK_SIZE=BLOCK_SIZE,
        d_head=d_head,
        n_heads=H,
    )
    return output


def triton_state_to_output(A_cumsum, C, states, T, BLOCK_SIZE=128, d_state=16, d_head=64):
    """Compute Y_off = einsum('bclhn,bchpn,bhcl->bclhp')"""
    B, C, L, H = states.shape
    output = torch.empty(B, C, L, H, device=states.device, dtype=torch.float32)

    grid = lambda meta: (B * C,)

    state_to_output_kernel[grid](
        A_cumsum,
        C,
        states,
        output,
        T=L,
        BLOCK_SIZE=BLOCK_SIZE,
        d_state=d_state,
        d_head=d_head,
        n_heads=H,
    )
    return output


class ModelNew(nn.Module):
    def __init__(self, batch_size, seq_length, n_heads, d_head, d_state, block_len=64):
        super().__init__()
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

        # Compute A_cumsum per block
        A_cumsum = torch.cumsum(A_blocks, dim=-1)

        # 1. Compute diagonal block outputs: Y_diag = einsum('bclhn,bcshn,bhcls,bcshp->bclhp')
        # Use Triton fused kernel
        L = triton_segsum(A_blocks, T=self.block_len, BLOCK_SIZE=128)
        Y_diag = triton_matmul_abc(C_blocks, B_blocks, L, X_blocks, T=self.block_len)

        # 2. Compute intra-chunk states
        states = triton_state_update(A_cumsum, B_blocks, X_blocks, T=self.block_len)

        # 3. Compute inter-chunk recurrence
        if initial_states is None:
            initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)
        new_states = triton_inter_chunk(A_cumsum, states, T=self.block_len)
        states = new_states[:, :-1]

        # 4. Compute state-to-output conversion
        Y_off = triton_state_to_output(A_cumsum, C_blocks, states, T=self.block_len)

        # Combine
        Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")

        return Y