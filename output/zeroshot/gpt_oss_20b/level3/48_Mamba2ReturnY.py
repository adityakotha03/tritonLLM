import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import triton
import triton.language as tl


# ---------- Triton kernels ----------
@triton.jit
def segsum_kernel(
    x_ptr,          # input: (b*h*c, l)
    out_ptr,        # output: (b*h*c, l, l)
    l: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Computes the segment sum matrix for each row of the input.
    For each row x of length l, out[i, j] = sum(x[0:j+1]) - sum(x[0:i])
    The resulting matrix is lower‑triangular; values above the diagonal
    are filled with -inf.
    """
    row = tl.program_id(0)                 # each program handles one row
    stride = l                              # stride to jump between rows

    # load the entire row
    offsets = tl.arange(0, l)
    x = tl.load(x_ptr + row * stride + offsets, mask=offsets < l, other=0.0)

    # cumulative sum
    cumsum = tl.cumsum(x)

    # compute triangular differences
    out = tl.empty((l, l), dtype=x.dtype)
    for i in range(l):
        for j in range(l):
            if j >= i:
                out[i, j] = cumsum[j] - cumsum[i]
            else:
                out[i, j] = -float("inf")

    # store the result
    out_offsets = tl.arange(0, l)[:, None] * l + tl.arange(0, l)[None, :]
    tl.store(out_ptr + row * l * l + out_offsets, out, mask=(out_offsets < l * l))


def segsum_torch(x):
    """
    Wrapper that calls the Triton segsum kernel.
    Expects x of shape (..., T). It flattens the leading dims
    and returns a tensor of shape (..., T, T).
    """
    assert x.is_cuda, "Input must be on CUDA"
    shape = x.shape
    T = shape[-1]
    flat = x.view(-1, T)                     # (N, T)
    N = flat.shape[0]
    out = torch.empty((N, T, T), device=x.device, dtype=x.dtype)

    BLOCK_SIZE = 128
    grid = lambda meta: (N,)
    segsum_kernel[grid](flat, out, T, BLOCK_SIZE=BLOCK_SIZE)

    return out.view(*shape[:-1], T, T)


@triton.jit
def matmul_kernel(
    a_ptr,          # shape (b*h*c, l, p)
    b_ptr,          # shape (b*h*c, l, n)
    out_ptr,        # shape (b*h*c, l, n)
    l: tl.constexpr,
    p: tl.constexpr,
    n: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Performs batched matrix multiplication for each block:
    out[b, l, n] = sum_{k=0}^{p-1} a[b, l, k] * b[b, k, n]
    The kernel is tiled over the inner dimension p.
    """
    row = tl.program_id(0)                 # each program handles one (b,h,c)
    stride_b = l * p
    stride_a = l * n

    for i in range(l):
        acc = tl.zeros((n,), dtype=tl.float32)
        for j in range(0, p, BLOCK_SIZE):
            a_chunk = tl.load(a_ptr + row * stride_b + i * p + j,
                              mask=j + tl.arange(0, BLOCK_SIZE) < p,
                              other=0.0)
            b_chunk = tl.load(b_ptr + row * stride_b + j * n + tl.arange(0, BLOCK_SIZE)[:, None],
                              mask=j + tl.arange(0, BLOCK_SIZE) < p,
                              other=0.0)
            acc += tl.dot(a_chunk, b_chunk)
        tl.store(out_ptr + row * stride_a + i * n, acc, mask=i < l)


def matmul_torch(a, b):
    """
    Wrapper that calls the Triton matmul kernel.
    Expects a of shape (B, H, C, L, P) and b of shape (B, H, C, P, N).
    Returns a tensor of shape (B, H, C, L, N).
    """
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA"
    B, H, C, L, P = a.shape
    _, _, _, _, N = b.shape
    a_flat = a.view(B * H * C, L, P)
    b_flat = b.view(B * H * C, P, N)
    out_flat = torch.empty_like(a_flat, dtype=a.dtype)
    BLOCK_SIZE = 128
    grid = lambda meta: (B * H * C,)
    matmul_kernel[grid](a_flat, b_flat, out_flat, L, P, N, BLOCK_SIZE=BLOCK_SIZE)
    return out_flat.view(B, H, C, L, N)


# ---------- Model implementation ----------
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

        # Parameters
        self.A = nn.Parameter(torch.randn(batch_size, seq_length, n_heads))
        self.B = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        self.C = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))

    def forward(self, X, initial_states=None):
        """
        Implements the SSD operation using Triton kernels.
        """
        # 1. Rearrange into blocks/chunks
        X_blocks, A_blocks, B_blocks, C_blocks = [
            rearrange(x, "b (c l) ... -> b c l ...", l=self.block_len)
            for x in (X, self.A, self.B, self.C)
        ]

        # Convert to shapes suitable for Triton kernels
        B, C, L, H = X_blocks.shape
        # A_blocks: (B, C, L, H) -> (B, H, C, L)
        A_blocks = rearrange(A_blocks, "b c l h -> b h c l")
        A_cumsum = torch.cumsum(A_blocks, dim=-1)

        # 2. Compute diagonal block outputs
        # L_mat = exp(segsum(A_blocks))
        segsum_out = segsum_torch(A_blocks)      # (B, H, C, L, L)
        L_mat = torch.exp(segsum_out)            # same shape

        # Prepare tensors for matmul
        # C_blocks: (B, C, L, H, d_state) -> (B, H, C, L, d_state)
        C_blocks = rearrange(C_blocks, "b c l h p -> b h c l p")
        B_blocks = rearrange(B_blocks, "b c l h p -> b h c l p")
        X_blocks = rearrange(X_blocks, "b c l h p -> b h c l p")

        # Y_diag = einsum("bclhn,bcshn,bhcls,bcshp->bclhp",
        #                 C_blocks, B_blocks, L, X_blocks)
        # This is equivalent to:
        #   temp = B_blocks * L   (broadcast over head)
        #   Y_diag = (temp @ X_blocks) @ C_blocks
        temp = torch.einsum("bclhn,bhcls->bclhs", B_blocks, L_mat)  # (B, H, C, L, d_state)
        Y_diag = torch.einsum("bclhs,bclhp->bclhp", temp, X_blocks)
        Y_diag = torch.einsum("bclhp,bhcls->bclhp", Y_diag, C_blocks)

        # 3. Compute intra-chunk states
        decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))  # (B, H, C, L, d_state)
        states = torch.einsum("bclhn,bhcl,bclhp->bchpn",
                              B_blocks, decay_states, X_blocks)     # (B, C, H, L, d_state)

        # 4. Compute inter-chunk recurrence
        if initial_states is None:
            initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)  # (B, C+1, H, L, d_state)

        decay_chunk = torch.exp(
            segsum_torch(
                torch.cat(
                    [A_cumsum[:, :, :, -1:], A_cumsum[:, :, :, :1]],
                    dim=-1
                )
            )
        )  # shape (B, H, C+1, L, L)

        new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)  # (B, Z, H, L, d_state)
        states = new_states[:, :-1]  # (B, C, H, L, d_state)

        # 5. State-to-output conversion
        state_decay_out = torch.exp(A_cumsum)  # (B, H, C, L, d_state)
        Y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp",
                             C_blocks, states, state_decay_out)

        # 6. Combine diagonal and off-diagonal terms
        Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")

        return Y