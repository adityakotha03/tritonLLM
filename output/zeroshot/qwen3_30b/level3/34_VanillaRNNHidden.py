import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def linear_kernel(
    x_ptr, w_ptr, out_ptr,
    n_rows, n_cols, n_features,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    ACTIVATION: tl.constexpr,
    USE_BIAS: tl.constexpr,
    bias_ptr,
    use_fp16: tl.constexpr
):
    # Calculate program ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(n_rows, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(n_cols, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(n_features, BLOCK_SIZE_K)

    # Calculate block index
    block_id = pid // num_pid_n
    block_m = block_id // num_pid_k
    block_n = pid % num_pid_n
    block_k = pid % num_pid_k

    # Calculate offsets for the block
    row_offset = block_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_offset = block_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    feature_offset = block_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    # Create masks for bounds
    row_mask = row_offset < n_rows
    col_mask = col_offset < n_cols
    feature_mask = feature_offset < n_features

    # Load weights (W) in a block-wise manner
    w = tl.load(
        w_ptr + (block_n * BLOCK_SIZE_N + col_offset)[:, None] * n_features +
        (block_k * BLOCK_SIZE_K + feature_offset)[None, :],
        mask=col_mask[:, None] & feature_mask[None, :],
        other=0.0
    )

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Perform matrix multiplication with fusion of reduction
    for k in range(0, num_pid_k):
        # Load input x block
        x = tl.load(
            x_ptr + (block_m * BLOCK_SIZE_M + row_offset)[:, None] * n_features +
            (k * BLOCK_SIZE_K + feature_offset)[None, :],
            mask=row_mask[:, None] & feature_mask[None, :],
            other=0.0
        )

        # Perform fused matmul
        acc += tl.dot(x, w)

    # Apply bias if needed
    if USE_BIAS:
        bias = tl.load(
            bias_ptr + col_offset,
            mask=col_mask,
            other=0.0
        )
        acc += bias[None, :]

    # Apply activation function
    if ACTIVATION == "TANH":
        acc = tl.tanh(acc)
    elif ACTIVATION == "NONE":
        pass
    else:
        raise ValueError("Unsupported activation")

    # Store output
    out_ptrs = out_ptr + (row_offset[:, None] * n_cols + col_offset[None, :])
    tl.store(out_ptrs, acc, mask=row_mask[:, None] & col_mask[None, :])


@triton.jit
def concat_kernel(
    x_ptr, y_ptr, out_ptr,
    batch_size, seq_len, input_size, hidden_size,
    BLOCK_SIZE: tl.constexpr
):
    # Each program processes a block of elements
    block_id = tl.program_id(0)
    block_offset = block_id * BLOCK_SIZE

    # Create indices for current block
    indices = block_offset + tl.arange(0, BLOCK_SIZE)
    mask = indices < seq_len * batch_size * (input_size + hidden_size)

    # Offset for x and y tensors
    x_offset = (indices % (seq_len * batch_size)) * input_size + (indices // (seq_len * batch_size)) * input_size
    y_offset = (indices % (seq_len * batch_size)) * hidden_size + (indices // (seq_len * batch_size)) * hidden_size

    # Load and concatenate
    x_vals = tl.load(x_ptr + x_offset, mask=indices < seq_len * batch_size * input_size, other=0.0)
    y_vals = tl.load(y_ptr + y_offset, mask=indices < seq_len * batch_size * hidden_size, other=0.0)

    # Write concatenated output
    tl.store(out_ptr + indices, x_vals, mask=indices < seq_len * batch_size * input_size)
    tl.store(out_ptr + indices + seq_len * batch_size * input_size, y_vals, mask=indices < seq_len * batch_size * hidden_size)


@triton.jit
def tanh_kernel(
    x_ptr, out_ptr,
    n_elements, BLOCK_SIZE: tl.constexpr
):
    block_id = tl.program_id(0)
    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.tanh(x)
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_linear(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor, activation: str = "NONE", use_fp16: bool = True) -> torch.Tensor:
    assert x.is_cuda and w.is_cuda and (bias is None or bias.is_cuda), "All tensors must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    n_rows, n_cols = x.shape[0], w.shape[1]
    n_features = w.shape[0]

    # Use fp16 or bf16 if requested
    dtype = torch.float16 if use_fp16 else torch.float32
    x = x.to(dtype)
    w = w.to(dtype)
    if bias is not None:
        bias = bias.to(dtype)

    out = torch.empty(n_rows, n_cols, device=x.device, dtype=dtype)

    # Determine block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64

    # Grid definition
    num_pid_m = triton.cdiv(n_rows, BLOCK_SIZE_M)
    num_pid_n = triton.cdiv(n_cols, BLOCK_SIZE_N)
    num_pid_k = triton.cdiv(n_features, BLOCK_SIZE_K)
    grid = (num_pid_m * num_pid_n * num_pid_k,)

    # Launch kernel
    linear_kernel[grid](
        x, w, out, n_rows, n_cols, n_features,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        ACTIVATION=activation,
        USE_BIAS=(bias is not None),
        bias_ptr=bias.data_ptr() if bias is not None else 0,
        use_fp16=use_fp16
    )

    return out


def triton_concat(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    y = y.contiguous()

    batch_size, seq_len, input_size = x.shape
    _, _, hidden_size = y.shape

    out = torch.empty(seq_len, batch_size, input_size + hidden_size, device=x.device, dtype=x.dtype)

    # Use BLOCK_SIZE of 1024 for good memory coalescing
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(seq_len * batch_size * (input_size + hidden_size), BLOCK_SIZE),)

    concat_kernel[grid](
        x, y, out,
        batch_size, seq_len, input_size, hidden_size,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out


def triton_tanh(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(x.numel(), BLOCK_SIZE),)

    tanh_kernel[grid](x, out, x.numel(), BLOCK_SIZE=BLOCK_SIZE)

    return out


class ModelNew(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights with Triton-friendly types
        self.i2h_weight = nn.Parameter(torch.randn(input_size + hidden_size, hidden_size, dtype=torch.float16))
        self.i2h_bias = nn.Parameter(torch.randn(hidden_size, dtype=torch.float16))
        self.h2o_weight = nn.Parameter(torch.randn(hidden_size, output_size, dtype=torch.float16))
        self.h2o_bias = nn.Parameter(torch.randn(output_size, dtype=torch.float16))

    def forward(self, x: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        seq_len, batch_size, _ = x.size()
        hidden = h0.to(x.device).to(torch.float16)  # Keep in fp16
        outputs = []

        # Fuse: [concat + linear + tanh + linear] into one RNN step
        for t in range(seq_len):
            # Concat input and hidden state
            combined = triton_concat(x[t], hidden)

            # Compute i2h with tanh fusion (i2h + tanh)
            hidden = triton_linear(
                combined, self.i2h_weight, self.i2h_bias,
                activation="TANH", use_fp16=True
            )

            # Compute output (h2o) with bias
            output = triton_linear(
                hidden, self.h2o_weight, self.h2o_bias,
                activation="NONE", use_fp16=True
            )

            outputs.append(output)

        # Stack outputs
        return torch.stack(outputs, dim=0).to(torch.float32)