import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr = 8,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    assert a.is_cuda and b.is_cuda
    assert a.shape[2] == b.shape[1], "Incompatible dimensions"
    assert a.is_contiguous() and b.is_contiguous()
    B, M, K = a.shape
    _, _, N = b.shape
    c = torch.empty((B, M, N), device=a.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(1), a.stride(2),
        b.stride(1), b.stride(2),
        c.stride(1), c.stride(2),
    )
    return c


@triton.jit
def conv1d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    bias_ptr,
    batch,
    out_channels,
    out_length,
    in_channels,
    length,
    kernel_size,
    stride,
    dilation,
    padding,
    has_bias: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_OUT: tl.constexpr,
    BLOCK_SIZE_IN: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)

    batch_start = pid_b * BLOCK_SIZE_BATCH
    oc_start = pid_oc * BLOCK_SIZE_OUT

    offs_b = batch_start + tl.arange(0, BLOCK_SIZE_BATCH)
    offs_oc = oc_start + tl.arange(0, BLOCK_SIZE_OUT)
    offs_ic = tl.arange(0, BLOCK_SIZE_IN)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    mask_b = offs_b < batch
    mask_oc = offs_oc < out_channels

    weight_ptrs = weight_ptr + offs_oc[:, None, None] * in_channels * kernel_size + \
                  offs_ic[None, :, None] * kernel_size + offs_k[None, None, :]
    weight_mask = mask_oc[:, None, None] & (offs_ic[None, :, None] < in_channels) & (offs_k[None, None, :] < kernel_size)

    for ol in range(0, out_length, BLOCK_SIZE_K):
        offs_ol = ol + offs_k
        mask_ol = offs_ol < out_length
        input_offsets = offs_b[None, :, None] * in_channels * length + \
                        offs_ic[None, None, :] * length + \
                        (offs_ol[None, None, :] * stride - padding + offs_k[None, None, :] * dilation)
        input_mask = mask_b[None, :, None] & (offs_ic[None, None, :] < in_channels) & mask_ol[None, None, :] & \
                     (input_offsets >= 0) & (input_offsets < length)
        input_ptrs = input_ptr + input_offsets

        output = tl.zeros((BLOCK_SIZE_OUT, BLOCK_SIZE_BATCH, BLOCK_SIZE_K), dtype=tl.float32)

        for ic in range(0, in_channels, BLOCK_SIZE_IN):
            input_ptrs_ic = input_ptrs + (ic - offs_ic[0]) * length
            weight_ptrs_ic = weight_ptrs + (ic - offs_ic[0]) * kernel_size

            input_chunk = tl.load(input_ptrs_ic, mask=input_mask, other=0.0)
            weight_chunk = tl.load(weight_ptrs_ic, mask=weight_mask, other=0.0)

            output += tl.dot(weight_chunk, input_chunk, out_dtype=tl.float32)

            input_mask = input_mask & ((offs_ic + ic + BLOCK_SIZE_IN) < in_channels)[:, None, None]

        if has_bias:
            bias = tl.load(bias_ptr + offs_oc, mask=mask_oc, other=0.0)
            output = output + bias[:, None, None]

        output = output.to(tl.float16)

        output_offsets = offs_b[None, :, None] * out_channels * out_length + \
                         offs_oc[:, None, None] * out_length + offs_ol[None, None, :]
        output_ptrs = output_ptr + output_offsets
        output_mask = mask_b[None, :, None] & mask_ol[None, None, :] & mask_oc[:, None, None]
        tl.store(output_ptrs, output, mask=output_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = dilation * (kernel_size - 1) // 2
        self.use_bias = bias

        # Initialize weight and bias
        k = 1.0 / (in_channels * kernel_size)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        self.weight.data.uniform_(-k**0.5, k**0.5)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            self.bias.data.uniform_(-k**0.5, k**0.5)
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        batch, in_channels, length = x.shape
        assert in_channels == self.in_channels

        out_length = (length + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1

        # Reshape weight to (out_channels, in_channels * kernel_size)
        weight_flat = self.weight.view(self.out_channels, -1).contiguous()

        # Im2col-like transformation using unfold
        # (batch, in_channels, kernel_size, out_length)
        x_unfold = torch.nn.functional.unfold(
            x.unsqueeze(-1),  # (batch, in_channels, length, 1)
            kernel_size=(self.kernel_size, 1),
            dilation=(self.dilation, 1),
            padding=(self.padding, 0),
            stride=(self.stride, 1)
        )  # (batch, in_channels * kernel_size, out_length)

        x_unfold = x_unfold.view(batch, self.in_channels * self.kernel_size, out_length).transpose(1, 2).contiguous()

        # Use Triton matmul
        output = triton_matmul(x_unfold, weight_flat.t().contiguous())

        # Add bias if needed
        if self.bias is not None:
            output = output + self.bias[None, None, :]

        return output.transpose(1, 2)  # (batch, out_channels, out_length)