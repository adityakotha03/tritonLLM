import torch
import torch.nn as nn
import triton
import triton.language as tl
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_mean_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = x2 % 128
    x1 = x2 // 128
    x3 = x2
    tmp0 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 0)
    tmp5 = 65536.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr0 + x1, tmp6, xmask)


@triton.jit
def triton_poi_fused_mean_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = x2 % 1
    x1 = x2 // 1
    x3 = x2
    tmp0 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 0)
    tmp5 = 1.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr0 + x1, tmp6, xmask)


def triton_mean(input_0, output_0):
    arg0_1, arg1_1 = input_0, output_0
    args.clear()
    assert_size_stride(arg0_1, (16, 128, 256, 256), (65536, 512, 2, 1))
    assert_size_stride(arg1_1, (16, 128, 1, 1), (128, 1, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 128, 1, 1), (128, 1, 1, 1), torch.float32)
        get_raw_buf = buf0
        triton_poi_fused_mean_0[grid(2048)](arg0_1, get_raw_buf, 2048,
            XBLOCK=128, num_warps=4, num_stages=1)
        del arg0_1
        buf1 = empty_strided_cuda((16, 128, 1, 1), (128, 1, 1, 1), torch.float32)
        triton_poi_fused_mean_1[grid(2048)](get_raw_buf, buf1, 2048,
            XBLOCK=1, num_warps=1, num_stages=1)
        del get_raw_buf
    return buf1


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(64, 128, 3, stride=2,
            padding=1, output_padding=1)
        self.multiplier = 0.5

    def forward(self, input_0):
        arg0_1 = input_0
        arg0_2 = arg0_1
        arg0_3 = arg0_2
        arg0_4 = arg0_3
        arg0_5 = arg0_4
        arg0_6 = arg0_5
        arg0_7 = arg0_6
        output_0 = self.conv_transpose(arg0_1)
        output_1 = output_0 * self.multiplier
        output_2 = triton_mean(output_1, empty_strided_cuda((16, 128, 1, 1),
            (128, 1, 1, 1), torch.float32))
        return output_2