import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime import triton_helpers
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_add_0(in_ptr0, in_ptr1, out_ptr0, xnumel, ynumel, xoffset,
    yoffset, rnumel):
    xoffset = xoffset + tl.program_id(1) * rnumel
    xoffset = xoffset + tl.arange(0, rnumel)[None, :]
    yoffset = yoffset + tl.program_id(0) * rnumel
    yoffset = yoffset + tl.arange(0, rnumel)[:, None]
    tl.full([rnumel, rnumel], True, tl.int1)
    tl.full([rnumel, rnumel], True, tl.int1)
    yindex = yoffset % xnumel
    yindex = yindex + xoffset
    ymask = yindex < xnumel
    xmask = xindex < ynumel
    x2 = xindex
    x3 = xindex
    x0 = yindex
    x1 = yindex
    tmp0 = tl.load(in_ptr0 + x2, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + x3, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


@triton.jit
def triton_poi_fused_hard_swish_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, xnumel, xoffset):
    xoffset = xoffset + tl.program_id(0) * xnumel
    xoffset = xoffset + tl.arange(0, xnumel)[:, None]
    tl.full([xnumel, 1], True, tl.int1)
    xindex = xoffset % xnumel
    x2 = xindex
    x3 = xindex
    x0 = xindex
    x1 = xindex
    tmp0 = tl.load(in_ptr0 + x2, None)
    tmp1 = tl.load(in_ptr1 + x3, None)
    tmp2 = tl.load(in_ptr2 + x1, None)
    tmp3 = tl.load(in_ptr3 + x0, None, eviction_policy='evict_last')
    tmp4 = 3.0
    tmp5 = tmp1 + tmp4
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp6, tmp5)
    tmp8 = tmp7 * tmp2
    tmp9 = tmp0 * tmp8
    tl.store(in_out_ptr0 + x0, tmp9, None)


@triton.jit
def triton_poi_fused_logsumexp_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, 1], True, tl.int1)
    xindex = xoffset
    x2 = xindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x2, None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, 1])
    tmp3 = tl.where(tl.full([XBLOCK, 1], True, tl.int1), tmp1, 0)
    tmp4 = tl.broadcast_to(tmp3, [1, XBLOCK])
    tmp5 = triton_helpers.max(tmp4, 1)[:, None]
    tmp6 = tmp0 - tmp5
    tmp7 = tl_math.exp(tmp6)
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, 1])
    tmp10 = tl.where(tl.full([XBLOCK, 1], True, tl.int1), tmp8, 0)
    tmp11 = tl.broadcast_to(tmp10, [1, XBLOCK])
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp13 = tmp5 + tmp12
    tmp14 = tl_math.log(tmp13)
    tl.store(out_ptr0 + x0, tmp14, None)


def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (64, 8, 3, 3), (576, 72, 9, 1))
    assert_size_stride(primals_2, (16, 64, 1, 1), (64, 1, 1, 64))
    assert_size_stride(primals_3, (64,), (1,))
    assert_size_stride(primals_4, (64,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 64, 128, 128), (1048576, 16384, 128,
            1), torch.float32)
        buf1 = empty_strided_cuda((128, 64, 128, 128), (1048576, 16384, 128,
            1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_0[grid(128, 64, 128, 128, 128, 128, 128)](primals_1
            .contiguous(), primals_2.contiguous(), buf0, 128, 64, 0, 0, 128,
            num_warps=4, num_stages=1)
        del primals_1
        del primals_2
        buf2 = buf0
        del buf0
        buf3 = buf2
        del buf2
        buf4 = buf3
        del buf3
        buf5 = buf4
        del buf4
        buf6 = buf5
        del buf5
        buf7 = buf6
        del buf6
        triton_poi_fused_hard_swish_1[grid(128 * 128 * 64)](buf7, primals_3,
            primals_4, primals_3, primals_4, 128 * 128 * 64, 0, num_warps=4,
            num_stages=1)
        buf8 = buf7
        del buf7
        buf9 = empty_strided_cuda((128, 64, 128, 128), (1048576, 16384, 128,
            1), torch.float32)
        buf10 = empty_strided_cuda((128, 64, 128, 128), (1048576, 16384, 128,
            1), torch.float32)
        buf11 = reinterpret_tensor(buf9, (128, 64, 128, 128), (1048576,
            16384, 128, 1), 0)
        triton_poi_fused_add_0[grid(128, 64, 128, 128, 128, 128, 128)](buf8,
            buf11, buf9, 128, 64, 0, 0, 128, num_warps=4, num_stages=1)
        del buf8
        buf12 = buf9
        del buf9
        buf13 = buf12
        del buf12
        buf14 = buf13
        del buf13
        buf15 = buf14
        del buf14
        buf16 = buf15
        del buf15
        buf17 = buf16
        del buf16
        triton_poi_fused_logsumexp_2[grid(128 * 128 * 64)](buf17,
            reinterpret_tensor(buf10, (128, 64, 128, 128), (1048576,
            16384, 128, 1), 0), 128 * 128 * 64, XBLOCK=256, num_warps=4,
            num_stages=1)
        del buf17
        del primals_3
        del primals_4
    return reinterpret_tensor(buf10, (128, 1, 128, 128), (16384, 16384, 1,
        16384), 0), reinterpret_tensor(buf11, (128, 64, 128, 128), (
        1048576, 16384, 128, 1), 0), reinterpret_tensor(buf12, (128, 64,
        128, 128), (1048576, 16384, 128, 1), 0), reinterpret_tensor(buf13,
        (128, 64, 128, 128), (1048576, 16384, 128, 1), 0), reinterpret_tensor(
        buf14, (128, 64, 128, 128), (1048576, 16384, 128, 1), 0), buf15, buf16,
        buf17


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(groups, out_channels, eps=eps)
        self.tanh = nn.Tanh()
        self.hard_swish = nn.Hardswish()

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_2 = self.conv.bias
        primals_3 = self.group_norm.weight
        primals_4 = self.group_norm.bias
        primals_5 = self.tanh
        primals_6 = self.hard_swish
        output = call([primals_1, primals_2, primals_3, primals_4,
            input_0])
        return output[0]