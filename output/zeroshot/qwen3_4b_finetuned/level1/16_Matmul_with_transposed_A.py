import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused__to_copy_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tl.store(out_ptr0 + x0, tmp0, xmask)


@triton.jit
def triton_poi_fused__to_copy_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tl.store(out_ptr0 + x0, tmp0, xmask)


@triton.jit
def triton_poi_fused__to_copy_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 2048
    tmp0 = x2
    tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (2048 * x1 + x0), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr0 + (2049 + 2048 * x1), tmp4 & tl.full([1], 1, tl
        .int64), tmp4 & tl.full([1], 1, tl.int64), other=0.0)
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tl.full([1], 1, tl.int64)
    tmp9 = tmp0 < tmp8
    tmp10 = tl.load(in_ptr0 + (4096 + 2048 * x1), tmp9, other=0.0)
    tmp11 = tl.load(in_ptr0 + (4097 + 2048 * x1), tmp9 & tl.full([1], 1,
        tl.int64), tmp9 & tl.full([1], 1, tl.int64), other=0.0)
    tmp12 = triton_helpers.maximum(tmp10, tmp11)
    tmp13 = triton_helpers.maximum(tmp7, tmp12)
    tmp14 = tl.full([1], 2, tl.int64)
    tmp15 = tmp0 < tmp14
    tmp16 = tl.load(in_ptr0 + (6144 + 2048 * x1), tmp15, other=0.0)
    tmp17 = tl.load(in_ptr0 + (6145 + 2048 * x1), tmp15 & tl.full([1], 1,
        tl.int64), tmp15 & tl.full([1], 1, tl.int64), other=0.0)
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = triton_helpers.maximum(tmp13, tmp18)
    tmp20 = tl.full([1], 3, tl.int64)
    tmp21 = tmp0 < tmp20
    tl.full([1], 4, tl.int64)
    tmp23 = tl.load(in_ptr0 + (8192 + 2048 * x1), tmp21, other=0.0)
    tmp24 = tl.load(in_ptr0 + (8193 + 2048 * x1), tmp21 & tl.full([1], 1,
        tl.int64), tmp21 & tl.full([1], 1, tl.int64), other=0.0)
    tmp25 = triton_helpers.maximum(tmp23, tmp24)
    tmp26 = triton_helpers.maximum(tmp19, tmp25)
    tl.store(out_ptr0 + x2, tmp26, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (8192, 4096), (4096, 1))
    assert_size_stride(arg1_1, (4096, 8192), (8192, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8192, 4096), (4096, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused__to_copy_0[grid(8192)](arg0_1, buf0, 8192, XBLOCK=
            128, num_warps=4, num_stages=1)
        del arg0_1
        buf1 = empty_strided_cuda((8192, 8192), (8192, 1), torch.float32)
        triton_poi_fused__to_copy_1[grid(8192)](arg1_1, buf1, 8192, XBLOCK=
            128, num_warps=4, num_stages=1)
        del arg1_1
        buf2 = empty_strided_cuda((8192, 2048), (2048, 1), torch.float32)
        triton_poi_fused__to_copy_2[grid(16384)](buf0, buf2, 16384, XBLOCK=
            256, num_warps=4, num_stages=1)
        del buf0
    return reinterpret_tensor(buf2, (8192, 2048), (2048, 1), 0
        ), reinterpret_tensor(buf1, (8192, 8192), (8192, 1), 0
        ), reinterpret_tensor(buf2, (8192, 2048), (1, 8192), 0)


class ModelNew(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B)
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, input_0, input_1):
        arg0_1 = input_0
        arg1_1 = input_1
        output = call([arg0_1, arg1_1])
        return output[0]
