import torch
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
def triton_poi_fused_relu_threshold_backward_0(in_out_ptr0, in_ptr0,
    out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 16384
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(in_out_ptr0 + x2, tmp4, xmask)
    tl.store(out_ptr0 + x2, tmp6, xmask)


@triton.jit
def triton_poi_fused_relu_threshold_backward_1(in_out_ptr0, in_ptr0,
    out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 8192
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(in_out_ptr0 + x2, tmp4, xmask)
    tl.store(out_ptr0 + x2, tmp6, xmask)


def call(args):
    (primals_1, primals_2, primals_3, primals_4, primals_5, primals_6,
        primals_7) = args
    args.clear()
    assert_size_stride(primals_1, (16384, 16384), (16384, 1))
    assert_size_stride(primals_2, (16384,), (1,))
    assert_size_stride(primals_3, (128, 16384), (16384, 1))
    assert_size_stride(primals_4, (16384,), (1,))
    assert_size_stride(primals_5, (16384, 8192), (8192, 1))
    assert_size_stride(primals_6, (8192,), (1,))
    assert_size_stride(primals_7, (8192, 8192), (8192, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf0,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf1 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf1,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf2 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf2,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf3 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf3,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf4 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf4,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf5 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf5,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf6 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf6,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf7 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf7,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf8 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf8,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf9 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf9,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf10 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf10,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf11 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf11,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf12 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf12,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf13 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf13,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf14 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf14,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf15 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf15,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf16 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf16,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf17 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf17,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf18 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf18,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf19 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf19,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf20 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf20,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf21 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf21,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf22 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf22,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf23 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf23,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf24 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf24,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf25 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf25,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf26 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf26,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf27 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf27,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf28 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf28,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf29 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf29,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf30 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf30,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf31 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf31,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf32 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf32,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf33 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf33,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf34 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf34,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf35 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf35,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf36 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf36,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf37 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf37,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf38 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf38,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf39 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf39,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf40 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf40,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf41 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf41,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf42 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf42,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf43 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf43,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf44 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf44,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf45 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf45,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf46 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf46,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf47 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf47,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf48 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf48,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf49 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf49,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf50 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf50,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf51 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf51,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf52 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf52,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf53 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf53,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf54 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf54,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf55 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf55,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf56 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf56,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf57 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf57,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf58 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf58,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf59 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf59,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf60 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf60,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf61 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf61,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf62 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf62,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf63 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf63,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf64 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(2097152)](buf64,
            primals_3, primals_2, 2097152, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        buf65 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(20