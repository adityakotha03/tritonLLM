import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_clone_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 40960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1024
    x1 = xindex // 1024
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask)
    tl.store(out_ptr0 + x2, tmp0, xmask)


@triton.jit
def triton_poi_fused_clone_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 73728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 384
    x1 = xindex // 384
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 384 * x1), xmask)
    tl.store(out_ptr0 + x2, tmp0, xmask)


@triton.jit
def triton_poi_fused_clone_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tl.store(out_ptr0 + x0, tmp0, xmask)


@triton.jit
def triton_poi_fused_clone_3(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tl.store(out_ptr0 + x0, tmp0, xmask)


@triton.jit
def triton_poi_fused_clone_4(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1536
    x1 = xindex // 1536
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1536 * x1), xmask)
    tl.store(out_ptr0 + x2, tmp0, xmask)


@triton.jit
def triton_poi_fused_clone_5(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 3072
    x1 = xindex // 3072
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 3072 * x1), xmask)
    tl.store(out_ptr0 + x2, tmp0, xmask)


@triton.jit
def triton_poi_fused_clone_6(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1024
    x1 = xindex // 1024
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask)
    tl.store(out_ptr0 + x2, tmp0, xmask)


def call(args):
    (primals_1, primals_2, primals_3, primals_4, primals_5, primals_6,
        primals_7, primals_8, primals_9, primals_10, primals_11, primals_12,
        primals_13, primals_14, primals_15) = args
    args.clear()
    assert_size_stride(primals_1, (512, 10, 128), (1280, 128, 1))
    assert_size_stride(primals_2, (10, 128), (128, 1))
    assert_size_stride(primals_3, (10, 128, 256), (32768, 256, 1))
    assert_size_stride(primals_4, (6, 10, 256), (2560, 256, 1))
    assert_size_stride(primals_5, (6, 256), (256, 1))
    assert_size_stride(primals_6, (6, 256, 256), (65536, 256, 1))
    assert_size_stride(primals_7, (10, 10, 256), (2560, 256, 1))
    assert_size_stride(primals_8, (10, 256), (256, 1))
    assert_size_stride(primals_9, (10, 256, 256), (65536, 256, 1))
    assert_size_stride(primals_10, (10, 10, 256), (2560, 256, 1))
    assert_size_stride(primals_11, (10, 256), (256, 1))
    assert_size_stride(primals_12, (10, 256, 256), (65536, 256, 1))
    assert_size_stride(primals_13, (10, 10, 256), (2560, 256, 1))
    assert_size_stride(primals_14, (10, 256), (256, 1))
    assert_size_stride(primals_15, (10, 256, 256), (65536, 256, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((10, 128, 10), (1280, 10, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_clone_0[grid(40960)](primals_1, buf0, 40960, XBLOCK=
            256, num_warps=4, num_stages=1)
        buf1 = empty_strided_cuda((10, 256), (256, 1), torch.float32)
        extern_kernels.mm(reinterpret_tensor(buf0, (10, 10), (10, 1), 0),
            reinterpret_tensor(primals_2, (10, 128), (1, 128), 0), out=buf1)
        del primals_2
        buf2 = empty_strided_cuda((10, 256, 256), (65536, 256, 1), torch.float32)
        triton_poi_fused_clone_1[grid(73728)](primals_3, buf2, 73728, XBLOCK
            =512, num_warps=8, num_stages=1)
        buf3 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.bmm(reinterpret_tensor(buf1, (1024, 10), (10, 1), 0
            ), reinterpret_tensor(buf2, (10, 256, 256), (262144, 256, 1), 0),
            out=buf3)
        buf4 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.bmm(reinterpret_tensor(buf1, (1024, 10), (10, 1), 0
            ), reinterpret_tensor(buf2, (10, 256, 256), (262144, 256, 1), 0),
            out=buf4)
        buf5 = empty_strided_cuda((10, 256), (256, 1), torch.float32)
        extern_kernels.mm(reinterpret_tensor(buf3, (10, 1024), (1024, 1), 0),
            reinterpret_tensor(primals_5, (10, 256), (1, 256), 0), out=buf5)
        buf6 = reinterpret_tensor(buf3, (1024, 10), (10, 1), 0)
        del buf3
        buf7 = empty_strided_cuda((10, 256, 256), (65536, 256, 1), torch.float32
            )
        extern_kernels.bmm(reinterpret_tensor(buf5, (1024, 10), (10, 1), 0),
            reinterpret_tensor(buf6, (10, 256, 256), (262144, 256, 1), 0), out=
            buf7)
        buf8 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.bmm(reinterpret_tensor(buf4, (1024, 10), (10, 1), 0),
            reinterpret_tensor(buf6, (10, 256, 256), (262144, 256, 1), 0), out=
            buf8)
        buf9 = reinterpret_tensor(buf4, (1024, 10), (10, 1), 0)
        del buf4
        buf10 = empty_strided_cuda((10, 256), (256, 1), torch.float32)
        extern_kernels.mm(reinterpret_tensor(buf8, (1024, 10), (10, 1), 0),
            reinterpret_tensor(primals_8, (10, 256), (1, 256), 0), out=buf10)
        buf11 = empty_strided_cuda((10, 256, 256), (65536, 256, 1), torch.float32
            )
        extern_kernels.bmm(reinterpret_tensor(buf10, (1024, 10), (10, 1), 0
            ), reinterpret_tensor(buf9, (10, 256, 256), (262144, 256, 1), 0),
            out=buf11)
        buf12 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.bmm(reinterpret_tensor(buf7, (1024, 10), (10, 1), 0),
            reinterpret_tensor(buf9, (10, 256, 256), (262144, 256, 1), 0), out=
            buf12)
        buf13 = reinterpret_tensor(buf7, (1024, 10), (10, 1), 0)
        del buf7
        buf14 = empty_strided_cuda((10, 256), (256, 1), torch.float32)
        extern_kernels.mm(reinterpret_tensor(buf12, (1024, 10), (10, 1), 0),
            reinterpret_tensor(primals_11, (10, 256), (1, 256), 0), out=buf14)
        buf15 = empty_strided_cuda((10, 256, 256), (65536, 256, 1), torch.float32
            )
        extern_kernels.bmm(reinterpret_tensor(buf14, (1024, 10), (10, 1), 0
            ), reinterpret_tensor(buf13, (10, 256, 256), (262144, 256, 1), 0),
            out=buf15)
        buf16 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.bmm(reinterpret_tensor(buf11, (1024, 10), (10, 1), 0
            ), reinterpret_tensor(buf13, (10, 256, 256), (262144, 256, 1), 0),
            out=buf16)
        buf17 = reinterpret_tensor(buf11, (1024, 10), (10, 1), 0)
        del buf11
        buf18 = empty_strided_cuda((10, 256), (256, 1), torch.float32)
        extern_kernels.mm(reinterpret_tensor(buf16, (1024, 10), (10, 1), 0),
            reinterpret_tensor(primals_14, (10, 256), (1, 256), 0), out=buf18)
        buf19 = empty_strided_cuda((10, 256, 256), (65536, 256, 1), torch.float32
            )
        extern_kernels.bmm(reinterpret_tensor(buf18, (1024, 10), (10, 1), 0
            ), reinterpret_tensor(buf17, (10, 256, 256), (262144, 256, 1), 0),
            out=buf19)
        buf20 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.bmm(reinterpret_tensor(buf15, (1024, 10), (10, 1), 0
            ), reinterpret_tensor(buf17, (10, 256, 256), (262144, 256, 1), 0),
            out=buf20)
        buf21 = reinterpret_tensor(buf15, (1024, 10), (10, 1), 0)
        del buf15
        buf22 = empty_strided_cuda((10, 256), (256, 1), torch.float32)
        extern_kernels.mm(reinterpret_tensor(buf20, (1024, 10), (10, 1), 0),
            reinterpret_tensor(primals_17, (10, 256), (1, 256), 0), out=buf22)
        buf23 = empty_strided_cuda((10, 256, 256), (65536, 256, 1), torch.float32
            )
        extern_kernels.bmm(reinterpret_tensor(buf22, (1024, 10), (10, 1), 0
            ), reinterpret_tensor(buf21, (10, 256, 256), (262144, 256, 1), 0),
            out=buf23)
        buf24 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.bmm(reinterpret_tensor(buf19, (1024, 10), (10, 1), 0
            ), reinterpret_tensor(buf21, (10, 256, 256), (262144, 256, 1), 0),
            out=buf24)
        buf25 = reinterpret_tensor(buf19, (1024, 10), (10, 1), 0)
        del buf19
        buf26 = empty_strided_cuda((10, 256), (256, 1), torch.float32)
        extern_kernels.mm(reinterpret_tensor(buf24, (1024, 10), (10, 1), 0),
            reinterpret_tensor(primals_18, (10, 256), (1, 256), 0), out=buf26)
        buf27 = empty_strided_cuda((10, 256, 256), (65536, 256, 1), torch.float32
            )
        extern_kernels.bmm(reinterpret_tensor(buf26, (1024, 10), (10, 1), 0
            ), reinterpret_tensor(buf25, (10, 256, 256), (262144, 256, 1), 0),
            out=buf27)
        buf28 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.bmm(reinterpret_tensor(buf23, (1024, 10), (10, 1), 0
            ), reinterpret_tensor(buf25, (10, 256, 256), (262144, 256, 1), 0),
            out=buf28)
        buf29 = reinterpret_tensor(buf23, (1024, 10), (10, 1), 0)
        del buf23
        buf30 = empty_strided_cuda((10, 256), (256, 1), torch.float32)
        extern_kernels.mm(reinterpret_tensor(buf28, (1024, 10), (10, 1), 0),
            reinterpret_tensor(primals_20, (10, 256), (1, 256), 0), out=buf30)
        buf31 = empty_strided_cuda((10, 256, 256), (65536, 256, 1), torch.float32
            )
        extern_kernels.bmm(reinterpret_tensor(buf30, (1024, 10), (10, 1), 0
            ), reinterpret_tensor(buf29, (10, 256, 256), (262144, 256, 1), 0),
            out=buf31)
        buf32 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.bmm(reinterpret_tensor(buf27, (1024, 10), (10, 1), 0
            ), reinterpret_tensor(buf29, (10, 256, 256), (262144, 256, 1), 0),
            out=buf32)
        buf33 = reinterpret_tensor(buf27, (1024, 10), (10, 1), 0)
        del buf27
        buf34 = empty_strided_cuda((10, 256), (256, 1), torch.float32)
        extern_kernels.mm(reinterpret_tensor(buf32, (1024, 10), (10, 1), 0),
            reinterpret_tensor(primals_21, (10, 256), (1, 256), 0), out=buf34)
        buf35 = empty_strided_cuda((10, 256, 256), (65536, 256, 1), torch.float32
            )
        extern_kernels.bmm(reinterpret_tensor(buf34, (1024, 10), (10, 1), 0
            ), reinterpret_tensor(buf33, (10, 256, 256), (262144, 256, 1), 0),
            out=buf35)
        buf36 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.bmm(reinterpret_tensor(buf31, (1024, 10), (10, 1), 0
            ), reinterpret_tensor(buf33, (10, 256, 256), (262144, 256, 1), 0),
            out=buf36)
        buf37 = reinterpret_tensor(buf31, (1024, 10), (10, 1), 0)
        del buf31
        buf38 = empty_strided_cuda((10, 256), (256, 1), torch.float32)
        extern_kernels.mm(reinterpret_tensor(buf36, (1024, 10), (10, 1), 0),
            reinterpret_tensor(primals_22, (10, 256), (1, 256), 0), out=buf38)
        buf39 = empty_strided_cuda((10, 256, 256), (65536, 256, 1), torch.float32
            )
        extern_kernels.bmm(reinterpret_tensor(buf38, (1024, 10), (10, 1), 0
            ), reinterpret_tensor(buf37, (10, 256, 256), (262144, 256, 1), 0),
            out=buf39)
        buf40 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.bmm(reinterpret_tensor(buf35, (1024, 10), (10, 1), 0
            ), reinterpret_tensor(buf37, (10, 256, 256), (262144, 256, 1), 0),
            out=buf40)
        buf41 = reinterpret_tensor(buf35, (1024, 10), (10, 1), 0)
        del buf35
        buf42 = empty_strided_cuda((10, 256), (256, 1), torch.float32)
        extern_kernels.mm(reinterpret_tensor(buf40, (1024, 10), (10, 1), 0),
            reinterpret_tensor(primals_23, (10, 256), (1, 256), 0), out=buf42)
        buf43 = empty_strided_cuda((10, 256, 256), (65536, 256, 1), torch.float32
            )
        extern_kernels.bmm(reinterpret_tensor(buf42, (1024, 10), (10, 1), 0
            ), reinterpret_tensor(buf41, (10, 256, 256), (262144, 256, 1), 0),
            out=buf43)
        buf44 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.bmm(reinterpret_tensor(buf39, (1024, 10), (10, 1), 0
            ), reinterpret_tensor(buf41, (10, 256, 256), (262144, 256, 1), 0),
            out=buf44)
        buf45 = reinterpret_tensor(buf39, (1024, 10), (10, 1), 0)
        del buf39
        buf46 = empty_strided_cuda((10, 256), (256, 1), torch.float32)
        extern_kernels.mm(reinterpret_tensor(buf44, (1024, 10), (10, 1), 0),
            reinterpret_tensor(primals_24, (10, 256), (1, 256), 0), out=buf46)
        buf47 = empty_strided_cuda((10, 256, 256), (65536, 256, 1), torch.float32
            )
        extern_kernels.bmm(reinterpret_tensor(buf46, (1024, 10), (10, 1), 0
            ), reinterpret_tensor(buf45, (10, 256, 256), (262144, 256, 1), 0),
            out=buf47)
        buf48 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.bmm(reinterpret_tensor(buf43, (1024, 10), (10, 1), 0
            ), reinterpret_tensor(buf45, (10, 256, 256), (262144, 256, 1), 0),
            out=buf48)
        buf49 = reinterpret_tensor(buf43, (1024, 10), (10, 1), 0)
        del buf43
        buf50 = empty_strided_cuda((10, 256), (256, 1), torch.float32)
        extern_kernels.mm(reinterpret_tensor(buf48, (1024, 10), (10, 1), 0),
            reinterpret_tensor(primals_25, (10, 256), (1, 256), 0), out=buf50)
        buf51 = empty_strided_cuda((10, 256, 256), (65536, 256, 1), torch.float32
            )
        extern_kernels.bmm(reinterpret_tensor(buf50, (1024, 10), (10, 1), 0
            ), reinterpret_tensor(buf49, (10, 256, 256), (262144, 256, 1), 0),
            out=buf51)
        buf52 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.bmm(reinterpret_tensor(buf47, (1024, 10), (10, 1), 0
            ), reinterpret_tensor(buf49, (10, 256, 256), (262144, 256, 1), 0),
            out=buf52)
        buf53 = reinterpret_tensor(buf47, (1024, 10), (10, 1), 0)
        del buf47
        buf54 = empty_strided_cuda((10, 256), (256, 1), torch.float32)
        extern_kernels.mm(reinterpret_tensor(buf52, (1024, 10), (10, 1), 0),
            reinterpret_tensor(primals_26, (10, 256), (1, 256), 0), out=buf54)
        buf55 = empty_strided_cuda((10, 256, 256), (65536, 256, 1), torch.float32
            )
        extern_kernels.bmm(reinterpret_tensor(buf54, (1024, 10), (10, 1), 0
            ), reinterpret_tensor(buf53, (10, 256, 256), (262144, 256, 1), 0),
            out=buf55)
        buf56 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.bmm(reinterpret_tensor(buf51, (1024, 10), (10, 1), 0
            ), reinterpret_tensor(buf53, (10, 256, 256), (262144, 256, 1), 0),
            out=buf56)
        buf57 = reinterpret_tensor(buf51, (1024, 10), (10, 1), 0)
        del buf51
        buf58 = empty_strided_cuda((10, 256), (256, 1), torch.float32)
        extern_kernels.mm(reinterpret_tensor(buf56, (1024, 10), (10, 1), 0),
            reinterpret_tensor(primals_27, (10, 256), (1, 256), 0), out=buf58)
        buf59 = empty_strided_cuda((10, 256, 256), (65536, 256, 1), torch.float32
            )
        extern_kernels.bmm(reinterpret_tensor(buf58, (1024, 10), (10, 1), 0
            ), reinterpret_tensor(buf57, (10, 256, 256), (262144, 256, 1), 0),
            out=buf59)
        buf60 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.bmm(reinterpret_tensor(buf55, (1024, 10), (10, 1), 0
            ), reinterpret_tensor(buf57, (10, 256, 256), (262144, 256, 1), 0),
            out=buf60)
        buf61 = reinterpret_tensor(buf55, (1024, 10), (10, 1), 0)
        del buf55
        buf62 = empty_strided_cuda((10, 256), (256, 1), torch.float32)
        extern_kernels.mm(reinterpret_tensor(buf60, (1024, 10), (10, 1), 0),
            reinterpret_tensor(primals_28, (10, 256), (1, 256), 0), out=buf62)
        buf63 = empty_strided_cuda((10, 256, 256), (65536, 256, 1), torch.float32
            )
        extern_kernels.bmm(reinterpret_tensor(buf62, (1024, 10), (10, 1), 0
            ), reinterpret_tensor(buf61, (10, 256, 256), (262144, 256, 1), 0),
            out=buf63)
        buf64 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.bmm(reinterpret_tensor(buf59, (1024, 10), (10, 1), 0
            ), reinterpret_tensor(buf61, (10, 256, 256), (262144, 256, 1), 0),
            out=buf64)
        buf65 = reinterpret_tensor(buf59, (1024, 10), (10, 1), 0)
        del buf59
        buf66 = empty_strided_cuda((10, 256), (256, 1), torch.float32)
        extern_kernels.mm(reinterpret_tensor(buf64, (1024, 10), (10, 1), 0),
            reinterpret_tensor(primals_29, (10, 256), (1, 256), 0), out=buf66)
        buf67 = empty_strided_cuda((10, 256, 256), (65536, 256, 1), torch.float32
            )
        extern_kernels.bmm(reinterpret_tensor(buf66, (1024, 10), (10, 1), 0
            ), reinterpret_tensor(buf65, (10, 256, 256), (262144, 256, 1), 0),
            out=buf67)
        buf68