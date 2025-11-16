import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_clone_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    ynumel = 8192
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 8192 * y0), xmask & ymask, eviction_policy
        ='evict_last')
    tl.store(out_ptr0 + (y0 + 8192 * x1), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_relu_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 8192024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1024
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + x2, tmp4, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7 = (
        args)
    args.clear()
    assert_size_stride(primals_1, (8192,), (1,))
    assert_size_stride(primals_2, (1024, 8192), (8192, 1))
    assert_size_stride(primals_3, (1024,), (1,))
    assert_size_stride(primals_4, (1024, 1024), (1024, 1))
    assert_size_stride(primals_5, (1024,), (1,))
    assert_size_stride(primals_6, (1024, 1024), (1024, 1))
    assert_size_stride(primals_7, (8192,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        extern_kernels.mm(primals_1, reinterpret_tensor(primals_2, (8192, 
            1024), (1, 8192), 0), out=buf0)
        del primals_2
        buf1 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf0, reinterpret_tensor(primals_4, (1024, 1024),
            (1, 1024), 0), out=buf1)
        del primals_4
        buf2 = buf1
        del buf1
        get_raw_stream(0)
        triton_poi_fused_relu_1[grid(8192024)](buf2, primals_5, 8192024,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_5
        buf3 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf2, reinterpret_tensor(primals_6, (1024, 1024),
            (1, 1024), 0), out=buf3)
        del primals_6
        buf4 = buf3
        del buf3
        triton_poi_fused_relu_1[grid(8192024)](buf4, primals_3, 8192024,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_3
        buf5 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf4, reinterpret_tensor(primals_7, (1024, 1024),
            (1, 1024), 0), out=buf5)
        del primals_7
        buf6 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf2, reinterpret_tensor(primals_4, (1024, 1024),
            (1, 1024), 0), out=buf6)
        buf7 = buf6
        del buf6
        triton_poi_fused_relu_1[grid(8192024)](buf7, primals_5, 8192024,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_5
        buf8 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf7, reinterpret_tensor(primals_6, (1024, 1024),
            (1, 1024), 0), out=buf8)
        buf9 = buf8
        del buf8
        triton_poi_fused_relu_1[grid(8192024)](buf9, primals_3, 8192024,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_3
        buf10 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf9, reinterpret_tensor(primals_7, (1024, 1024),
            (1, 1024), 0), out=buf10)
        buf11 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf9, reinterpret_tensor(primals_4, (1024, 1024),
            (1, 1024), 0), out=buf11)
        buf12 = buf11
        del buf11
        triton_poi_fused_relu_1[grid(8192024)](buf12, primals_5, 8192024,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_5
        buf13 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf12, reinterpret_tensor(primals_6, (1024, 1024),
            (1, 1024), 0), out=buf13)
        buf14 = buf13
        del buf13
        triton_poi_fused_relu_1[grid(8192024)](buf14, primals_3, 8192024,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_3
        buf15 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf14, reinterpret_tensor(primals_7, (1024, 1024),
            (1, 1024), 0), out=buf15)
        buf16 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf4, reinterpret_tensor(primals_4, (1024, 1024),
            (1, 1024), 0), out=buf16)
        buf17 = buf16
        del buf16
        triton_poi_fused_relu_1[grid(8192024)](buf17, primals_5, 8192024,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_5
        buf18 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf17, reinterpret_tensor(primals_6, (1024, 1024),
            (1, 1024), 0), out=buf18)
        buf19 = buf18
        del buf18
        triton_poi_fused_relu_1[grid(8192024)](buf19, primals_3, 8192024,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_3
        buf20 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf19, reinterpret_tensor(primals_7, (1024, 1024),
            (1, 1024), 0), out=buf20)
        buf21 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf14, reinterpret_tensor(primals_4, (1024, 1024),
            (1, 1024), 0), out=buf21)
        buf22 = buf21
        del buf21
        triton_poi_fused_relu_1[grid(8192024)](buf22, primals_5, 8192024,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_5
        buf23 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf22, reinterpret_tensor(primals_6, (1024, 1024),
            (1, 1024), 0), out=buf23)
        buf24 = buf23
        del buf23
        triton_poi_fused_relu_1[grid(8192024)](buf24, primals_3, 8192024,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_3
        buf25 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf24, reinterpret_tensor(primals_7, (1024, 1024),
            (1, 1024), 0), out=buf25)
        buf26 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf19, reinterpret_tensor(primals_4, (1024, 1024),
            (1, 1024), 0), out=buf26)
        buf27 = buf26
        del buf26
        triton_poi_fused_relu_1[grid(8192024)](buf27, primals_5, 8192024,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_5
        buf28 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf27, reinterpret_tensor(primals_6, (1024, 1024),
            (1, 1024), 0), out=buf28)
        buf29 = buf28
        del buf28
        triton_poi_fused_relu_1[grid(8192024)](buf29, primals_3, 8192024,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_3
        buf30 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf29, reinterpret_tensor(primals_7, (1024, 1024),
            (1, 1024), 0), out=buf30)
        buf31 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf24, reinterpret_tensor(primals_4, (1024, 1024),
            (1, 1024), 0), out=buf31)
        buf32 = buf31
        del buf31
        triton_poi_fused_relu_1[grid(8192024)](buf32, primals_5, 8192024,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_5
        buf33 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf32, reinterpret_tensor(primals_6, (1024, 1024),
            (1, 1024), 0), out=buf33)
        buf34 = buf33
        del buf33
        triton_poi_fused_relu_1[grid(8192024)](buf34, primals_3, 8192024,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_3
        buf35 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf34, reinterpret_tensor(primals_7, (1024, 1024),
            (1, 1024), 0), out=buf35)
        buf36 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf29, reinterpret_tensor(primals_4, (1024, 1024),
            (1, 1024), 0), out=buf36)
        buf37 = buf36
        del buf36
        triton_poi_fused_relu_1[grid(8192024)](buf37, primals_5, 8192024,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_5
        buf38 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf37, reinterpret_tensor(primals_6, (1024, 1024),
            (1, 1024), 0), out=buf38)
        buf39 = buf38
        del buf38
        triton_poi_fused_relu_1[grid(8192024)](buf39, primals_3, 8192024,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_3
        buf40 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf39, reinterpret_tensor(primals_7, (1024, 1024),
            (1, 1024), 0), out=buf40)
        buf41 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf34, reinterpret_tensor(primals_4, (1024, 1024),
            (1, 1024), 0), out=buf41)
        buf42 = buf41
        del buf41
        triton_poi_fused_relu_1[grid(8192024)](buf42, primals_5, 8192024,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_5
        buf43 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf42, reinterpret_tensor(primals_6, (1024, 1024),
            (1, 1024), 0), out=buf43)
        buf44 = buf43
        del buf43
        triton_poi_fused_relu_1[grid(8192024)](buf44, primals_3, 8192024,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_3
        buf45 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf44, reinterpret_tensor(primals_7, (1024, 1024),
            (1, 1024), 0), out=buf45)
        buf46 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf39, reinterpret_tensor(primals_4, (1024, 1024),
            (1, 1024), 0), out=buf46)
        buf47 = buf46
        del buf46
        triton_poi_fused_relu_1[grid(8192024)](buf47, primals_5, 8192024,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_5
        buf48 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf47, reinterpret_tensor(primals_6, (1024, 1024),
            (1, 1024), 0), out=buf48)
        buf49 = buf48
        del buf48
        triton_poi_fused_relu_1[grid(8192024)](buf49, primals_3, 8192024,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_3
        buf50 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf49, reinterpret_tensor(primals_7, (1024, 1024),
            (1, 1024), 0), out=buf50)
        buf51 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf44, reinterpret_tensor(primals_4, (1024, 1024),
            (1, 1024), 0), out=buf51)
        buf52 = buf51
        del buf51
        triton_poi_fused_relu_1[grid(8192024)](buf52, primals_5, 8192024,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_5
        buf53 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf52, reinterpret_tensor(primals_6, (1024, 1024),
            (1, 1024), 0), out=buf53)
        buf54 = buf53
        del buf53
        triton_poi_fused_relu_1[grid(8192024)](buf54, primals_3, 8192024,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_3
        buf55 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf54, reinterpret_tensor(primals_7, (1024, 1024),
            (1, 1024), 0), out=buf55)
        buf56 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf49, reinterpret_tensor(primals_4, (1024, 1024),
            (1, 1024), 0), out=buf56)
        buf57 = buf56
        del buf56
        triton_poi_fused_relu_1[grid(8192024)](buf57, primals_5, 8192024,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_5
        buf58 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf57, reinterpret_tensor(primals_6, (1024, 1024),
            (1, 1024), 0), out=buf58)
        buf59 = buf58
        del buf58
        triton_poi_fused_relu_1[grid(8192024)](buf59, primals_3, 8192024,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_3
        buf60 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf59, reinterpret_tensor(primals_7, (1024, 1024),
            (1, 1024), 0), out=buf60)
        buf61 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf54, reinterpret_tensor(primals_4, (1024, 1024),
            (1, 1024), 0), out=buf61)
        buf62 = buf61
        del buf61
        triton_poi_fused_relu_1[grid(8192024)](buf62, primals_5, 8192024,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_5
        buf63 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf62, reinterpret_tensor(primals_6, (1024, 1024),
            (1, 1024), 0), out=buf63)
        buf64 = buf63
        del buf63
        triton_poi_fused_relu_1[grid(8192024)](buf64, primals_3, 8192024,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_3
        buf65 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf64, reinterpret_tensor(primals_7, (1024, 1024),
            (1, 1024), 0), out=buf65)
        buf66 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf59, reinterpret_tensor(primals_4, (1024, 1024),
            (1, 1024), 0), out=buf66)
        buf67 = buf66
        del buf66
        triton_poi_fused_relu_1[grid(8192024)](buf67, primals_5, 8192024,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_5
        buf68 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf67, reinterpret_tensor(primals_6, (1024, 1024),
            (1, 1024), 0), out=buf68)
        buf69 = buf68
        del buf68
        triton_poi_fused_relu_1[grid(8192024)](buf69, primals_3, 8192024,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_3
        buf70 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf69, reinterpret_tensor(primals_7, (1024, 1024),
            (1, 1024), 0), out=buf70)
        buf71 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf64, reinterpret_tensor(primals_4, (1024, 1024),
            (1, 1024), 0), out=buf71)
        buf72 = buf71
        del buf71
        triton_poi_fused_relu_1[grid(8192024)](buf72, primals_5, 8192024,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_5
        buf73 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf72, reinterpret_tensor(primals_6, (1024, 1024),
            (1, 1024), 0), out=buf73)
        buf74 = buf73
        del buf73
        triton_poi_fused_relu_1[grid(8192024)](buf74, primals_3, 8192024,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_3
        buf75 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf74, reinterpret_tensor(primals_7, (1024, 1024),
            (1, 1024), 0), out=buf75)
        buf76 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf69, reinterpret_tensor(primals_4, (1024, 1024),
            (1, 1024), 0), out=buf76)
        buf77 = buf76
        del buf76
        triton_poi_fused_relu_1[grid(8192024)](buf77, primals_5, 8192024,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_5
        buf78 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf77, reinterpret_tensor(primals_6, (1024, 1024),
            (1, 1024), 0), out=buf78)
        buf79 = buf78
        del buf78
        triton_poi_fused_relu_1[grid(8192024)](buf79, primals_3, 8192024,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_3
        buf80 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf79, reinterpret_tensor(primals_7, (1024, 1024),
            (1, 1024), 0), out=buf80)
        buf81 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf74, reinterpret_tensor(primals_4, (1024, 1024),
            (1, 1024), 0), out=buf81)
        buf82 = buf81
        del buf81
        triton_poi_fused_relu_1[grid(8192024)](buf82, primals_5, 8192024,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_5
        buf83 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf82, reinterpret_tensor(primals_6, (1024, 1024),
            (1, 1024), 0), out=buf83)
        buf84 = buf83
        del buf83
        triton_poi_fused_relu_1[grid(8192024)](buf84, primals_3, 8192024,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_3
        buf85 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf84, reinterpret_tensor(primals_7, (1024, 1024),
            (1, 1024), 0), out=buf85)
        buf86 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf79, reinterpret_tensor(primals_4, (1024, 1024),
            (1, 1024), 0), out=buf86)
        buf87 = buf86
        del buf86
        triton_poi_fused_relu_1[grid(8192024)](buf87, primals_5, 8192024,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_5
        buf88 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf87, reinterpret_tensor(primals_6, (1024, 1024),
            (1, 1024), 0