import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_mul_sigmoid_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp3 = 2.0
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + x0, tmp4, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (32768, 32768), (32768, 1))
    assert_size_stride(primals_2, (32768,), (1,))
    assert_size_stride(primals_3, (128, 32768), (32768, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](primals_3, buf0, 
            4194304, XBLOCK=512, num_warps=8, num_stages=1)
        del primals_3
        buf1 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf0, buf1, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        buf2 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf1, buf2, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        buf3 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf2, buf3, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        buf4 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf3, buf4, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf3
        buf5 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf4, buf5, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf4
        buf6 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf5, buf6, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf5
        buf7 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf6, buf7, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf6
        buf8 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf7, buf8, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf7
        buf9 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf8, buf9, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf8
        buf10 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf9, buf10, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf9
        buf11 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf10, buf11, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf10
        buf12 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf11, buf12, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf11
        buf13 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf12, buf13, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf12
        buf14 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf13, buf14, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf13
        buf15 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf14, buf15, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf14
        buf16 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf15, buf16, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf15
        buf17 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf16, buf17, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf16
        buf18 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf17, buf18, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf17
        buf19 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf18, buf19, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf18
        buf20 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf19, buf20, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf19
        buf21 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf20, buf21, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf20
        buf22 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf21, buf22, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf21
        buf23 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf22, buf23, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf22
        buf24 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf23, buf24, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf23
        buf25 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf24, buf25, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf24
        buf26 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf25, buf26, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf25
        buf27 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf26, buf27, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf26
        buf28 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf27, buf28, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf27
        buf29 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf28, buf29, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf28
        buf30 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf29, buf30, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf29
        buf31 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf30, buf31, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf30
        buf32 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf31, buf32, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf31
        buf33 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf32, buf33, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf32
        buf34 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf33, buf34, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf33
        buf35 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf34, buf35, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf34
        buf36 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf35, buf36, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf35
        buf37 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf36, buf37, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf36
        buf38 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf37, buf38, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf37
        buf39 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf38, buf39, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf38
        buf40 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf39, buf40, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf39
        buf41 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf40, buf41, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf40
        buf42 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf41, buf42, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf41
        buf43 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf42, buf43, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf42
        buf44 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf43, buf44, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf43
        buf45 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf44, buf45, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf44
        buf46 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf45, buf46, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf45
        buf47 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf46, buf47, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf46
        buf48 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf47, buf48, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf47
        buf49 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf48, buf49, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf48
        buf50 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf49, buf50, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf49
        buf51 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf50, buf51, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf50
        buf52 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf51, buf52, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf51
        buf53 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf52, buf53, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf52
        buf54 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf53, buf54, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf53
        buf55 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf54, buf55, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf54
        buf56 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf55, buf56, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf55
        buf57 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf56, buf57, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf56
        buf58 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf57, buf58, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf57
        buf59 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf58, buf59, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf58
        buf60 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf59, buf60, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf59
        buf61 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf60, buf61, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf60
        buf62 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf61, buf62, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf61
        buf63 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf62, buf63, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf62
        buf64 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf63, buf64, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf63
        buf65 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf64, buf65, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf64
        buf66 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf65, buf66, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf65
        buf67 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf66, buf67, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf66
        buf68 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf67, buf68, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf67
        buf69 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf68, buf69, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf68
        buf70 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf69, buf70, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf69
        buf71 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf70, buf71, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf70
        buf72 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf71, buf72, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf71
        buf73 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf72, buf73, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf72
        buf74 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf73, buf74, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf73
        buf75 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf74, buf75, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf