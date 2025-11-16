import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_per_fused_clone_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl
    .constexpr, XBLOCK: tl.constexpr):
    ynumel = 65536
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 64
    y1 = yindex // 64
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 64 * x2 + 262144 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 4096 * y3), tmp0, xmask & ymask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (64, 64, 512, 512), (16384, 256, 512, 1))
    assert_size_stride(primals_2, (64, 64), (64, 1))
    assert_size_stride(primals_3, (64,), (1,))
    assert_size_stride(primals_4, (64,), (1,))
    assert_size_stride(primals_5, (64, 64, 512, 512), (16384, 256, 512, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((64, 64, 512, 512), (16384, 256, 512, 1),
            torch.float32)
        get_raw_stream(0)
        triton_per_fused_clone_0[grid(65536, 4096)](primals_1, buf0, 65536,
            4096, XBLOCK=16, YBLOCK=32, num_warps=4, num_stages=1)
        del primals_1
        buf1 = extern_kernels.mm(reinterpret_tensor(buf0, (64, 64), (64, 1),
            0), reinterpret_tensor(primals_2, (64, 64), (1, 64), 0))
        del primals_2
        buf2 = empty_strided_cuda((64, 64, 512, 512), (16384, 1, 64, 1), 
            torch.float32)
        extern_kernels.mm(reinterpret_tensor(buf0, (64, 64), (64, 1), 0),
            reinterpret_tensor(primals_3, (64, 64), (1, 64), 0), out=buf2)
        del primals_3
        buf4 = empty_strided_cuda((64, 64, 512, 512), (16384, 256, 512, 1),
            torch.float32)
        triton_per_fused_clone_0[grid(65536, 4096)](primals_5, buf4, 65536,
            4096, XBLOCK=16, YBLOCK=32, num_warps=4, num_stages=1)
        del primals_5
        buf3 = extern_kernels.mm(reinterpret_tensor(buf4, (64, 64), (64, 1),
            0), reinterpret_tensor(primals_4, (64, 64), (1, 64), 0))
        del primals_4
        buf5 = empty_strided_cuda((64, 64, 512, 512), (16384, 256, 512, 1),
            torch.float32)
        extern_kernels.bmm(reinterpret_tensor(buf1, (64, 64, 512), (1, 512,
            1), 0), reinterpret_tensor(primals_5, (64, 512, 512), (16384, 512
            , 1), 0), out=buf5)
        buf6 = empty_strided_cuda((64, 64, 512, 512), (16384, 1, 1, 1), 
            torch.float32)
        extern_kernels.bmm(reinterpret_tensor(buf3, (64, 64, 512), (1, 512,
            1), 0), reinterpret_tensor(buf2, (64, 512, 512), (1, 512, 1), 0
            ), out=buf6)
        buf7 = empty_strided_cuda((64, 64, 512, 512), (16384, 256, 512, 1),
            torch.float32)
        triton_per_fused_clone_0[grid(65536, 4096)](primals_5, buf7, 65536,
            4096, XBLOCK=16, YBLOCK=32, num_warps=4, num_stages=1)
        del primals_5
        buf8 = empty_strided_cuda((64, 64, 512, 512), (16384, 256, 512, 1),
            torch.float32)
        extern_kernels.bmm(reinterpret_tensor(buf5, (64, 64, 512), (1, 512,
            1), 0), reinterpret_tensor(buf7, (64, 512, 512), (16384, 512, 1),
            0), out=buf8)
    return reinterpret_tensor(buf8, (64, 64, 512, 512), (16384, 1, 64, 1), 0
        ), reinterpret_tensor(buf1, (64, 64, 512), (1, 512, 1), 0
        ), reinterpret_tensor(buf3, (64, 64, 512), (1, 512, 1), 0
        ), reinterpret_tensor(buf4, (64, 64), (64, 1), 0
        ), buf0, buf2, reinterpret_tensor(buf7, (64, 512, 512), (16384, 1, 
        1), 0), buf6, buf5


class ModelNew(nn.Module):
    """
    Simple model that performs Batch Normalization.
    """
    def __init__(self, num_features: int):
        """
        Initializes the BatchNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(ModelNew, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=num_features)

    def forward(self, input_0):
        primals_2 = self.bn.weight
        primals_3 = self.bn.bias
        primals_1 = input_0
        primals_4 = self.bn.weight_1
        primals_5 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4, primals_5])
        return output[0]
