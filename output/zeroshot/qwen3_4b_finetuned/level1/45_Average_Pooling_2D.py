import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_avg_pool2d_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 2048
    x1 = xindex // 2048
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 2048 * x1), xmask)
    tmp1 = tl.load(in_ptr0 + (1024 + x0 + 2048 * x1), xmask)
    tmp3 = tl.load(in_ptr0 + (2048 + x0 + 2048 * x1), xmask)
    tmp5 = tl.load(in_ptr0 + (3072 + x0 + 2048 * x1), xmask)
    tmp7 = tl.load(in_ptr0 + (4096 + x0 + 2048 * x1), xmask)
    tmp9 = tl.load(in_ptr0 + (5120 + x0 + 2048 * x1), xmask)
    tmp11 = tl.load(in_ptr0 + (6144 + x0 + 2048 * x1), xmask)
    tmp13 = tl.load(in_ptr0 + (7168 + x0 + 2048 * x1), xmask)
    tmp15 = tl.load(in_ptr0 + (8192 + x0 + 2048 * x1), xmask)
    tmp17 = tl.load(in_ptr0 + (9216 + x0 + 2048 * x1), xmask)
    tmp19 = tl.load(in_ptr0 + (10240 + x0 + 2048 * x1), xmask)
    tmp21 = tl.load(in_ptr0 + (11264 + x0 + 2048 * x1), xmask)
    tmp23 = tl.load(in_ptr0 + (12288 + x0 + 2048 * x1), xmask)
    tmp25 = tl.load(in_ptr0 + (13312 + x0 + 2048 * x1), xmask)
    tmp27 = tl.load(in_ptr0 + (14336 + x0 + 2048 * x1), xmask)
    tmp29 = tl.load(in_ptr0 + (15360 + x0 + 2048 * x1), xmask)
    tmp31 = tl.load(in_ptr0 + (16384 + x0 + 2048 * x1), xmask)
    tmp33 = tl.load(in_ptr0 + (17408 + x0 + 2048 * x1), xmask)
    tmp35 = tl.load(in_ptr0 + (18432 + x0 + 2048 * x1), xmask)
    tmp37 = tl.load(in_ptr0 + (19456 + x0 + 2048 * x1), xmask)
    tmp40 = tmp0 + tmp1
    tmp41 = 2.0
    tmp42 = tmp40 / tmp41
    tmp43 = tmp2 + tmp3
    tmp44 = tmp43 / tmp41
    tmp45 = tmp42 + tmp44
    tmp46 = tmp45 / tmp41
    tmp47 = tmp5 + tmp7
    tmp48 = tmp47 / tmp41
    tmp49 = tmp46 + tmp48
    tmp50 = tmp49 / tmp41
    tmp51 = tmp9 + tmp11
    tmp52 = tmp51 / tmp41
    tmp53 = tmp50 + tmp52
    tmp54 = tmp53 / tmp41
    tmp55 = tmp13 + tmp15
    tmp56 = tmp55 / tmp41
    tmp57 = tmp54 + tmp56
    tmp58 = tmp57 / tmp41
    tmp59 = tmp17 + tmp19
    tmp60 = tmp59 / tmp41
    tmp61 = tmp58 + tmp60
    tmp62 = tmp61 / tmp41
    tmp63 = tmp21 + tmp23
    tmp64 = tmp63 / tmp41
    tmp65 = tmp62 + tmp64
    tmp66 = tmp65 / tmp41
    tmp67 = tmp25 + tmp27
    tmp68 = tmp67 / tmp41
    tmp69 = tmp66 + tmp68
    tmp70 = tmp69 / tmp41
    tmp71 = tmp29 + tmp31
    tmp72 = tmp71 / tmp41
    tmp73 = tmp70 + tmp72
    tmp74 = tmp73 / tmp41
    tmp75 = tmp33 + tmp35
    tmp76 = tmp75 / tmp41
    tmp77 = tmp74 + tmp76
    tmp78 = tmp77 / tmp41
    tmp79 = tmp37 + tmp39
    tmp80 = tmp79 / tmp41
    tmp81 = tmp78 + tmp80
    tmp82 = tmp81 / tmp41
    tl.store(out_ptr0 + x2, tmp82, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (16, 64, 2048, 2048), (268435456, 4194304, 
        2048, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 64, 2048, 2048), (268435456, 4194304,
            1, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_avg_pool2d_0[grid(4194304)](arg0_1, buf0, 4194304,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del arg0_1
    return buf0,


class ModelNew(nn.Module):
    """
    Simple model that performs 2D Average Pooling.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling operation. Defaults to None (same as kernel_size).
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
        """
        super(ModelNew, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride,
            padding=padding)

    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]
