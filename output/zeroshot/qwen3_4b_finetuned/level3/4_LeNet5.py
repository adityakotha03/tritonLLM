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
def triton_poi_fused_convolution_relu_0(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 10368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 324 % 6
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + x3, tmp4, xmask)


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_1(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 2592
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 9
    x1 = xindex // 9 % 9
    x2 = xindex // 81
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2 * x0 + 36 * x1 + 1296 * x2), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2 * x0 + 36 * x1 + 1296 * x2), xmask,
        eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (18 + 2 * x0 + 36 * x1 + 1296 * x2), xmask,
        eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (19 + 2 * x0 + 36 * x1 + 1296 * x2), xmask,
        eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = tl.full([1], 1, tl.int8)
    tmp9 = tl.full([1], 0, tl.int8)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 > tmp2
    tmp12 = tl.full([1], 2, tl.int8)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tmp5 > tmp4
    tmp15 = tl.full([1], 3, tl.int8)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tl.store(out_ptr0 + x3, tmp6, xmask)
    tl.store(out_ptr1 + x3, tmp16, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_2(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 5184
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 81 % 16
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + x3, tmp4, xmask)


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_3(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 1296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 3
    x1 = xindex // 3 % 3
    x2 = xindex // 9
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2 * x0 + 12 * x1 + 486 * x2), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2 * x0 + 12 * x1 + 486 * x2), xmask,
        eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (6 + 2 * x0 + 12 * x1 + 486 * x2), xmask,
        eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (7 + 2 * x0 + 12 * x1 + 486 * x2), xmask,
        eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = tl.full([1], 1, tl.int8)
    tmp9 = tl.full([1], 0, tl.int8)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 > tmp2
    tmp12 = tl.full([1], 2, tl.int8)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tmp5 > tmp4
    tmp15 = tl.full([1], 3, tl.int8)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tl.store(out_ptr0 + x3, tmp6, xmask)
    tl.store(out_ptr1 + x3, tmp16, xmask)


@triton.jit
def triton_poi_fused_relu_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 120
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_relu_5(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 84
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_add_6(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 80
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 20
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, xmask)


def call(args):
    (primals_1, primals_2, primals_3, primals_4, primals_5, primals_6,
        primals_7, primals_8, primals_9, primals_10, primals_11, primals_12,
        primals_13) = args
    args.clear()
    assert_size_stride(primals_1, (6, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_2, (6,), (1,))
    assert_size_stride(primals_3, (4096, 1, 32, 32), (1024, 1024, 32, 1))
    assert_size_stride(primals_4, (16, 6, 5, 5), (150, 25, 5, 1))
    assert_size_stride(primals_5, (16,), (1,))
    assert_size_stride(primals_6, (120, 400), (400, 1))
    assert_size_stride(primals_7, (120,), (1,))
    assert_size_stride(primals_8, (84, 120), (120, 1))
    assert_size_stride(primals_9, (84,), (1,))
    assert_size_stride(primals_10, (20, 84), (84, 1))
    assert_size_stride(primals_11, (20,), (1,))
    assert_size_stride(primals_12, (20,), (1,))
    assert_size_stride(primals_13, (20,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 
            1), padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4096, 6, 32, 32), (61440, 10240, 320, 1))
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_poi_fused_convolution_relu_0[grid(10368)](buf1, primals_2, 
            10368, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_2
        buf2 = empty_strided_cuda((4096, 6, 9, 9), (4896, 81, 9, 1), torch.
            float32)
        buf3 = empty_strided_cuda((4096, 6, 9, 9), (4896, 81, 9, 1), torch.int8
            )
        triton_poi_fused_max_pool2d_with_indices_1[grid(2592)](buf1, buf2,
            buf3, 2592, XBLOCK=256, num_warps=4, num_stages=1)
        buf4 = extern_kernels.convolution(buf2, primals_4, stride=(1, 1),
            padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (4096, 16, 9, 9), (12960, 810, 90, 1))
        buf5 = buf4
        del buf4
        triton_poi_fused_convolution_relu_2[grid(5184)](buf5, primals_5, 
            5184, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_5
        buf6 = empty_strided_cuda((4096, 16, 3, 3), (1458, 90, 30, 1),
            torch.float32)
        buf7 = empty_strided_cuda((4096, 16, 3, 3), (1458, 90, 30, 1),
            torch.int8)
        triton_poi_fused_max_pool2d_with_indices_3[grid(1296)](buf5, buf6,
            buf7, 1296, XBLOCK=128, num_warps=4, num_stages=1)
        buf8 = empty_strided_cuda((4096, 120), (120, 1), torch.float32)
        extern_kernels.mm(reinterpret_tensor(buf6, (4096, 400), (400, 1), 0
            ), reinterpret_tensor(primals_6, (400, 120), (1, 400), 0), out=buf8
            )
        buf9 = buf8
        del buf8
        triton_poi_fused_relu_4[grid(16000)](buf9, primals_7, 16000, XBLOCK
            =256, num_warps=4, num_stages=1)
        del primals_7
        buf10 = empty_strided_cuda((4096, 84), (84, 1), torch.float32)
        extern_kernels.mm(buf9, reinterpret_tensor(primals_8, (120, 84), (1,
            120), 0), out=buf10)
        buf11 = buf10
        del buf10
        triton_poi_fused_relu_5[grid(16000)](buf11, primals_9, 16000, XBLOCK
            =256, num_warps=4, num_stages=1)
        del primals_9
        buf12 = empty_strided_cuda((4096, 20), (20, 1), torch.float32)
        extern_kernels.addmm(primals_11, buf11, reinterpret_tensor(
            primals_10, (84, 20), (1, 84), 0), alpha=1, beta=1, out=buf12)
        del primals_11
        buf13 = buf12
        del buf12
        triton_poi_fused_add_6[grid(80)](buf13, primals_12, 80, XBLOCK=64,
            num_warps=1, num_stages=1)
        del primals_12
        del primals_13
    return (buf13, primals_1, primals_3, primals_4, buf1, buf2, buf3, buf5,
        buf6, buf7, reinterpret_tensor(buf6, (4096, 400), (400, 1), 0),
        buf9, buf11, primals_10, primals_8, primals_6)


class ModelNew(nn.Module):
    def __init__(self, num_classes):
        """
        LeNet-5 architecture implementation in PyTorch.

        :param num_classes: The number of output classes.
        """
        super(ModelNew, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)
    
    def forward(self, input_0):
        primals_1 = self.conv1.weight
        primals_2 = self.conv1.bias
        primals_4 = self.conv2.weight
        primals_5 = self.conv2.bias
        primals_6 = self.fc1.weight
        primals_7 = self.fc1.bias
        primals_8 = self.fc2.weight
        primals_9 = self.fc2.bias
        primals_10 = self.fc3.weight
        primals_11 = self.fc3.bias
        primals_12 = primals_10
        primals_13 = primals_11
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4,
            primals_5, primals_6, primals_7, primals_8, primals_9,
            primals_10, primals_11, primals_12, primals_13])
        return output[0]
