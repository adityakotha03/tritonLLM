import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_convolution_relu_0(in_ptr0, in_ptr1, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = xindex // 32
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x1 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = tmp3 < tmp2
    tl.store(out_ptr0 + x2, tmp2, xmask)
    tl.store(out_ptr0 + (32 + x2), tmp4, xmask)


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_1(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 4 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 4 * x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 4 * x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 4 * x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 > tmp0
    tmp4 = tmp3 > tmp1
    tmp6 = tmp5 > tmp3
    tmp7 = tl.full([1], 1, tl.int64)
    tmp8 = tmp7 < tmp5
    tmp9 = tmp4 & tmp2
    tmp10 = tmp6 & tmp4
    tmp11 = tmp8 & tmp6
    tmp12 = tl.where(tmp10, tmp3, tmp1)
    tmp13 = tl.where(tmp11, tmp5, tmp3)
    tmp14 = tl.where(tmp9, tmp1, tmp3)
    tmp15 = tl.where(tmp11, tmp13, tmp14)
    tmp16 = tl.where(tmp10, tmp12, tmp14)
    tmp17 = tl.where(tmp11, tmp13, tmp16)
    tmp18 = tl.where(tmp9, tmp15, tmp17)
    tmp19 = tl.where(tmp10, tmp16, tmp18)
    tl.store(out_ptr0 + x0, tmp19, xmask)
    tl.store(out_ptr1 + x0, tmp17, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_2(in_ptr0, in_ptr1, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = tmp3 < tmp2
    tl.store(out_ptr0 + x2, tmp2, xmask)
    tl.store(out_ptr0 + (16 + x2), tmp4, xmask)


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_3(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 4 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 4 * x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 4 * x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 4 * x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 > tmp0
    tmp4 = tmp3 > tmp1
    tmp6 = tmp5 > tmp3
    tmp7 = tl.full([1], 1, tl.int64)
    tmp8 = tmp7 < tmp5
    tmp9 = tmp4 & tmp2
    tmp10 = tmp6 & tmp4
    tmp11 = tmp8 & tmp6
    tmp12 = tl.where(tmp10, tmp3, tmp1)
    tmp13 = tl.where(tmp11, tmp5, tmp3)
    tmp14 = tl.where(tmp9, tmp1, tmp3)
    tmp15 = tl.where(tmp11, tmp13, tmp14)
    tmp16 = tl.where(tmp10, tmp12, tmp14)
    tmp17 = tl.where(tmp11, tmp13, tmp16)
    tmp18 = tl.where(tmp9, tmp15, tmp17)
    tmp19 = tl.where(tmp10, tmp16, tmp18)
    tl.store(out_ptr0 + x0, tmp19, xmask)
    tl.store(out_ptr1 + x0, tmp17, xmask)


@triton.jit
def triton_poi_fused_relu_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 491520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 120
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = tmp3 < tmp2
    tl.store(in_out_ptr0 + x2, tmp2, xmask)
    tl.store(in_out_ptr0 + (120 + x2), tmp4, xmask)


@triton.jit
def triton_poi_fused_relu_5(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 344064
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 84
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = tmp3 < tmp2
    tl.store(in_out_ptr0 + x2, tmp2, xmask)
    tl.store(in_out_ptr0 + (84 + x2), tmp4, xmask)


@triton.jit
def triton_poi_fused_convolution_6(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 8000
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
        primals_7, primals_8, primals_9, primals_10, primals_11) = args
    args.clear()
    assert_size_stride(primals_1, (6, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_2, (6,), (1,))
    assert_size_stride(primals_3, (4096, 1, 32, 32), (1024, 1024, 32, 1))
    assert_size_stride(primals_4, (16, 6, 5, 5), (150, 25, 5, 1))
    assert_size_stride(primals_5, (16,), (1,))
    assert_size_stride(primals_6, (120, 4096), (4096, 1))
    assert_size_stride(primals_7, (120,), (1,))
    assert_size_stride(primals_8, (84, 120), (120, 1))
    assert_size_stride(primals_9, (84,), (1,))
    assert_size_stride(primals_10, (20, 84), (84, 1))
    assert_size_stride(primals_11, (20,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4096, 6, 5, 5), (150, 25, 5, 1), torch.
            float32)
        get_input = primals_3
        triton_poi_fused_convolution_relu_0[grid] = lambda meta: (1024,)
        triton_poi_fused_convolution_relu_0[grid](
            primals_1, get_input, buf0, 1024, XBLOCK=128, num_warps=4,
            num_stages=1)
        buf1 = empty_strided_cuda((4096, 6, 2, 2), (24, 4, 2, 1), torch.int64)
        buf2 = empty_strided_cuda((4096, 6, 2, 2), (24, 4, 2, 1), torch.float32
            )
        triton_poi_fused_max_pool2d_with_indices_1[grid] = lambda meta: (256,)
        triton_poi_fused_max_pool2d_with_indices_1[grid](
            buf0, buf1, buf2, 256, XBLOCK=128, num_warps=4, num_stages=1)
        del buf0
        buf3 = empty_strided_cuda((4096, 16, 5, 5), (400, 25, 5, 1), torch.
            float32)
        triton_poi_fused_convolution_relu_2[grid] = lambda meta: (4096,)
        triton_poi_fused_convolution_relu_2[grid](
            primals_4, buf2, buf3, 4096, XBLOCK=512, num_warps=8, num_stages=1)
        buf4 = empty_strided_cuda((4096, 16, 2, 2), (64, 4, 2, 1), torch.int64)
        buf5 = empty_strided_cuda((4096, 16, 2, 2), (64, 4, 2, 1), torch.float32
            )
        triton_poi_fused_max_pool2d_with_indices_3[grid] = lambda meta: (1024,)
        triton_poi_fused_max_pool2d_with_indices_3[grid](
            buf3, buf4, buf5, 1024, XBLOCK=512, num_warps=8, num_stages=1)
        del buf3
        buf6 = empty_strided_cuda((4096, 120), (120, 1), torch.float32)
        triton_poi_fused_relu_4[grid] = lambda meta: (491520,)
        triton_poi_fused_relu_4[grid](buf6, primals_6, 491520, XBLOCK=1024,
            num_warps=4, num_stages=1)
        del primals_6
        buf7 = empty_strided_cuda((4096, 84), (84, 1), torch.bool)
        triton_poi_fused_relu_5[grid] = lambda meta: (344064,)
        triton_poi_fused_relu_5[grid](buf7, primals_8, 344064, XBLOCK=1024,
            num_warps=4, num_stages=1)
        del primals_8
        buf8 = empty_strided_cuda((4096, 20), (20, 1), torch.float32)
        triton_poi_fused_convolution_6[grid] = lambda meta: (8000,)
        triton_poi_fused_convolution_6[grid](buf8, primals_10, 8000, XBLOCK
            =1024, num_warps=4, num_stages=1)
        del primals_10
    return buf8, primals_1, primals_2, primals_4, primals_5, primals_7, buf1,
    buf2, buf4, buf5, buf6, buf7, primals_9, primals_11


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
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4,
            primals_5, primals_6, primals_7, primals_8, primals_9,
            primals_10, primals_11])
        return output[0]
