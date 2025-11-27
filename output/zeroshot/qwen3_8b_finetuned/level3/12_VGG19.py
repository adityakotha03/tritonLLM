import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_relu_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 12288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 > tmp1
    tmp3 = tl.where(tmp2, tmp0, tmp1)
    tl.store(out_ptr0 + x0, tmp3, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15 = args
    args.clear()
    assert_size_stride(primals_1, (3, 64, 3, 3), (576, 3, 9, 1))
    assert_size_stride(primals_2, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_3, (64, 64, 3, 3), (576, 3, 9, 1))
    assert_size_stride(primals_4, (128, 64, 3, 3), (576, 3, 9, 1))
    assert_size_stride(primals_5, (128, 128, 3, 3), (576, 3, 9, 1))
    assert_size_stride(primals_6, (256, 128, 3, 3), (576, 3, 9, 1))
    assert_size_stride(primals_7, (256, 256, 3, 3), (576, 3, 9, 1))
    assert_size_stride(primals_8, (256, 256, 3, 3), (576, 3, 9, 1))
    assert_size_stride(primals_9, (256, 256, 3, 3), (576, 3, 9, 1))
    assert_size_stride(primals_10, (512, 256, 3, 3), (576, 3, 9, 1))
    assert_size_stride(primals_11, (512, 512, 3, 3), (576, 3, 9, 1))
    assert_size_stride(primals_12, (512, 512, 3, 3), (576, 3, 9, 1))
    assert_size_stride(primals_13, (512, 512, 3, 3), (576, 3, 9, 1))
    assert_size_stride(primals_14, (512, 512, 3, 3), (576, 3, 9, 1))
    assert_size_stride(primals_15, (512, 512, 3, 3), (576, 3, 9, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 64, 112, 112), (78400, 112, 112, 1), torch.float32)
        buf1 = empty_strided_cuda((1, 64, 112, 112), (78400, 112, 112, 1), torch.float32)
        buf2 = empty_strided_cuda((1, 128, 56, 56), (3136, 56, 56, 1), torch.float32)
        buf3 = empty_strided_cuda((1, 128, 56, 56), (3136, 56, 56, 1), torch.float32)
        buf4 = empty_strided_cuda((1, 256, 28, 28), (784, 28, 28, 1), torch.float32)
        buf5 = empty_strided_cuda((1, 256, 28, 28), (784, 28, 28, 1), torch.float32)
        buf6 = empty_strided_cuda((1, 512, 14, 14), (21, 14, 14, 1), torch.float32)
        buf7 = empty_strided_cuda((1, 512, 14, 14), (21, 14, 14, 1), torch.float32)
        del primals_1
        del primals_2
        get_raw_stream(0)
        triton_poi_fused_relu_0[grid(12288)](buf1, buf0, 12288, XBLOCK=128, num_warps=4, num_stages=1)
        del buf0
        buf8 = buf1
        buf9 = buf8
        del buf1
        buf10 = empty_strided_cuda((1, 128, 56, 56), (3136, 56, 56, 1), torch.float32)
        buf11 = empty_strided_cuda((1, 128, 56, 56), (3136, 56, 56, 1), torch.float32)
        triton_poi_fused_relu_0[grid(125824)](buf11, buf10, 125824, XBLOCK=128, num_warps=4, num_stages=1)
        del buf10
        buf12 = buf11
        buf13 = buf12
        del buf11
        buf14 = empty_strided_cuda((1, 256, 28, 28), (784, 28, 28, 1), torch.float32)
        buf15 = empty_strided_cuda((1, 256, 28, 28), (784, 28, 28, 1), torch.float32)
        triton_poi_fused_relu_0[grid(35840)](buf15, buf14, 35840, XBLOCK=128, num_warps=4, num_stages=1)
        del buf14
        buf16 = buf15
        buf17 = buf16
        del buf15
        buf18 = empty_strided_cuda((1, 512, 14, 14), (21, 14, 14, 1), torch.float32)
        buf19 = empty_strided_cuda((1, 512, 14, 14), (21, 14, 14, 1), torch.float32)
        triton_poi_fused_relu_0[grid(25088)](buf19, buf18, 25088, XBLOCK=128, num_warps=4, num_stages=1)
        del buf18
        buf20 = buf19
        buf21 = buf20
        del buf19
        del primals_3
        del primals_4
        del primals_5
        buf22 = empty_strided_cuda((1, 4096), (4096, 1), torch.float32)
        buf23 = empty_strided_cuda((1, 4096), (4096, 1), torch.float32)
        triton_poi_fused_relu_0[grid(4096)](buf23, buf22, 4096, XBLOCK=128, num_warps=4, num_stages=1)
        del buf22
        buf24 = buf23
        buf25 = buf24
        del buf23
        buf26 = empty_strided_cuda((1, 4096), (4096, 1), torch.float32)
        buf27 = empty_strided_cuda((1, 4096), (4096, 1), torch.float32)
        triton_poi_fused_relu_0[grid(4096)](buf27, buf26, 4096, XBLOCK=128, num_warps=4, num_stages=1)
        del buf26
        buf28 = buf27
        buf29 = buf28
        del buf27
        del primals_6
        del primals_7
        del primals_8
        del primals_9
        del primals_10
        del primals_11
        del primals_12
        del primals_13
        del primals_14
        del primals_15
        del primals_2
    return buf9, buf13, buf17, buf21, buf25, buf29, primals_1, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.0),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.0),
            nn.Linear(4096, num_classes)
        )

    def forward(self, input_0):
        primals_1 = self.features[0].weight
        primals_2 = self.features[0].bias
        primals_3 = self.features[2].weight
        primals_4 = self.features[2].bias
        primals_5 = self.features[4].weight
        primals_6 = self.features[4].bias
        primals_7 = self.features[6].weight
        primals_8 = self.features[6].bias
        primals_9 = self.features[8].weight
        primals_10 = self.features[8].bias
        primals_11 = self.features[10].weight
        primals_12 = self.features[10].bias
        primals_13 = self.features[12].weight
        primals_14 = self.features[12].bias
        primals_15 = self.features[14].weight
        primals_16 = self.features[14].bias
        primals_17 = self.classifier[0].weight
        primals_18 = self.classifier[0].bias
        primals_19 = self.classifier[2].weight
        primals_20 = self.classifier[2].bias
        primals_21 = self.classifier[4].weight
        primals_22 = self.classifier[4].bias
        primals_23 = self.classifier[6].weight
        primals_24 = self.classifier[6].bias
        output = call([input_0, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24])
        return output[0]