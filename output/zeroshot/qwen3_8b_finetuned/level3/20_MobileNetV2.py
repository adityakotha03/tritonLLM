import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_add_relu6_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex // 1280
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = tl.full([1], 6, tl.int32)
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tl.store(in_out_ptr0 + x2, tmp6, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10 = args
    args.clear()
    assert_size_stride(primals_1, (1280, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_2, (1280,), (1,))
    assert_size_stride(primals_3, (3, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(primals_4, (1280, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(primals_5, (1280,), (1,))
    assert_size_stride(primals_6, (1280, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(primals_7, (1280,), (1,))
    assert_size_stride(primals_8, (1280, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(primals_9, (1280,), (1,))
    assert_size_stride(primals_10, (1280, 1, 1, 1), (1, 1, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1280, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_poi_fused_add_relu6_0[grid(1280)](buf1, primals_10, 1280,
            XBLOCK=128, num_warps=4, num_stages=1)
        buf2 = empty_strided_cuda((1, 1280, 1, 1), (1280, 1, 1280, 1280),
            torch.float32)
        buf3 = torch.ops.aten.convolution.convolution(primals_3, primals_1,
            stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (1, 1280, 1, 1), (1280, 1, 1280, 1280))
        buf4 = empty_strided_cuda((1, 1280, 1, 1), (1280, 1, 1280, 1280),
            torch.float32)
        buf5 = buf4
        del buf4
        triton_poi_fused_add_relu6_0[grid(1280)](buf5, primals_9, 1280,
            XBLOCK=128, num_warps=4, num_stages=1)
        buf6 = torch.ops.aten.convolution.convolution(buf3, primals_4,
            stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        buf7 = torch.ops.aten.convolution.convolution(buf6, primals_6,
            stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        buf8 = torch.ops.aten.convolution.convolution(buf7, primals_8,
            stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        buf9 = buf8
        del buf8
        triton_poi_fused_add_relu6_0[grid(1280)](buf9, primals_7, 1280,
            XBLOCK=128, num_warps=4, num_stages=1)
        buf10 = torch.ops.aten.convolution.convolution(buf9, primals_1, stride
            =(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        del buf9
    return buf10, primals_1, primals_10, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, buf1, buf5, buf7, buf3, buf6, buf10


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        MobileNetV2 architecture implementation in PyTorch.

        :param num_classes: The number of output classes. Default is 1000.
        """
        super(ModelNew, self).__init__()
        
        def _make_divisible(v, divisor, min_value=None):
            """
            This function ensures that the number of channels is divisible by the divisor.
            """
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            # Make sure that round down does not go down by more than 10%.
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        def _inverted_residual_block(inp, oup, stride, expand_ratio):
            """
            Inverted Residual Block for MobileNetV2.
            """
            hidden_dim = int(inp * expand_ratio)
            use_res_connect = stride == 1 and inp == oup

            layers = []
            if expand_ratio != 1:
                # Pointwise convolution
                layers.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
                layers.append(nn.BatchNorm2d(hidden_dim))
                layers.append(nn.ReLU6(inplace=True))

            layers.extend([
                # Depthwise convolution
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # Pointwise linear convolution
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ])

            if use_res_connect:
                return nn.Sequential(*layers), True
            else:
                return nn.Sequential(*layers), False

        # MobileNetV2 architecture
        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # Building first layer
        features = [nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(input_channel),
                    nn.ReLU6(inplace=True)]

        # Building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c, 8)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(_inverted_residual_block(input_channel, output_channel, stride, expand_ratio=t)[0])
                input_channel = output_channel

        # Building last several layers
        features.append(nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False))
        features.append(nn.BatchNorm2d(last_channel))
        features.append(nn.ReLU6(inplace=True))

        # Final layer
        features.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.features = nn.Sequential(*features)

        # Linear layer
        self.classifier = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(last_channel, num_classes),
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

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
        primals_11 = self.classifier[1].weight
        primals_12 = self.classifier[1].bias
        primals_13 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4, primals_5,
            primals_6, primals_7, primals_8, primals_9, primals_10,
            primals_11, primals_12, primals_13])
        return output[0]