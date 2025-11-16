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


@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 5120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 16 % 512
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 20480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4096 % 128
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + (1024 + x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (2049 + x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (3072 + x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = tmp0 + tmp5
    tl.store(out_ptr0 + x2, tmp6, xmask)


@triton.jit
def triton_poi_fused_cat_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 50560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 128
    x2 = xindex % 128
    x0 = xindex % 5120
    x4 = xindex
    tmp0 = x1
    tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 2, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + x0, tmp4 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp6 = tmp4 & xmask
    tl.full([1], 1, tl.int64)
    tmp9 = tl.load(in_ptr1 + x4, tmp6, eviction_policy='evict_last',
        other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tl.store(out_ptr0 + x4, tmp10, tmp6)


@triton.jit
def triton_poi_fused__to_copy_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK:
    tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 10
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 10 * x2), xmask & ymask, eviction_policy
        ='evict_last')
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x2 + 128 * y0), tmp2, xmask & ymask)


@triton.jit
def triton_poi_fused_clone_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    ynumel = 10
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 10 * x2), xmask & ymask, eviction_policy
        ='evict_last')
    tl.store(out_ptr0 + (x2 + 128 * y0), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_clone_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    ynumel = 10
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 10 * x2), xmask & ymask, eviction_policy
        ='evict_last')
    tl.store(out_ptr0 + (y0 + 10 * x2), tmp0, xmask & ymask)


def call(args):
    (primals_1, primals_2, primals_3, primals_4, primals_5, primals_6,
        primals_7, primals_8, primals_9, primals_10) = args
    args.clear()
    assert_size_stride(primals_1, (5120,), (1,))  # This line is added by the
    # test
    assert_size_stride(primals_2, (5120,), (1,))  # This line is added by the
    # test
    assert_size_stride(primals_3, (5120,), (1,))  # This line is added by the
    # test
    assert_size_stride(primals_4, (5120,), (1,))  # This line is added by the
    # test
    assert_size_stride(primals_5, (128,), (1,))  # This line is added by the
    # test
    assert_size_stride(primals_6, (128, 128, 4, 4), (2048, 16, 4, 1))  # This
    # line is added by the test
    assert_size_stride(primals_7, (128, 4096), (4096, 1))  # This line is
    # added by the test
    assert_size_stride(primals_8, (1, 1, 128), (128, 128, 1))  # This line is
    # added by the test
    assert_size_stride(primals_9, (1000,), (1,))  # This line is added by the
    # test
    assert_size_stride(primals_10, (1000, 128), (128, 1))  # This line is
    # added by the test
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_6, primals_1, stride=(4, 
            4), padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (10, 128, 8, 8), (8192, 64, 8, 1))
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(5120)](buf1, primals_2, 5120,
            XBLOCK=128, num_warps=4, num_stages=1)
        del primals_2
        buf2 = empty_strided_cuda((10, 128, 8, 8), (8192, 64, 8, 1), torch.
            float32)
        extern_kernels.convolution(buf1, primals_3, stride=(1, 1), padding=(
            0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0
            ), groups=1, bias=None, out=buf2)
        assert_size_stride(buf2, (10, 128, 8, 8), (8192, 64, 8, 1))
        buf3 = empty_strided_cuda((10, 512, 16), (10240, 2048, 1), torch.
            float32)
        triton_poi_fused_convolution_1[grid(20480)](buf2, buf3, 20480,
            XBLOCK=256, num_warps=4, num_stages=1)
        buf4 = empty_strided_cuda((10, 128, 2), (256, 2, 1), torch.float32)
        triton_poi_fused_cat_2[grid(50560)](primals_5, buf3, buf4, 50560,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_5
        buf5 = empty_strided_cuda((10, 128, 2), (256, 2, 1), torch.float32)
        triton_poi_fused__to_copy_3[grid(10, 128)](buf4, buf5, 10, 128,
            XBLOCK=16, XOFFSET=64, num_warps=2, num_stages=1)
        del buf4
        buf6 = empty_strided_cuda((10, 128, 2), (256, 2, 1), torch.float32)
        triton_poi_fused_clone_4[grid(10, 128)](buf5, buf6, 10, 128, XBLOCK
            =32, YBLOCK=16, num_warps=4, num_stages=1)
        buf7 = buf5
        del buf5
        buf8 = buf6
        del buf6
        triton_poi_fused_clone_5[grid(10, 128)](buf7, buf8, 10, 128, XBLOCK
            =32, YBLOCK=16, num_warps=4, num_stages=1)
        buf9 = buf8
        del buf8
        buf10 = buf7
        del buf7
        triton_poi_fused_clone_5[grid(10, 128)](buf9, buf10, 10, 128, XBLOCK
            =32, YBLOCK=16, num_warps=4, num_stages=1)
        buf11 = empty_strided_cuda((10, 128, 2), (256, 2, 1), torch.float32)
        extern_kernels.bmm(buf10, primals_8, out=buf11)
        del primals_8
        buf12 = extern_kernels.addmm(primals_9, buf11, primals_10, alpha=1,
            beta=1)
        assert_size_stride(buf12, (10, 1000), (1000, 1))
        del primals_9
    return (buf12, primals_1, primals_3, primals_4, primals_6, primals_7,
        buf1, buf2, buf3, buf10, buf11)


class ModelNew(nn.Module):
    def __init__(self, num_classes, embed_dim=512, num_heads=8, num_layers=6, 
                 mlp_ratio=4.0, patch_size=4, in_channels=3, image_size=32):
        """
        Convolutional Vision Transformer (CViT) implementation.
        :param num_classes: Number of output classes for classification.
        :param embed_dim: Dimensionality of the embedding space.
        :param num_heads: Number of attention heads.
        :param num_layers: Number of transformer layers.
        :param mlp_ratio: Ratio of the MLP hidden dimension to the embedding dimension.
        :param patch_size: Size of the convolutional patches.
        :param in_channels: Number of input channels (e.g., 3 for RGB images).
        :param image_size: Height/width of the square input image.
        """
        super(ModelNew, self).__init__()

        self.patch_size = patch_size
        self.image_size = image_size
        self.embed_dim = embed_dim

        self.conv1 = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2  # Total number of patches after conv
        self.linear_proj = nn.Linear(embed_dim * num_patches, embed_dim)

        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=0.0,
                batch_first=True
            ) for _ in range(num_layers)
        ])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.fc_out = nn.Linear(embed_dim, num_classes)

    def forward(self, input_0):
        primals_8 = self.cls_token
        primals_9 = self.fc_out.weight
        primals_10 = self.fc_out.bias
        primals_1 = self.conv1.weight
        primals_3 = self.conv1.bias
        primals_2 = self.linear_proj.weight
        primals_4 = self.linear_proj.bias
        primals_6 = self.transformer_layers[0].self_attn.in_proj_weight
        primals_7 = self.transformer_layers[0].self_attn.in_proj_bias
        primals_5 = self.transformer_layers[0].self_attn.out_proj.weight
        primals_10 = self.fc_out.bias
        output = call([primals_1, primals_2, primals_3, primals_4,
            primals_5, primals_6, primals_7, primals_8, primals_9, primals_10
            ])
        return output[0]
