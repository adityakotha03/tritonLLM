import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_clone_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 100000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = xindex // 512 % 14
    x2 = xindex // 7168
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512 * x1 + 7168 * x2), xmask)
    tl.store(out_ptr0 + x3, tmp0, xmask)


@triton.jit
def triton_poi_fused_add_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 100000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = xindex // 512 % 14
    x2 = xindex // 7168
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512 * x1 + 7168 * x2), xmask)
    tmp1 = tl.load(in_ptr1 + x3, xmask)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_gelu_add_native_cat_mul_sub_native_layer_norm_0(in_out_ptr0
    , in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask)
    tmp3 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = 0.5
    tmp5 = tmp2 * tmp4
    tmp6 = tmp5 * tmp5
    tmp7 = 0.03567764166454006
    tmp8 = tmp5 * tmp7
    tmp9 = tmp6 + tmp8
    tmp10 = tmp2 + tmp9
    tmp11 = tmp10 - tmp3
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK])
    tmp14 = tl.broadcast_to(tmp12, [XBLOCK])
    tmp16 = tl.broadcast_to(tmp14, [XBLOCK])
    tmp18 = tl.where(xmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 0)[:, None]
    tmp20 = tl.broadcast_to(tmp14, [XBLOCK])
    tmp22 = tl.broadcast_to(tmp20, [XBLOCK])
    tmp24 = tl.where(xmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 0)[:, None]
    tmp26 = 14.0
    tmp27 = tmp25 / tmp26
    tmp28 = tmp14 - tmp27
    tmp29 = tmp28 * tmp28
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK])
    tmp31 = tl.where(xmask, tmp30, 0)
    tmp32 = tl.sum(tmp31, 0)[:, None]
    tmp33 = tmp32 / tmp26
    tmp34 = 1e-05
    tmp35 = tmp33 + tmp34
    tmp36 = libdevice.rsqrt(tmp35)
    tl.store(in_out_ptr0 + x0, tmp10, xmask)
    tl.store(out_ptr0 + x0, tmp25, xmask)
    tl.store(out_ptr1 + x0, tmp33, xmask)
    tl.store(out_ptr2 + x0, tmp36, xmask)


@triton.jit
def triton_poi_fused_native_layer_norm_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 14
    x1 = xindex // 14
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr4 + x1, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr5 + x2, xmask)
    tmp6 = tmp0 + tmp1
    tmp7 = tmp6 - tmp2
    tmp8 = tmp7 * tmp3
    tmp9 = tmp8 + tmp4
    tmp10 = tmp5 + tmp9
    tl.store(out_ptr0 + x2, tmp10, xmask)


def call(args):
    (primals_1, primals_2, primals_3, primals_4, primals_5, primals_6,
        primals_7, primals_8, primals_9, primals_10, primals_11) = args
    args.clear()
    assert_size_stride(primals_1, (2, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(primals_2, (512,), (1,))
    assert_size_stride(primals_3, (512, 1568), (1568, 1))
    assert_size_stride(primals_4, (1, 1, 512), (512, 512, 1))
    assert_size_stride(primals_5, (1, 1, 512), (512, 512, 1))
    assert_size_stride(primals_6, (512,), (1,))
    assert_size_stride(primals_7, (512, 512), (512, 1))
    assert_size_stride(primals_8, (512,), (1,))
    assert_size_stride(primals_9, (512,), (1,))
    assert_size_stride(primals_10, (10,), (1,))
    assert_size_stride(primals_11, (2048, 512), (512, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((2, 14, 512), (7168, 512, 1), torch.float32)
        extern_kernels.addmm(primals_4, reinterpret_tensor(primals_1, (2, 
            10096), (10096, 1), 0), reinterpret_tensor(primals_3, (10096, 
            512), (1, 10096), 0), alpha=1, beta=1, out=buf0)
        del primals_3
        buf1 = empty_strided_cuda((2, 14, 512), (7168, 512, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_clone_0[grid(100000)](primals_1, buf1, 100000,
            XBLOCK=128, num_warps=4, num_stages=1)
        buf2 = empty_strided_cuda((2, 15, 512), (7680, 512, 1), torch.float32)
        triton_poi_fused_add_1[grid(100000)](buf0, primals_5, buf2, 100000,
            XBLOCK=128, num_warps=4, num_stages=1)
        del primals_5
        buf3 = extern_kernels.addmm(primals_6, buf2, reinterpret_tensor(
            primals_7, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf3)
        del primals_6
        buf4 = buf2
        del buf2
        buf5 = empty_strided_cuda((2, 14, 512), (7168, 512, 1), torch.float32)
        buf6 = empty_strided_cuda((2, 14, 1), (14, 1, 512), torch.float32)
        buf7 = empty_strided_cuda((2, 14, 1), (14, 1, 512), torch.float32)
        triton_poi_fused_gelu_add_native_cat_mul_sub_native_layer_norm_0[
            grid(512)](buf4, primals_8, primals_9, buf5, buf6, buf7, 512,
            XBLOCK=256, num_warps=4, num_stages=1)
        del primals_8
        del primals_9
        buf8 = extern_kernels.addmm(primals_10, buf4, reinterpret_tensor(
            primals_11, (512, 10), (1, 512), 0), alpha=1, beta=1, out=buf8)
        del primals_10
        buf9 = buf4
        del buf4
        triton_poi_fused_native_layer_norm_1[grid(512)](buf9, buf5, buf6,
            buf7, primals_11, primals_11, buf8, 512, XBLOCK=256, num_warps=
            4, num_stages=1)
        del buf5
        del buf6
        del buf7
        del primals_11
    return buf8, primals_1, primals_2, primals_4, primals_7, buf0, buf1, buf3,
        buf8, buf9


class ModelNew(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dropout=0.1, emb_dropout=0.1):
        """
        Vision Transformer (ViT) model.

        :param image_size: The size of the input image (assumed to be square).
        :param patch_size: The size of each patch (assumed to be square).
        :param num_classes: The number of output classes.
        :param dim: The dimensionality of the embedding space.
        :param depth: The number of transformer layers.
        :param heads: The number of attention heads.
        :param mlp_dim: The dimensionality of the MLP (Multi-Layer Perceptron) in the transformer.
        :param channels: The number of channels in the input image (default is 3 for RGB).
        :param dropout: Dropout rate applied in the MLP.
        :param emb_dropout: Dropout rate applied to the embedded patches.
        """
        super(ModelNew, self).__init__()
        
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        
        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout),
            num_layers=depth
        )
        
        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes)
        )
    
    def forward(self, input_0):
        primals_3 = self.patch_to_embedding.weight
        primals_2 = self.patch_to_embedding.bias
        primals_4 = self.cls_token
        primals_5 = self.pos_embedding
        primals_6 = self.transformer.layers[0].self_attn.in_proj_weight
        primals_7 = self.transformer.layers[0].self_attn.in_proj_bias
        primals_8 = self.transformer.layers[0].self_attn.out_proj.weight
        primals_9 = self.transformer.layers[0].self_attn.out_proj.bias
        primals_10 = self.transformer.layers[0].linear1.weight
        primals_11 = self.transformer.layers[0].linear1.bias
        primals_12 = self.transformer.layers[0].linear2.weight
        primals_13 = self.transformer.layers[0].linear2.bias
        primals_14 = self.transformer.layers[1].self_attn.in_proj_weight
        primals_15 = self.transformer.layers[1].self_attn.in_proj_bias
        primals_16 = self.transformer.layers[1].self_attn.out_proj.weight
        primals_17 = self.transformer.layers[1].self_attn.out_proj.bias
        primals_18 = self.transformer.layers[1].linear1.weight
        primals_19 = self.transformer.layers[1].linear1.bias
        primals_20 = self.transformer.layers[1].linear2.weight
        primals_21 = self.transformer.layers[1].linear2.bias
        primals_22 = self.transformer.layers[2].self_attn.in_proj_weight
        primals_23 = self.transformer.layers[2].self_attn.in_proj_bias
        primals_24 = self.transformer.layers[2].self_attn.out_proj.weight
        primals_25 = self.transformer.layers[2].self_attn.out_proj.bias
        primals_26 = self.transformer.layers[2].linear1.weight
        primals_27 = self.transformer.layers[2].linear1.bias
        primals_28 = self.transformer.layers[2].linear2.weight
        primals_29 = self.transformer.layers[2].linear2.bias
        primals_30 = self.transformer.layers[3].self_attn.in_proj_weight
        primals_31 = self.transformer.layers[3].self_attn.in_proj_bias
        primals_32 = self.transformer.layers[3].self_attn.out_proj.weight
        primals_33 = self.transformer.layers[3].self_attn.out_proj.bias
        primals_34 = self.transformer.layers[3].linear1.weight
        primals_35 = self.transformer.layers[3].linear1.bias
        primals_36 = self.transformer.layers[3].linear2.weight
        primals_37 = self.transformer.layers[3].linear2.bias
        primals_38 = self.transformer.layers[4].self_attn.in_proj_weight
        primals_39 = self.transformer.layers[4].self_attn.in_proj_bias
        primals_40 = self.transformer.layers[4].self_attn.out_proj.weight
        primals_41 = self.transformer.layers[4].self_attn.out_proj.bias
        primals_42 = self.transformer.layers[4].linear1.weight
        primals_43 = self.transformer.layers[4].linear1.bias
        primals_44 = self.transformer.layers[4].linear2.weight
        primals_45 = self.transformer.layers[4].linear2.bias
        primals_46 = self.transformer.layers[5].self_attn.in_proj_weight
        primals_47 = self.transformer.layers[5].self_attn.in_proj_bias
        primals_48 = self.transformer.layers[5].self_attn.out_proj.weight
        primals_49 = self.transformer.layers[5].self_attn.out_proj.bias
        primals_50 = self.transformer.layers[5].linear1.weight
        primals_51 = self.transformer.layers[5].linear1.bias
        primals_52 = self.transformer.layers[5].linear2.weight
        primals_53 = self.transformer.layers[5].linear2.bias
        primals_54 = self.mlp_head[0].weight
        primals_55 = self.mlp_head[0].bias
        primals_56 = self.mlp_head[2].weight
        primals_57 = self.mlp_head[2].bias
        primals_1 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4,
            primals_5, primals_6, primals_7, primals_8, primals_9,
            primals_10, primals_11, primals_12, primals_13, primals_14,
            primals_15, primals_16, primals_17, primals_18, primals_19,
            primals_20, primals_21, primals_22, primals_23, primals_24,
            primals_25, primals_26, primals_27, primals_28, primals_29,
            primals_30, primals_31, primals_32, primals_33, primals_34,
            primals_35, primals_36, primals_37, primals_38, primals_39,
            primals_40, primals_41, primals_42, primals_43, primals_44,
            primals_45, primals_46, primals_47, primals_48, primals_49,
            primals_50, primals_51, primals_52, primals_53, primals_54,
            primals_55, primals_56, primals_57])
        return output[0]