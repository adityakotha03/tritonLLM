import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_add_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x2 = xindex // 512
    x1 = xindex % 196
    tmp0 = tl.load(in_ptr0 + (x2, x1, x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2, x1, x0), tmp2, xmask)


@triton.jit
def triton_poi_fused_erf_mul_sub_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.4142135623731027
    tmp2 = tmp0 / tmp1
    tmp3 = tl.erf(tmp2)
    tmp4 = 1.0 + tmp3
    tmp5 = 0.5 * tmp4
    tmp6 = tmp0 * tmp5
    tl.store(out_ptr0 + x0, tmp6, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10 = args
    args.clear()
    assert_size_stride(primals_1, (196, 512), (512, 1))
    assert_size_stride(primals_2, (512,), (1,))
    assert_size_stride(primals_3, (2, 1, 512), (512, 1, 512))
    assert_size_stride(primals_4, (2, 1, 512), (512, 1, 512))
    assert_size_stride(primals_5, (2, 196, 512), (100352, 512, 1))
    assert_size_stride(primals_6, (2, 512, 1024), (524288, 1024, 1))
    assert_size_stride(primals_7, (1024,), (1,))
    assert_size_stride(primals_8, (1024,), (1,))
    assert_size_stride(primals_9, (1, 1, 512), (512, 1, 1))
    assert_size_stride(primals_10, (2, 1024), (1024, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((2, 196, 512), (100352, 512, 1), torch.float32)
        get_rawbuf0 = buf0
        del buf0
        buf1 = empty_strided_cuda((2, 196, 512), (100352, 512, 1), torch.float32)
        get_rawbuf1 = buf1
        del buf1
        buf2 = empty_strided_cuda((2, 196, 512), (100352, 512, 1), torch.float32)
        get_rawbuf2 = buf2
        del buf2
        buf3 = empty_strided_cuda((2, 512), (512, 1), torch.float32)
        buf4 = empty_strided_cuda((2, 512), (512, 1), torch.float32)
        buf5 = empty_strided_cuda((2, 1024), (1024, 1), torch.float32)
        buf6 = empty_strided_cuda((2, 1024), (1024, 1), torch.float32)
        buf7 = empty_strided_cuda((2, 1024), (1024, 1), torch.float32)
        buf8 = empty_strided_cuda((2, 1024), (1024, 1), torch.float32)
        del primals_3
        del primals_4
        triton_poi_fused_add_0[grid(200704)](primals_5, primals_2, buf0, 200704,
            XBLOCK=128, num_warps=4, num_stages=1)
        del primals_2
        buf0 = buf0 + primals_9
        del primals_9
        del primals_5
        del primals_6
        triton_poi_fused_add_0[grid(200704)](buf0, primals_1, buf1, 200704,
            XBLOCK=128, num_warps=4, num_stages=1)
        del primals_1
        buf1 = reinterpret_tensor(buf1, (2, 196, 512), (100352, 512, 1), 0)
        del buf0
        del buf1
        buf1 = buf1 + primals_4
        del primals_4
        buf2 = buf1 + primals_3
        del primals_3
        del primals_4
        del primals_5
        buf3 = buf2
        del buf2
        del buf1
        buf4 = buf3
        del buf3
        buf5 = buf4
        del buf4
        buf6 = buf5
        del buf5
        buf7 = buf6
        del buf6
        buf8 = buf7
        del buf7
        del primals_8
        del primals_7
    return buf8, primals_10, primals_6, primals_1, primals_8, primals_9, primals_4, primals_2, buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, primals_5


class ModelNew(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dropout=0.1, emb_dropout=0.1):
        super(ModelNew, self).__init__()
        assert image_size % patch_size == 0
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
        primals_1 = self.patch_to_embedding.weight
        primals_2 = self.patch_to_embedding.bias
        primals_3 = self.cls_token
        primals_4 = self.dropout.weight
        primals_5 = self.pos_embedding
        primals_6 = self.mlp_head[0].weight
        primals_7 = self.mlp_head[0].bias
        primals_8 = self.mlp_head[3].weight
        primals_9 = self.mlp_head[2].weight
        primals_10 = self.mlp_head[1].bias
        primals_11 = self.mlp_head[3].bias
        output = call([input_0, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11])
        return output[0]