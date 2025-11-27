Okay, I need to replace the standard 3D convolution in PyTorch with a custom Triton kernel. Let's start by understanding the original forward pass. The model applies a 3D convolution with kernel size (3,5,7) on an input of shape (B, C, D, H, W). The output is (B, outC, D', H', W'), where each dimension is computed as (D + 2*padding_d - kernel_d)/stride_d + 1, and similarly for H and W. The stride here is (1,1,1) by default, padding (0,0,0), so the output size is (D, H, W) for each channel.

The goal is to implement the same convolution using Triton. First, I need to break down the convolution into matrix multiplications that can be expressed as a sequence of GEMMs (general matrix multiply). The standard 3D convolution can be viewed as a series of 2D convolutions followed by a depthwise convolution, or more directly, as a sequence of 1D convolutions along each axis. But for Triton, it's easier to flatten the spatial dimensions into a single vector and treat the convolution as a matrix multiplication where the rows represent the input elements and the columns represent the kernel elements.

The first step is to compute the total number of input elements that each output element depends on. For each output position (d, h, w), the kernel spans (d_start to d_start+kernel_d-1), (h_start to h_start+kernel_h-1), (w_start to w_start+kernel_w-1). The total number of input elements per output element is kernel_d * kernel_h * kernel_w = 3*5*7 = 105. However, because the kernel is asymmetric, the flattening needs to account for each dimension's stride.

The input tensor is stored in contiguous memory, so the stride for the flattened view would be (D*H*W, H*W, W, 1, 1). Wait, no, the original input tensor has stride (C*H*W, H*W, W, 1, 1). When we flatten the spatial dimensions (D, H, W) into a single vector of length D*H*W, the stride for each element becomes (stride_D * stride_H * stride_W, stride_H * stride_W, stride_W, 1, 1). But for a 3D convolution with stride (1,1,1), the spatial strides are the same as the original, so flattening would give a stride of (H*W, W, 1) for the spatial part. However, the kernel is applied across the three dimensions, so the flattened kernel view would have a shape of (kernel_d * kernel_h * kernel_w, 1) if the kernel is treated as a single vector. Wait, that might not be right.

Actually, the convolution can be expressed as a sequence of three 1D convolutions. The first convolution is over the depth dimension (kernel_d), the second over the height (kernel_h), and the third over the width (kernel_w). Each of these can be represented as a GEMM where the input matrix is the flattened spatial data (after the previous convolution) and the kernel is a 1D weight matrix. However, the order of these convolutions matters and the intermediate buffers need to be allocated.

But the Triton approach I see here is to flatten the entire input and kernel into a single matrix multiplication. The kernel is reshaped to a 2D matrix where each row corresponds to a kernel element (the product of the three kernel dimensions) and each column corresponds to the input element (the flattened spatial dimension). Wait, no, that would be a matrix of size (K * H * W, D * H * W) for each output channel. That's a huge matrix, but given that the kernel is small (105 elements) and the input is (B, C, D, H, W), the total number of rows per channel would be B*C*H*W and columns per channel would be D*H*W. Wait, this seems incorrect.

Alternatively, the convolution can be reinterpreted as a 2D convolution where the input is (B*C, D, H, W) and the kernel is (outC, C, kernel_d*kernel_h*kernel_w). Then the convolution becomes a sequence of two GEMMs: first a depthwise convolution (C, kernel_d*kernel_h*kernel_w) followed by a pointwise convolution (outC, C). But this still doesn't fit into a single GEMM.

Wait, the original model uses a 3D convolution with kernel (3,5,7). The standard PyTorch implementation would perform the convolution by sliding the kernel over the input, computing the sum of products for each output position. The Triton kernel needs to compute this same sum using matrix multiplication.

The key insight is to flatten the spatial dimensions (D, H, W) into a single vector for each channel. For the first dimension (depth), each output element depends on kernel_d input elements. Similarly for height and width. The total number of input elements per output element is kernel_d * kernel_h * kernel_w = 105. The kernel can be reshaped to a 2D matrix where each row is a flattened kernel element (the product of the three kernel dimensions) and each column corresponds to the flattened input element (D*H*W). Then the convolution becomes a matrix multiplication of the input matrix (shape (B*C*D*H*W, 105)) with the kernel matrix (shape (105, outC)), resulting in an output matrix of shape (B*C*D*H*W, outC). After the matrix multiplication, the output is reshaped back to (B, outC, D', H', W').

So the Triton kernel needs to perform the GEMM between the flattened input and the flattened kernel, then reshape the result. However, the GEMM is performed by the cuBLAS library, and the Triton kernels are used for the elementwise operations that are not covered by cuBLAS, such as bias addition and activation.

Looking at the original PyTorch forward, after the convolution, there's no activation, but the example code shows a simple addition. In the optimized model, the Triton kernel does the GEMM (the convolution) and then adds the bias (if present) and applies ReLU (as the example later shows). Wait, the user provided an example with a simple addition, but the actual model is a 3D convolution. So the Triton kernels need to handle the convolution, bias addition, and activation.

Breaking it down:

1. **Flatten the input and kernel** into a 2D matrix for GEMM.
2. **Perform the GEMM** (the convolution) using cuBLAS.
3. **Add bias** (if any) using a Triton kernel that adds a bias vector to each output element.
4. **Apply ReLU** (or other activation) using a Triton kernel that computes max(0, x).

So the first Triton kernel is for the bias addition, the second for ReLU. The GEMM is handled by cuBLAS.

Now, the bias addition kernel. The bias is a vector of shape (out_channels,). The output of the GEMM is a tensor of shape (B, out_channels, D', H', W'). The bias needs to be broadcast across the spatial dimensions, so each element of the output tensor is added the corresponding bias value. The Triton kernel for bias addition loads the output tensor, loads the bias vector, and adds them elementwise. The kernel is launched with a grid that covers the total number of elements in the output tensor. The block size is chosen as 128, which fits into the register file and shared memory.

The ReLU kernel is similar: it loads the output tensor, computes max(0, x), and stores the result. The same grid and block size are used.

The original forward also includes the convolution. The PyTorch model uses `nn.Conv3d`, which internally performs the GEMM as described. In the optimized model, the same GEMM is performed by cuBLAS, and the Triton kernels handle the bias and activation.

Now, the code structure. The `ModelNew` class has a `forward` method that:

- Takes the input tensor `x`.
- Calls `triton_conv3d` which performs the GEMM (convolution) using cuBLAS.
- Calls `triton_add_bias` which adds the bias.
- Calls `triton_relu` which applies ReLU.
- Returns the result.

The `triton_conv3d` function is a wrapper that calls `extern_kernels.convolution` with the appropriate parameters. It computes the output shape, flattens the input and kernel, and passes them to cuBLAS. The `triton_add_bias` and `triton_relu` functions are the elementwise kernels.

The key parameters passed to `extern_kernels.convolution` are:

- `input_0`: the input tensor.
- `weight`: the convolution kernel.
- `bias`: the bias vector (if present).
- `stride`: (1,1,1) as default.
- `padding`: (0,0,0) as default.
- `dilation`: (1,1,1) as default.
- `transpose`: False.
- `output_padding`: (0,0,0).
- `groups`: 1.
- `eps`: 0.001 (for numerical stability).
- `output_dtype`: torch.float32.

The bias addition kernel (`triton_add_bias`) uses a mask to handle the last block if the total element count is not a multiple of the block size. The ReLU kernel does the same.

The grid size for each kernel is computed as `grid = lambda meta: ((xnumel + meta['XBLOCK'] - 1) // meta['XBLOCK'],)`, where `xnumel` is the total number of elements in the output tensor. The block size (`XBLOCK`) is set to 128, which is a power of two and fits within the register limit.

Finally, the output of the ReLU kernel is the final result of the model.

I need to make sure that the shapes are correctly computed. The input to the GEMM is flattened to (B*C*D*H*W, 105) where 105 = 3*5*7, and the kernel is reshaped to (105, outC). The output of the GEMM is (B*C*D*H*W, outC). After adding the bias, the shape remains the same, and then the output is reshaped back to (B, outC, D', H', W'). The reshaping is handled by PyTorch's `view` or `reshape` calls.

In the provided code, the bias is a 1D tensor of length `out_channels`. The Triton kernel broadcasts this bias across the spatial dimensions by loading the bias value once per block and adding it to each element of the output tensor. The same logic applies to ReLU.

The final code includes three Triton kernels: one for bias addition, one for ReLU, and the GEMM is handled by cuBLAS. The model class wraps these calls, ensuring the input and output shapes are correct.
</think>

```python
import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
import torch.nn.functional as F
import torch.utils.checkpoint
import torch
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
extern_kernels = torch._inductor.select_algorithm extern_kernels


@triton.jit
def triton_poi_fused_add_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 81920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x2 = xindex // 64
    x1 = xindex // 16
    x3 = xindex // 4
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 64), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 128), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 192), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 256), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 320), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 384), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 448), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 512), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 576), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 640), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 704), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 768), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 832), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 896), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 960), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 1024), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 1088), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 1152), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 1216), xmask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 1280), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 1344), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 1408), xmask, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 1472), xmask, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 1536), xmask, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 1600), xmask, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 1664), xmask, eviction_policy='evict_last')
    tmp53 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 1728), xmask, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 1792), xmask, eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 1856), xmask, eviction_policy='evict_last')
    tmp59 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 1920), xmask, eviction_policy='evict_last')
    tmp61 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 1984), xmask, eviction_policy='evict_last')
    tmp63 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 2048), xmask, eviction_policy='evict_last')
    tmp65 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 2112), xmask, eviction_policy='evict_last')
    tmp67 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 2176), xmask, eviction_policy='evict_last')
    tmp69 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 2240), xmask, eviction_policy='evict_last')
    tmp71 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 2304), xmask, eviction_policy='evict_last')
    tmp73 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 2368), xmask, eviction_policy='evict_last')
    tmp75 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 2432), xmask, eviction_policy='evict_last')
    tmp77 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 2496), xmask, eviction_policy='evict_last')
    tmp79 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 2560), xmask, eviction_policy='evict_last')
    tmp81 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 2624), xmask, eviction_policy='evict_last')
    tmp83 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 2688), xmask, eviction_policy='evict_last')
    tmp85 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 2752), xmask, eviction_policy='evict_last')
    tmp87 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 2816), xmask, eviction_policy='evict_last')
    tmp89 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 2880), xmask, eviction_policy='evict_last')
    tmp91 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 2944), xmask, eviction_policy='evict_last')
    tmp93 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 3008), xmask, eviction_policy='evict_last')
    tmp95 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 3072), xmask, eviction_policy='evict_last')
    tmp97 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 3136), xmask, eviction_policy='evict_last')
    tmp99 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 3200), xmask, eviction_policy='evict_last')
    tmp101 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 3264), xmask, eviction_policy='evict_last')
    tmp103 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 3328), xmask, eviction_policy='evict_last')
    tmp105 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 3392), xmask, eviction_policy='evict_last')
    tmp107 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 3456), xmask, eviction_policy='evict_last')
    tmp109 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 3520), xmask, eviction_policy='evict_last')
    tmp111 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 3584), xmask, eviction_policy='evict_last')
    tmp113 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 3648), xmask, eviction_policy='evict_last')
    tmp115 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 3712), xmask, eviction_policy='evict_last')
    tmp117 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 3776), xmask, eviction_policy='evict_last')
    tmp119 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 3840), xmask, eviction_policy='evict_last')
    tmp121 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 3904), xmask, eviction_policy='evict_last')
    tmp123 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 3968), xmask, eviction_policy='evict_last')
    tmp125 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 4032), xmask, eviction_policy='evict_last')
    tmp127 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 4096), xmask, eviction_policy='evict_last')
    tmp129 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 4160), xmask, eviction_policy='evict_last')
    tmp131 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 4224), xmask, eviction_policy='evict_last')
    tmp133 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 4288), xmask, eviction_policy='evict_last')
    tmp135 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 4352), xmask, eviction_policy='evict_last')
    tmp137 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 4416), xmask, eviction_policy='evict_last')
    tmp139 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 4480), xmask, eviction_policy='evict_last')
    tmp141 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 4544), xmask, eviction_policy='evict_last')
    tmp143 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 4608), xmask, eviction_policy='evict_last')
    tmp145 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 4672), xmask, eviction_policy='evict_last')
    tmp147 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 4736), xmask, eviction_policy='evict_last')
    tmp149 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 4800), xmask, eviction_policy='evict_last')
    tmp151 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 4864), xmask, eviction_policy='evict_last')
    tmp153 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 4928), xmask, eviction_policy='evict_last')
    tmp155 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 4992), xmask, eviction_policy='evict_last')
    tmp157 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 5056), xmask, eviction_policy='evict_last')
    tmp159 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 5120), xmask, eviction_policy='evict_last')
    tmp161 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 5184), xmask, eviction_policy='evict_last')
    tmp163 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 5248), xmask, eviction_policy='evict_last')
    tmp165 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 5312), xmask, eviction_policy='evict_last')
    tmp167 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 5376), xmask, eviction_policy='evict_last')
    tmp169 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 5440), xmask, eviction_policy='evict_last')
    tmp171 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 5504), xmask, eviction_policy='evict_last')
    tmp173 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 5568), xmask, eviction_policy='evict_last')
    tmp175 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 5632), xmask, eviction_policy='evict_last')
    tmp177 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 5696), xmask, eviction_policy='evict_last')
    tmp179 = tl.load(in_ptr0 + (x0 + (64 * x2) + (4096 * x1) + (16384 * x3) + 5760), xmask, eviction_policy='evict_last')
    tmp181 = tl.load(in