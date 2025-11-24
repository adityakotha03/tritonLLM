@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, out_ptr, bias_ptr,
    B, C, H, W, 
    KERNEL_H, KERNEL_W,
    STRIDE_H, STRIDE_W,
    DILATION_H, DILATION_W,
    PAD_H, PAD_W,
    OUT_C,
    OUT_H, OUT_W,
    BLOCK_SIZE: tl.constexpr,
):