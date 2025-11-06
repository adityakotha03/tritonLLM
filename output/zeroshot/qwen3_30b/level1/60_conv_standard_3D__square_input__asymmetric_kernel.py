import torch
import torch.nn as nn
import triton
import triton.language as tl


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        # Convert to float16 to use Tensor Cores
        self.kernel = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size).half())
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels).half())
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure the input is contiguous and on GPU
        x = x.contiguous()
        # Convert to float16 for better performance
        x = x.half()
        # If the kernel is not on the same device, move it
        self.kernel = self.kernel.to(x.device)
        if self.bias is not None:
            self.bias = self.bias.to(x.device)

        # Calculate output dimensions
        batch_size, in_channels, width, height, depth = x.shape
        k_w, k_h, k_d = self.kernel_size
        out_w = (width + 2 * self.padding - k_w) // self.stride + 1
        out_h = (height + 2 * self.padding - k_h) // self.stride + 1
        out_d = (depth + 2 * self.padding - k_d) // self.stride + 1

        # Output tensor
        out = torch.empty(batch_size, self.out_channels, out_w, out_h, out_d, dtype=torch.float16, device=x.device)

        # Parameters for the kernel
        total_ow = out_w // 8
        total_oh = out_h // 8
        total_od = out_d // 8
        total_blocks = batch_size * self.out_channels * total_ow * total_oh * total_od
        grid = lambda meta: ( (total_blocks + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"], )

        # Launch the kernel
        conv3d_kernel[grid](
            x, self.kernel, out,
            batch_size, in_channels, width, height, depth,
            out_w, out_h, out_d,
            k_w, k_h, k_d,
            self.stride, self.padding,
            BLOCK_SIZE=512,
            num_warps=4
        )

        # Convert back to float32 if needed
        if x.dtype == torch.float32:
            out = out.float()

        # Add bias if present
        if self.bias is not None:
            out = out + self.bias.view(1, self.out_channels, 1, 1, 1)

        return out


@triton.jit
def conv3d_kernel(
    x_ptr, kernel_ptr, out_ptr,
    batch_size: tl.int32, in_channels: tl.int32, width: tl.int32, height: tl.int32, depth: tl.int32,
    out_w: tl.int32, out_h: tl.int32, out_d: tl.int32,
    k_w: tl.int32, k_h: tl.int32, k_d: tl.int32,
    stride: tl.int32, padding: tl.int32,
    BLOCK_SIZE: tl.constexpr,
    num_warps: tl.constexpr
):
    # Each block handles one spatial tile (8x8x8) and one output channel
    # Compute the base index for the block
    base_index = tl.program_id(0)
    # Decompose base_index into (b, oc, ow_base, oh_base, od_base)
    total_ow = out_w // 8
    total_oh = out_h // 8
    total_od = out_d // 8
    total_per_oc = total_ow * total_oh * total_od
    total_per_channel = total_per_oc * batch_size
    b = base_index // total_per_channel
    index = base_index % total_per_channel
    oc = index // total_per_oc
    index = index % total_per_oc
    ow_base = index // (total_oh * total_od)
    index = index % (total_oh * total_od)
    oh_base = index // total_od
    od_base = index % total_od

    # Shared memory for kernel and input patch
    # Kernel: (oc, ic, k_w, k_h, k_d) -> we only need for the current oc
    # We'll load the entire kernel for the current oc into shared memory
    kernel_shared = tl.allocate_shared((in_channels * k_w * k_h * k_d,), dtype=tl.float16)
    # Input patch: (in_channels, k_w, k_h, k_d) for the current spatial tile
    # But we need to cover a larger region to account for the stride
    # The input region is from (ow_base*stride - padding, ...) to (ow_base*stride - padding + 8*stride + k_w - 1, ...)
    # We'll allocate shared memory for the entire input patch for the tile
    # The input patch size is: in_channels * (8*stride + k_w - 1) * (8*stride + k_h - 1) * (8*stride + k_d - 1)
    # But we'll use a simpler approach: we only need the part that overlaps with the kernel
    # We'll use: (in_channels, k_w, k_h, k_d) for the kernel and a larger region for the input
    # But we are not using the larger region for now.
    # Instead, we'll load the input patch on-demand.

    # For now, we'll load the kernel weights for the current output channel into shared memory
    # The kernel is (out_channels, in_channels, k_w, k_h, k_d), so we only need the oc-th channel
    # Compute the offset in the kernel tensor
    kernel_offset = oc * in_channels * k_w * k_h * k_d
    # Load kernel weights into shared memory
    for i in range(k_w * k_h * k_d):
        for j in range(in_channels):
            idx = j * k_w * k_h * k_d + i
            kernel_shared[idx] = tl.load(kernel_ptr + kernel_offset + idx)

    # Each thread in the block computes one output element in the tile
    thread_id = tl.load(tl.arange(0, BLOCK_SIZE))
    # Decompose thread_id into (t_w, t_h, t_d)
    t_w = thread_id % 8
    t_h = (thread_id // 8) % 8
    t_d = (thread_id // 64) % 8

    # Compute output spatial position
    ow = ow_base + t_w
    oh = oh_base + t_h
    od = od_base + t_d

    # Check bounds
    if ow >= out_w or oh >= out_h or od >= out_d:
        return

    # Compute input spatial indices
    iw_start = ow * stride - padding
    ih_start = oh * stride - padding
    id_start = od * stride - padding

    # Initialize output
    acc = tl.zeros((1,), dtype=tl.float32)

    # Compute the convolution
    for ic in range(in_channels):
        for kw in range(k_w):
            for kh in range(k_h):
                for kd in range(k_d):
                    # Compute input index
                    iw = iw_start + kw
                    ih = ih_start + kh
                    id = id_start + kd
                    # Check bounds for input
                    if iw < 0 or iw >= width or ih < 0 or ih >= height or id < 0 or id >= depth:
                        continue
                    # Load input and kernel
                    input_val = tl.load(x_ptr + b * in_channels * width * height * depth +
                                        ic * width * height * depth +
                                        iw * height * depth +
                                        ih * depth +
                                        id)
                    kernel_val = kernel_shared[ic * k_w * k_h * k_d + kw * k_h * k_d + kh * k_d + kd]
                    acc = acc + input_val * kernel_val

    # Store output
    out_offset = b * self.out_channels * out_w * out_h * out_d +
                 oc * out_w * out_h * out_d +
                 ow * out_h * out_d +
                 oh * out_d +
                 od
    tl.store(out_ptr + out_offset, acc)