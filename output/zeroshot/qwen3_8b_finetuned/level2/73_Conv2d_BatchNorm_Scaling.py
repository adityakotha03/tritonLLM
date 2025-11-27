@triton.jit
def bn_scale_kernel(
    input_ptr, mean_ptr, var_ptr, gamma_ptr, beta_ptr, scaling_factor,
    out_ptr, n_elements, BLOCK_SIZE: tl.constexpr,
):
    xoffset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    xmask = xoffset < n_elements
    xindex = xoffset
    xchannel = xindex // (B*H*W)  # Compute channel index
    x = tl.load(input_ptr + xindex, xmask, other=0.0)
    mean = tl.load(mean_ptr + xchannel, xmask, eviction_policy='evict_last')
    var = tl.load(var_ptr + xchannel, xmask, eviction_policy='evict_last')
    gamma = tl.load(gamma_ptr + xchannel, xmask, eviction_policy='evict_last')
    beta = tl.load(beta_ptr + xchannel, xmask, eviction_policy='evict_last')
    # Compute normalized value
    normalized = (x - mean) / tl.sqrt(var + eps)
    # Apply scaling
    scaled = normalized * (gamma * scaling_factor) + (beta * scaling_factor)
    tl.store(out_ptr + xindex, scaled, xmask)