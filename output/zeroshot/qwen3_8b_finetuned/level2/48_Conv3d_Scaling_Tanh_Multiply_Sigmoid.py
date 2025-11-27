scaling_factor = 2
bias_shape = (out_channels, 1, 1, 1)
self.scaling_factor = nn.Parameter(torch.randn(bias_shape))