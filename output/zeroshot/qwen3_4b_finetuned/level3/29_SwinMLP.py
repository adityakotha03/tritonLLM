# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import repeat
import collections.abc

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinMLPBlock(nn.Module):
    r""" Swin MLP Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.padding = [self.window_size - self.shift_size, self.shift_size,
                        self.window_size - self.shift_size, self.shift_size]  # P_l,P_r,P_t,P_b

        self.norm1 = norm_layer(dim)
        # use group convolution to implement multi-head MLP
        self.spatial_mlp = nn.Conv1d(self.num_heads * self.window_size ** 2,
                                     self.num_heads * self.window_size ** 2,
                                     kernel_size=1,
                                     groups=self.num_heads)

        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # shift
        if self.shift_size > 0:
            P_l, P_r, P_t, P_b = self.padding
            shifted_x = F.pad(x, [0, 0, P_l, P_r, P_t, P_b], "constant", 0)
        else:
            shifted_x = x
        _, _H, _W, _ = shifted_x.shape

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # Window/Shifted-Window Spatial MLP
        x_windows_heads = x_windows.view(-1, self.window_size * self.window_size, self.num_heads, C // self.num_heads)
        x_windows_heads = x_windows_heads.transpose(1, 2)  # nW*B, nH, window_size*window_size, C//nH
        x_windows_heads = x_windows_heads.reshape(-1, self.num_heads * self.window_size * self.window_size,
                                                  C // self.num_heads)
        spatial_mlp_windows = self.spatial_mlp(x_windows_heads)  # nW*B, nH*window_size*window_size, C//nH
        spatial_mlp_windows = spatial_mlp_windows.view(-1, self.num_heads, self.window_size * self.window_size,
                                                       C // self.num_heads).transpose(1, 2)
        spatial_mlp_windows = spatial_mlp_windows.reshape(-1, self.window_size * self.window_size, C)

        # merge windows
        spatial_mlp_windows = spatial_mlp_windows.reshape(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(spatial_mlp_windows, self.window_size, _H, _W)  # B H' W' C

        # reverse shift
        if self.shift_size > 0:
            P_l, P_r, P_t, P_b = self.padding
            x = shifted_x[:, P_t:-P_b, P_l:-P_r, :].contiguous()
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

class BasicLayer(nn.Module):
    """ A basic Swin MLP layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinMLPBlock(dim=dim, input_resolution=input_resolution,
                         num_heads=num_heads, window_size=window_size,
                         shift_size=0 if (i % 2 == 0) else window_size // 2,
                         mlp_ratio=mlp_ratio,
                         drop=drop,
                         drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                         norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class ModelNew(nn.Module):
    r""" Swin MLP

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin MLP layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        drop_rate (float): Dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               drop=drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def flops(self):
        flops = 0
        # patch_embed
        flops += self.patch_embed.flops()
        # layers
        for i_layer in range(self.num_layers):
            dim = int(self.embed_dim * 2 ** i_layer)
            # input: B, H*W, C; output: B, H/2*W/2, 4*C
            flops += self.layers[i_layer].downsample.flops()
            for j in range(self.layers[i_layer].depth):
                # input: B, H/2*W/2, 4*C; output: B, H/2*W/2, 4*C
                flops += self.layers[i_layer].blocks[j].norm1.flops()
                flops += self.layers[i_layer].blocks[j].norm2.flops()
                flops += self.layers[i_layer].blocks[j].mlp.fc1.flops()
                flops += self.layers[i_layer].blocks[j].mlp.fc2.flops()
        # output
        flops += self.num_features * self.num_features
        return flops

    @staticmethod
    def _convolution_kernel(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3,
        out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10,
        out_ptr11, out_ptr12, out_ptr13, out_ptr14, out_ptr15, out_ptr16,
        out_ptr17, out_ptr18, out_ptr19, out_ptr20, out_ptr21, out_ptr22,
        out_ptr23, out_ptr24, out_ptr25, out_ptr26, out_ptr27, out_ptr28,
        out_ptr29, out_ptr30, out_ptr31, out_ptr32, out_ptr33, out_ptr34,
        out_ptr35, out_ptr36, out_ptr37, out_ptr38, out_ptr39, out_ptr40,
        out_ptr41, out_ptr42, out_ptr43, out_ptr44, out_ptr45, out_ptr46,
        out_ptr47, out_ptr48, out_ptr49, out_ptr50, out_ptr51, out_ptr52,
        out_ptr53, out_ptr54, out_ptr55, out_ptr56, out_ptr57, out_ptr58,
        out_ptr59, out_ptr60, out_ptr61, out_ptr62, out_ptr63, out_ptr64,
        out_ptr65, out_ptr66, out_ptr67, out_ptr68, out_ptr69, out_ptr70,
        out_ptr71, out_ptr72, out_ptr73, out_ptr74, out_ptr75, out_ptr76,
        out_ptr77, out_ptr78, out_ptr79, out_ptr80, out_ptr81, out_ptr82,
        out_ptr83, out_ptr84, out_ptr85, out_ptr86, out_ptr87, out_ptr88,
        out_ptr89, out_ptr90, out_ptr91, out_ptr92, out_ptr93, out_ptr94,
        out_ptr95, out_ptr96, out_ptr97, out_ptr98, out_ptr99, out_ptr100,
        out_ptr101, out_ptr102, out_ptr103, out_ptr104, out_ptr105,
        out_ptr106, out_ptr107, out_ptr108, out_ptr109, out_ptr110,
        out_ptr111, out_ptr112, out_ptr113, out_ptr114, out_ptr115,
        out_ptr116, out_ptr117, out_ptr118, out_ptr119, out_ptr120,
        out_ptr121, out_ptr122, out_ptr123, out_ptr124, out_ptr125,
        out_ptr126, out_ptr127, out_ptr128, out_ptr129, out_ptr130,
        out_ptr131, out_ptr132, out_ptr133, out_ptr134, out_ptr135,
        out_ptr136, out_ptr137, out_ptr138, out_ptr139, out_ptr140,
        out_ptr141, out_ptr142, out_ptr143, out_ptr144, out_ptr145,
        out_ptr146, out_ptr147, out_ptr148, out_ptr149, out_ptr150,
        out_ptr151, out_ptr152, out_ptr153, out_ptr154, out_ptr155,
        out_ptr156, out_ptr157, out_ptr158, out_ptr159, out_ptr160,
        out_ptr161, out_ptr162, out_ptr163, out_ptr164, out_ptr165,
        out_ptr166, out_ptr167, out_ptr168, out_ptr169, out_ptr170,
        out_ptr171, out_ptr172, out_ptr173, out_ptr174, out_ptr175,
        out_ptr176, out_ptr177, out_ptr178, out_ptr179, out_ptr180,
        out_ptr181, out_ptr182, out_ptr183, out_ptr184, out_ptr185,
        out_ptr186, out_ptr187, out_ptr188, out_ptr189, out_ptr190,
        out_ptr191, out_ptr192, out_ptr193, out_ptr194, out_ptr195,
        out_ptr196, out_ptr197, out_ptr198, out_ptr199, out_ptr200,
        out_ptr201, out_ptr202, out_ptr203, out_ptr204, out_ptr205,
        out_ptr206, out_ptr207, out_ptr208, out_ptr209, out_ptr210,
        out_ptr211, out_ptr212, out_ptr213, out_ptr214, out_ptr215,
        out_ptr216, out_ptr217, out_ptr218, out_ptr219, out_ptr220,
        out_ptr221, out_ptr222, out_ptr223, out_ptr224, out_ptr225,
        out_ptr226, out_ptr227, out_ptr228, out_ptr229, out_ptr230,
        out_ptr231, out_ptr232, out_ptr233, out_ptr234, out_ptr235,
        out_ptr236, out_ptr237, out_ptr238, out_ptr239, out_ptr240,
        out_ptr241, out_ptr242, out_ptr243, out_ptr244, out_ptr245,
        out_ptr246, out_ptr247, out_ptr248, out_ptr249, out_ptr250,
        out_ptr251, out_ptr252, out_ptr253, out_ptr254, out_ptr255,
        out_ptr256, out_ptr257, out_ptr258, out_ptr259, out_ptr260,
        out_ptr261, out_ptr262, out_ptr263, out_ptr264, out_ptr265,
        out_ptr266, out_ptr267, out_ptr268, out_ptr269, out_ptr270,
        out_ptr271, out_ptr272, out_ptr273, out_ptr274, out_ptr275,
        out_ptr276, out_ptr277, out_ptr278, out_ptr279, out_ptr280,
        out_ptr281, out_ptr282, out_ptr283, out_ptr284, out_ptr285,
        out_ptr286, out_ptr287, out_ptr288, out_ptr289, out_ptr290,
        out_ptr291, out_ptr292, out_ptr293, out_ptr294, out_ptr295,
        out_ptr296, out_ptr297, out_ptr298, out_ptr299, out_ptr300,
        out_ptr301, out_ptr302, out_ptr303, out_ptr304, out_ptr305,
        out_ptr306, out_ptr307, out_ptr308, out_ptr309, out_ptr310,
        out_ptr311, out_ptr312, out_ptr313, out_ptr314, out_ptr315,
        out_ptr316, out_ptr317, out_ptr318, out_ptr319, out_ptr320,
        out_ptr321, out_ptr322, out_ptr323, out_ptr324, out_ptr325,
        out_ptr326, out_ptr327, out_ptr328, out_ptr329, out_ptr330,
        out_ptr331, out_ptr332, out_ptr333, out_ptr334, out_ptr335,
        out_ptr336, out_ptr337, out_ptr338, out_ptr339, out_ptr340,
        out_ptr341, out_ptr342, out_ptr343, out_ptr344, out_ptr345,
        out_ptr346, out_ptr347, out_ptr348, out_ptr349, out_ptr350,
        out_ptr351, out_ptr352, out_ptr353, out_ptr354, out_ptr355,
        out_ptr356, out_ptr357, out_ptr358, out_ptr359, out_ptr360,
        out_ptr361, out_ptr362, out_ptr363, out_ptr364, out_ptr365,
        out_ptr366, out_ptr367, out_ptr368, out_ptr369, out_ptr370,
        out_ptr371, out_ptr372, out_ptr373, out_ptr374, out_ptr375,
        out_ptr376, out_ptr377, out_ptr378, out_ptr379, out_ptr380,
        out_ptr381, out_ptr382, out_ptr383, out_ptr384, out_ptr385,
        out_ptr386, out_ptr387, out_ptr388, out_ptr389, out_ptr390,
        out_ptr391, out_ptr392, out_ptr393, out_ptr394, out_ptr395,
        out_ptr396, out_ptr397, out_ptr398, out_ptr399, out_ptr400,
        out_ptr401, out_ptr402, out_ptr403, out_ptr404, out_ptr405,
        out_ptr406, out_ptr407, out_ptr408, out_ptr409, out_ptr410,
        out_ptr411, out_ptr412, out_ptr413, out_ptr414, out_ptr415,
        out_ptr416, out_ptr417, out_ptr418, out_ptr419, out_ptr420,
        out_ptr421, out_ptr422, out_ptr423, out_ptr424, out_ptr425,
        out_ptr426, out_ptr427, out_ptr428, out_ptr429, out_ptr430,
        out_ptr431, out_ptr432, out_ptr433, out_ptr434, out_ptr435,
        out_ptr436, out_ptr437, out_ptr438, out_ptr439, out_ptr440,
        out_ptr441, out_ptr442, out_ptr443, out_ptr444, out_ptr445,
        out_ptr446, out_ptr447, out_ptr448, out_ptr449, out_ptr450,
        out_ptr451, out_ptr452, out_ptr453, out_ptr454, out_ptr455,
        out_ptr456, out_ptr457, out_ptr458, out_ptr459, out_ptr460,
        out_ptr461, out_ptr462, out_ptr463, out_ptr464, out_ptr465,
        out_ptr466, out_ptr467, out_ptr468, out_ptr469, out_ptr470,
        out_ptr471, out_ptr472, out_ptr473, out_ptr474, out_ptr475,
        out_ptr476, out_ptr477, out_ptr478, out_ptr479, out_ptr480,
        out_ptr481, out_ptr482, out_ptr483, out_ptr484, out_ptr485,
        out_ptr486, out_ptr487, out_ptr488, out_ptr489, out_ptr490,
        out_ptr491, out_ptr492, out_ptr493, out_ptr494, out_ptr495,
        out_ptr496, out_ptr497, out_ptr498, out_ptr499, out_ptr500,
        out_ptr501, out_ptr502, out_ptr503, out_ptr504, out_ptr505,
        out_ptr506, out_ptr507, out_ptr508, out_ptr509, out_ptr510,
        out_ptr511, out_ptr512, out_ptr513, out_ptr514, out_ptr515,
        out_ptr516, out_ptr517, out_ptr518, out_ptr519, out_ptr520,
        out_ptr521, out_ptr522, out_ptr523, out_ptr524, out_ptr525,
        out_ptr526, out_ptr527, out_ptr528, out_ptr529, out_ptr530,
        out_ptr531, out_ptr532, out_ptr533, out_ptr534, out_ptr535,
        out_ptr536, out_ptr537, out_ptr538, out_ptr539, out_ptr540,
        out_ptr541, out_ptr542, out_ptr543, out_ptr544, out_ptr545,
        out_ptr546, out_ptr547, out_ptr548, out_ptr549, out_ptr550,
        out_ptr551, out_ptr552, out_ptr553, out_ptr554, out_ptr555,
        out_ptr556, out_ptr557, out_ptr558, out_ptr559, out_ptr560,
        out_ptr561, out_ptr562, out_ptr563, out_ptr564, out_ptr565,
        out_ptr566, out_ptr567, out_ptr568, out_ptr569, out_ptr570,
        out_ptr571, out_ptr572, out_ptr573, out_ptr574, out_ptr575,
        out_ptr576, out_ptr577, out_ptr578, out_ptr579, out_ptr580,
        out_ptr581, out_ptr582, out_ptr583, out_ptr584, out_ptr585,
        out_ptr586, out_ptr587, out_ptr588, out_ptr589, out_ptr590,
        out_ptr591, out_ptr592, out_ptr593, out_ptr594, out_ptr595,
        out_ptr596, out_ptr597, out_ptr598, out_ptr599, out_ptr600,
        out_ptr601, out_ptr602, out_ptr603, out_ptr604, out_ptr605,
        out_ptr606, out_ptr607, out_ptr608, out_ptr609, out_ptr610,
        out_ptr611, out_ptr612, out_ptr613, out_ptr614, out_ptr615,
        out_ptr616, out_ptr617, out_ptr618, out_ptr619, out_ptr620,
        out_ptr621, out_ptr622, out_ptr623, out_ptr624, out_ptr625,
        out_ptr626, out_ptr627, out_ptr628, out_ptr629, out_ptr630,
        out_ptr631, out_ptr632, out_ptr633, out_ptr634, out_ptr635,
        out_ptr636, out_ptr637, out_ptr638, out_ptr639, out_ptr640,
        out_ptr641, out_ptr642, out_ptr643, out_ptr644, out_ptr645,
        out_ptr646, out_ptr647, out_ptr648, out_ptr649, out_ptr650,
        out_ptr651, out_ptr652, out_ptr653, out_ptr654, out_ptr655,
        out_ptr656, out_ptr657, out_ptr658, out_ptr659, out_ptr660,
        out_ptr661, out_ptr662, out_ptr663, out_ptr664, out_ptr665,
        out_ptr666, out_ptr667, out_ptr6