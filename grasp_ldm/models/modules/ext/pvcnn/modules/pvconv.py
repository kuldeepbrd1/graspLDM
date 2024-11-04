# Adapted from https://github.com/alexzhou907/PVD/blob/9747265a5f141e5546fd4f862bfa66aa59f1bd33/modules/pvconv.py
import torch.nn as nn

from ....modules import Attention, Swish
from . import functional as F
from .se import SE3d
from .shared_mlp import SharedMLP
from .voxelization import Voxelization

__all__ = ["PVConv"]


class PVConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        resolution,
        use_attention=False,
        dropout=0.1,
        with_se=False,
        with_se_relu=False,
        normalize=True,
        eps=0,
    ):
        """PVConv

        Args:
                in_channels (int): Number of input channels.
                out_channels (int): Number of output channels.
                kernel_size (int): Kernel size of the convolution.
                resolution (int): Voxel resolution.
                attention (bool, optional): Whether to use attention. Defaults to False.
                dropout (float, optional): Dropout rate. Defaults to 0.1.
                with_se (bool, optional): Whether to use SE. Defaults to False.
                with_se_relu (bool, optional): Whether to use ReLU in SE. Defaults to False.
                eps (float, optional): Epsilon for normalization. Defaults to 0.

        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.resolution = resolution

        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)
        voxel_layers = [
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=kernel_size // 2,
            ),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            Swish(),
        ]
        voxel_layers += [nn.Dropout(dropout)] if dropout is not None else []
        voxel_layers += [
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=kernel_size // 2,
            ),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            Attention(out_channels, 8) if use_attention else Swish(),
        ]
        if with_se:
            voxel_layers.append(SE3d(out_channels, use_relu=with_se_relu))
        self.voxel_layers = nn.Sequential(*voxel_layers)
        self.point_features = SharedMLP(in_channels, out_channels)

    def forward(self, inputs):
        features, coords = inputs
        voxel_features, voxel_coords = self.voxelization(features, coords)
        voxel_features = self.voxel_layers(voxel_features)
        voxel_features = F.trilinear_devoxelize(
            voxel_features, voxel_coords, self.resolution, self.training
        )
        fused_features = voxel_features + self.point_features(features)
        return fused_features, coords


class PVConvReLU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        resolution,
        attention=False,
        leak=0.2,
        dropout=0.1,
        with_se=False,
        with_se_relu=False,
        normalize=True,
        eps=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.resolution = resolution

        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)
        voxel_layers = [
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(leak, True),
        ]
        voxel_layers += [nn.Dropout(dropout)] if dropout is not None else []
        voxel_layers += [
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm3d(out_channels),
            Attention(out_channels, 8) if attention else nn.LeakyReLU(leak, True),
        ]
        if with_se:
            voxel_layers.append(SE3d(out_channels, use_relu=with_se_relu))
        self.voxel_layers = nn.Sequential(*voxel_layers)
        self.point_features = SharedMLP(in_channels, out_channels)

    def forward(self, inputs):
        features, coords, temb = inputs
        voxel_features, voxel_coords = self.voxelization(features, coords)
        voxel_features = self.voxel_layers(voxel_features)
        voxel_features = F.trilinear_devoxelize(
            voxel_features, voxel_coords, self.resolution, self.training
        )
        fused_features = voxel_features + self.point_features(features)
        return fused_features, coords, temb
