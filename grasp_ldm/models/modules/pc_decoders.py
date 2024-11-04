from typing import Sequence

from torch import nn

from .ext.pvcnn.pvcnn_base import PVCNN, PVCNN2
from .ext.pvcnn.utils import (
    create_mlp_components,
    create_pointnet2_fp_modules,
    create_pointnet2_sa_components,
    create_pointnet_components,
)


class PVCNNInvert(nn.Module):
    def __init__(
        self,
        in_channels=3,
        extra_feature_channels=0,
        width_multiplier=1,
        voxel_resolution_multiplier=1,
        scale_channels=0.25,
        scale_voxel_resolution=0.75,
        num_blocks=...,
        extra_block_channels=None,
    ):
        super().__init__()
        assert extra_feature_channels >= 0

        if not isinstance(num_blocks, Sequence):
            raise TypeError("num_blocks must be of type List or Tuple")
        elif len(num_blocks) != 4:
            raise ValueError(
                "PVCNN is configured with 4 PVConv modules. The num_blocks sequence must of length 4."
            )

        self.in_channels = in_channels + extra_feature_channels
        self.block_spec = self.get_blocks_spec(
            c_mul=scale_channels,
            r_mul=scale_voxel_resolution,
            num_blocks=num_blocks,
            extra_block_channels=extra_block_channels,
        )

        self.out_channels = self.block_spec[-1][0]

        layers, channels_point, concat_channels_point = create_pointnet_components(
            blocks=self.block_spec,
            in_channels=self.in_channels,
            with_se=True,
            normalize=False,
            width_multiplier=width_multiplier,
            voxel_resolution_multiplier=voxel_resolution_multiplier,
        )

        self.point_features = nn.ModuleList(layers)

    def get_blocks_spec(self, c_mul, r_mul, num_blocks, extra_block_channels=None):
        # base_blocks = ((64, 1, 32), (128, 2, 16), (1024, 1, None), (2048, 1, None))
        assert len(num_blocks) == 4 and isinstance(num_blocks, Sequence)

        nb1, nb2, nb3, nb4 = num_blocks

        c1 = int(64 * c_mul)
        c2 = int(512 * c_mul)
        c3 = int(256 * c_mul)
        c4 = int(128 * c_mul)  # 512?

        r1 = int(16 * r_mul)
        r2 = int(32 * r_mul)

        assert c1 % 2 == 0 and c2 % 2 == 0 and c3 % 2 == 0 and c4 % 2 == 0
        assert r1 % 2 == 0 and r2 % 2 == 0

        if extra_block_channels is None:
            blocks = ((c1, nb1, r1), (c2, nb2, r2), (c3, nb3, None), (c4, nb4, None))
        else:
            extra_blocks = ((c, 1, None) for c in extra_block_channels)
            blocks = (
                (c1, nb1, None),
                (c2, nb2, None),
                (c3, nb3, r1),
                (c4, nb4, r2),
                *extra_blocks,
            )

        return blocks

    def forward(self, inputs, cond=None):
        assert (
            inputs.dim() == 3 and inputs.shape[1] >= 3
        ), f"Invalid inputs: {inputs.shape}"
        coords, features = inputs[:, :3, :].contiguous(), inputs

        for i in range(len(self.point_features)):
            features, coords = self.point_features[i]((features, coords))

        return features, coords


class PVCNN2Invert(nn.Module):
    # Set Abstraction Modules
    # (PVconv config, SA config)
    # PV Conv config: (out_channels, num_blocks, voxel_resolution),
    # SA config (num_centers, radius, num_neighbors, out_channels))
    # out_channels from SA config are shared MLP channels
    sa_blocks = [
        ((32, 1, 32), (1024, 0.1, 32, (32, 64))),
        ((64, 2, 16), (256, 0.2, 32, (64, 128))),
        ((128, 1, 8), (64, 0.4, 32, (128, 256))),
        (None, (16, 0.8, 32, (256, 256, 512))),
    ]

    # Feature Propagation Modules
    # (fp config, pvconv config)
    # fp_config: (in_channels, out_channels)
    # PVConv config: (out_channels, num_blocks, voxel_resolution)
    fp_blocks = [
        ((256, 256), (256, 1, 8)),
        ((256, 256), (256, 1, 8)),
        ((256, 128), (128, 2, 16)),
        ((128, 128, 64), (64, 1, 32)),
    ]

    def __init__(
        self,
        in_channels=3,
        extra_feature_channels=0,
        width_multiplier=1,
        voxel_resolution_multiplier=1,
        use_attention=False,
        dropout=0.1,
    ):
        super().__init__()
        self.in_channels = in_channels + extra_feature_channels

        ## Set Abstraction Modules
        (
            sa_layers,
            sa_in_channels,
            channels_sa_features,
            _,
        ) = create_pointnet2_sa_components(
            sa_blocks=self.sa_blocks,
            embed_dim=0,
            extra_feature_channels=extra_feature_channels,
            with_se=True,
            voxelization_normalize=True,
            use_attention=use_attention,
            dropout=dropout,
            width_multiplier=width_multiplier,
            voxel_resolution_multiplier=voxel_resolution_multiplier,
        )
        self.sa_layers = nn.ModuleList(sa_layers)

        # only use extra features in the last fp module
        sa_in_channels[0] = extra_feature_channels

        # Feature Propagation Modules
        fp_layers, channels_fp_features = create_pointnet2_fp_modules(
            fp_blocks=self.fp_blocks,
            in_channels=channels_sa_features,
            sa_in_channels=sa_in_channels,
            with_se=True,
            width_multiplier=width_multiplier,
            voxel_resolution_multiplier=voxel_resolution_multiplier,
        )
        self.fp_layers = nn.ModuleList(fp_layers)

        # Determine out channels from the last PVConv module
        self.out_channels = self.fp_layers[-1][-1].out_channels

    def get_blocks_spec(c_mul, r_mul, num_blocks, extra_block_channels=None):
        return PVCNNInvert.get_blocks_spec(
            c_mul, r_mul, num_blocks, extra_block_channels
        )

    def forward(self, inputs, cond=None):
        assert (
            inputs.dim() == 3 and inputs.shape[1] >= 3
        ), f"Invalid inputs: {inputs.shape}"

        coords, features = inputs[:, :3, :].contiguous(), inputs

        coords_list, in_features_list = [], []

        # Forward pass through the Set Abstraction Modules
        for sa_blocks in self.sa_layers:
            in_features_list.append(features)
            coords_list.append(coords)
            features, coords = sa_blocks((features, coords))

        in_features_list[0] = inputs[:, 3:, :].contiguous()

        # Forward pass through the Feature Propagation Modules
        for fp_idx, fp_blocks in enumerate(self.fp_layers):
            features, coords = fp_blocks(
                (
                    coords_list[-1 - fp_idx],
                    coords,
                    features,
                    in_features_list[-1 - fp_idx],
                )
            )

        return features, coords


class PVCNNDecoder(nn.Module):
    def __init__(
        self,
        in_features=32,
        in_channels=1,
        n_points=1024,
        extra_feature_channels=0,
        scale_channels=0.25,
        scale_voxel_resolution=0.75,
        num_blocks=(1, 1, 1, 1),
        extra_block_channels=None,
        use_global_attention=True,
    ) -> None:
        """PVCNN decoder

        Args:
            in_features (int, optional): Number of input features. Defaults to 3.
            out_features (int, optional): Number of output features. Defaults to 32.
            n_points (int, optional): Number of points in the pointcloud. Defaults to 1024.
            extra_feature_channels (int, optional): Number of extra features to add. Defaults to 0.
            scale_channels (float, optional): Scale factor for the number of channels. Defaults to 0.25.
            scale_voxel_resolution (float, optional): Scale factor for the voxel resolution. Defaults to 0.75.
            num_blocks (tuple, optional): Number of blocks in each scale. Defaults to (1, 1, 1, 1).
            is_conditioned (bool, optional): If the model is conditioned on extra data. Defaults to False.
            cond_dims (int, optional): Conditioning dimensions. Defaults to None.
            extra_block_channels (tuple, optional): Extra channels (i.e. normals) for each block. Defaults to None.
            use_global_attention (bool, optional): If to use global attention after PVCNN. Defaults to True.
        """
        super().__init__()

        self.in_channels = in_channels
        self.in_features = in_features

        self._pvcnn_in_channels = 16

        # Converts squeezed features [B,L] -> [B,N]
        self.in_layer = nn.Linear(
            in_features=self.in_features,
            out_features=n_points,
        )

        # Converts latents [B,1,N] -> [B,3,N]
        self.conv_layer_expand = nn.Sequential(
            nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self._pvcnn_in_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm1d(self._pvcnn_in_channels),
            nn.ReLU(inplace=True),
        )

        self.pvcnn_modules = PVCNNInvert(
            in_channels=self._pvcnn_in_channels,
            extra_feature_channels=extra_feature_channels,
            scale_channels=scale_channels,
            scale_voxel_resolution=scale_voxel_resolution,
            num_blocks=num_blocks,
            extra_block_channels=extra_block_channels,
        )

        self.out_channels = self.pvcnn_modules.out_channels

    def forward(self, out, cond=None):
        """Propagate a batch pointcloud [B,N,3] to batch latents [B,O],
        where O is the number of output features

        Args:
            out (Tensor): pointcloud points xyz: [B,N,3]

        Returns:
            Tensor: [B,O]
        """
        # [B,L] -> [B, 1, L] -> [B, 3, L] -> [B, 3, N]
        out = self.in_layer(out)
        out = out.unsqueeze(1) if out.ndim == 2 else out

        # [B, 1, N] -> [B, 3, N]
        out = self.conv_layer_expand(out)

        # [B, 3, N] -> [B, C, N]
        feats, coords = self.pvcnn_modules(out, cond)

        return feats


class PVCNN2Decoder(PVCNNDecoder):
    def __init__(
        self,
        in_features=3,
        out_features=32,
        n_points=1024,
        extra_feature_channels=0,
        scale_channels=0.25,
        scale_voxel_resolution=0.75,
        num_blocks=(1, 1, 1, 1),
        is_conditioned=False,
        cond_dims=None,
        extra_block_channels=None,
        use_global_attention=True,
        use_local_attention=True,
    ) -> None:
        """PVCNN2 Encoder

        Overrides the PVCNN modules in `PVCNNDecoder` to use the PVCNN2 model.

        Args:
            in_features (int, optional): Number of input features. Defaults to 3.
            out_features (int, optional): Number of output features. Defaults to 32.
            n_points (int, optional): Number of points in the pointcloud. Defaults to 1024.
            extra_feature_channels (int, optional): Number of extra features to add. Defaults to 0.
            scale_channels (float, optional): Scale factor for the number of channels. Defaults to 0.25.
            scale_voxel_resolution (float, optional): Scale factor for the voxel resolution. Defaults to 0.75.
            num_blocks (tuple, optional): Number of blocks in each scale. Defaults to (1, 1, 1, 1).
            is_conditioned (bool, optional): If the model is conditioned on extra data. Defaults to False.
            cond_dims (int, optional): Conditioning dimensions. Defaults to None.
            extra_block_channels (tuple, optional): Extra channels (i.e. normals) for each block. Defaults to None.
            use_global_attention (bool, optional): If to use global attention after PVCNN. Defaults to True.
            use_local_attention (bool, optional): If to use local attention in PVConv modules. Defaults to True.
        """
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            n_points=n_points,
            extra_feature_channels=extra_feature_channels,
            scale_channels=scale_channels,
            scale_voxel_resolution=scale_voxel_resolution,
            num_blocks=num_blocks,
            is_conditioned=is_conditioned,
            cond_dims=cond_dims,
            extra_block_channels=extra_block_channels,
            use_global_attention=use_global_attention,
        )

        # Override pvcnn_modules with PVCNN2
        self.pvcnn_modules = PVCNN2Invert(
            extra_feature_channels=extra_feature_channels,
            scale_channels=scale_channels,
            scale_voxel_resolution=scale_voxel_resolution,
            num_blocks=num_blocks,
            is_conditioned=is_conditioned,
            cond_dims=cond_dims,
            extra_block_channels=extra_block_channels,
            use_attention=use_local_attention,
        )
