from typing import Sequence

import torch
import torch.nn as nn
from einops import rearrange

from .utils import (
    create_mlp_components,
    create_pointnet2_fp_modules,
    create_pointnet2_sa_components,
    create_pointnet_components,
)


class PVCNN(nn.Module):
    # Blocks design, C100 original PVCNN configuration, others are

    def __init__(
        self,
        in_channels=3,
        extra_feature_channels=0,
        scale_channels=0.25,
        scale_voxel_resolution=0.75,
        num_blocks=(1, 2, 1, 1),
        is_conditioned=False,
        cond_dims=None,
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
            width_multiplier=1,
            voxel_resolution_multiplier=1,
        )

        self.point_features = nn.ModuleList(layers)

        self.is_conditioned = is_conditioned

        if is_conditioned:
            assert None not in (cond_dims,), "Conditioning dims was not set "
            self.is_conditioned = True

            _channel_specs = [self.in_channels] + [spec[0] for spec in self.block_spec]

            # FiLM style embedding: https://arxiv.org/pdf/1709.07871.pdf
            # This will work only for 2D  tensors atm: [ B, D]
            # [B, D] -> [B, 2*C_block], C_block: block channels out
            self.emb_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(cond_dims, 2 * block_channels),
                        nn.SiLU(),
                        nn.Linear(2 * block_channels, 2 * block_channels),
                    )
                    for block_channels in _channel_specs[:-1]
                ]
            )

    def get_blocks_spec(self, c_mul, r_mul, num_blocks, extra_block_channels=None):
        # TODO: This is redundant with create_pointnet_components accepting multipliers
        # base_blocks = ((64, 1, 32), (128, 2, 16), (1024, 1, None), (2048, 1, None))
        assert len(num_blocks) == 4 and isinstance(num_blocks, Sequence)

        nb1, nb2, nb3, nb4 = num_blocks

        c1 = int(64 * c_mul)
        c2 = int(128 * c_mul)
        c3 = int(1024 * c_mul)
        c4 = int(2048 * c_mul)  # 512?

        r1 = int(32 * r_mul)
        r2 = int(16 * r_mul)

        assert c1 % 2 == 0 and c2 % 2 == 0 and c3 % 2 == 0 and c4 % 2 == 0
        assert r1 % 2 == 0 and r2 % 2 == 0

        if extra_block_channels is None:
            blocks = ((c1, nb1, r1), (c2, nb2, r2), (c3, nb3, None), (c4, nb4, None))
        else:
            extra_blocks = ((c, 1, None) for c in extra_block_channels)
            blocks = (
                (c1, nb1, r1),
                (c2, nb2, r2),
                (c3, nb3, None),
                (c4, nb4, None),
                *extra_blocks,
            )

        return blocks

    def forward(self, inputs, *, cond=None):
        """Forward pass through the PVCNN network

        Args:
            inputs (Tensor): [B, 3+C, N] Point cloud with extra features
                        The first 3 channels are the point coordinates.
                        The rest are extra features.
            cond (Tensor): [B, D] Conditioning tensor

        Returns:
            Tensor: [B, C, N] Features
        """

        # Feature tensor with points and extra features: [B, 3+C, N]
        features = inputs[:, : self.in_channels, :]

        # Point Coordinates: [B, 3, N]
        coords = features[:, :3, :]

        for i in range(len(self.point_features)):
            if self.is_conditioned:
                features = self.condition_features(
                    emb_module=self.emb_layers[i], features=features, cond=cond
                )
            features, _ = self.point_features[i]((features, coords))

        return features

    def condition_features(self, emb_module, features, cond):
        """Condition the features with the conditioning tensor

        Args:
            emb_module (nn.Module): Embedding module
            features (Tensor): [B, C, N] Features
            cond (Tensor): [B, D] Conditioning tensor

        Returns:
            Tensor: [B, C, N] Conditioned features
        """
        assert (
            cond is not None
        ), "Initialized conditioning layers, but no conditioning was provided"

        # [B,D] -> [B, 2*C_block]
        emb = emb_module(cond)

        # [B, 2*C_block] -> [ B, 2*C_block, 1]
        emb = rearrange(emb, "b c -> b c 1")

        # [ B, 2*C_block, 1] -> ([B, C_block,1], [B, C_block, 1])
        scale, shift = emb.chunk(2, dim=1)

        features = features * (1 + scale) + shift
        return features

    def _forward_debug(self, inputs):
        features = inputs[:, : self.in_features, :]

        coords = features[:, :3, :]
        out_features_list = []
        for i in range(len(self.point_features)):
            features, _ = self.point_features[i]((features, coords))
            out_features_list.append(features)
        return out_features_list


class PVCNN2(nn.Module):
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

    def forward(self, inputs, cond=None):
        if isinstance(inputs, dict):
            inputs = inputs["features"]

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

        return features


if __name__ == "__main__":
    import time

    import torch
    from ptflops import get_model_complexity_info

    model = PVCNN(
        extra_feature_channels=0,
        scale_channels=0.25,
        scale_voxel_resolution=0.75,
        num_blocks=(1, 1, 1, 1),
    )
    model.to(device="cuda:0")
    b = 128
    n_pts = 1024
    n_channel = 3

    x_in = torch.randn([b, n_channel, n_pts], device="cuda:0")
    # x_features =
    start = time.time()
    out = model(x_in)
    print(f"forward: {time.time() -start}")

    macs, params = get_model_complexity_info(
        model,
        (n_channel, n_pts),
        as_strings=True,
        print_per_layer_stat=True,
        verbose=True,
    )
    print("{:<30}  {:<8}".format("Computational complexity: ", macs))
    print("{:<30}  {:<8}".format("Number of parameters: ", params))

    # print(f"Out shape: ({features_out.shape}")
