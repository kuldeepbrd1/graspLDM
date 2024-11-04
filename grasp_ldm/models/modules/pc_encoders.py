import torch
from torch import nn

from .ext.pvcnn.pvcnn_base import PVCNN, PVCNN2
from .modules import Attention, FCLayers


class PVCNNEncoder(nn.Module):
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
        use_global_attention=False,
        out_channels=1,
        load_from_ckpt_path=None,
    ) -> None:
        """PVCNN Encoder

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
            out_channels (int, optional): Number of output channels. Defaults to 1.
            load_from_ckpt_path (str, optional): Path to checkpoint to load encoder weights. Defaults to None.
        """
        super().__init__()

        self.pvcnn_modules = PVCNN(
            extra_feature_channels=extra_feature_channels,
            scale_channels=scale_channels,
            scale_voxel_resolution=scale_voxel_resolution,
            num_blocks=num_blocks,
            is_conditioned=is_conditioned,
            cond_dims=cond_dims,
            extra_block_channels=extra_block_channels,
        )

        self.in_features = in_features
        self.out_features = out_features

        # To reduce the exploding params requirements of the global attention, reduce channels of the output
        _conv_downscale_channels = int(self.pvcnn_modules.out_channels / 2)

        self.conv_downscale = nn.Conv1d(
            in_channels=self.pvcnn_modules.out_channels,
            out_channels=_conv_downscale_channels,
            kernel_size=1,
        )
        self.global_attention = (
            Attention(in_ch=_conv_downscale_channels, num_groups=8, D=1)
            if use_global_attention
            else None
        )

        # Converts out pvcnn features [B,C,N] -> [B,C_out, out_feats]
        self.out_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=_conv_downscale_channels,
                out_channels=out_channels,
                kernel_size=1,
            ),
            nn.Linear(
                in_features=n_points,
                out_features=self.out_features,
            ),
        )

        if load_from_ckpt_path is not None:
            self.load_ckpt_and_freeze(load_from_ckpt_path)

    def forward(self, out, cond=None):
        """Propagate a batch pointcloud [B,N,3] to batch latents [B,O],
        where O is the number of output features

        Args:
            out (Tensor): pointcloud points xyz: [B,N,3]

        Returns:
            Tensor: [B,O]
        """
        # [B, N, 3] -> [B, 3, N]
        out = torch.transpose(out, 1, 2).contiguous()

        # [B, 3, N] -> [B, C, N]
        out = self.pvcnn_modules(out, cond=cond)

        # Intermediate downscaling: [B, C, N] -> [B, C, Ni]
        out = self.conv_downscale(out)

        if self.global_attention is not None:
            # [B, C, Ni] -> [B, C, Ni]
            out = self.global_attention(out)

        # [B, C, Ni] -> [B, C_out , out_features]
        out = self.out_layer(out)

        out = out.squeeze(1) if out.shape[-2] == 1 else out

        return out

    def load_ckpt_and_freeze(self, ckpt_path, fine_tune=False):
        """Load a checkpoint and freeze the weights of the PVCNN modules

        Args:
            ckpt_path (str): Path to checkpoint
        """
        ckpt = torch.load(ckpt_path)
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        self.load_state_dict(state_dict)

        if fine_tune:
            freeze_modules = [self.pvcnn_modules, self.conv_downscale] + (
                [self.global_attention] if self.global_attention else []
            )
        else:
            freeze_modules = self.modules()

        for mod in freeze_modules:
            for param in mod.parameters():
                param.requires_grad = False


class PVCNN2Encoder(PVCNNEncoder):
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

        Overrides the PVCNN modules in `PVCNNEncoder` to use the PVCNN2 model.

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
        self.pvcnn_modules = PVCNN2(
            extra_feature_channels=extra_feature_channels,
            scale_channels=scale_channels,
            scale_voxel_resolution=scale_voxel_resolution,
            num_blocks=num_blocks,
            is_conditioned=is_conditioned,
            cond_dims=cond_dims,
            extra_block_channels=extra_block_channels,
            use_attention=use_local_attention,
        )


class PointNet2Encoder(nn.Module):
    def __init__(
        self,
        model_scale: int,
        pointnet_radius=0.02,
        pointnet_nclusters=128,
        in_features=3,
        out_features=6,
        is_normal_channel=False,
    ) -> None:
        """_summary_

        Args:
            model_scale (int): scale factor for scaling model size
            pointnet_radius (float, optional): pointnet2 neighbourhood grouping radius . Defaults to 0.02.
            pointnet_nclusters (int, optional): number of sampling clusters. Defaults to 128.
            in_features (int, optional): input channels/dimensions: x.shape[-1]. Defaults to 19.
            out_features (int, optional): Encoder output dims. Defaults to 6.
            is_normal_channel (bool, optional): input contains a normal channel. Defaults to False.
        """
        super().__init__()

        self.pointnet2_module = PointNet2Base(
            model_scale,
            pointnet_radius,
            pointnet_nclusters,
            in_features,
            is_normal_channel,
        )

        self.in_features = in_features
        self.out_features = out_features

        self._fc_layer_specs = [256, out_features]
        self.fc_layers = FCLayers(
            in_features=self.pointnet2_module.out_features,
            layer_outs_specs=self._fc_layer_specs,
        )

    def forward(self, xyz):
        _, xyz_features = self.pointnet2_module(xyz)
        out = self.fc_layers(xyz_features.squeeze(-1))
        return out


if __name__ == "__main__":
    import time

    from ptflops import get_model_complexity_info

    test = 2
    b = 128
    x_num = 1024
    x_dim = 3

    x_in = torch.randn([b, x_num, x_dim], device="cuda:0")
    expect_out_features = 6

    if test == 1:
        ## Pointnet2 Encoder
        model = PointNet2Encoder(
            model_scale=1, in_features=3, out_features=expect_out_features
        )
    elif test == 2:
        ## PVCNN Encoder
        model = PVCNNEncoder()
    else:
        raise ValueError

    model.to(device="cuda:0")

    start_t = time.time()
    out = model(x_in)
    dt = time.time() - start_t

    assert out.shape == torch.Size([b, expect_out_features])

    macs, params = get_model_complexity_info(
        model,
        (x_num, x_dim),
        as_strings=True,
        print_per_layer_stat=True,
        verbose=True,
    )
    print("{:<30}  {:<8}".format("Computational complexity: ", macs))
    print("{:<30}  {:<8}".format("Number of parameters: ", params))
    print(f"Forward pass: {dt} s")
