import warnings
from typing import Sequence, Tuple, Union

import numpy as np
import torch
from addict import Dict
from torch import Tensor, nn

from grasp_ldm.utils.config import Config

from grasp_ldm.losses.builder import build_loss_from_cfg
from .modules.base_network import BaseGraspSampler
from .modules.pc_encoders import PVCNN2Encoder, PVCNNEncoder
from .modules.resnets import ResNet1D, Unet1D


class GraspCVAE(BaseGraspSampler):
    def __init__(
        self,
        grasp_latent_size: int,
        pc_latent_size: int,
        grasp_encoder_config: dict,
        pc_encoder_config: dict,
        decoder_config: dict,
        loss_config: dict,
        intermediate_feature_resolution: int = 16,
        num_output_qualities: Union[int, None] = None,
    ) -> None:
        """Grasp VAE

        Clarify features channels and batch
        Args:
            latent_dims (int): _description_
            encoder_config (dict): _description_
            decoder_config (dict): _description_
            loss_config (dict): _description_
            model_scale (int, optional): model scale factor that scales layer channels across models. Defaults to 1.

        """
        ## Using super in the following might fail if the base classes have arguments
        super().__init__()

        # Encoders latent feature dims
        self.grasp_latent_size = grasp_latent_size
        self.pc_latent_size = pc_latent_size

        ## Losses
        self.loss_config = loss_config

        assert (
            "reconstruction_loss" in loss_config and "latent_loss" in loss_config
        ), "`reconstruction_loss` and `latent_loss` must be specified in loss_config"

        # TODO: Build only if in train mode
        self.reconstruction_loss = build_loss_from_cfg(loss_config.reconstruction_loss)
        self.latent_loss = build_loss_from_cfg(loss_config.latent_loss)

        # Optional losses
        self.classification_loss = (
            build_loss_from_cfg(loss_config.classification_loss)
            if hasattr(loss_config, "classification_loss")
            else None
        )

        self.quality_loss = (
            build_loss_from_cfg(loss_config.quality_loss)
            if hasattr(loss_config, "quality_loss")
            else None
        )

        ## Sub-networks
        self.encoder = PcConditionedGraspEncoder(
            pc_encoder_config=pc_encoder_config,
            grasp_encoder_config=grasp_encoder_config,
            pc_latent_size=self.pc_latent_size,
            grasp_latent_size=self.grasp_latent_size,
        )

        self.bottleneck = VAEBottleneck(
            in_features=self.encoder.out_features,
            latent_size=self.grasp_latent_size,
        )

        # Learn qualities?
        self.num_output_qualities = num_output_qualities
        self.decoder = ConditionalGraspPoseDecoder(
            in_features=grasp_latent_size,
            config=decoder_config,
            num_output_qualities=self.num_output_qualities,
            feature_resolution=intermediate_feature_resolution,
        )

        self.out_features = self.decoder.out_features

    @property
    def latent_losses(self):
        """Get latent losses for annealing etc"""
        return [self.latent_loss]

    @property
    def use_grasp_qualities(self) -> bool:
        return True if self.decoder._use_qualities else False

    def encode(self, xyz: Tensor, grasp: Tensor) -> Sequence:
        """Encode inputs

        Args:
            input (Tensor): input tensor

        Returns:
            Sequence: [mu, logvar] - mean and log variance
        """
        z_grasp, z_pc = self.encoder(xyz, grasp)

        mu, logvar = self.bottleneck(z_grasp.squeeze(dim=-2))
        z_grasp = self.bottleneck.reparameterize(mu, logvar)
        return (mu, logvar, z_grasp), (None, None, z_pc)

    def forward(
        self, xyz: Tensor, grasp: Tensor, compute_loss: bool = True, **kwargs
    ) -> Union[Tensor, Tuple[Tensor, Dict]]:
        """Forward

        Args:
            xyz (Tensor): batch point clouds
            grasp (Tensor): batch grasp [B,6], where dim 1 is [t(3) mrp(3)]
            compute_loss (bool, optional): Compute loss for training? Defaults to True.

        Returns:
            Tensor: _description_
        """
        # Encode to latent distribution
        (mu_h, logvar_h, z_h), (_, _, z_pc) = self.encode(xyz, grasp)

        # out: [tmrp(B,6), class_logits(B,2), Qualities(B,4)]
        out = self.decoder(z_h, z_pc)

        if compute_loss:
            loss_dict = self.loss_fn(
                x_in=grasp,
                x_out=torch.concat(out, -1),
                grasp_mu_logvar=(mu_h, logvar_h),
                **kwargs,
            )
            return out, loss_dict
        else:
            return out

    def loss_fn(
        self,
        x_in: Tensor,
        x_out: Tensor,
        grasp_mu_logvar: Tuple[Tensor, Tensor],
        **kwargs,
    ) -> Dict:
        """Loss fn
            B: Batch size
            D_g: Input Grasp representation dimensions
            D_zg: Grasp encoder latent dimensions

        Args:
            x_in (Tensor): [B, D_g] ground truth grasp pose paramters
            x_out (Tensor): [B, D_g] predicted grasp pose parameters
            grasp_mu_logvar (tuple): encoded mu logvar from grasp encoder
                                    (mu:[B, D_zg], logvar:[B, D_zg])

        Returns:
            dict: loss dictionary
        """
        loss_dict = Dict(loss=0)

        grasps_in = x_in[..., :6]
        grasps_out = x_out[..., :6]

        # Reconstruction loss
        loss_dict.reconstruction_loss = self.reconstruction_loss(
            grasps_in.squeeze(), grasps_out.squeeze(), **kwargs
        )

        # Latent loss
        (
            loss_dict.latent_loss,
            loss_dict._unweighted_kld,
        ) = self.latent_loss(*grasp_mu_logvar, return_unweighted=True, **kwargs)

        # Classification loss
        if self.classification_loss is not None:
            cls_in = x_in[..., 6]
            cls_out = x_out[..., 6]

            loss_dict.classification_loss = self.classification_loss(
                output=cls_out.squeeze(), targets=cls_in.squeeze(), **kwargs
            )

        # Quality loss
        if self.quality_loss is not None:
            quals_in = x_in[..., 7:]
            quals_out = x_out[..., 7:]

            loss_dict.quality_loss = self.quality_loss(
                quals_in.squeeze(), quals_out.squeeze(), **kwargs
            )

        # Do not add unweighted KL loss- only for monitoring
        loss_dict.loss += (
            loss_dict.latent_loss
            + loss_dict.reconstruction_loss
            + (loss_dict.quality_loss if self.quality_loss is not None else 0)
            + (
                loss_dict.classification_loss
                if self.classification_loss is not None
                else 0
            )
        )

        return loss_dict

    def encode_pc(self, xyz: Tensor) -> Tensor:
        """Helper function to encode pointclouds"""
        return self.encoder.encode_pc(xyz)

    def sample_grasp_latent(self, batch_size: int, device: int) -> Tensor:
        """Helper function to sample grasp latent codes"""
        return torch.randn(batch_size, self.grasp_latent_size).to(device)

    def generate_grasps(self, xyz: Tensor, num_grasps: int = 10) -> Tensor:
        """Generates grasps for a given pointcloud

        Args:batch_size_pc
            xyz (Tensor): [B, N, 3] pointcloud
            num_grasps (int, optional): Number of grasps to generate. Defaults to 10.

        Returns:
            Tensor: [B, num_grasps, 6] grasp pose
        """
        assert (
            xyz.ndim == 3
        ), f"Input pointcloud should be  3-dim tensor of shape [B, N, 3]. Found a {xyz.ndim} dimensional tensor."

        # num objects/pcs
        num_pcs = xyz.shape[0]

        # Encode input pointcloud
        z_pc_cond = self.encode_pc(xyz)

        # Repeat pointcloud conditioning latent code for each grasp
        z_pc_cond = z_pc_cond.repeat_interleave(num_grasps, dim=0)

        # Conditional Grasp latent
        z_h = torch.randn(num_pcs * num_grasps, self.grasp_latent_size).to(xyz.device)

        # Decode grasp latent code
        out = self.decoder(z_h, z_pc_cond)

        return out


class PcConditionedGraspEncoder(nn.Module):
    """
    Encodes grasp input into a pc-conditional/unconditional latent code
    """

    PC_ENCODERS = {
        "PVCNNEncoder": PVCNNEncoder,
        "PVCNN2Encoder": PVCNN2Encoder,
    }  # "PointNet2Encoder": PointNet2Encoder, }

    def __init__(
        self,
        pc_encoder_config: Config,
        grasp_encoder_config: Config,
        pc_latent_size: int = 64,
        grasp_latent_size: int = 4,
    ) -> None:
        """Initialize GraspVAEEncoders object

        Args:
                grasp_input_dims (int): dimensions of chose grasp pose vector representation
                latent_dims (int): Final latent vector dimensions
                pc_latent_dims (int): Pc encoded latent vector dimensions from pc_encoder
                pc_encoder_model_args (dict): pointcloud encoder model arguments
                pc_encoder_arch_type (str, optional): architecture. Defaults to "pointnet2".
                grasp_pose_encoder_outs (Sequence, optional): Grasp pose encoder layer out specs
        Raises:
                NotImplementedError: _description_
        """
        super().__init__()

        if pc_encoder_config.type not in self.PC_ENCODERS:
            raise NotImplementedError(
                f"Pointcloud encoder network arch of type=`{pc_encoder_config.type}` is not implemented. \
				Available base network types are: {list(self.PC_ENCODERS)}"
            )

        self.pc_encoder = self.PC_ENCODERS[pc_encoder_config.type](
            out_features=pc_latent_size, **pc_encoder_config.args
        )

        self.grasp_encoder = ConditionalGraspPoseEncoder(
            config=grasp_encoder_config, latent_size=grasp_latent_size
        )

        self.out_features = grasp_latent_size

    def forward(
        self,
        xyz: Tensor,
        h: Tensor,
        z_pc: Tensor = None,
    ) -> Tensor:
        """Forward
        TODO: Shape spec

        In case z_pc is already available at this method's call,
        the forward method will avoid double computation, by bypassing
        the encode_pc step

        If using simple encoder (FC) to encode grasp, there is no proper conditioning on z_pc.
        so, use a bunch of simple FC fusion layers on concatenated tensor
        NOTE: This is just to support older experimental VAE models

        When not using simple encoder, fuse

        Args:
            xyz (Tensor): pointcloud coordinates
            h (Tensor): grasps
            z_pc (Tensor, optional): pc_latent. Defaults to None.

        Returns:
            Tensor: Grasp latent (pc-conditioned)
        """
        batch_size_h = h.shape[0]
        batch_size_pc = xyz.shape[0]

        pc_repeats = batch_size_h // batch_size_pc

        h = h.unsqueeze(dim=1)
        if z_pc is None:
            assert xyz is not None
            z_pc = self.pc_encoder(xyz).repeat_interleave(pc_repeats, dim=0)

        z_grasp = self.grasp_encoder(h, cond=z_pc)

        return z_grasp, z_pc

    def encode_pc(self, xyz: Tensor) -> Tensor:
        return self.pc_encoder(xyz)

    def get_conditioning_latent(self, xyz: Tensor) -> Tensor:
        return self.encode_pc(xyz)


class ConditionalGraspPoseDecoder(nn.Module):
    """Decodes latent code to the output/input space"""

    MODELS = {"Unet1D": Unet1D, "ResNet1D": ResNet1D}

    def __init__(
        self, config, in_features, feature_resolution, num_output_qualities=None
    ) -> None:
        super().__init__()

        if config.type not in self.MODELS:
            raise NotImplementedError(
                f"Base network arch of type=`{config.type}` is not implemented. \
				Available base network types are: {list(self.MODELS)}"
            )

        # Base Decoder network
        self.in_features = in_features
        self.feature_resolution = feature_resolution

        # Input layer
        self.in_layer = nn.Linear(
            in_features=self.in_features, out_features=self.feature_resolution
        )

        # Core network
        self.net = self.MODELS[config.type](dim=feature_resolution, **config.args)
        _net_out_features = self.net.out_features

        # Output layers
        self.tmrp = nn.Linear(_net_out_features, 6)
        self.class_logits = nn.Linear(_net_out_features, 1)

        self._use_qualities = (
            True
            if num_output_qualities is not None and num_output_qualities > 0
            else False
        )

        # Qualities
        if self._use_qualities:
            self.num_qualities = num_output_qualities
            self.qualities = nn.Linear(_net_out_features, num_output_qualities)
            self.out_features = (6, 1, num_output_qualities)
        else:
            self.num_qualities = None
            self.out_features = (6, 1)

    def forward(
        self, z_h: Tensor, cond: Tensor = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward

        Args:
            z_h (Tensor): grasp latents
            cond (Tensor, optional): pointcloud conditioning latents.
                                Defaults to None.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor): tmrp, cls_logits, qualities
        """
        # Input layer [B, D] -> [B, D1]
        z_h = self.in_layer(z_h)

        # Unsqueeze to 1 channel tensor: [B, D1] -> [ B, 1, D1]
        z_h = z_h.unsqueeze(-2)

        # Core network
        z_h = self.net(z_h, z_cond=cond)

        # Squeeze single channel: [B, 1, D1] -> [B, 1, D1]
        z_h = z_h.squeeze(-2)

        # Decode Outputs
        tmrp = self.tmrp(z_h)
        cls_logits = self.class_logits(z_h)

        res = (tmrp, cls_logits)

        if self._use_qualities:
            quals = self.qualities(z_h)
            res += (quals,)

        return res


class ConditionalGraspPoseEncoder(nn.Module):
    MODELS = {"Unet1D": Unet1D, "ResNet1D": ResNet1D}

    def __init__(
        self,
        config: Config,
        latent_size: int,
        feature_resolution: int = 16,
    ) -> None:
        """Conditional Grasp Pose Encoder

        Args:
            config (Config): Encoder model config for the required model type
            latent_size (int): Size of the grasp latent
            feature_resolution (int, optional): Resolution of the features to be maintained
                        in the network. Defaults to 16.

        """
        super().__init__()

        # Input feature
        self.in_features = config.args.in_features
        self.out_features = latent_size

        assert config.type in list(
            self.MODELS
        ), f"Cannot build GraspPoseEncoder of model_type: {config.type} from supported models: {self.MODELS}"

        # Constant feature size at which it is propagated through the network
        self.feature_resolution = None
        self._resolve_feature_resolution(feature_resolution)

        ## Network layers

        # Linear layer to cast input features to feature_resolution
        self.in_layer = nn.Linear(
            in_features=self.in_features, out_features=self.feature_resolution
        )

        # Core network that operates on the features of size: feature_resolution
        # Remove the in_features argument from the config, in_features is already resolved in in_layer
        _ = config.args.pop("in_features") if "in_features" in config.args else None
        self.net = self.MODELS[config.type](dim=feature_resolution, **config.args)

        # Linear layer to cast output features of size=feature_resolution to size=latent_size
        self.out_layer = nn.Linear(
            in_features=self.net.out_features, out_features=self.out_features
        )

    def _resolve_feature_resolution(self, feature_resolution: int) -> None:
        """Resolves the feature resolution to be used in the network

        Args:
            feature_resolution (int): Resolution of the features to be maintained
                        in the network. Defaults to 16.

        Raises:
            ValueError: If feature_resolution is < 0

        Returns:
            None: Sets the feature_resolution attribute
        """

        assert feature_resolution is not None, "feature_resolution cannot be None"
        if feature_resolution < 0:
            raise ValueError(
                f"feature_resolution must be >= 0, got {feature_resolution}"
            )
        elif feature_resolution > 0 and feature_resolution < self.in_features:
            warnings.warn(
                f"feature_resolution ({feature_resolution}) is less than in_features ({self.in_features})"
                "This is not a good idea."
            )
            self.feature_resolution = feature_resolution
        else:
            self.feature_resolution = feature_resolution

        return

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Forward pass
            B: Batch size
            D: Input feature dims
            D_cond: Conditioning feature dims
            L: Latent size

        Args:
            x (torch.Tensor): Input tensor [B, 1, D]
            cond (torch.Tensor): Conditioning tensor [B, C_cond, D_cond]

        Returns:
            torch.Tensor: Output tensor [B, 1, L]
        """
        x = self.in_layer(x)
        x = self.net(x, z_cond=cond)
        x = self.out_layer(x)

        return x


class VAEBottleneck(nn.Module):
    def __init__(self, in_features: int, latent_size: int) -> None:
        """VAE Bottleneck

        Args:
            in_features (int): Input feature size
            latent_size (int): Latent size
        """
        super().__init__()

        self.mu = nn.Linear(in_features, latent_size)
        self.logvar = nn.Linear(in_features, latent_size)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample from N(mu, var) using the reparameterization trick

        Args:
            mu (torch.Tensor): Mean [B, L]
            logvar (torch.Tensor): Log variance [B, L]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass

        Args:
            z (torch.Tensor): Input tensor [B, L]

        Returns:
            torch.Tensor: Output tensor [B, L]
        """
        mu = self.mu(z)
        logvar = self.logvar(z)
        return mu, logvar
