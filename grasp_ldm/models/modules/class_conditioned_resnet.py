from typing import Sequence

import torch
from torch import nn

from .resnets import TimeConditionedResNet1D, default


class ClassTimeConditionedResNet1D(TimeConditionedResNet1D):
    def __init__(
        self,
        dim: int,
        init_dim: int = None,
        out_channels: int = None,
        block_channels: Sequence = ...,
        channels: int = 1,
        input_conditioning_dims: int = None,
        is_self_conditioned: bool = False,
        resnet_block_groups: int = 8,
        learned_variance: bool = False,
        dropout=None,
        is_time_conditioned: bool = True,
        learned_sinusoidal_cond: bool = False,
        random_fourier_features: bool = False,
        learned_sinusoidal_dim: int = 16,
    ) -> None:
        super().__init__(
            dim,
            init_dim,
            out_channels,
            block_channels,
            channels,
            input_conditioning_dims,
            is_self_conditioned,
            resnet_block_groups,
            learned_variance,
            dropout,
            is_time_conditioned,
            learned_sinusoidal_cond,
            random_fourier_features,
            learned_sinusoidal_dim,
        )
        self.cls_embed = nn.Sequential(
            nn.Linear(1, self.emb_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        time: torch.Tensor = None,
        z_cond: torch.Tensor = None,
        x_self_cond: torch.Tensor = None,
        cls_cond: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        """Forward

        Args:
            x (torch.Tensor): input
            time (torch.Tensor): timestep for diffusion
                Note: Set to None, when using the architecture outside diffusion.
                    i.e. self.is_time_conditioned = False
            z_cond (torch.Tensor, optional): conditioning latent. Defaults to None.
            x_self_cond (torch.Tensor, optional): self conditioning vector. Defaults to None.

        Returns:
            torch.Tensor: output
        """

        # Ugly: improve
        if cls_cond is None:
            assert (
                "mode_cls" in kwargs["metas"]
            ), "Class conditioning tensor is required"
            cls_cond = (
                kwargs["metas"]["mode_cls"]
                .unsqueeze(-1)
                .reshape(-1, 1)
                .to(dtype=x.dtype)
            )

        if self.is_self_conditioned:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        # r = x.clone()

        # Time embedding for diffusion, None for non-diffusion
        if self.is_time_conditioned and self.time_mlp is not None:
            assert time is not None
            latent_emb = self.time_mlp(time)
        else:
            latent_emb = None

        # Class embedding
        cls_emb = self.cls_embed(cls_cond).squeeze(1)
        latent_emb += cls_emb

        # Add input embedding if inupt conditioned
        if self.is_input_conditioned:
            input_emb = self.input_emb_layers(z_cond)
            if input_emb.ndim != 2 and input_emb.ndim == 3:
                latent_emb = latent_emb.unsqueeze(-2).repeat([1, 3, 1])
            else:
                raise NotImplementedError
            latent_emb = latent_emb + input_emb if latent_emb is not None else input_emb

        for block1, block2, attn, updownsample in self.blocks:
            x = block1(x, latent_emb)

            x = block2(x, latent_emb)
            x = attn(x)

            x = updownsample(x)
            if self.dropout:
                x = self.dropout(x)

        x = self.final_res_block(x, latent_emb)
        return self.final_conv(x)
