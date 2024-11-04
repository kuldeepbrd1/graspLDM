# This is adapted from [lucidrains/denoising_diffusion-pytorch] and [openai/improved-diffusion]
# --
# [lucidrains/denoising_diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)
# [openai/improved-diffusion](https://github.com/openai/improved-diffusion)
# --
# Mods add support for additional input conditioning and also allow using
# the same Unet class to build models without time conditioning outside of DDM

import math
from functools import partial
from typing import Sequence

import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from torch import einsum, nn


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv1d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    return nn.Conv1d(dim, default(dim_out, dim), 4, 2, 1)


class WeightStandardizedConv2d(nn.Conv1d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv1d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        """_summary_

            Input:  [B, D, C]

            Embedding :  [Be, De, 1] i.e. 1 channel squeezed tensor
                    or   [Be, De, Ce] i.e. multi-channel tensor, Ce>1

            Feature tranform is done for each channel,
            i.e. [B, D, C] -> [B, D, C] * [Be, De, 1] -> [B, D, C]

            When multi-channel embedding is used, the feature transform is done
            per channel:
            i.e. [B, D, C] -> [B, D, C, 1] -> repeat Ce times -> [B, D, C, Ce]  ->
            [B, D, C, Ce] * [Be, De, 1, Ce] -> [B, D, C, Ce] -> reduce over Ce -> [B, D, C]


        Args:
            x (Tensor): Input tensor
            scale_shift (Tensor, optional): Feature transform scale/shift. Defaults to None.



        Returns:
            Tensor: output
        """
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift

            assert scale.ndim == x.ndim, "Mismatch in dimensions"

            if scale.shape[-1] == 1:
                # If scale/shift is [B, D, 1]
                x = x * (scale + 1) + shift
            else:
                x = x.unsqueeze(-1).tile((1, 1, 1, scale.shape[-1])) * (
                    scale.unsqueeze(-2) + 1
                ) + shift.unsqueeze(-2)
                x = reduce(x, " b c d r -> b c d", "sum")
        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, dim_out * 2))
            if exists(emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(emb):
            emb = self.mlp(emb)
            emb = (
                rearrange(emb, "b d -> b d 1")
                if emb.ndim == 2
                else rearrange(emb, "b c d -> b d c")
            )
            scale_shift = emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv1d(hidden_dim, dim, 1), LayerNorm(dim))

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) n -> b h c n", h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c n -> b (h c) n", h=self.heads)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) n -> b h c n", h=self.heads), qkv)

        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b h d j -> b h i d", attn, v)

        out = rearrange(out, "b h n d -> b (h d) n")
        return self.to_out(out)


class ResNet1D(nn.Module):
    def __init__(
        self,
        dim: int,
        init_dim: int = None,
        out_channels: int = None,
        block_channels: Sequence = (16, 64, 128, 64, 16),
        channels: int = 1,
        input_conditioning_dims: int = None,
        is_self_conditioned: bool = False,
        resnet_block_groups: int = 8,
        learned_variance: bool = False,
        dropout=None,
    ) -> None:
        """Unet1D
            Notation:
                Tensor Shape: [..., B, C, D]
                    B: Batch size
                    C: Channels
                    D: Feature Dims
        Args:
            dim (int): input dims (D)
            init_dim (int, optional): init dim TODO. Defaults to None.
            out_channels (int, optional): output dim (D) . Defaults to None.
            dim_mults (Sequence, optional): Dimension multiplier per Residual block,
                Length of the sequence is the number of Residual Blocks. Defaults to (1, 2, 4, 8).

            channels (int, optional): input channels (C) [..., C, D]. Defaults to 3.
            input_conditioning_dims (int, optional): conditioning latent dims (D), If conditioning with an input, . Defaults to None.
            is_self_conditioned (bool, optional): enable self conditioning. Defaults to False.
                From: Generating discrete data using Diffusion Models with self-conditioning
                    https://arxiv.org/abs/2208.04202

            is_time_conditioned (bool, optional): Defaults to True.
                        True, if conditioned with time, when using Unet as a denoiser net in DDMs .
                        False if no time conditioning, i.e. normal Unet without Diffusion.

            resnet_block_groups (int, optional): Groups is residual blocks. Defaults to 8.
            learned_variance (bool, optional): Learned Variance. Defaults to False.
            learned_sinusoidal_cond (bool, optional): Learned Sinusoidal embeddings. Defaults to False.
            random_fourier_features (bool, optional): Random fourier projection. Defaults to False.
            learned_sinusoidal_dim (int, optional): Learned sinusoidal embedding dims. Defaults to 16.
        """
        super().__init__()

        # determine dimensions

        self.channels = channels

        self.is_self_conditioned = is_self_conditioned
        input_channels = channels * (2 if is_self_conditioned else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding=3)

        dims = (dim,) + block_channels
        in_out = list(zip(dims[:-1], dims[1:]))

        self.in_features = dim
        self.out_features = dim

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        self.dropout = (
            nn.Dropout(p=dropout, inplace=True) if dropout is not None else None
        )

        # Input embedding

        emb_dim = dim * 4
        self.emb_dim = emb_dim

        if input_conditioning_dims is not None:
            self.is_input_conditioned = True

            self.input_emb_layers = nn.Sequential(
                nn.Linear(input_conditioning_dims, emb_dim), nn.SiLU()
            )
        else:
            self.is_input_conditioned = False
            self.input_emb_layers = None

        # ResBlock layers
        self.blocks = nn.ModuleList([])
        # num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            # is_last = ind >= (num_resolutions - 1)
            module_list = nn.ModuleList(
                [
                    block_klass(dim_in, dim_in, emb_dim=emb_dim),
                    block_klass(dim_in, dim_in, emb_dim=emb_dim),
                    Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                    nn.Conv1d(dim_in, dim_out, 3, padding=1),
                ]
            )
            self.blocks.append(module_list)

        # mid_dim = dims[-1]
        # self.mid_block1 = block_klass(mid_dim, mid_dim, emb_dim=emb_dim)
        # self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        # self.mid_block2 = block_klass(mid_dim, mid_dim, emb_dim=emb_dim)

        default_out_channels = channels * (1 if not learned_variance else 2)
        self.out_channels = default(out_channels, default_out_channels)

        self.final_res_block = block_klass(dims[-1], dims[-1], emb_dim=emb_dim)
        self.final_conv = nn.Conv1d(dims[-1], self.out_channels, 1)
        # self.output = nn.Linear(dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        *,
        z_cond: torch.Tensor = None,
        x_self_cond: torch.Tensor = None,
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
        if self.is_self_conditioned:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        # r = x.clone()

        latent_emb = None

        # Add input embedding if inupt conditioned
        if self.is_input_conditioned:
            input_emb = self.input_emb_layers(z_cond)
            latent_emb = latent_emb + input_emb if latent_emb is not None else input_emb

        for block1, block2, attn, updownsample in self.blocks:
            x = block1(x, latent_emb)

            x = block2(x, latent_emb)
            x = attn(x)

            x = updownsample(x)
            if self.dropout:
                x = self.dropout(x)

        # x = self.mid_block1(x, latent_emb)
        # x = self.mid_attn(x)
        # x = self.mid_block2(x, latent_emb)

        # x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, latent_emb)
        return self.final_conv(x)


class TimeConditionedResNet1D(nn.Module):
    def __init__(
        self,
        dim: int,
        init_dim: int = None,
        out_channels: int = None,
        block_channels: Sequence = (16, 64, 128, 64, 16),
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
        """Unet1D
            Notation:
                Tensor Shape: [..., B, C, D]
                    B: Batch size
                    C: Channels
                    D: Dims
        Args:
            dim (int): input dims (D)
            init_dim (int, optional): init dim TODO. Defaults to None.
            out_channels (int, optional): output dim (D) . Defaults to None.
            dim_mults (Sequence, optional): Dimension multiplier per Residual block,
                Length of the sequence is the number of Residual Blocks. Defaults to (1, 2, 4, 8).

            channels (int, optional): input channels (C) [..., C, D]. Defaults to 3.
            input_conditioning_dims (int, optional): conditioning latent dims (D), If conditioning with an input, . Defaults to None.
            is_self_conditioned (bool, optional): enable self conditioning. Defaults to False.
                From: Generating discrete data using Diffusion Models with self-conditioning
                    https://arxiv.org/abs/2208.04202

            is_time_conditioned (bool, optional): Defaults to True.
                        True, if conditioned with time, when using Unet as a denoiser net in DDMs .
                        False if no time conditioning, i.e. normal Unet without Diffusion.

            resnet_block_groups (int, optional): Groups is residual blocks. Defaults to 8.
            learned_variance (bool, optional): Learned Variance. Defaults to False.
            learned_sinusoidal_cond (bool, optional): Learned Sinusoidal embeddings. Defaults to False.
            random_fourier_features (bool, optional): Random fourier projection. Defaults to False.
            learned_sinusoidal_dim (int, optional): Learned sinusoidal embedding dims. Defaults to 16.
        """
        super().__init__()

        # determine dimensions

        self.channels = channels

        self.is_self_conditioned = is_self_conditioned
        input_channels = channels * (2 if is_self_conditioned else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding=3)

        dims = (dim,) + block_channels
        in_out = list(zip(dims[:-1], dims[1:]))

        self.in_features = dim
        self.out_features = dim

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        self.dropout = (
            nn.Dropout(p=dropout, inplace=True) if dropout is not None else None
        )

        # Time and Input embedding

        emb_dim = dim * 4
        self.emb_dim = emb_dim

        self.random_or_learned_sinusoidal_cond = (
            learned_sinusoidal_cond or random_fourier_features
        )
        if is_time_conditioned:
            self.is_time_conditioned = True
            if self.random_or_learned_sinusoidal_cond:
                sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                    learned_sinusoidal_dim, random_fourier_features
                )
                fourier_dim = learned_sinusoidal_dim + 1
            else:
                sinu_pos_emb = SinusoidalPosEmb(dim)
                fourier_dim = dim

            self.time_mlp = nn.Sequential(
                sinu_pos_emb,
                nn.Linear(fourier_dim, emb_dim),
                nn.GELU(),
                nn.Linear(emb_dim, emb_dim),
            )
        else:
            self.is_time_conditioned = False
            self.time_mlp = None

        if input_conditioning_dims is not None:
            self.is_input_conditioned = True

            # TODO: Add linear layer at the end
            self.input_emb_layers = nn.Sequential(
                nn.Linear(input_conditioning_dims, emb_dim), nn.SiLU()
            )
        else:
            self.is_input_conditioned = False
            self.input_emb_layers = None

        # ResBlock layers
        self.blocks = nn.ModuleList([])

        for _, (dim_in, dim_out) in enumerate(in_out):
            module_list = nn.ModuleList(
                [
                    block_klass(dim_in, dim_in, emb_dim=emb_dim),
                    block_klass(dim_in, dim_in, emb_dim=emb_dim),
                    Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                    nn.Conv1d(dim_in, dim_out, 3, padding=1),
                ]
            )
            self.blocks.append(module_list)

        default_out_channels = channels * (1 if not learned_variance else 2)
        self.out_channels = default(out_channels, default_out_channels)

        self.final_res_block = block_klass(dims[-1], dims[-1], emb_dim=emb_dim)
        self.final_conv = nn.Conv1d(dims[-1], self.out_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        *,
        time: torch.Tensor = None,
        z_cond: torch.Tensor = None,
        x_self_cond: torch.Tensor = None,
        **kwargs,
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

        # Add input embedding if inupt conditioned
        if self.is_input_conditioned:
            input_emb = self.input_emb_layers(z_cond)
            if input_emb.ndim != 2 and input_emb.ndim == 3:
                latent_emb = latent_emb.unsqueeze(-2).repeat([1, input_emb.shape[1], 1])
            elif input_emb.ndim == 2:
                pass
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


# model


class Unet1D(nn.Module):
    def __init__(
        self,
        dim: int,
        init_dim: int = None,
        out_channels: int = None,
        dim_mults: Sequence = (1, 2, 4, 8),
        channels: int = 1,
        input_conditioning_dims: int = None,
        is_self_conditioned: bool = False,
        is_time_conditioned: bool = True,
        resnet_block_groups: int = 8,
        learned_variance: bool = False,
        learned_sinusoidal_cond: bool = False,
        random_fourier_features: bool = False,
        learned_sinusoidal_dim: int = 16,
        dropout=None,
    ) -> None:
        """Unet1D
            Notation:
                Tensor Shape: [..., B, C, D]
                    B: Batch size
                    C: Channels
                    D: Dims
        Args:
            dim (int): input dims (D)
            init_dim (int, optional): init dim TODO. Defaults to None.
            out_channels (int, optional): output dim (D) . Defaults to None.
            dim_mults (Sequence, optional): Dimension multiplier per Residual block,
                Length of the sequence is the number of Residual Blocks. Defaults to (1, 2, 4, 8).

            channels (int, optional): input channels (C) [..., C, D]. Defaults to 3.
            input_conditioning_dims (int, optional): conditioning latent dims (D), If conditioning with an input, . Defaults to None.
            is_self_conditioned (bool, optional): enable self conditioning. Defaults to False.
                From: Generating discrete data using Diffusion Models with self-conditioning
                    https://arxiv.org/abs/2208.04202

            is_time_conditioned (bool, optional): Defaults to True.
                        True, if conditioned with time, when using Unet as a denoiser net in DDMs .
                        False if no time conditioning, i.e. normal Unet without Diffusion.

            resnet_block_groups (int, optional): Groups is residual blocks. Defaults to 8.
            learned_variance (bool, optional): Learned Variance. Defaults to False.
            learned_sinusoidal_cond (bool, optional): Learned Sinusoidal embeddings. Defaults to False.
            random_fourier_features (bool, optional): Random fourier projection. Defaults to False.
            learned_sinusoidal_dim (int, optional): Learned sinusoidal embedding dims. Defaults to 16.
        """
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.in_features = dim
        self.out_features = dim

        self.is_self_conditioned = is_self_conditioned
        input_channels = channels * (2 if is_self_conditioned else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings

        emb_dim = dim * 4
        self.emb_dim = emb_dim
        self.dropout = (
            nn.Dropout(p=dropout, inplace=True) if dropout is not None else None
        )

        self.random_or_learned_sinusoidal_cond = (
            learned_sinusoidal_cond or random_fourier_features
        )
        if is_time_conditioned:
            self.is_time_conditioned = True
            if self.random_or_learned_sinusoidal_cond:
                sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                    learned_sinusoidal_dim, random_fourier_features
                )
                fourier_dim = learned_sinusoidal_dim + 1
            else:
                sinu_pos_emb = SinusoidalPosEmb(dim)
                fourier_dim = dim

            self.time_mlp = nn.Sequential(
                sinu_pos_emb,
                nn.Linear(fourier_dim, emb_dim),
                nn.GELU(),
                nn.Linear(emb_dim, emb_dim),
            )
        else:
            self.is_time_conditioned = False
            self.time_mlp = None

        # Input embedding
        if input_conditioning_dims is not None:
            self.is_input_conditioned = True

            self.input_emb_layers = nn.Sequential(
                nn.Linear(input_conditioning_dims, emb_dim),
                nn.SiLU(),
                nn.Linear(emb_dim, emb_dim),
            )

        else:
            self.is_input_conditioned = False
            self.input_emb_layers = None

        # ResBlock layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            module_list = nn.ModuleList(
                [
                    block_klass(dim_in, dim_in, emb_dim=emb_dim),
                    block_klass(dim_in, dim_in, emb_dim=emb_dim),
                    Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                    Downsample(dim_in, dim_out)
                    if not is_last
                    else nn.Conv1d(dim_in, dim_out, 3, padding=1),
                ]
            )
            self.downs.append(module_list)

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, emb_dim=emb_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, emb_dim=emb_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            module_list = nn.ModuleList(
                [
                    block_klass(dim_out + dim_in, dim_out, emb_dim=emb_dim),
                    block_klass(dim_out + dim_in, dim_out, emb_dim=emb_dim),
                    Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                    Upsample(dim_out, dim_in)
                    if not is_last
                    else nn.Conv1d(dim_out, dim_in, 3, padding=1),
                ]
            )
            self.ups.append(module_list)

        default_out_channels = channels * (1 if not learned_variance else 2)
        self.out_channels = default(out_channels, default_out_channels)

        self.final_res_block = block_klass(dim * 2, dim, emb_dim=emb_dim)
        self.final_conv = nn.Conv1d(dim, self.out_channels, 1)
        # self.output = nn.Linear(dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        *,
        time: torch.Tensor = None,
        z_cond: torch.Tensor = None,
        x_self_cond: torch.Tensor = None,
        **kwargs,
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
        if self.is_self_conditioned:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        # Time embedding for diffusion, None for non-diffusion
        time_latent_emb = (
            self.time_mlp(time)
            if (self.is_time_conditioned and self.time_mlp is not None)
            else None
        )

        # Add input embedding if inupt conditioned
        if self.is_input_conditioned:
            input_emb = self.input_emb_layers(z_cond)
            time_latent_emb = (
                time_latent_emb + input_emb
                if time_latent_emb is not None
                else input_emb
            )

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, time_latent_emb)
            h.append(x)

            x = block2(x, time_latent_emb)
            x = attn(x)
            h.append(x)

            x = downsample(x)
            if self.dropout:
                x = self.dropout(x)

        x = self.mid_block1(x, time_latent_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, time_latent_emb)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, time_latent_emb)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, time_latent_emb)
            x = attn(x)

            x = upsample(x)
            if self.dropout:
                x = self.dropout(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, time_latent_emb)
        return self.final_conv(x)
