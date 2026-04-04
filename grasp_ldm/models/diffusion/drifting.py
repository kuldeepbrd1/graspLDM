"""Drifting Model for one-step generative modeling.

Based on: "Generative Modeling via Drifting"
Deng, Li, Li, Du, He — MIT/Harvard, arXiv:2602.04770

Core idea: instead of iterating a reverse diffusion at inference time (N scheduler steps),
the optimizer does the distributional work during training. A "drifting field" V is estimated
from mini-batches to push generated samples toward real data. When training converges, V ≈ 0
and the generator f maps noise → data in a single forward pass.

Training objective:
    1. Sample real data x and noise ε ~ N(0,I)
    2. Compute generated sample x̂ = f(ε, z_cond) via current generator
    3. Estimate drifting field V using a kernel that attracts x̂ toward x and repels x̂ from
       other generated samples:  V(x̂_i) = Σ_j k(x̂_i, x_j) * (x_j - x̂_i)
                                           - Σ_j k(x̂_i, x̂_j) * (x̂_j - x̂_i)
    4. Compute drifted target: x̂_drifted = x̂ + step_size * V(x̂)
    5. Regression loss: ||f(ε, z_cond) - sg(x̂_drifted)||²
       where sg(·) is stop-gradient — the target is not differentiated through

Inference: z_grasp = f(ε, z_cond)  — single forward pass, no loop.
"""

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class DriftingModel(nn.Module):
    """One-step generative model via drifting field training.

    The generator takes Gaussian noise and (optionally) conditioning and maps
    directly to the target distribution in a single forward pass.

    tensor notation: [B, C, D]
        B: Batch size
        C: Channels (1 for 1D)
        D: Feature dims
    """

    def __init__(
        self,
        model: nn.Module,
        n_dims: int,
        drift_step_size: float = 1.0,
        kernel_bandwidth: float = 1.0,
        loss_type: str = "l2",
    ) -> None:
        """
        Args:
            model (nn.Module): Generator network with signature
                ``forward(x, *, time, z_cond)`` — same interface as the DDPM denoiser.
                At training time, x is the noise input ε; time is zeros (unused).
                At inference time, a single call with ε produces the generated sample.
            n_dims (int): Dimensionality of the latent space (D).
            drift_step_size (float): Step size η for computing the drifted target.
                Controls how far generated samples are moved toward real data per step.
                Defaults to 1.0.
            kernel_bandwidth (float): Bandwidth σ for the RBF kernel used in the drifting
                field. Controls the smoothness of the distributional attraction/repulsion.
                Defaults to 1.0.
            loss_type (str): Regression loss. Valid: ["l1", "l2", "huber"]. Defaults to "l2".
        """
        super().__init__()
        self.model = model
        self.n_dims = n_dims
        self.channels = 1
        self.drift_step_size = drift_step_size
        self.kernel_bandwidth = kernel_bandwidth
        self.loss_type = loss_type

    def _rbf_kernel(self, a: Tensor, b: Tensor) -> Tensor:
        """Pairwise RBF kernel k(a_i, b_j) = exp(-||a_i - b_j||² / (2σ²))

        Args:
            a: [B, D]
            b: [B, D]

        Returns:
            Tensor: kernel matrix [B, B]
        """
        # Pairwise squared distances: [B, B]
        diff = a.unsqueeze(1) - b.unsqueeze(0)  # [B, B, D]
        sq_dist = (diff ** 2).sum(-1)            # [B, B]
        return torch.exp(-sq_dist / (2 * self.kernel_bandwidth ** 2))

    def _drifting_field(self, x_gen: Tensor, x_real: Tensor) -> Tensor:
        """Estimate drifting field V at generated samples.

        V(x̂_i) = Σ_j k(x̂_i, x_j)(x_j - x̂_i)   [attraction toward real data]
                - Σ_j k(x̂_i, x̂_j)(x̂_j - x̂_i)  [repulsion from other generated]

        Args:
            x_gen:  generated samples [B, D]
            x_real: real samples      [B, D]

        Returns:
            Tensor: drifting field V  [B, D]
        """
        k_gen_real = self._rbf_kernel(x_gen, x_real)   # [B, B]
        k_gen_gen  = self._rbf_kernel(x_gen, x_gen)    # [B, B]

        # Attraction: pull toward real data
        # k_gen_real[i, j] * (x_real[j] - x_gen[i])  →  sum over j
        attraction = torch.einsum("ij,jd->id", k_gen_real, x_real) - \
                     (k_gen_real.sum(1, keepdim=True) * x_gen)

        # Repulsion: push away from other generated samples
        repulsion = torch.einsum("ij,jd->id", k_gen_gen, x_gen) - \
                    (k_gen_gen.sum(1, keepdim=True) * x_gen)

        return attraction - repulsion

    def _loss_fn(self, pred: Tensor, target: Tensor) -> Tensor:
        if self.loss_type == "l1":
            return F.l1_loss(pred, target)
        elif self.loss_type == "l2":
            return F.mse_loss(pred, target)
        elif self.loss_type == "huber":
            return F.smooth_l1_loss(pred, target)
        raise NotImplementedError(f"Unknown loss_type: {self.loss_type}")

    def forward(self, x_real: Tensor, z_cond: Tensor = None, **kwargs) -> Tensor:
        """Training forward: compute drifting regression loss.

        Args:
            x_real (Tensor): real latent samples [B, C, D]
            z_cond (Tensor, optional): conditioning [B, z_cond_dim]. Defaults to None.

        Returns:
            Tensor: scalar loss
        """
        b, _, d = x_real.shape
        assert d == self.n_dims, f"Expected n_dims={self.n_dims}, got {d}"

        # Sample noise ε ~ N(0, I)
        noise = torch.randn_like(x_real)

        # Dummy timestep (generator is time-unconditional; reuses denoiser backbone)
        t_zeros = torch.zeros(b, device=x_real.device, dtype=torch.long)

        # Generate: x̂ = f(ε, z_cond)
        x_gen = self.model(noise, time=t_zeros, z_cond=z_cond, **kwargs)  # [B, C, D]

        # Flatten channel dim for drifting field computation
        x_gen_flat  = x_gen.squeeze(1)    # [B, D]
        x_real_flat = x_real.squeeze(1)   # [B, D]

        # Compute drifting field V and drifted target (stop-gradient)
        with torch.no_grad():
            V = self._drifting_field(x_gen_flat, x_real_flat)   # [B, D]
            x_drifted = (x_gen_flat + self.drift_step_size * V).unsqueeze(1)  # [B, C, D]

        # Regression loss: push f(ε) toward drifted target
        loss = self._loss_fn(x_gen, x_drifted)
        return loss

    @torch.no_grad()
    def sample(
        self,
        z_cond: Tensor = None,
        batch_size: int = 1,
        return_all: bool = False,
        device: torch.device = "cuda:0" if torch.cuda.is_available() else "cpu",
        **kwargs,
    ) -> Tuple[Tensor, list]:
        """Sample — single forward pass (1-NFE).

        Args:
            z_cond (Tensor, optional): conditioning [B, z_cond_dim]. Defaults to None.
            batch_size (int): number of samples to generate. Defaults to 1.
            return_all (bool): unused, kept for API compatibility with GaussianDiffusion1D.
            device: target device.

        Returns:
            Tuple[Tensor, list]: (generated_latents [B, C, D], [])
        """
        noise = torch.randn((batch_size, self.channels, self.n_dims), device=device)
        t_zeros = torch.zeros(batch_size, device=device, dtype=torch.long)
        x_gen = self.model(noise, time=t_zeros, z_cond=z_cond, **kwargs)
        return x_gen, []
