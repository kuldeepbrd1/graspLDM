from typing import Any, Tuple

import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler
from torch import Tensor, nn
from tqdm.auto import tqdm


class GaussianDiffusion1D(nn.Module):
    ALL_LOSSES = ["l1", "l2", "huber"]
    NOISE_SCHEDULERS = ["ddpm", "ddim"]
    BETA_SCHEDULES = ["linear", "scaled_linear", "squaredcos_cap_v2", "cosine"]
    VARIANCE_TYPES = [
        "fixed_small",
        "fixed_small_log",
        "fixed_large",
        "fixed_large_log",
        "learned",
        "learned_range",
    ]

    def __init__(
        self,
        model: nn.Module,
        n_dims: int,
        noise_scheduler_type: str = "ddpm",
        beta_schedule: str = "linear",
        variance_type: str = "fixed_small",
        pred_type: str = "epsilon",
        beta_start: str = 0.0001,
        beta_end: str = 0.02,
        num_steps: int = 1000,
        loss_type: str = "l1",
        clip_sample=True,
    ) -> None:
        """Gaussian Diffusion 1D

        tensor notation: [B,C,D]
            B: Batch size
            C: Channels (Default 1 for 1D)
            D: Feature dims

        Args:
            model (nn.Module): denoiser model
                denoiser model should have forward arugment structure:
                    ```
                        def forward (self, x, *, t=t, z_cond=z_cond):
                    ```
                    x: Input tensor [B,C,D]
                    t: Batched timestep tensor (long) [B,1]
                    z_cond: conditioning [B, ...]

            n_dims (int): input data dims (D)

            noise_scheduler (str, optional): Noise scheduler  type.
                    Valid:  ["ddpm", "ddim"]

            beta_schedule (str, optional): beta noise schedule. Defaults to "linear".
                    Valid: [linear, scaled_linear, or squaredcos_cap_v2]

            variance_type(str, optional): Type of variance used to add noise.
                    Valid:  [fixed_small, fixed_small_log, fixed_large, fixed_large_log, learned or learned_range

            pred_type(str,optional): prediction type of the scheduler function
                    Valid: "epsilon"   (predicting the noise of the diffusion process),
                            "sample"   (directly predicting the noisy sample)
                            "v_prediction" (https://imagen.research.google/video/paper.pdf)

            num_steps (int, optional): number of diffusion steps. Defaults to 1000.

            loss_type (str, optional): loss type.
                    Valid: ["l1", "l2", "linear"]. Defaults to "l1".
        """
        super().__init__()
        self.num_train_timesteps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = (
            beta_schedule if not beta_schedule == "cosine" else "squaredcos_cap_v2"
        )
        self.variance_type = variance_type
        self.num_steps = num_steps
        self.pred_type = pred_type
        self.clip_sample = clip_sample

        self.model = model
        self.n_dims = n_dims
        self.channels = 1

        self.noise_scheduler = self.configure_noise_scheduler(noise_scheduler_type)
        self._noise_scheduler_type = noise_scheduler_type

        assert (
            loss_type in self.ALL_LOSSES
        ), f"Invalid loss_type. Supported losses: {self.ALL_LOSSES} "

        self.loss_type = loss_type

        self._is_variance_learned = self.variance_type in ["learned", "learned_range"]
        if self._is_variance_learned:
            assert (
                self.model.out_channels == 2
            ), f"For learned variance mode ({self.variance_type}), the score model should 2 output channels: eps_pred and var_pred"
        else:
            assert (
                self.model.out_channels == 1
            ), f"For pre-defined variance type {self.variance_type}, the score model should have only one output channel: eps_pred"

    @property
    def num_inference_steps(self):
        inf_t = self.noise_scheduler.num_inference_steps
        return inf_t if inf_t is not None else self.num_steps

    def set_inference_timesteps(self, num_steps):
        """Set inference timesteps for diffusion
        Useful for custom time steps for DDIM

        Args:
            num_steps (int): number of time steps
        """
        self.noise_scheduler.set_timesteps(num_steps)

    def configure_noise_scheduler(self, noise_scheduler_type: str) -> Any:
        """Configure Noise Scheduler

        This allows use of multiple schedulers for training but mostly sampling

        TODO: Support more schedulers. Add scale model input to inter-operate between schedulers at sampling time

        Returns:
            _type_: _description_
        """
        assert (
            noise_scheduler_type in self.NOISE_SCHEDULERS
        ), f"{self.noise_scheduler} Not supported"

        assert (
            self.beta_schedule in self.BETA_SCHEDULES
        ), f"{self.beta_schedule} not supported"

        assert (
            self.variance_type in self.VARIANCE_TYPES
        ), f"{self.variance_type} not supported"

        kwargs = dict(
            num_train_timesteps=self.num_steps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            beta_schedule=self.beta_schedule,
            variance_type=self.variance_type,
            prediction_type=self.pred_type,
            clip_sample=self.clip_sample,
        )

        if noise_scheduler_type == "ddpm":
            scheduler = DDPMScheduler(**kwargs)
        elif noise_scheduler_type == "ddim":
            kwargs.pop("variance_type")
            scheduler = DDIMScheduler(**kwargs)
        else:
            raise NotImplementedError

        return scheduler

    def q_sample_at_t(self, x_0: Tensor, t: int) -> Tuple[Tensor, Tensor]:
        """Forward diffusion: Sampling from fixed posterior

        Args:
            x_0 (Tensor): input x @ t=0
            t (int): query time

        Returns:
            Tuple[Tensor, Tensor]: (x@t=t, noise)
        """
        noise = torch.randn_like(x_0)
        noisy_x_t = self.noise_scheduler.add_noise(x_0, noise, t)
        return noisy_x_t, noise

    def loss_fn(self, true_noise: Tensor, predicted_noise: Tensor) -> Tensor:
        """Loss function
        Supports: ["l1", "l2", "huber"]

        Args:
            true_noise (Tensor): injected true noise [B, C, D]
            predicted_noise (Tensor): noise predicted by the model [B, C, D]

        Returns:
            Tensor: Loss (1,)
        """
        if self.loss_type == "l1":
            loss = F.l1_loss(true_noise, predicted_noise)
        elif self.loss_type == "l2":
            loss = F.mse_loss(true_noise, predicted_noise)
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(true_noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss

    def forward(self, x_0: Tensor, z_cond: Tensor = None, **kwargs) -> Tensor:
        """Forward (train)

        Args:
            x_0 (Tensor): x @ t=0  [B,C,D]
            z_cond (Tensor, optional): Optional conditioning. Defaults to None.

        Returns:
            Tensor: _description_
        """
        b, _, d = x_0.shape

        assert (
            d == self.n_dims
        ), f"Got tensor with size {d} at index -1, expected {self.ndims} from self.ndims."

        t = torch.randint(0, self.num_steps, (b,), device=x_0.device).long()

        x_t, true_noise = self.q_sample_at_t(x_0, t)
        out = self.model(x_t, time=t, z_cond=z_cond, **kwargs)

        if self._is_variance_learned:
            noise_pred, var_pred = out.chunk(2, dim=1)
        else:
            noise_pred = out

        loss = self.loss_fn(true_noise, noise_pred)

        return loss

    @torch.no_grad()
    def sample(
        self,
        z_cond: Tensor = None,
        batch_size: int = 1,
        return_all: bool = False,
        device: torch.device = "cuda:0" if torch.cuda.is_available() else "cpu",
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        """Sample- reverse diffusion

        Args:
            z_cond (Tensor, optional): Conditioning. Defaults to None.
            batch_size (int, optional): Batch size (B) for sampling. Defaults to 1.
            return_all (bool, optional): Return output from at time steps. Defaults to True.
            device (torch.device, optional): Tensor device. Defaults to "cuda:0" if torch.cuda.is_available() else "cpu".

        Returns:
            Tuple[Tensor, Tensor]: (x@t=0, x_list@t=[0,...,T]) where x:[B,C,D]
        """

        x_T = torch.randn((batch_size, self.channels, self.n_dims)).to(device)

        x_noisy = x_T
        all_noisy = [x_T] if return_all else []

        for t in tqdm(
            reversed(
                range(
                    0, self.num_steps, int(self.num_steps // self.num_inference_steps)
                )
            ),
            desc="Sampling time step",
            total=self.num_inference_steps,
        ):
            t_batch = torch.full(
                (x_noisy.shape[0],), t, device=device, dtype=torch.long
            )

            pred_noise = self.model(x_noisy, time=t_batch, z_cond=z_cond, **kwargs)
            x_noisy = self.noise_scheduler.step(pred_noise, t, x_noisy).prev_sample

            if return_all:
                all_noisy.append(x_noisy)

        return x_noisy, all_noisy
