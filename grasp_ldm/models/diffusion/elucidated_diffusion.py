# Adapted from https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/elucidated_diffusion.py

import warnings
from math import sqrt

import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from torch import nn
from tqdm import tqdm


# helpers
def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


# tensor helpers
def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


# normalization
def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


class ElucidatedDiffusion(nn.Module):
    def __init__(
        self,
        net,
        *,
        seq_length,
        channels=1,
        num_sample_steps=32,
        sigma_min=0.002,
        sigma_max=80,
        sigma_data=0.5,
        rho=7,
        P_mean=-1.2,
        P_std=1.2,
        S_churn=80,
        S_tmin=0.05,
        S_tmax=50,
        S_noise=1.003,
    ):
        """Elucidated Diffusion Model
        https://arxiv.org/abs/2206.00364

        Args:
            net (nn.Module): the network to use for denoising
            seq_length (int): number of timesteps in the sequence
            channels (int, optional): number of channels. Defaults to 1.
            num_sample_steps (int, optional): number of sampling steps. Defaults to 32.
            sigma_min (float, optional): min noise level. Defaults to 0.002.
            sigma_max (float, optional): max noise level. Defaults to 80.
            sigma_data (float, optional): standard deviation of data distribution. Defaults to 0.5.
            rho (int, optional): controls the sampling schedule. Defaults to 7.
            P_mean (float, optional): mean of log-normal distribution from which noise is drawn for training. Defaults to -1.2.
            P_std (float, optional): standard deviation of log-normal distribution from which noise is drawn for training. Defaults to 1.2.
            S_churn (int, optional): parameters for stochastic sampling - depends on dataset, Table 5 in paper. Defaults to 80.
            S_tmin (float, optional): parameters for stochastic sampling - depends on dataset, Table 5 in paper. Defaults to 0.05.
            S_tmax (float, optional): parameters for stochastic sampling - depends on dataset, Table 5 in paper. Defaults to 50.
            S_noise (float, optional): parameters for stochastic sampling - depends on dataset, Table 5 in paper. Defaults to 1.003.
        """
        super().__init__()
        assert net.random_or_learned_sinusoidal_cond
        self.self_condition = False  # net.self_condition

        self.net = net

        # image dimensions

        self.channels = channels
        self.seq_length = seq_length

        # parameters

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        self.rho = rho

        self.P_mean = P_mean
        self.P_std = P_std

        self.num_sample_steps = num_sample_steps  # otherwise known as N in the paper

        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise

    @property
    def device(self):
        return next(self.net.parameters()).device

    # derived preconditioning params - Table 1

    def c_skip(self, sigma):
        return (self.sigma_data**2) / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma):
        return sigma * self.sigma_data * (self.sigma_data**2 + sigma**2) ** -0.5

    def c_in(self, sigma):
        return 1 * (sigma**2 + self.sigma_data**2) ** -0.5

    def c_noise(self, sigma):
        return log(sigma) * 0.25

    # preconditioned network output
    # equation (7) in the paper

    def preconditioned_network_forward(
        self, noised_x, sigma, *, z_cond=None, self_cond=None, clamp=False
    ):
        batch, device = noised_x.shape[0], noised_x.device

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device=device)

        padded_sigma = rearrange(sigma, "b -> b 1 1")

        net_out = self.net(
            self.c_in(padded_sigma) * noised_x,
            time=self.c_noise(sigma),
            z_cond=z_cond,
            x_self_cond=self_cond,
        )

        out = self.c_skip(padded_sigma) * noised_x + self.c_out(padded_sigma) * net_out

        if clamp:
            out = out.clamp(-1.0, 1.0)

        return out

    # sampling

    # sample schedule
    # equation (5) in the paper

    def sample_schedule(self, num_sample_steps=None):
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        N = num_sample_steps
        inv_rho = 1 / self.rho

        steps = torch.arange(num_sample_steps, device=self.device, dtype=torch.float32)
        sigmas = (
            self.sigma_max**inv_rho
            + steps / (N - 1) * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.rho

        sigmas = F.pad(sigmas, (0, 1), value=0.0)  # last step is sigma value of 0.
        return sigmas

    def sample(self, **kwargs):
        if kwargs.pop("use_dpmpp"):
            x, all_x = self.sample_using_dpmpp(**kwargs)
        else:
            x, all_x = self.sample_normal(**kwargs)

        return x, all_x

    @torch.no_grad()
    def sample_normal(
        self,
        batch_size=16,
        z_cond=None,
        num_sample_steps=None,
        clamp=False,
        return_all=False,
    ):
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        shape = (batch_size, self.channels, self.seq_length)

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma

        sigmas = self.sample_schedule(num_sample_steps)

        gammas = torch.where(
            (sigmas >= self.S_tmin) & (sigmas <= self.S_tmax),
            min(self.S_churn / num_sample_steps, sqrt(2) - 1),
            0.0,
        )

        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))

        # images is noise at the beginning

        init_sigma = sigmas[0]

        x = init_sigma * torch.randn(shape, device=self.device)

        # for self conditioning

        x_start = None
        all_x = [x]
        # gradually denoise

        for sigma, sigma_next, gamma in tqdm(
            sigmas_and_gammas, desc="sampling time step"
        ):
            sigma, sigma_next, gamma = map(
                lambda t: t.item(), (sigma, sigma_next, gamma)
            )

            eps = self.S_noise * torch.randn(
                shape, device=self.device
            )  # stochastic sampling

            sigma_hat = sigma + gamma * sigma
            x_hat = x + sqrt(sigma_hat**2 - sigma**2) * eps

            self_cond = x_start if self.self_condition else None

            model_output = self.preconditioned_network_forward(
                x_hat, sigma_hat, z_cond=z_cond, self_cond=self_cond, clamp=clamp
            )
            denoised_over_sigma = (x_hat - model_output) / sigma_hat

            x_next = x_hat + (sigma_next - sigma_hat) * denoised_over_sigma

            # second order correction, if not the last timestep

            if sigma_next != 0:
                self_cond = model_output if self.self_condition else None

                model_output_next = self.preconditioned_network_forward(
                    x_next, sigma_next, z_cond=z_cond, self_cond=self_cond, clamp=clamp
                )
                denoised_prime_over_sigma = (x_next - model_output_next) / sigma_next
                x_next = x_hat + 0.5 * (sigma_next - sigma_hat) * (
                    denoised_over_sigma + denoised_prime_over_sigma
                )

            x = x_next
            all_x += [x] if return_all else []
            x_start = model_output_next if sigma_next != 0 else model_output

        # x = x.clamp(-1.0, 1.0)
        # x = unnormalize_to_zero_to_one(x)
        return x, all_x

    @torch.no_grad()
    def sample_using_dpmpp(
        self,
        batch_size=16,
        z_cond=None,
        num_sample_steps=20,
        clamp=False,
        return_all=False,
    ):
        """
        thanks to Katherine Crowson (https://github.com/crowsonkb) for figuring it all out!
        https://arxiv.org/abs/2211.01095
        """
        if batch_size != z_cond.shape[0]:
            warnings.warn(
                f"The batch size for sample generation {batch_size} is different from conditioning batch_size {z_cond.shape[0]}."
                "\n This may be unreliable. If generation is being done for more than one pointcloud, batch_size should be (num_batch_pc*num_grasps_per_pc)."
            )

        device, num_sample_steps = self.device, default(
            num_sample_steps, self.num_sample_steps
        )

        sigmas = self.sample_schedule(num_sample_steps)

        shape = (batch_size, self.channels, self.seq_length)
        x = sigmas[0] * torch.randn(shape, device=device)
        all_x = [x]

        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sigma: sigma.log().neg()

        old_denoised = None
        for i in tqdm(range(len(sigmas) - 1)):
            denoised = self.preconditioned_network_forward(
                x, sigmas[i].item(), z_cond=z_cond, clamp=clamp
            )
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t

            if not exists(old_denoised) or sigmas[i + 1] == 0:
                denoised_d = denoised
            else:
                h_last = t - t_fn(sigmas[i - 1])
                r = h_last / h
                gamma = -1 / (2 * r)
                denoised_d = (1 - gamma) * denoised + gamma * old_denoised

            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
            all_x += [x] if return_all else []
            old_denoised = denoised

        # x = x.clamp(-1.0, 1.0)
        # return unnormalize_to_zero_to_one(x)
        return x, all_x

    # training

    def loss_weight(self, sigma):
        return (sigma**2 + self.sigma_data**2) * (sigma * self.sigma_data) ** -2

    def noise_distribution(self, batch_size):
        return (
            self.P_mean + self.P_std * torch.randn((batch_size,), device=self.device)
        ).exp()

    def forward(self, x, *, z_cond=None):
        b, c, n = x.shape
        assert (
            n == self.seq_length
        ), f"seq length must be {self.seq_length}, but got {n}"

        # x = normalize_to_neg_one_to_one(x)

        sigmas = self.noise_distribution(b)
        padded_sigmas = rearrange(sigmas, "b -> b 1 1")

        noise = torch.randn_like(x)

        noised_x = x + padded_sigmas * noise  # alphas are 1. in the paper

        self_cond = None

        if self.self_condition:
            raise NotImplementedError

        denoised = self.preconditioned_network_forward(
            noised_x, sigmas, z_cond=z_cond, self_cond=self_cond
        )

        losses = F.mse_loss(denoised, x, reduction="none")
        losses = reduce(losses, "b ... -> b", "mean")

        losses = losses * self.loss_weight(sigmas)

        return losses.mean()
