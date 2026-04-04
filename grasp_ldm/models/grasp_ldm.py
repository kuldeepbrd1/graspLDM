import warnings

import torch
from addict import Dict

from .diffusion import DriftingModel, ElucidatedDiffusion, GaussianDiffusion1D
from .modules.base_network import BaseGraspSampler


class GraspLatentDDM(BaseGraspSampler):
    def __init__(
        self,
        model,
        latent_in_features,
        diffusion_timesteps,
        diffusion_loss,
        beta_schedule="linear",
        noise_scheduler_type: str = "ddpm",
        num_inference_steps: int = None,
        pred_type: str = "epsilon",
        denoising_loss_weight=1,
        variance_type="fixed_small",
        elucidated_diffusion=False,
        beta_start=5e-5,
        beta_end=5e-2,
        diffusion_type: str = "ddpm",
        drift_step_size: float = 1.0,
        kernel_bandwidth: float = 1.0,
    ) -> None:
        """Grasp Latent Diffusion Model

        Args:
            model (nn.Module): denoiser model with signature
                ``forward(x, *, t, z_cond)`` where x is [B,C,D],
                t is [B,1] timestep tensor, z_cond is [B,...] conditioning.
            latent_in_features (int): input data dimensionality (D)
            diffusion_timesteps (int): number of diffusion timesteps
            diffusion_loss (str): diffusion loss type ("l1", "l2")
            beta_schedule (str, optional): beta noise schedule type.
                Valid: ["linear", "scaled_linear", "squaredcos_cap_v2"]. Defaults to "linear".
            noise_scheduler_type (str, optional): noise scheduler type.
                Valid: ["ddpm", "ddim"]. Defaults to "ddpm".
            num_inference_steps (int, optional): number of inference steps for DDIM/DDPM sampling.
                Defaults to None (uses num_train_timesteps).
            pred_type (str, optional): prediction type for diffusion model.
                Valid: "epsilon" (predict noise), "v_prediction" (predict velocity). Defaults to "epsilon".
            denoising_loss_weight (int, optional): weight for denoising loss. Defaults to 1.
            variance_type (str, optional): variance type for noise addition.
                Valid: ["fixed_small", "fixed_large", "learned", "learned_range"]. Defaults to "fixed_small".
            elucidated_diffusion (bool, optional): use ElucidatedDiffusion instead of DDPM. Defaults to False.
            beta_start (float, optional): starting beta value. Defaults to 5e-5.
            beta_end (float, optional): ending beta value. Defaults to 5e-2.
            diffusion_type (str, optional): generative model type.
                "ddpm" uses GaussianDiffusion1D (DDPM/DDIM scheduler, multi-step).
                "drifting" uses DriftingModel (one-step inference, arXiv:2602.04770).
                Defaults to "ddpm".
            drift_step_size (float, optional): step size for drifting field update.
                Only used when diffusion_type="drifting". Defaults to 1.0.
            kernel_bandwidth (float, optional): RBF kernel bandwidth for drifting field.
                Only used when diffusion_type="drifting". Defaults to 1.0.
        """
        super().__init__()
        self.vae_model = None
        self._diffusion_type = diffusion_type

        if diffusion_type == "drifting":
            self.diffusion_model = DriftingModel(
                model=model,
                n_dims=latent_in_features,
                drift_step_size=drift_step_size,
                kernel_bandwidth=kernel_bandwidth,
                loss_type=diffusion_loss,
            )
        elif elucidated_diffusion:
            self.diffusion_model = ElucidatedDiffusion(
                net=model, seq_length=latent_in_features
            )
        else:
            self.diffusion_model = GaussianDiffusion1D(
                model=model,
                n_dims=latent_in_features,
                num_steps=diffusion_timesteps,
                num_inference_steps=num_inference_steps,
                loss_type=diffusion_loss,
                beta_schedule=beta_schedule,
                beta_start=beta_start,
                beta_end=beta_end,
                noise_scheduler_type=noise_scheduler_type,
                variance_type=variance_type,
                pred_type=pred_type,
            )

        self.loss_weight = denoising_loss_weight
        self.is_vae_frozen = False

    @property
    def use_grasp_qualities(self):
        return self.vae_model.use_grasp_qualities

    @property
    def scheduler_type(self):
        return getattr(self.diffusion_model, "_noise_scheduler_type", self._diffusion_type)

    @property
    def _latent_loss_objects(self):
        return self.vae_model._latent_loss_objects

    def set_vae_model(self, vae_model):
        self.vae_model = vae_model

    def load_vae_weights(self, state_dict):
        self.vae_model.load_state_dict(state_dict, strict=True)

    def set_inference_timesteps(self, num_inference_steps):
        self.diffusion_model.set_inference_timesteps(num_inference_steps)

    def freeze_vae_model(self):
        for param in self.vae_model.parameters():
            param.requires_grad = False
        self.vae_model.eval()
        self.is_vae_frozen = True

    def forward(self, pc, grasps, compute_loss=None, **kwargs):
        """Training forward: compute denoising loss for a batch of pc and grasps.

        Args:
            pc (torch.Tensor): point cloud [batch_size, num_points, 3]
            grasps (torch.Tensor): grasps [batch_size, 6/7]

        Returns:
            tuple: (None, loss_dict) where loss_dict has keys "loss" and "denoising_loss"
        """
        if not self.is_vae_frozen:
            self.freeze_vae_model()
            warnings.warn("VAE model was frozen manually after loading")
            self.print_params_info()

        (_, _, z_h), (_, _, z_pc_cond) = self.vae_model.encode(pc, grasps)

        denoising_loss = self.diffusion_model(
            z_h.unsqueeze(1), z_cond=z_pc_cond, **kwargs
        )

        loss_dict = Dict(loss=denoising_loss, denoising_loss=denoising_loss)
        return None, loss_dict

    @torch.no_grad()
    def generate_grasps(self, xyz, num_grasps=10, return_intermediate=False, **kwargs):
        """Generate grasps for a given point cloud via reverse diffusion.

        Args:
            xyz (torch.Tensor): point cloud [batch_size, num_points, 3]
            num_grasps (int): number of grasps to generate per point cloud. Defaults to 10.
            return_intermediate (bool): return intermediate diffusion steps. Defaults to False.

        Returns:
            tuple: (decoder_output, intermediates)
                decoder_output is (tmrp, cls_logits[, qualities])
                intermediates is [] if return_intermediate=False
        """
        z_pc_cond = self.vae_model.encode_pc(xyz)
        z_pc_cond = z_pc_cond.repeat_interleave(num_grasps, dim=0)

        out, all_outs = self.diffusion_model.sample(
            z_cond=z_pc_cond,
            batch_size=z_pc_cond.shape[0],
            return_all=return_intermediate,
            **kwargs,
        )
        out = self.vae_model.decoder(out.squeeze(-2), z_pc_cond)

        if not return_intermediate:
            return (out, [])

        step_outs = []
        for idx in torch.linspace(0, len(all_outs) - 1, steps=50, dtype=torch.int):
            _out = self.vae_model.decoder(all_outs[idx].squeeze(-2), z_pc_cond)
            step_outs.append([t.detach().cpu() for t in _out])
        return out, step_outs

    def print_params_info(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        print("------------------------------------------------")
        print(f"Trainable parameters:     {trainable:,}")
        print(f"Non-trainable parameters: {frozen:,}")
        print("------------------------------------------------")
