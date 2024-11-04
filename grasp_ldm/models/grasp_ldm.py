import warnings

import torch
from addict import Dict

from .diffusion import ElucidatedDiffusion, GaussianDiffusion1D
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
        is_conditioned=True,
        joint_training=False,
        denoising_loss_weight=1,
        variance_type="fixed_small",
        elucidated_diffusion=False,
        beta_start=5e-5,
        beta_end=5e-2,
    ) -> None:
        """Grasp Latent Diffusion Model

        Args:
            model (nn.Module): denoiser model
                denoiser model should have forward arugment structure:
                    ```
                        def forward (self, x, *, t=t, z_cond=z_cond):
                    ```
                    x: Input tensor [B,C,D]
                    t: Batched timestep tensor (long) [B,1]
                    z_cond: conditioning [B, ...]

            latent_in_features (int): input data dims (D)
            diffusion_timesteps (int): Number of diffusion timesteps
            diffusion_loss (str): Diffusion loss type
            beta_schedule (str, optional): beta noise schedule. Defaults to "linear".
                    Valid: [linear, scaled_linear, or squaredcos_cap_v2]
            noise_scheduler_type (str, optional): Noise scheduler  type.
                    Valid:  ["ddpm", "ddim"]
            is_conditioned (bool, optional): Whether the diffusion model is conditioned. Defaults to True.
            joint_training (bool, optional): Whether the diffusion model is trained jointly with the denoiser. Defaults to False. TODO: Deprecate
            denoising_loss_weight (int, optional): Weight of the denoising loss. Defaults to 1.
            variance_type (str, optional): Type of variance used to add noise. Defaults to "fixed_small".
                    Valid:  [fixed_small, fixed_small_log, fixed_large, fixed_large_log, learned or learned_range
            elucidated_diffusion (bool, optional): Whether to use Elucidated Diffusion. Defaults to False. TODO: Deprecate
            beta_start (float, optional): Starting beta value. Defaults to 5e-5.
            beta_end (float, optional): Ending beta value. Defaults to 5e-2.
        """
        super().__init__()
        self.vae_model = None

        self.is_elucidated_diffusion = elucidated_diffusion
        if elucidated_diffusion:
            self.diffusion_model = ElucidatedDiffusion(
                net=model, seq_length=latent_in_features
            )
        else:
            self.diffusion_model = GaussianDiffusion1D(
                model=model,
                n_dims=latent_in_features,
                num_steps=diffusion_timesteps,
                loss_type=diffusion_loss,
                beta_schedule=beta_schedule,
                beta_start=beta_start,
                beta_end=beta_end,
                noise_scheduler_type=noise_scheduler_type,
                variance_type=variance_type,
            )

        self.is_conditioned = is_conditioned

        # TODO: Deprecate this
        self.joint_training = joint_training

        self.loss_weight = denoising_loss_weight

        # self.vae_model_loaded = False
        self.is_vae_frozen = False

    @property
    def use_grasp_qualities(self):
        """Get whether grasp qualities are used for training"""
        return self.vae_model.use_grasp_qualities

    @property
    def scheduler_type(self):
        """Get Diffusion noise scheduler type"""
        return self.diffusion_model._noise_scheduler_type

    @property
    def _latent_loss_objects(self):
        """Hotfix to check where annealing should be applied"""
        return self.vae_model._latent_loss_objects

    def set_vae_model(self, vae_model):
        """Set VAE model

        Args:
            vae_model (nn.Module): VAE model
        """
        self.vae_model = vae_model
        return

    def load_vae_weights(self, state_dict):
        """Update VAE weights

        Args:
            state_dict (dict): VAE state dict
        """
        self.vae_model.load_state_dict(state_dict, strict=True)
        return

    def set_inference_timesteps(self, num_inference_steps):
        """Set the number of inference steps for reverse diffusion sampler

        Args:
            num_inference_steps (int): Number of inference steps
        """
        self.diffusion_model.set_inference_timesteps(num_inference_steps)
        return

    def freeze_vae_model(self):
        for param in self.vae_model.parameters():
            param.requires_grad = False
        self.vae_model.eval()
        self.is_vae_frozen = True
        return

    def forward(self, pc, grasps, compute_loss=None, **kwargs):
        """Training Forward Pass: Computes loss for batched pc and grasps

        Args:
            pc (torch.Tensor): Input point cloud (batch_size, num_points, 3)
            grasps (torch.Tensor): Input grasps (batch_size*num_grasps, 6/7)
            compute_loss (bool, optional): Whether to compute loss. Defaults to None.
                        TODO: Remove this dummy flag and harmonize other derived classes

        Returns:
            torch.Tensor: Generated grasps of shape (batch_size, 6/7)

        """
        # This is to avoid having unfrozen vae model
        # This can happen when resuming training from a model checkpoint
        # TODO: Improve vae freezing from train.py or callbacks
        if not self.is_vae_frozen:
            self.freeze_vae_model()
            warnings.warn("VAE model was frozen manually after loading")
            self.print_params_info()
            self.is_vae_frozen = True

        z_pc_cond = None

        (mu_h, logvar_h, z_h), (
            mu_pc,
            logvar_pc,
            z_pc_cond,
        ) = self.vae_model.encode(pc, grasps)

        denoising_loss = self.diffusion_model(
            z_h.unsqueeze(1), z_cond=z_pc_cond, **kwargs
        )

        if self.joint_training:
            denoising_loss *= self.loss_weight
            grasps_out = self.vae_model.decoder(z_h, z_pc_cond)
            loss_dict = self.vae_model._loss_fn(
                x_in=grasps,
                x_out=grasps_out,
                mu_h=mu_h,
                logvar_h=logvar_h,
                mu_pc=mu_pc,
                logvar_pc=logvar_pc,
            )
            loss_dict.denoising_loss = denoising_loss
            loss_dict.loss = loss_dict.loss + denoising_loss

        else:
            grasps_out = None
            loss_dict = Dict(loss=denoising_loss, denoising_loss=denoising_loss)

        return grasps_out, loss_dict

    @torch.no_grad()
    def generate_grasps(self, xyz, num_grasps=10, return_intermediate=False, **kwargs):
        """Generation/Sampling : Generates grasps from a given point cloud

        Args:
            xyz (torch.Tensor): Input point cloud of shape (batch_size, num_points, 3)
            num_grasps (int, optional): Number of grasps to generate per point cloud. Defaults to 10.
            return_intermediate (bool, optional): Whether to return intermediate outputs. Defaults to False.

        Returns:
            torch.Tensor: Generated grasps of shape (batch_size*num_grasps, 6/7)
            torch.Tensor: Intermediate outputs of shape (batch_size, num_grasps, num_steps, latent_dim)
                            or empty list [] if return_intermediate is False
        """
        # batch_size = xyz.shape[0]
        z_pc_cond = self.vae_model.encode_pc(xyz)

        # Repeat interleave latents as per num_grasps
        z_pc_cond = z_pc_cond.repeat_interleave(num_grasps, dim=0)

        # Sampling batch size i.e. how many random vectors are sampled from prior
        # Should be of size (num_grasps * num_pc_batches) == z_pc_cond.shape[0]
        # num_grasps generation per pc is taken care of in score network
        # where z_pc_cond is appropriately applied

        sampling_batch_size = z_pc_cond.shape[0]
        out, all_outs = self.diffusion_model.sample(
            z_cond=z_pc_cond,
            batch_size=sampling_batch_size,
            return_all=return_intermediate,
            **kwargs
        )
        out = self.vae_model.decoder(out.squeeze(-2), z_pc_cond)

        if not return_intermediate:
            return (out, [])
        else:
            step_outs = []

            # Cannot do this whole loop on gpu if return all is True
            for idx in torch.linspace(0, len(all_outs) - 1, steps=50, dtype=torch.int):
                _out = self.vae_model.decoder(all_outs[idx].squeeze(-2), z_pc_cond)
                inter_out = [_out_tensor.detach().cpu() for _out_tensor in _out]
                step_outs.append(inter_out)
        return out, step_outs

    def print_params_info(self):
        """Prints model parameters information"""

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable_params = sum(
            p.numel() for p in self.parameters() if not p.requires_grad
        )

        print("------------------------------------------------")
        print("Model Trainable Parameters: ", trainable_params)
        print("Model Non-Trainable Parameters: ", non_trainable_params)
        print("------------------------------------------------")
