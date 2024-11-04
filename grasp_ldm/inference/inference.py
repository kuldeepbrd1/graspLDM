import warnings

import torch
from torch import nn

from ..models.builder import build_model_from_cfg
from ..utils.config import Config
from ..utils.rotations import tmrp_to_H
from .inference_base import Inference, unnormalize_grasps, unnormalize_pc


class InferenceLDM(Inference):
    def __init__(
        self,
        config_path,
        ckpt_path,
        num_inference_steps=None,
        augment_pc=False,
        use_fast_sampler=False,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    ) -> None:
        super().__init__()

        # Config
        self.config = Config.fromfile(config_path)
        self.do_augment = augment_pc

        # Model and weights
        self.ckpt_path = ckpt_path
        self.device = device
        self.ddm_mode = "ddm"
        self.model = self.load_model()
        self._setup_ldm_sampler(
            num_inference_steps=num_inference_steps, use_fast_sampler=use_fast_sampler
        )

        # sigmoid for cls layer
        self._sigmoid = nn.Sigmoid()

    def _setup_ldm_sampler(self, num_inference_steps, use_fast_sampler):
        if use_fast_sampler:
            if self.ddm_mode == "ddm":
                self.config.models.ddm.model.args.noise_scheduler_type = "ddim"
                fast_sampler = "DDIM"
                num_inference_steps = (
                    100 if num_inference_steps is None else num_inference_steps
                )

            elif self.ddm_mode == "elucidated_ddm":
                fast_sampler = "DPMPP"
                num_inference_steps = (
                    32 if num_inference_steps is None else num_inference_steps
                )

            else:
                NotImplementedError
        else:
            if self.ddm_mode == "ddm" and num_inference_steps is None:
                num_inference_steps = 1000
            if self.ddm_mode == "elucidated_ddm" and num_inference_steps is None:
                num_inference_steps = 32
            fast_sampler = None

        self.num_inference_steps = num_inference_steps
        self.fast_sampler = fast_sampler

        return

    def load_model(self):
        vae_model = build_model_from_cfg(self.config.models.vae)

        # if use_vae_ckpt_path is not None and os.path.isfile(use_vae_ckpt_path):
        #     pl_vae_model.load_state_dict(torch.load(use_vae_ckpt_path)["state_dict"])

        ldm_model = build_model_from_cfg(self.config.models.ddm)

        ldm_model.vae_model = vae_model
        ldm_model.load_state_dict(torch.load(self.ckpt_path)["state_dict"])

        return ldm_model.eval().to(device=self.device)

    def generate_grasps(self, pc, metas, num_grasps=10, cls_cond=None, **kwargs):
        batch_pcs = (pc.unsqueeze(0) if pc.ndim == 2 else pc).to(self.device)

        metas = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in metas.items()
        }

        num_pcs_in = batch_pcs.shape[0]

        in_kwargs = dict(xyz=batch_pcs, metas=metas, **kwargs)

        if self.fast_sampler == "DPMPP":
            in_kwargs["use_dpmpp"] = True
            in_kwargs["num_sample_steps"] = self.num_inference_steps
        elif self.fast_sampler == "DDIM":
            self.model.set_inference_timesteps(self.num_inference_steps)

        # start = time.time()
        final_grasps, all_diffusion_grasps = self.model.generate_grasps(
            num_grasps=num_grasps, **in_kwargs
        )
        # print(f"Sampling time: {time.time() - start}")

        if self.model.vae_model.decoder._use_qualities:
            tmrp, cls_logit, qualities = final_grasps
            qualities = qualities.view((num_pcs_in, num_grasps, qualities.shape[-1]))
        else:
            tmrp, cls_logit = final_grasps
            qualities = None

        # Results
        pc = batch_pcs.view((num_pcs_in, pc.shape[-2], pc.shape[-1]))
        tmrp = tmrp.view((num_pcs_in, num_grasps, tmrp.shape[-1]))
        grasp_unnorm = unnormalize_grasps(tmrp, metas)
        H_grasps = tmrp_to_H(grasp_unnorm)

        if all_diffusion_grasps:
            if batch_pcs.shape[0] > 1:
                raise NotImplementedError(
                    "Batched grasps for all diffusion steps are not implemented"
                )

            all_steps_grasps = []
            for step_grasp in all_diffusion_grasps:
                all_steps_grasps.append(
                    tmrp_to_H(unnormalize_grasps(step_grasp[0], metas))
                )
        else:
            all_steps_grasps = []

        confidence = cls_logit.view((num_pcs_in, num_grasps, cls_logit.shape[-1]))
        confidence = self._sigmoid(confidence)

        pc_unnorm = unnormalize_pc(pc, metas)

        return dict(
            grasps=H_grasps,
            grasp_tmrp=grasp_unnorm,
            confidence=confidence,
            qualities=qualities,
            pc=pc_unnorm,
            all_steps_grasps=all_steps_grasps,
        )


class InferenceVAE(Inference):
    def __init__(
        self,
        config_path,
        ckpt_path,
        augment_pc=False,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    ) -> None:
        super().__init__()

        self.device = device
        self.do_augment = augment_pc
        self.config = Config.fromfile(config_path)
        self.ckpt_path = ckpt_path
        self.model = self.load_model()

        # Take normalization coeffs from config
        self.set_normalization_params(norm_config=self.config.data.norm_config)

        # Sigmoid for cls logits
        self._sigmoid = nn.Sigmoid()

    @property
    def exp_dir(self):
        return self.experiment.exp_dir

    def load_model(self):
        model = build_model_from_cfg(self.config.models.vae)

        missing_keys, unexpected_keys = model.load_state_dict(
            torch.load(self.ckpt_path)["state_dict"], strict=False
        )

        if missing_keys:
            warnings.warn(f"Missing keys while loading state dict: {missing_keys}")

        if unexpected_keys:
            warnings.warn(
                f"Found unexpected keys while loading state dict: {unexpected_keys}"
            )
        return model.eval().to(device=self.device)

    def generate_grasps(self, pc, metas, num_grasps=10, **kwargs):
        # Batch pcs
        batch_pcs = (pc.unsqueeze(0) if pc.ndim == 2 else pc).to(self.device)

        metas = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in metas.items()
        }

        num_pcs_in = batch_pcs.shape[0]

        # Infer
        final_grasps = self.model.generate_grasps(batch_pcs, num_grasps)

        # Split outputs
        if self.model.decoder._use_qualities:
            tmrp, cls_logit, qualities = final_grasps
            qualities = qualities.view((num_pcs_in, num_grasps, qualities.shape[-1]))
        else:
            tmrp, cls_logit = final_grasps
            qualities = None

        # Split batches
        pc = batch_pcs.view((num_pcs_in, pc.shape[-2], pc.shape[-1])).clone().detach()
        tmrp = tmrp.view((num_pcs_in, num_grasps, tmrp.shape[-1])).clone().detach()

        # Unnormalize
        pc_unnorm = pc * metas["pc_std"].unsqueeze(-2) + metas["pc_mean"].unsqueeze(-2)
        grasp_unnorm = tmrp[..., :6].to(pc.device) * metas["grasp_std"].unsqueeze(
            -2
        ) + metas["grasp_mean"].unsqueeze(-2)

        # Convert to 4x4 homogenous matrices
        H_grasps = tmrp_to_H(grasp_unnorm.clone().detach())

        # Get class confidence
        confidence = cls_logit.view((num_pcs_in, num_grasps, cls_logit.shape[-1]))
        confidence = self._sigmoid(confidence)

        return dict(
            grasps=H_grasps,
            grasp_tmrp=grasp_unnorm,
            confidence=confidence,
            qualities=qualities,
            pc=pc_unnorm,
        )
