import glob
import os
import warnings
from abc import abstractmethod
from enum import Enum
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from grasp_ldm.dataset.builder import build_dataset_from_cfg
from grasp_ldm.models.builder import build_model_from_cfg
from grasp_ldm.utils.config import Config
from grasp_ldm.utils.rotations import tmrp_to_H
from grasp_ldm.utils.torch_utils import fix_state_dict_prefix
from grasp_ldm.utils.vis import visualize_pc_grasps


class Conditioning(Enum):
    UNCONDITIONAL = "NORMAL"
    CLASS_CONDITIONED = "CLASS_CONDITIONED"
    REGION_CONDITIONED = "REGION_CONDITIONED"


class ModelType(Enum):
    LDM = "LDM"
    VAE = "VAE"


def unnormalize_pc(pc: Tensor, metas: dict) -> Tensor:
    """Unnormalize pointcloud per dataset statistics (in metas)

        N: pointcloud num points

    Args:
        pc (Tensor): pointcloud [N,3]
        metas (dict): meta information with statistics `pc_std`
                and `pc_mean` by which input pointcloud was normalized

    Raises:
        NotImplementedError: if `dataset_normalized` is not mentioned in metas, unnormalization may not be correct

    Returns:
        Tensor: Unnormalized Tensor [N,3]
    """
    assert isinstance(
        pc, torch.Tensor
    ), "Torch implementation for unnormalize requires a pointcloud as a Tensor"

    if pc.ndim == 2:
        pc_unnorm = pc * metas["pc_std"].to(pc.device) + metas["pc_mean"].to(pc.device)
    elif pc.ndim == 3:
        # batches of pc[B,N,3], pc_mean[B,3] and pc_std[B,3]
        pc_unnorm = pc * metas["pc_std"].unsqueeze(-2).to(pc.device) + metas[
            "pc_mean"
        ].unsqueeze(-2).to(pc.device)
    else:
        raise NotImplementedError

    return pc_unnorm


def unnormalize_grasps(grasps: Tensor, metas: dict) -> Tensor:
    """Unnormalize pointcloud per dataset statistics (in metas)

        B: Batch size / num grasps

    Args:
        pc (Tensor): Grasp as batches of t-mrp [B,6]
        metas (dict): meta information with statistics `grasp_std`
                and `grasp_mean` by which input grasp pose is normalized in the dataset

    Raises:
        NotImplementedError: if `dataset_normalized` is not mentioned in metas, unnormalization may not be correct

    Returns:
        Tensor: Unnormalized Tensor [N,3]
    """
    assert (
        isinstance(grasps, torch.Tensor) and grasps.shape[-1] == 6
    ), "Torch implementation for unnormalize expects grasps as a [..., 6] Tensor"

    # if grasps.ndim == 2:
    #     pc_unnorm = pc * metas["pc_std"] + metas["pc_mean"]
    # elif pc.ndim == 3:
    #     # batches of pc[B,N,3], pc_mean[B,3] and pc_std[B,3]
    #     pc_unnorm = pc * metas["pc_std"].unsqueeze(1) + metas["pc_mean"](1)
    # else:
    #     raise NotImplementedError

    return grasps * metas["grasp_std"].unsqueeze(-2).to(grasps.device) + metas[
        "grasp_mean"
    ].unsqueeze(-2).to(grasps.device)


class Experiment:
    def __init__(
        self,
        exp_name,
        exp_out_root=f"output",
        modes=["vae", "ddm", "elucidated_ddm"],
        vae_ckpt_path=None,
        ddm_ckpt_path=None,
        elucidated_ckpt_path=None,
    ) -> None:
        self.exp_name = exp_name

        self.exp_dir = os.path.join(exp_out_root, exp_name)
        self._modes = modes

        assert os.path.isdir(self.exp_dir), FileNotFoundError(
            f"No experiment directory `{exp_name}` found in `output/`"
        )

        self._config_paths = {
            mode: glob.glob(f"{self.exp_dir}/{mode}/*.py") for mode in self._modes
        }

        _manual_ckpt_paths = dict(
            vae=vae_ckpt_path if "vae" in modes else None,
            ddm=ddm_ckpt_path if "ddm" in modes else None,
            elucidated_ddm=elucidated_ckpt_path if "elucidated_ddm" in modes else None,
        )

        self._ckpt_paths = self._resolve_ckpt_paths(_manual_ckpt_paths)

    def get_config(self, mode):
        assert mode in self._modes, f"Could not find mode ({mode}) in experiment modes "
        config_path = self._config_paths[mode][0]
        return Config.fromfile(config_path)

    def get_ckpt_path(self, mode):
        return self._ckpt_paths[mode]

    def _resolve_ckpt_paths(self, manual_ckpt_paths):
        _default_ckpt_paths = {}

        for mode in self._modes:
            default_path = f"{self.exp_dir}/{mode}/checkpoints/last.ckpt"

            enforce_path = manual_ckpt_paths[mode]

            enforce_path_exists = (
                os.path.isfile(enforce_path) if enforce_path is not None else False
            )

            selected_path = default_path if not enforce_path_exists else enforce_path

            if not os.path.isfile(selected_path):
                raise FileNotFoundError(
                    f"For given mode ({mode}) in `modes`:"
                    f"Could not find any checkpoint in ckpt path: {selected_path}"
                )

            _default_ckpt_paths[mode] = selected_path

        return _default_ckpt_paths


class Inference:
    def __init__(self) -> None:
        # Settable from derived class
        self._model = None
        self.dataset = None

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, set_model: nn.Module):
        assert isinstance(set_model, nn.Module)
        self._model = set_model
        return

    def build_model(
        self, model_config, device="cuda:0" if torch.cuda.is_available() else "cpu"
    ):
        model = build_model_from_cfg(model_config)
        model.eval()
        return model

    def build_dataset(self, config, split: str = "test"):
        # This copies all the init params from train dataset and changes the split from which samples are drawn
        if not hasattr(config.data, split):
            config.data.test = config.data.train.copy()
            config.data.test.args.split = split

        dataset = build_dataset_from_cfg(
            config.data,
            split=split,
        )
        dataset.load_grasp_data()

        return dataset

    def generate_grasps(self, **kwargs):
        assert (
            self.model is not None
        ), "`model` property was not initialized. Set the model before calling `generate_grasps`"

        return self.model.generate_grasps(**kwargs)

    def to_numpy(self, tensor):
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()
        elif isinstance(tensor, np.ndarray):
            pass
        else:
            raise TypeError(
                f"Input should be of type torch.Tensor, but found {type(tensor)}"
            )

        return tensor

    def visualize(self, pointcloud, grasps, confidences, return_scene=False):
        pointcloud = self.to_numpy(pointcloud)
        grasps = self.to_numpy(grasps)
        confidences = self.to_numpy(confidences)

        scene = visualize_pc_grasps(pointcloud, grasps, confidences)

        if not return_scene:
            scene.show()
            return

        return scene

    def visualize_fancy(
        self, pointcloud, grasps, confidences, window_size=(1280, 960), label="grasps"
    ):
        # TODO: Glooey vis rotating
        scene = self.visualize(pointcloud, grasps, confidences, return_scene=True)
        _ = GlooeyWidget(size=window_size, scenes=[scene], labels=[label])
        return

    @abstractmethod
    def generate_grasps(self, **kwargs):
        raise NotImplementedError

    def to_cpu(self, tensors):
        "Detaches and moves all tensors to cpu"
        if isinstance(tensors, Sequence):
            cpu_outs = []
            for out in tensors:
                cpu_outs.append(self.to_cpu(out))
        elif isinstance(tensors, torch.Tensor):
            cpu_outs = tensors.detach().cpu()
        elif isinstance(tensors, dict):
            cpu_outs = {
                k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                for k, v in tensors.items()
            }
        else:
            raise NotImplementedError(
                f"`to_cpu()` not implemented for input type: {type(tensors)}"
            )
        return cpu_outs

    def infer(
        self,
        data_idx=None,
        num_grasps: int = 10,
        visualize: bool = False,
        condition_type: Conditioning = Conditioning.UNCONDITIONAL,
        conditioning: Any = None,
        **kwargs,
    ):
        """Infer a given index in the Dataset
        if data_idx is given, infer on that example
        else, use a random data_idx in the dataset

        Args:
            data_idx (_type_, optional): _description_. Defaults to None.
            num_grasps (int, optional): _description_. Defaults to 10.
            visualize (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        data_idx = (
            data_idx
            if data_idx is not None
            else np.random.randint(0, len(self.dataset))
        )
        assert data_idx < len(self.dataset), "data index out of range"

        dataitem = self.dataset.__getitem__(index=data_idx)
        cache = dataitem.copy()
        pc = dataitem["pc"]
        metas = dataitem["metas"]

        if condition_type == Conditioning.CLASS_CONDITIONED:
            results = self.generate_class_conditioned_grasps(
                pc,
                num_grasps=num_grasps,
                metas=metas,
                class_label=conditioning,
                **kwargs,
            )
        elif condition_type == Conditioning.REGION_CONDITIONED:
            results = self.generate_region_conditioned_grasps(
                pc, num_grasps=num_grasps, metas=metas, region_id=conditioning
            )
        else:
            results = self.generate_grasps(
                pc, num_grasps=num_grasps, metas=metas, **kwargs
            )

        results["inputs"] = cache

        if visualize:
            scene = self.visualize(
                pointcloud=results["pc"].squeeze(0),
                grasps=results["grasps"].squeeze(0),
                confidences=results["confidence"].squeeze(0),
                return_scene=True,
            )

            # scene.add_geometry(
            #     trimesh.points.PointCloud(
            #         results["region_pc"].squeeze().cpu().numpy(), colors=(255, 0, 0)
            #     )
            # )
            return scene
        else:
            return results

    def generate_class_conditioned_grasps(
        self,
        pc,
        num_grasps: int = 10,
        metas=None,
        data_idx=None,
        class_label: int = 0,
        **kwargs,
    ):
        """Infer a given index in the Dataset
        if data_idx is given, infer on that example
        else, use a random data_idx in the dataset

        Args:
            data_idx (_type_, optional): _description_. Defaults to None.
            num_grasps (int, optional): _description_. Defaults to 10.
            visualize (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """

        # Pre-process conditioning labels
        class_cond_labels = torch.LongTensor([class_label])
        class_cond_labels = (
            class_cond_labels.unsqueeze(0)
            .repeat((num_grasps, 1))
            .to(self.device, dtype=torch.float32)
        )

        # Add to metas
        metas["mode_cls"] = class_cond_labels

        # Generate grasps
        return self.generate_grasps(pc, metas=metas, num_grasps=num_grasps, **kwargs)

    def generate_region_conditioned_grasps(
        self,
        pc,
        num_grasps: int = 10,
        metas=None,
        data_idx=None,
        region_id: int = 0,
    ):
        """Region conditioned grasp selection

        Currently, regions are obtained from the dataset __getitem__ method

        """

        # Pre-process conditioning labels
        region_cond_labels = torch.LongTensor([region_id])
        region_cond_labels = (
            region_cond_labels.unsqueeze(0).repeat((1, num_grasps)).to(self.device)
        )

        # Add to metas
        metas["grasp_region_labels"] = region_cond_labels.to(self.device)

        # Hot fix: unsqueeze to give a fake batch dimension for shape
        # tuple to work in PointsTimeConditionedResNet1D
        metas["region_points"] = metas["region_points"].unsqueeze(0).to(self.device)
        metas["num_grasps"] = torch.Tensor([num_grasps])
        # Generate grasps
        results = self.generate_grasps(pc, metas, num_grasps=num_grasps)
        results["region_pc"] = unnormalize_pc(
            metas["region_points"].squeeze(0)[region_id].cpu(), metas
        )
        return results


class InferenceLDM(Inference):
    def __init__(
        self,
        exp_name,
        exp_out_root,
        data_root=None,
        data_split="test",
        use_ema_model=True,
        ddm_ckpt_path=None,
        vae_ckpt_path=None,
        elucidated_ckpt_path=None,
        use_elucidated=False,
        use_fast_sampler=True,
        num_inference_steps=None,
        augment_pc=False,
        load_dataset=True,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    ) -> None:
        super().__init__()
        _modes = ["ddm"] if not use_elucidated else ["elucidated_ddm"]
        _modes = _modes + ["vae"] if vae_ckpt_path is not None else _modes

        self.experiment = Experiment(
            exp_name=exp_name,
            exp_out_root=exp_out_root,
            modes=_modes,
            vae_ckpt_path=vae_ckpt_path,
            ddm_ckpt_path=ddm_ckpt_path,
            elucidated_ckpt_path=elucidated_ckpt_path,
        )

        self.use_ema_model = use_ema_model
        self.ddm_mode = "ddm" if not use_elucidated else "elucidated_ddm"
        self.device = device
        self.do_augment = augment_pc
        self.config = self.experiment.get_config(self.ddm_mode)

        self._setup_ldm_sampler(
            num_inference_steps=num_inference_steps, use_fast_sampler=use_fast_sampler
        )

        self.num_inference_steps = num_inference_steps

        self.ckpt_path = self.experiment.get_ckpt_path(self.ddm_mode)

        self.model = self.load_model(use_vae_ckpt_path=vae_ckpt_path)

        if load_dataset:
            if data_root is not None:
                self._patch_data_root(data_root)
            self._patch_data_split(data_split, load_dataset=load_dataset)
            self.dataset = self.build_dataset(config=self.config, split=data_split)

        else:
            self.dataset = None

        self._sigmoid = nn.Sigmoid()

    @property
    def exp_dir(self):
        return self.experiment.exp_dir

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

    def _patch_data_root(self, data_root):
        assert (
            hasattr(self, "config") and self.config is not None
        ), "Method was called out of order, no config found"

        # Patch data root dir
        self.config.data.train.args.data_root_dir = data_root
        return

    def _patch_data_split(self, split="test", load_dataset=False, augs=None):
        # Patch split
        if not hasattr(self.config.data, split):
            self.config.data[split] = self.config.data.train.copy()
            self.config.data[split].args.split = split
            self.config.data[split].args.augs_config = augs
            self.config.data[split].args.num_repeat_dataset = 1

        if self.config.data.train.type == "AcronymPartialPointclouds":
            self.config.data[split].args.preempt_load_data = load_dataset
        return

    # TODO: load weights without PL wrapper
    def load_model(self, use_vae_ckpt_path=None):
        model = build_model_from_cfg(self.config.model.ddm)
        model.set_vae_model(build_model_from_cfg(self.config.model.vae))

        # State dict contains weights of both normal model and ema model at that ckpt
        # Use appropriate prefix to load weights
        state_dict = torch.load(self.ckpt_path)["state_dict"]
        model_prefix = "model" if not self.use_ema_model else "ema_model.online_model"
        state_dict = fix_state_dict_prefix(
            state_dict, model_prefix, ignore_all_others=True
        )

        # cfg_vae_ckpt_path = self.experiment.vae.ckpt_path
        # if cfg_vae_ckpt_path is not None:
        #     assert os.path.exists(
        #         cfg_vae_ckpt_path
        #     ), f"Checkpoint {cfg_vae_ckpt_path} does not exist."
        # else:
        #     cfg_vae_ckpt_path = f"{self.experiment.exp_dir}/vae/checkpoints/last.ckpt"

        #     # Load VAE weights
        #     state_dict = torch.load(cfg_vae_ckpt_path)["state_dict"]
        #     if self.use_ema_model:
        #         state_dict = self._fix_state_dict_prefix(
        #             state_dict, "ema_model.online_model", ignore_all_others=True
        #         )
        #     else:
        #         state_dict = self._fix_state_dict_prefix(
        #             state_dict, "model", ignore_all_others=True
        #         )
        #     vae_model.load_vae_weights(state_dict=state_dict)

        try:
            missing_keys, unexpected_keys = model.load_state_dict(
                state_dict, strict=True
            )
            if missing_keys:
                warnings.warn(f"Missing keys while loading state dict: {missing_keys}")

            if unexpected_keys:
                warnings.warn(
                    f"Found unexpected keys while loading state dict: {unexpected_keys}"
                )
        except Exception as e:
            msg = f"Error while loading state dict: You might be using an incompatible state dict. \n"
            if self.use_ema_model:
                msg += f"EMA model is requested but may not be available. Check and set the `use_ema_model` flag appropriately."

            msg += f"Error: {e}"

            raise RuntimeError(msg)

        return model.eval().cuda()

        return pl_ddm_model.model.eval().cuda()

    def normalize_input(self, pc):
        assert pc.ndim in (2, 3)

        # Center on pc mean
        pc_mean = torch.mean(pc, dim=-2)
        pc -= pc_mean.unsqueeze(1) if pc.ndim == 3 else pc_mean

        # Normalize per dataset statistics
        pc = (pc - self.PC_MEAN) / self.PC_STD

        # Compute metas for unnormalization
        grasp_mean = self.GRASP_MEAN.unsqueeze(0).repeat(pc.shape[0], 1)
        grasp_mean[..., :3] += pc_mean

        metas = dict(
            pc_mean=self.PC_MEAN + pc_mean,
            pc_std=self.PC_STD.unsqueeze(0),
            grasp_mean=grasp_mean,
            grasp_std=self.GRASP_STD.unsqueeze(0),
            dataset_normalized=True,
        )
        return (pc, metas)

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

    def infer_on_pointcloud(self, pc, num_grasps=10, return_intermediate=False):
        pc_normalized, metas = self.normalize_input(pc)

        return self.generate_grasps(
            pc_normalized,
            metas,
            num_grasps=num_grasps,
            return_intermediate=return_intermediate,
        )


class InferenceVAE(Inference):
    def __init__(
        self,
        exp_name,
        exp_out_root,
        use_ema_model=True,
        data_root=None,
        data_split="test",
        ddm_ckpt_path=None,
        vae_ckpt_path=None,
        augment_pc=False,
        load_dataset=True,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    ) -> None:
        super().__init__()

        _modes = ["vae"]
        self.experiment = Experiment(
            exp_name=exp_name,
            exp_out_root=exp_out_root,
            modes=_modes,
            vae_ckpt_path=vae_ckpt_path,
            ddm_ckpt_path=ddm_ckpt_path,
        )

        self.device = device
        self.do_augment = augment_pc
        self.config = self.experiment.get_config("vae")
        self.ckpt_path = self.experiment.get_ckpt_path("vae")
        self.use_ema_model = use_ema_model
        self.model = self.load_model()

        if load_dataset:
            if data_root is not None:
                self._patch_data_root(data_root)
            self._patch_data_split(data_split, load_dataset=load_dataset)
            self.dataset = self.build_dataset(config=self.config, split=data_split)
        else:
            self.dataset = None

        self._sigmoid = nn.Sigmoid()

    @property
    def exp_dir(self):
        return self.experiment.exp_dir

    def load_model(self):
        # compatibility issue with old configs
        model_key = "model" if "model" in self.config else "models"
        model = build_model_from_cfg(self.config[model_key].vae)

        # State dict contains weights of both normal model and ema model at that ckpt
        # Use appropriate prefix to load weights
        state_dict = torch.load(self.ckpt_path)["state_dict"]
        model_prefix = "model" if not self.use_ema_model else "ema_model.online_model"
        state_dict = fix_state_dict_prefix(
            state_dict, model_prefix, ignore_all_others=True
        )
        try:
            missing_keys, unexpected_keys = model.load_state_dict(
                state_dict, strict=True
            )
            if missing_keys:
                warnings.warn(f"Missing keys while loading state dict: {missing_keys}")

            if unexpected_keys:
                warnings.warn(
                    f"Found unexpected keys while loading state dict: {unexpected_keys}"
                )
        except Exception as e:
            msg = f"Error while loading state dict: You might be using an incompatible state dict. \n"
            if self.use_ema_model:
                msg += f"EMA model is requested but may not be available. Check and set the `use_ema_model` flag appropriately. \n"

            msg += f"Error: {e}"

            raise RuntimeError(msg)

        return model.eval().cuda()

    def _patch_data_root(self, data_root):
        assert (
            hasattr(self, "config") and self.config is not None
        ), "Method was called out of order, no config found"

        # Patch data root dir
        self.config.data.train.args.data_root_dir = data_root
        return

    def _patch_data_split(self, split="test", load_dataset=True, augs=None):
        # Patch split
        if not hasattr(self.config.data, split):
            self.config.data[split] = self.config.data.train.copy()
            self.config.data[split].args.split = split
            self.config.data[split].args.augs_config = augs
            self.config.data[split].args.num_repeat_dataset = 1

        if self.config.data.train.type == "AcronymPartialPointclouds":
            self.config.data[split].args.preempt_load_data = load_dataset
        return

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
        pc = batch_pcs.view((num_pcs_in, pc.shape[-2], pc.shape[-1]))
        tmrp = tmrp.view((num_pcs_in, num_grasps, tmrp.shape[-1]))

        # Unnormalize
        pc_unnorm = pc * metas["pc_std"].unsqueeze(-2) + metas["pc_mean"].unsqueeze(-2)
        grasp_unnorm = tmrp[..., :6].to(pc.device) * metas["grasp_std"].unsqueeze(
            -2
        ) + metas["grasp_mean"].unsqueeze(-2)

        # Convert to 4x4 homogenous matrices
        H_grasps = tmrp_to_H(grasp_unnorm)

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


class AcronymGroundTruthModel:
    class DummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.op = nn.Identity()

        def forward(self, x):
            return self.op(x)

    def __init__(
        self, exp_name, split, data_root, config_path=None, debug=False
    ) -> None:
        self.model = self.DummyModel()
        self.exp_name = exp_name
        self.exp_dir = os.path.join("output", exp_name)
        self.config = self.get_config(config_path=config_path)
        self.dataset = self.build_dataset(
            config=self.config, split=split, data_root=data_root
        )
        self.debug = debug

    def infer(self, data_idx, num_grasps=10, randomize=True):
        dataitem = self.dataset.__getitem__(data_idx)
        grasps = dataitem["grasps"][..., :6]
        if randomize:
            grasps = grasps[torch.randperm(grasps.shape[0])]

        grasps = unnormalize_grasps(grasps, dataitem["metas"])
        grasps = tmrp_to_H(grasps[:num_grasps][..., :6])

        if self.debug:
            from grasp_ldm.utils.vis import visualize_pc_grasps

            pc_vis = unnormalize_pc(dataitem["pc"], dataitem["metas"]).cpu().numpy()
            grasps_vis = grasps.cpu().numpy()
            scene = visualize_pc_grasps(pc_vis, grasps_vis)
            scene.show()

        return dict(grasps=grasps, inputs=dataitem)

    def get_config(self, config_path=None):
        if config_path is None:
            # Take config from one of the inference directories
            config_path = glob.glob(f"{self.exp_dir}/vae/*.py") + glob.glob(
                f"{self.exp_dir}/ddm/*.py"
            )

            assert (
                len(config_path) > 0
            ), "Could not find a config file in inference directories. Provide an explicit path."

        return Config.fromfile(config_path[0])

    def build_dataset(self, config, data_root, split):
        if not hasattr(self.config.data, split):
            self.config.data[split] = self.config.data.train.copy()
            self.config.data[split].args.split = split

        self.config.data[split].args.num_repeat_dataset = 1
        self.config.data[split].args.data_root_dir = data_root
        self.config.data[split].args.batch_num_grasps_per_pc = 200
        self.config.data[split].args.augs_config = None

        dataset = (
            build_dataset_from_cfg(config.data.train)
            if split == "train"
            else build_dataset_from_cfg(config, split)
        )
        dataset.load_grasp_data()
        return dataset
