import glob
import os
from abc import abstractmethod
from enum import Enum
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from grasp_ldm.models.builder import build_model_from_cfg
from grasp_ldm.utils.config import Config
from grasp_ldm.utils.rotations import tmrp_to_H
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

    return grasps * metas["grasp_std"].unsqueeze(-2).to(grasps.device) + metas[
        "grasp_mean"
    ].unsqueeze(-2).to(grasps.device)


class Inference:
    def __init__(self) -> None:
        # Settable from derived class
        self._model = None
        self.dataset = None
        self.device = None

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, set_model: nn.Module):
        assert isinstance(set_model, nn.Module)
        self._model = set_model
        return

    def set_normalization_params(self, norm_config):
        # Check norm config
        assert hasattr(norm_config, "pc_shift"), "norm_config should have `pc_shift`"
        assert hasattr(
            norm_config, "grasp_shift"
        ), "norm_config should have `grasp_shift`"
        assert hasattr(
            norm_config, "translation_scale"
        ), "norm_config should have `translation_scale`"
        assert hasattr(
            norm_config, "rotation_scale"
        ), "norm_config should have `rotation_scale`"

        # set norm params
        self._INPUT_PC_SHIFT = torch.tensor(
            norm_config.pc_shift, dtype=torch.float32, device=self.device
        )
        self._INPUT_GRASP_SHIFT = torch.tensor(
            norm_config.grasp_shift, dtype=torch.float32, device=self.device
        )
        _ones = torch.ones((3,), dtype=torch.float32, device=self.device)
        self._INPUT_PC_SCALE = _ones.clone() * norm_config.translation_scale
        self._INPUT_GRASP_SCALE = torch.cat(
            (
                _ones.clone() * norm_config.translation_scale,
                _ones.clone() * norm_config.rotation_scale,
            ),
        )
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
            config,
            split=split,
        )
        dataset.load_grasp_data()

        return dataset

    def generate_grasps(self, **kwargs):
        assert (
            self.model is not None
        ), "`model` property was not initialized. Set the model before calling `generate_grasps`"

        return self.model.generate_grasps(**kwargs)

    def generate_on_pointcloud(self, pc, num_grasps=10, return_intermediate=False):
        """Generate grasps on a given pointcloud

        Args:
            pc (torch.Tensor): pointcloud [N,3]
            num_grasps (int, optional): Number of grasps to generate. Defaults to 10.
            return_intermediate (bool, optional): Return intermediate grasps from diffusion. Defaults to False.
                                    Unused in others

        Returns:
            dict: Results dictionary from generate_grasps
        """
        pc_normalized, metas = self.normalize_input(pc)

        return self.generate_grasps(
            pc_normalized,
            metas,
            num_grasps=num_grasps,
            return_intermediate=return_intermediate,
        )

    def normalize_input(self, pc):
        """Normalize input pointcloud

        Args:
            pc (torch.Tensor): raw input pointcloud [N,3]

        Returns:
            tuple: (pc, metas) where pc is normalized pointcloud and metas is a dict with normalization statistics
        """

        assert pc.ndim in (2, 3)

        # Center on pc mean
        pc_mean = torch.mean(pc, dim=-2)
        pc -= pc_mean.unsqueeze(1) if pc.ndim == 3 else pc_mean

        # Normalize per dataset statistics
        pc = (pc - self._INPUT_PC_SHIFT) / self._INPUT_PC_SCALE

        # Compute metas for unnormalization
        grasp_mean = self._INPUT_GRASP_SHIFT
        grasp_mean[..., :3] += pc_mean

        metas = {}
        metas["pc_mean"] = self._INPUT_PC_SHIFT + pc_mean
        metas["pc_std"] = self._INPUT_PC_SCALE
        metas["grasp_mean"] = grasp_mean
        metas["grasp_std"] = self._INPUT_GRASP_SCALE
        metas["use_dataset_statistics"] = False

        return (pc, metas)

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
