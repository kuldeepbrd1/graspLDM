from enum import Enum
import enum
from typing import Tuple

import torch

from grasp_ldm.dataset.acronym import FILTER_63_CATEGORIES
from grasp_ldm.dataset.augmentations import BaseAugmentation, BaseGeneralAugmentation
from grasp_ldm.utils.gripper import SimplePandaGripper
from grasp_ldm.utils.rotations import tmrp_to_H, get_random_rotations_in_angle_limit

from .acronym_pointclouds import AcronymShapenetPointclouds
from .acronym_partial_pointclouds import AcronymPartialPointclouds
import numpy as np


class BaseGraspPointsClassificationMixin:
    """These methods will be called first over inherited methods due to MRO

    Still probably better to rename them
    """

    def make_data_item(self, pc, grasps, metas, qualities, use_bogus=False):
        # Gripper point cloud
        grasp_points = (
            tmrp_to_H(grasps[..., :6]) @ self.gripper_points.transpose(-1, -2)
        ).transpose(-1, -2)
        grasp_points = grasp_points[..., :3]

        # Grasp success labels
        success_labels = grasps[..., 6].view(-1)

        if use_bogus:
            # Add grasp points for bogus grasps (colliding and free-space)
            bogus_grasps = self.get_bogus_grasps(
                pc,
                grasps,
                int(self.bogus_grasp_fraction * self.batch_num_grasps_per_pc),
                fraction_perturbed=0.5,
            )

            bogus_grasps = bogus_grasps[..., :3]

            grasp_points = torch.cat((grasp_points, bogus_grasps), dim=0)
            success_labels = torch.cat(
                (
                    success_labels,
                    torch.zeros(bogus_grasps.shape[0], dtype=torch.long),
                ),
                dim=0,
            )

            random_indices = torch.randperm(grasp_points.shape[0])
            grasp_points = grasp_points[random_indices]
            success_labels = success_labels[random_indices]

        # self.visualize_raw(pc, grasps)
        # Enforce float32 for all tensors
        pc = pc.to(torch.float32)
        qualities = qualities.to(torch.float32)

        # Pre-proc and augmentations
        (
            pc,
            grasp_points,
            preproc_metas,
        ) = self.preprocess_data(pc, grasp_points)

        metas.update(preproc_metas)

        # Last step sanity check for dataloader
        if grasps.shape[0] < self.batch_num_grasps_per_pc:
            raise RuntimeError(
                "Fatal: `grasps` was empty. This should not happen in data loading"
            )

        # Debug visualization
        # scene = self.visualize(pc, grasp_points, success_labels, show=True)

        return dict(
            pc=pc,
            grasps=grasp_points,
            success=success_labels,
            qualities=qualities,
            metas=metas,
        )

    def preprocess_data(
        self, pc: torch.Tensor, grasp_points: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Preprocess data
            Center pc/grasps on pc mean, apply augmentations and normalize inputs

        Args:
            pc (torch.Tensor): pointcloud tensor of shape [1, N, 3]
            grasps (torch.Tensor): grasp points tensor of shape [G, P, 3]

        Returns:
            Tuple[torch.Tensor, torch.Tensor, dict]: pc, grasps, metas
        """

        metas = {}

        # Center points and grasp translations on pc mean
        pc_mean = pc.mean(dim=-2)
        pc -= pc_mean
        grasp_points -= pc_mean

        # Apply augmentations
        if self.augmentations is not None:
            pc = pc.unsqueeze(0) if pc.ndim == 2 else pc
            pc, grasp_points = self.apply_augmentations(pc, grasp_points)
            pc = pc.squeeze(0) if pc.ndim == 3 and pc.shape[0] == 1 else pc

        # Input normalization
        pc = pc = (pc - self._INPUT_PC_SHIFT) / self._INPUT_PC_SCALE
        grasp_points = (grasp_points - self._INPUT_PC_SHIFT) / self._INPUT_PC_SCALE

        # meta info stores exact normalization params used for this object
        # total mean/shift = centering mean + input normalization mean
        metas["pc_mean"] = self._INPUT_PC_SHIFT + pc_mean
        metas["pc_std"] = self._INPUT_PC_SCALE
        metas["grasp_mean"] = []
        metas["grasp_std"] = []
        metas["use_dataset_statistics"] = False

        return pc, grasp_points, metas

    def apply_augmentations(
        self, pc: torch.Tensor, grasp_points: torch.Tensor
    ) -> torch.Tensor:
        """Apply augmentations

        TODO: Add support for other Pose Representations (H)

        Args:
            pc (torch.Tensor): pointcloud tensor of shape [1, N, 3]
            grasps (torch.Tensor): grasps tensor of shape [G, 6(+1)]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: pc[1, N, 3], grasps[G, 6(+1))]
        """

        assert pc.shape[-1] == 3

        for aug in self.augmentations:
            if isinstance(aug, BaseAugmentation):
                if aug._transforms_pc and aug._transforms_grasps:
                    raise Exception(
                        "Grasp poses are not used and therefore cannot be augmented in `AcronymGraspToPoints`"
                    )
                elif aug._transforms_pc and not aug._transforms_grasps:
                    pc = aug(pc)
                else:
                    raise NotImplementedError
            elif isinstance(aug, BaseGeneralAugmentation):
                aug.reset()
                pc = aug.apply(pc)
                grasp_points = aug.apply(grasp_points)
            else:
                raise NotImplementedError

        return pc, grasp_points

    def get_bogus_grasps(
        self,
        pc: torch.Tensor,
        grasp_poses: torch.Tensor,
        num_bogus: int,
        fraction_perturbed=0.5,
    ):
        """Get bogus grasps

        Args:
            pc (torch.Tensor): pointcloud tensor of shape [N, 3]
            grasps (torch.Tensor): grasp poses tensor of shape [G, 4, 4]
            num_bogus (int): Number of bogus grasps to generate
            fraction_perturbed (float, optional): Fraction of bogus grasps
                        that are randomly perturbed. Defaults to 0.5.

        """
        grasp_poses = tmrp_to_H(grasp_poses[:num_bogus, :6])

        # Pull or push the gripper points to bottom center
        # Pulled grasps give us free-space grasps
        # Pushed grasps give us collision grasps
        gripper_points_z_pull = self.gripper_points.clone()
        gripper_points_z_pull[..., :3] -= torch.from_numpy(
            np.array(SimplePandaGripper.BOTTOM_CENTER)
        )

        gripper_points_z_push = self.gripper_points.clone()
        gripper_points_z_push[..., :3] += torch.from_numpy(
            np.array(SimplePandaGripper.BOTTOM_CENTER)
        )

        # Concatenate push and pull gripper points, to select randomly between the two
        gripper_points = torch.cat(
            (gripper_points_z_pull.unsqueeze(0), gripper_points_z_push.unsqueeze(0)), 0
        )

        # Get number of grasps perturbed by a random transformation
        num_perturbed_grasps = int(num_bogus * fraction_perturbed)
        random_indices = torch.randperm(grasp_poses.shape[0])[:num_perturbed_grasps]

        # Apply a normally distributed random shift such that the
        # 3-sigma bounds are the dimensions of the pc bounding box
        # i.e 99.7% of the bogus grasps will be shifted not more than
        # the approximate size of the object
        pc_size = pc.max(-2).values - pc.min(-2).values
        random_shift = torch.rand((num_perturbed_grasps, 3)) * pc_size / 3

        # Get random rotations
        random_rotations = get_random_rotations_in_angle_limit(
            360, batch_size=num_perturbed_grasps
        )

        # Make H matrices for random rotations and shift
        H_aug = (
            torch.eye(4)
            .repeat((grasp_poses.shape[0], 1, 1))
            .to(dtype=pc.dtype, device=pc.device)
        )
        H_aug[random_indices, :3, :3] = random_rotations
        H_aug[random_indices, :3, 3] = random_shift

        # Use one of the push/pull gripper points configurations
        # and apply random H_aug to get bogus grasps
        # Apply grasp pose to get bogus grasps in desired (e.g. camera) frame
        bogus_grasps = (
            grasp_poses
            @ H_aug
            @ gripper_points[
                torch.randint(len(gripper_points), (num_bogus,))
            ].transpose(-1, -2)
        ).transpose(-1, -2)

        return bogus_grasps

    def visualize(self, pc, grasps, success_labels=None, show=True):
        import trimesh

        pc = pc.squeeze(0).cpu().numpy()
        grasps = grasps.squeeze(0).cpu().numpy()

        r = pc[..., 0] * 255 / max(pc[..., 0])
        g = pc[..., 1] * 200 / max(pc[..., 1])
        b = pc[..., 2] * 175 / max(pc[..., 2])
        a = np.ones(pc.shape[0]) * 200

        pc_colors = np.clip(np.vstack((r, g, b, a)).T, 0, 255)

        pc = trimesh.points.PointCloud(pc, colors=pc_colors)

        grasp_pcs = []
        for idx in range(min(10, grasps.shape[0])):
            colors = (
                ([0, 255, 0] if success_labels[idx] == 1 else [255, 0, 0])
                if success_labels is not None
                else [0, 0, 255]
            )
            grasp_pcs.append(trimesh.points.PointCloud(grasps[idx], colors=colors))

        scene = trimesh.Scene([pc] + grasp_pcs)

        if not show:
            return scene
        else:
            scene.show(
                viewer="gl", line_settings=dict(point_size=5), flags=dict(axis=True)
            )
            return None


class AcronymFullPcGraspPointsClassification(
    BaseGraspPointsClassificationMixin,
    AcronymShapenetPointclouds,
):
    def __init__(
        self,
        data_root_dir: str,
        split: list = "train",
        filter_categories: list = ...,
        rotation_repr="mrp",
        augs_config=None,
        batch_num_points_per_pc=1024,
        batch_num_grasps_per_pc=20,
        batch_failed_grasps_ratio: float = 0.3,
        load_fixed_subset_grasps_per_obj=None,
        use_dataset_statistics_for_norm: bool = False,
        num_repeat_dataset: int = 2,
        gripper_points_file: str = "data/gripper/gripper_points_76.npy",
    ) -> None:
        super().__init__(
            data_root_dir,
            split,
            filter_categories,
            rotation_repr,
            augs_config,
            batch_num_points_per_pc,
            batch_num_grasps_per_pc,
            batch_failed_grasps_ratio,
            load_fixed_subset_grasps_per_obj,
            use_dataset_statistics_for_norm,
            num_repeat_dataset,
        )

        _gripper_points = np.load(gripper_points_file)
        _gripper_points = torch.from_numpy(_gripper_points)
        _gripper_points = torch.cat(
            (_gripper_points, torch.ones(_gripper_points.shape[0], 1)), dim=1
        ).to(torch.float32)

        self.gripper_points = _gripper_points

        self.num_gripper_points = self.gripper_points.shape[0]

    def __getitem__(self, index: int):
        """Get Item for dataloader

            This provides normalized inputs for the networks, with
            pointcloud and grasps centered at the pointcloud mean.

            Depending on the `use_dataset_statistics` flag, additional scaling (std)
            may be done using dataset statistics or fixed normalization parameters.

            When unnormalizing, use `pc_mean`, `pc_std`, `grasp_mean` and `grasp_std`
            in `metas` to recover unnormalized pc and grasps as follows:

            ```
            pc_original = pc_normalized * metas["pc_std"] + metas["pc_mean"]
            grasps_original = grasps_normalized * metas["grasp_std"] + metas["grasp_mean"]
            ```

            G: Number of grasps
            N: Number of points in pointcloud

        Args:
            index (int): data index

        Raises:
            RuntimeError: If `grasps` was empty. This should not happen in data loading

        Returns:
            dict: Batch sample dict with keys:
                "pc": (torch.Tensor) pointcloud tensor of shape [1, N, 3]
                "grasps": (torch.Tensor) grasps tensor of shape [G, 6 (+1)]
                "qualities": (torch.Tensor) grasp qualities tensor of shape [G, 4]
                "metas": (dict) metadata dict with keys:
                    "pc_mean": (torch.Tensor) pointcloud mean of shape [3]
                    "pc_std": (torch.Tensor) pointcloud std of shape [3]
                    "grasp_mean": (torch.Tensor) grasp mean of shape [6]
                    "grasp_std": (torch.Tensor) grasp std of shape [6]

        """
        grasp_info = self.__supergetitem__(index)

        # Clone to avoid modifying the original dataitem
        dataitem = grasp_info.copy()

        mesh = dataitem["mesh"]
        grasps = dataitem["grasps"]
        qualities = dataitem["qualities"]
        metas = dataitem["metas"]

        # Pointcloud sample from mesh surface
        pc, sample_indices = mesh.sample(
            self.batch_num_points_per_pc, return_index=True
        )
        pc = torch.from_numpy(pc)

        return self.make_data_item(
            pc=pc, grasps=grasps, qualities=qualities, metas=metas
        )


class AcronymPartialPcGraspPointsClassification(
    BaseGraspPointsClassificationMixin,
    AcronymPartialPointclouds,
):
    def __init__(
        self,
        data_root_dir: str,
        camera_json: str,
        split: list = "train",
        rotation_repr="mrp",
        augs_config=None,
        batch_num_points_per_pc=1024,
        batch_num_grasps_per_pc=20,
        depth_px_scale=10000,
        scene_prefix="scene_",
        batch_failed_grasps_ratio: float = 0.3,
        num_repeat_dataset: int = 2,
        max_scenes=None,
        max_num_pc_per_scene=20,
        gripper_points_file: str = "data/gripper/gripper_points_76.npy",
    ) -> None:
        super().__init__(
            data_root_dir,
            camera_json,
            num_points_per_pc=batch_num_points_per_pc,
            num_grasps_per_obj=batch_num_grasps_per_pc,
            rotation_repr=rotation_repr,
            max_scenes=max_scenes,
            augs_config=augs_config,
            split=split,
            depth_px_scale=depth_px_scale,
            scene_prefix=scene_prefix,
            min_usable_pc_points=1024,
            preempt_load_data=True,
            use_failed_grasps=True
            if batch_failed_grasps_ratio is not None and batch_failed_grasps_ratio > 0
            else False,
            failed_grasp_ratio=batch_failed_grasps_ratio,
            load_fixed_grasp_transforms=None,
            is_input_dataset_normalized=False,
            num_pc_per_scene=max_num_pc_per_scene,
            num_repeat_dataset=num_repeat_dataset,
        )

        _gripper_points = np.load(gripper_points_file)
        _gripper_points = torch.from_numpy(_gripper_points)
        _gripper_points = torch.cat(
            (_gripper_points, torch.ones(_gripper_points.shape[0], 1)), dim=1
        ).to(torch.float32)

        self.gripper_points = _gripper_points

        self.num_gripper_points = self.gripper_points.shape[0]

        self.bogus_grasp_fraction = 0.2

    def __getitem__(self, index: int):
        if not hasattr(self, "depth_data"):
            raise RuntimeError("The dataset was not loaded")

        # Map query index to data index, if dataset is repeated
        mapped_idx = self._map_to_data_index(index)

        pc, grasps, metas = self.get_pc_grasps(mapped_idx)

        pc = pc.to(torch.float32)
        grasps = grasps.to(torch.float32)
        grasp_qualities = torch.tensor([])  # grasp_qualities.to(torch.float32)

        return self.make_data_item(
            pc=pc, grasps=grasps, qualities=grasp_qualities, metas=metas, use_bogus=True
        )
