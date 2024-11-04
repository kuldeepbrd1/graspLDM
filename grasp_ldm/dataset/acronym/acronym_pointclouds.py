from typing import Tuple

import torch
import tqdm
import trimesh
from addict import Dict

from ..augmentations import Augmentations, BaseAugmentation
from . import FILTER_63_CATEGORIES
from .acronym import AcronymBaseDataset


class AcronymShapenetPointclouds(AcronymBaseDataset):
    # Different scale factors because of imbalance of mrp values vs translation values
    _TRANSLATION_SCALE_FACTOR = 0.05
    _ROTATION_SCALE_FACTOR = 0.5

    def __init__(
        self,
        data_root_dir: str,
        split: list = "train",
        filter_categories: list = FILTER_63_CATEGORIES,
        rotation_repr="mrp",
        augs_config=None,
        batch_num_points_per_pc=1024,
        batch_num_grasps_per_pc=20,
        batch_failed_grasps_ratio: float = 0.3,
        load_fixed_subset_grasps_per_obj=None,
        use_dataset_statistics_for_norm: bool = False,
        num_repeat_dataset: int = 2,
    ) -> None:
        """Acronym Dataset for Pointclouds
            Inherited from AcronymBaseDataset, which provides grasps and meshes
            Overridden __getitem__ to return pointclouds and implements normalization, augmentation etc helpers

        Args:
            data_root_dir (str): Path to the root directory of the dataset. Must contain the "grasps" and "meshes" folders.
            split (list, optional): Dataset split to use. Defaults to "train".
            filter_categories (list, optional): List of categories to use from all ShapenetSem categories. Defaults to FILTER_63_CATEGORIES.
            rotation_repr (str, optional): Rotation representation to use. Defaults to "mrp".
            augs_config (dict, optional): Augmentations config. Defaults to None.
                    The config should be a list of config dict, that follows the normal initialization config format
                    for classes in the augmentations.py file.:
                Example:
                    augs_config = [
                        dict(type="RandomRotation", args(p=0.5, max_angle=180, is_degree=True)),
                        dict(type="PointcloudJitter", args(p=0.25, sigma=0.01, clip=0.05)),
                        dict(type="RandomPointcloudDropout", args(p=0.5, max_dropout_ratio=0.6)),
                    ]

            batch_num_points_per_pc (int, optional): Number of points to sample from the pointcloud. Defaults to 1024.
            batch_num_grasps_per_pc (int, optional): Number of grasps to sample for the pointcloud. Defaults to 20.
            batch_failed_grasps_ratio (float, optional): Ratio of failed grasps to sample from the pointcloud. Defaults to 0.3.
            load_fixed_subset_grasps_per_obj (int, optional): If a fixed subset of grasps should be loaded. Defaults to None.
                    If set to 0. or None, all grasps are loaded. Otherwise specified number of grasps to load in the fixed subset.

            use_dataset_statistics_for_norm (bool, optional): Whether to use mean/shift std/scale computed over
                    all the data samples for input normalization or to use fixed scaling at input.
                    Defaults to False.
            num_repeat_dataset (int, optional): Number of times to repeat the dataset. Defaults to 0.
                    This is useful to speed up training by not having to restart dataloader workers.
        """

        super().__init__(
            data_root_dir=data_root_dir,
            split=split,
            filter_categories=filter_categories,
            rotation_repr=rotation_repr,
            min_num_grasps=batch_num_grasps_per_pc,
            num_grasps_fixed_grasp_subset=load_fixed_subset_grasps_per_obj,
        )

        # Batch sample attributes
        self.batch_num_points_per_pc = batch_num_points_per_pc
        self.batch_num_grasps_per_pc = batch_num_grasps_per_pc
        self.batch_failed_grasp_ratio = batch_failed_grasps_ratio

        # Normalization
        self._use_norm_dataset_statistics = False
        self._set_normalization_params(use_dataset_statistics_for_norm)

        # Augmentations
        self._augs_config = augs_config
        self.augmentations = (
            Augmentations.build_augmentations_from_cfg(augs_cfg=augs_config)
            if augs_config is not None
            else None
        )

        # Repeat dataset
        self.num_repeat_dataset = (
            num_repeat_dataset
            if num_repeat_dataset is not None and num_repeat_dataset > 0
            else 1
        )

    def __len__(self) -> int:
        return len(self.grasp_infos) * self.num_repeat_dataset

    def pre_load(self) -> None:
        """Pre-load dataset to memory if __getitem__ depends on it"""
        if self.grasp_infos is None:
            self.load_grasp_data()

        return

    def _map_to_data_index(self, idx: int) -> int:
        """Map index to orginal index for repeated dataset"""
        return idx % super().__len__()

    def __supergetitem__(self, index: int, num_grasps: int = None) -> dict:
        index = index if not self.num_repeat_dataset else self._map_to_data_index(index)

        # Get mesh and grasps for data index from base class
        # dataitem is a dict with keys: "mesh", "grasps", "qualities" and "metas"
        grasp_info = super().__getitem__(
            index,
            num_grasps=self.batch_num_grasps_per_pc
            if num_grasps is None
            else num_grasps,
            ratio_bad_grasps=self.batch_failed_grasp_ratio,
        )
        return grasp_info

    def __getitem__(self, index: int) -> dict:
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

        ## Add sampled indices to metas
        # metas["pc_sample_indices"] = sample_indices

        # Enforce float32 for all tensors
        pc = pc.to(torch.float32)
        grasps = grasps.to(torch.float32)
        qualities = qualities.to(torch.float32)

        # Pre-proc and augmentations
        pc, grasps, preproc_metas = self.preprocess_data(pc, grasps)
        metas.update(preproc_metas)

        # Last step sanity check for dataloader
        if grasps.shape[0] < self.batch_num_grasps_per_pc:
            raise RuntimeError(
                "Fatal: `grasps` was empty. This should not happen in data loading"
            )

        return dict(
            pc=pc,
            grasps=grasps,
            qualities=qualities,
            metas=metas,
        )

    def preprocess_data(
        self, pc: torch.Tensor, grasps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Preprocess data
            Center pc/grasps on pc mean, apply augmentations and normalize inputs

        Args:
            pc (torch.Tensor): pointcloud tensor of shape [1, N, 3]
            grasps (torch.Tensor): grasps tensor of shape [G, 6 (+1)]

        Returns:
            Tuple[torch.Tensor, torch.Tensor, dict]: pc, grasps, metas
        """

        metas = {}

        # Center points and grasp translations on pc mean
        pc, grasps, pc_mean = self.center_on_pc_mean(pc, grasps)

        # Apply augmentations
        if self.augmentations is not None:
            pc = pc.unsqueeze(0) if pc.ndim == 2 else pc
            pc, grasps = self.apply_augmentations(pc, grasps)
            pc = pc.squeeze(0) if pc.ndim == 3 and pc.shape[0] == 1 else pc

        # Input normalization
        pc, grasps = self.normalize_inputs(pc, grasps)

        # final grasp mean = centering mean + input normalization mean
        grasp_mean = self._INPUT_GRASP_SHIFT.clone()

        grasp_mean[..., :3] += pc_mean

        # meta info stores exact normalization params used for this object
        # total mean/shift = centering mean + input normalization mean
        metas["pc_mean"] = self._INPUT_PC_SHIFT + pc_mean
        metas["pc_std"] = self._INPUT_PC_SCALE
        metas["grasp_mean"] = grasp_mean
        metas["grasp_std"] = self._INPUT_GRASP_SCALE
        metas["use_dataset_statistics"] = self._use_norm_dataset_statistics

        return pc, grasps, metas

    def center_on_pc_mean(
        self, pc: torch.Tensor, grasps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Center pointcloud and grasps on pointcloud mean

        Args:
            pc (torch.Tensor): pointcloud tensor of shape [1, N, 3]
            grasps (torch.Tensor): grasps tensor of shape [G, 6]

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: pc[1, N, 3], grasps[G, 7], pc_mean[3,]
        """

        assert pc.ndim in (2, 3)

        pc_mean = torch.mean(pc, dim=-2)

        pc -= pc_mean.unsqueeze(1) if pc.ndim == 3 else pc_mean
        grasps[..., :3] -= pc_mean
        return pc, grasps, pc_mean

    def normalize_inputs(
        self, pc: torch.Tensor, grasps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalize inputs

        Args:
            pc (torch.Tensor): pointcloud tensor of shape [1, N, 3]
            grasps (torch.Tensor): grasps tensor of shape [G, 6]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: pc [1, N, 3], grasps [G, 6]
        """
        assert pc.shape[-1] == 3
        assert grasps.shape[-1] == 6 or grasps.shape[-1] == 7

        pc = (pc - self._INPUT_PC_SHIFT) / self._INPUT_PC_SCALE
        grasps[..., :6] = (
            grasps[..., :6] - self._INPUT_GRASP_SHIFT
        ) / self._INPUT_GRASP_SCALE

        return pc, grasps

    def unnormalize_outputs(
        self, pc: torch.Tensor, grasps: torch.Tensor, metas: dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unnormalize outputs

        Args:
            pc (torch.Tensor): pointcloud tensor of shape [1, N, 3]
            grasps (torch.Tensor): grasps tensor of shape [G, 6(+1)]
            metas (dict): metadata dict with keys:
                "pc_mean": (torch.Tensor) pointcloud mean of shape [3]
                "pc_std": (torch.Tensor) pointcloud std of shape [3]
                "grasp_mean": (torch.Tensor) grasp mean of shape [6]
                "grasp_std": (torch.Tensor) grasp std of shape [6]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: pc[1, N, 3], grasps[G, 7]
        """
        assert pc.shape[-1] == 3
        assert grasps.shape[-1] == 6 or grasps.shape[-1] == 7

        pc = pc * metas["pc_std"] + metas["pc_mean"]
        grasps[..., :6] = grasps[..., :6] * metas["grasp_std"] + metas["grasp_mean"]

        return pc, grasps

    def apply_augmentations(
        self, pc: torch.Tensor, grasps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply augmentations

        TODO: Add support for other Pose Representations (H)

        Args:
            pc (torch.Tensor): pointcloud tensor of shape [1, N, 3]
            grasps (torch.Tensor): grasps tensor of shape [G, 6(+1)]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: pc[1, N, 3], grasps[G, 6(+1))]
        """

        assert pc.shape[-1] == 3
        assert grasps.shape[-1] == 6 or grasps.shape[-1] == 7

        for aug in self.augmentations:
            assert isinstance(
                aug, BaseAugmentation
            ), "Augmentations must be of type BaseAugmentation"

            if aug._transforms_pc and aug._transforms_grasps:
                pc, grasps[..., :6] = aug(pc, grasps[..., :6])
            elif aug._transforms_pc and not aug._transforms_grasps:
                pc = aug(pc)
            else:
                raise NotImplementedError
        return pc, grasps

    def _set_normalization_params(self, use_dataset_statistics: bool) -> None:
        """Set dataset-wide normalization (shift/scale) params

        Args:
            use_dataset_statistics (bool): If True, use dataset statistics
                for normalization. If False, use fixed normalization parameters.
        """
        self._use_norm_dataset_statistics = use_dataset_statistics

        # Since pc and grasps are centered around pc_mean on per object basis,
        # set dataset-wide shift to zero
        self._INPUT_PC_SHIFT = torch.zeros((3,))
        self._INPUT_GRASP_SHIFT = torch.zeros((6,))

        if use_dataset_statistics:
            if self.grasp_infos is None:
                self.grasp_infos = self._load_all_obj_grasps()

            (_, pc_std), (_, grasp_std) = self.get_dataset_statistics()

            self._INPUT_PC_SCALE = torch.Tensor(pc_std)
            self._INPUT_GRASP_SCALE = torch.Tensor(grasp_std)
        else:
            self._INPUT_PC_SCALE = torch.ones((3,)) * self._TRANSLATION_SCALE_FACTOR
            self._INPUT_GRASP_SCALE = torch.cat(
                (
                    torch.ones((3,)) * self._TRANSLATION_SCALE_FACTOR,
                    torch.ones((3,)) * self._ROTATION_SCALE_FACTOR,
                ),
            )
        return

    def get_dataset_statistics(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get dataset statistics

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                pc_mean[3,], pc_std[3,], grasp_mean[6,], grasp_std[6,]
        """

        grasps_dataset = torch.zeros((1, 6))
        pcs_dataset = torch.zeros(1, 3)

        for index in tqdm.tqdm(
            range(len(self.grasp_infos)), desc="Collecting dataset input statistics"
        ):
            grasp_filename = list(self.grasp_infos)[index]

            ## Mesh
            obj_scale = self.grasp_infos[grasp_filename]["obj_scale"]
            mesh_path = self.grasp_infos[grasp_filename]["mesh_path"]
            mesh = self.get_object_mesh(mesh_path, obj_scale)

            # Pointcloud
            pc, _ = trimesh.sample.sample_surface(mesh, self.batch_num_points_per_pc)
            pc = torch.from_numpy(pc).to(torch.float32)

            ## Grasps
            grasps = self.grasp_infos[grasp_filename]["grasps"].clone()
            # bad_grasps = self.obj_grasp_infos[grasp_filename]["failed_grasps"]

            pc, grasps, _ = self.center_on_pc_mean(pc, grasps)

            pcs_dataset = torch.concatenate((pcs_dataset, pc), dim=0)
            grasps_dataset = torch.concatenate((grasps_dataset, grasps), dim=0)

        pc_mean = torch.mean(pcs_dataset, dim=0)
        pc_std = torch.std(pcs_dataset, dim=0)

        grasp_mean = torch.mean(grasps_dataset, dim=0)
        grasp_std = torch.std(grasps_dataset, dim=0)

        # print(
        #     f"\nPointcloud Statistics: \nMean: {pc_mean.numpy()} \nSD: {pc_std.numpy()}"
        #     f"\n\nGrasp Representation Statistics \nMean: {grasp_mean.numpy()} \n SD: {grasp_std.numpy()}"
        # )

        return (pc_mean, pc_std), (grasp_mean, grasp_std)
