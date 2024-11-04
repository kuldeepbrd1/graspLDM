import glob
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import torch
import tqdm
import trimesh
from torch.utils.data import Dataset

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from grasp_ldm.utils.rotations import PoseRepresentation, H_to_tmrp
from grasp_ldm.utils.torch_utils import minmax_normalize


def load_data_splits(root_dir: str):
    """Load train/test data splits

    Args:
        root_dir (str): path to acronym data root

    Returns:
        dict: dict of category-wise train/test object grasp files
    """
    split_dict = {}
    split_paths = glob.glob(os.path.join(root_dir, "splits/*.json"))
    for split_p in split_paths:
        category = os.path.basename(split_p).split(".json")[0]
        splits = json.load(open(split_p, "r"))
        split_dict[category] = {}
        split_dict[category]["train"] = [
            obj_p.replace(".json", ".h5") for obj_p in splits["train"]
        ]
        split_dict[category]["test"] = [
            obj_p.replace(".json", ".h5") for obj_p in splits["test"]
        ]
    return split_dict


class AcronymBaseDataset(Dataset):
    # Valid rotation representations for grasp transforms (used in _getitem_)
    VALID_REPRESENTATIONS = {"mrp": PoseRepresentation.TMRP, "H": PoseRepresentation.H}

    def __init__(
        self,
        data_root_dir: str,
        split: list = "train",
        filter_categories: list = None,
        rotation_repr="mrp",
        min_num_grasps=100,
        num_grasps_fixed_grasp_subset=None,
    ) -> None:
        """Base class for Acronym dataset

        Args:
            data_root_dir (str): path to acronym data root
            split (list, optional): train/test split. Defaults to "train".
            filter_categories (list, optional): list of categories to filter. Defaults to None.
            rotation_repr (str, optional): rotation representation for grasp transforms. Defaults to "mrp".
            min_num_grasps (int, optional): minimum number of grasps to return. Defaults to 100.
            num_grasps_fixed_grasp_subset (int, optional): number of grasps to use from a fixed grasp subset. Defaults to None.
        """
        super().__init__()

        # Logging: Settable logger. print() if None
        self.logger = None

        # Directories
        self.root_dir = data_root_dir
        self.acronym_grasps_dir = os.path.join(self.root_dir, "grasps")
        self.mesh_dir = os.path.join(self.root_dir, "meshes")

        # Load data splits: category -> train/test
        self.data_splits = load_data_splits(data_root_dir)
        self.split = split

        # Filter categories
        self._full_category_list = list(self.data_splits.keys())

        if filter_categories:
            self._set_filtered_category_list(filter_categories)
        else:
            self.category_list = self._full_category_list

        # Pose representation
        self._pose_representation = self.VALID_REPRESENTATIONS[rotation_repr]

        # Default number of output grasps on _getitem_ call
        # Used to discard objects with num_grasps < min_num_grasps to avoid errors in dataloader
        self._min_num_grasps = min_num_grasps

        # Optional: Using a small fixed grasp subset to learn from
        # Has shown better generation results at low model capacity
        if num_grasps_fixed_grasp_subset is not None:
            assert (num_grasps_fixed_grasp_subset > 0) and (
                num_grasps_fixed_grasp_subset >= min_num_grasps
            ), f"`num_grasps_fixed_grasp_subset` value must be greater than the given `min_num_grasps` ({min_num_grasps}))"

        self._use_fixed_grasp_subset = num_grasps_fixed_grasp_subset is not None
        self._num_grasps_fixed_grasp_subset = num_grasps_fixed_grasp_subset

        # Restrict dataset to 5 items for debugging
        # # TODO: REMOVE THIS PLEASE AFTER DEBUGGING
        # num_debug_samples = 10
        # Pre-load all grasps from relevant h5 files
        if self.split == "train":
            # self.data_splits["Cup"]["train"] = self.data_splits["Cup"]["train"][
            #     :num_debug_samples
            # ]
            self.grasp_infos = self._load_all_obj_grasps()
        else:
            self.grasp_infos = None

    def __len__(self):
        return len(self.grasp_infos)

    def __getitem__(self, index, num_grasps=100, ratio_bad_grasps=0.0) -> dict:
        """Get item at index

        Args:
            index (int): index
            num_grasps (int, optional): number of total grasps to return. Defaults to 100.
            ratio_bad_grasps (float, optional): ratio of bad grasps to return in the batch sample. Defaults to None.

        Returns:
            dict: dict of grasps, qualities, mesh, metas
                dict is of the form:
                {
                    "grasps": torch.Tensor of shape [num_grasps, 7],
                    "qualities": torch.Tensor of shape [num_grasps, 1],
                    "mesh": trimesh.Trimesh,
                    "metas": dict of meta infos
                }

        Raises:
            AssertionError: if `grasp_infos` is None, i.e. grasps were not loaded
        """

        assert (
            self.grasp_infos is not None
        ), "Grasp infos not loaded. Cannot call __getitem__"

        # Get dataitem at query index
        grasp_filename = list(self.grasp_infos)[index]
        dataitem = self.grasp_infos[grasp_filename]

        # Mesh
        mesh = self.get_object_mesh(
            mesh_path=dataitem["mesh_path"],
            scale=dataitem["obj_scale"],
        )

        # Grasps
        grasps, grasp_qualities = self.get_grasps(
            dataitem, num_grasps, ratio_bad_grasps
        )

        # Meta-infos
        metas = dict(
            quality_order=dataitem["quality_order"],
            scale=dataitem["obj_scale"],
            category=dataitem["mesh_category"],
            mesh_path=dataitem["mesh_path"],
            num_grasps=num_grasps,
            index=index,
        )

        return dict(grasps=grasps, qualities=grasp_qualities, mesh=mesh, metas=metas)

    def get_obj_category(self, index: int) -> str:
        """Get object category from index

        Args:
            index (int): index

        Returns:
            str: object category
        """
        return list(self.grasp_infos)[index].split("_")[0]

    def load_grasp_data(self):
        """Alias public method to allow user-control of when to load or not load dataset apriori"""
        self.grasp_infos = self._load_all_obj_grasps()

    def set_logger(self, logger: Any) -> None:
        """Set logger

        Args:
            logger (Any): logger, must have a `log` method
        """
        self.logger = logger
        return

    def get_grasps(
        self, dataitem: dict, num_grasps: int = 100, ratio_bad_grasps: float = 0.0
    ):
        """Get grasps and qualities

        Args:
            dataitem (dict): dict of dataitem
            num_grasps (int, optional): Number of grasps to sample. Defaults to 100.
            ratio_bad_grasps (float, optional): Ratio of bad grasps to sample. Defaults to 0..

        Returns:
            tuple: tuple of grasps and grasp qualities
        """
        assert (
            ratio_bad_grasps >= 0.0 and ratio_bad_grasps <= 1.0
        ), "ratio_bad_grasps must be between 0 and 1"

        mix_good_bad_grasps = ratio_bad_grasps != 0.0

        # Splits
        num_good_grasps = int(num_grasps * (1 - ratio_bad_grasps))
        num_bad_grasps = num_grasps - num_good_grasps

        # Good grasps
        good_grasps = dataitem["grasps"]
        good_grasp_qualities = dataitem["grasp_qualities"]

        # Append success flag =1 to good grasps
        good_grasps = torch.concat(
            (good_grasps, torch.ones((good_grasps.shape[0], 1))), dim=-1
        )

        if mix_good_bad_grasps:
            # Bad grasps
            bad_grasps = dataitem["bad_grasps"]
            bad_grasp_qualities = dataitem["bad_grasp_qualities"]

            # Append success flag =0 to bad grasps
            bad_grasps = torch.concat(
                (bad_grasps, torch.zeros((bad_grasps.shape[0], 1))), dim=-1
            )

            # Randomly sample good and bad grasps
            good_idxs = torch.randperm(good_grasps.shape[0])[:num_good_grasps]
            bad_idxs = torch.randperm(bad_grasps.shape[0])[:num_bad_grasps]

            # Concatenate
            grasps = torch.concat((good_grasps[good_idxs], bad_grasps[bad_idxs]), dim=0)
            grasp_qualities = torch.concat(
                (good_grasp_qualities, bad_grasp_qualities), dim=0
            )
        else:
            grasps = good_grasps
            grasp_qualities = good_grasp_qualities

        # Shuffle grasp order
        random_idcs = torch.randperm(grasps.shape[0])[:num_grasps]
        grasps = grasps[random_idcs]
        grasp_qualities = grasp_qualities[random_idcs]

        return grasps, grasp_qualities

    def get_meshname_from_acronym_file(self, acronym_file: str) -> Tuple[str, str]:
        """Get mesh filename from acronym h5 file/path

        Args:
            acronym_file (str): Acronym h5 filename/path

        Returns:
            Tuple[str, str]: (category name, mesh filename)
        """

        filename = acronym_file.split("_")[1] + ".obj"
        cat = acronym_file.split("_")[0]
        return cat, filename

    def get_object_mesh(self, mesh_path: str, scale: float) -> trimesh.Trimesh:
        """Get object mesh

        Args:
            mesh_path (str): mesh path
            scale (float): mesh scale

        Returns:
            trimesh.Trimesh: object trimesh
        """

        mesh = trimesh.load(mesh_path, file_type="obj", force="mesh")
        mesh.apply_scale(scale)

        return mesh

    def _set_filtered_category_list(self, filter_categories: list) -> None:
        """Filter categories and set category list and data splits

        Args:
            filter_categories (list): List of categories to filter

        Raises:
            AssertionError: If category not found in valid ShapeNetSem categories

        Returns:
            None
        """
        filtered_splits = {}
        for cat in filter_categories:
            assert (
                cat in self._full_category_list
            ), f"{cat} not found in valid ShapeNetSem categories"
            filtered_splits[cat] = self.data_splits[cat]

        self.data_splits = filtered_splits
        self._filtered_categories = filter_categories
        self.category_list = filter_categories

        return

    def _load_all_obj_grasps(self):
        """Load grasps from h5 files


        Returns:
            dict: object grasp infos dict

        Raises:
            AssertionError: If mesh or grasp file not found

        Notes:
            returned dict has the following structure:

            obj_grasp_infos = {
                "obj_grasp_path": {
                    "grasps": torch.Tensor,
                    "grasp_qualities": torch.Tensor,
                    "bad_grasps": torch.Tensor,
                    "bad_grasp_qualities": torch.Tensor,
                    "mesh_path": str,
                    "obj_scale": float,
                }
            }
        """
        obj_grasp_infos = {}
        good_grasps_count = 0
        bad_grasps_count = 0

        for category in tqdm.tqdm(
            self.data_splits.values(),
            desc=f"Loading grasps for {len(self.data_splits.values())} ACRONYM categories ",
        ):
            for grasp_filename in category[self.split]:
                # Get grasp and mesh file paths
                grasp_fp = os.path.join(self.acronym_grasps_dir, grasp_filename)
                mesh_cat, mesh_file = self.get_meshname_from_acronym_file(
                    grasp_filename
                )
                mesh_fp = os.path.join(self.mesh_dir, mesh_cat, mesh_file)

                if os.path.exists(grasp_fp) and os.path.exists(mesh_fp):
                    # Load grasp data
                    data = h5py.File(grasp_fp, "r")
                    obj_scale = data["object/scale"][()]

                    # Get good and bad grasps
                    (
                        good_grasps,
                        good_grasp_qualities,
                        bad_grasps,
                        bad_grasp_qualities,
                        grasp_quality_type_list,
                    ) = self._prepare_grasps_and_qualities(data)

                    # Discard this dataitem with less than min_num_grasps to avoid data loading issues
                    if good_grasps.shape[0] <= self._min_num_grasps:
                        continue

                    # Qualities: Lower is better (weird lexical usage, I know):
                    # See- https://github.com/NVlabs/acronym/issues/14#issuecomment-1064859636
                    # Reverse and normalize in [0,1]
                    good_grasp_qualities = minmax_normalize(
                        -good_grasp_qualities, dim=0, v_min=0.0, v_max=1.0
                    )
                    bad_grasp_qualities = minmax_normalize(
                        -bad_grasp_qualities, dim=0, v_min=0.0, v_max=1.0
                    )

                    # Count grasps for logging
                    good_grasps_count += good_grasps.shape[0]
                    bad_grasps_count += bad_grasps.shape[0]

                    # Add to obj_grasp_infos dict
                    obj_grasp_infos[grasp_filename] = dict(
                        grasps=good_grasps,
                        grasp_qualities=good_grasp_qualities,
                        bad_grasps=bad_grasps,
                        bad_grasp_qualities=bad_grasp_qualities,
                        mesh_path=mesh_fp,
                        mesh_category=mesh_cat,
                        obj_scale=obj_scale,
                        quality_order=grasp_quality_type_list,
                    )

        self._log_message(
            f"Recorded {len(obj_grasp_infos)} objects from {len(self.data_splits.values())}"
            " categories with valid grasp annotations."
            f"\n Total:{good_grasps_count+bad_grasps_count} \nSuccessful grasps count: {good_grasps_count} "
            f"\nUnsuccessful grasps count: {bad_grasps_count}"
        )

        return obj_grasp_infos

    def _prepare_grasps_and_qualities(
        self, data: h5py.File
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        """Prepare grasps and qualities as tensors

        Args:
            data (h5py.File): h5 data

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
                    good grasp transforms [N, 4, 4],
                    good grasp qualities [N, 4],
                    bad grasp transforms [M, 4, 4],
                    bad grasp qualities [M, 4],
                    grasp_quality_type_list [4]
        """

        # Extract form h5 grasps files
        qualities = data["grasps/qualities/flex"]
        transforms = np.array(data["grasps/transforms"])
        success = np.array(qualities["object_in_gripper"])

        # Separate successful and unsuccessful grasps and qualities
        good_grasps = transforms[success > 0]
        bad_grasps = transforms[success == 0]
        good_grasp_qualities = []
        bad_grasp_qualities = []
        grasp_quality_order = []

        # Extract qualities
        for q_key, q_vals in qualities.items():
            if q_key == "object_in_gripper":
                # Already accounted for in successful/unsuccessful
                continue
            good_grasp_qualities.append(np.array(q_vals)[success > 0])
            bad_grasp_qualities.append(np.array(q_vals)[success == 0])
            grasp_quality_order.append(q_key)

        # Convert to Tensors
        good_grasps = self._prepare_grasps_transforms_tensor(good_grasps)
        bad_grasps = self._prepare_grasps_transforms_tensor(bad_grasps)
        good_grasp_qualities = torch.tensor(
            np.array(good_grasp_qualities), dtype=torch.float32
        ).T
        bad_grasp_qualities = torch.tensor(
            np.array(bad_grasp_qualities), dtype=torch.float32
        ).T

        # Cutoff fixed indexes if loading fixed/finite subset of grasps
        if self._use_fixed_grasp_subset:
            # Cutoff at min of num_grasps_fixed_grasp_subset and num good grasps
            cutoff_index = min(self._num_grasps_fixed_grasp_subset, len(good_grasps))
            good_grasps = good_grasps[:cutoff_index]
            bad_grasps = bad_grasps[:cutoff_index]
            good_grasp_qualities = good_grasp_qualities[:cutoff_index]
            bad_grasp_qualities = bad_grasp_qualities[:cutoff_index]

        return (
            good_grasps,
            good_grasp_qualities,
            bad_grasps,
            bad_grasp_qualities,
            grasp_quality_order,
        )

    def _prepare_grasps_transforms_tensor(
        self,
        grasp_transforms: np.ndarray,
    ) -> torch.Tensor:
        """Convert grasp pose matrix to input grasp representation tensors
        Currently only supports MRP representation

            N: Number of grasps

        Args:
            grasp_transforms (np.ndarray): [N, 4, 4] grasp pose matrices in world

        Returns:
            torch.Tensor: [N, 6] tensor with [t(3) mrp(3)] representation
        """

        if self._pose_representation == PoseRepresentation.TMRP:
            transforms = torch.from_numpy(grasp_transforms).to(torch.float32)
            cam_grasp_tmrps = H_to_tmrp(transforms)
            return cam_grasp_tmrps
        elif self._pose_representation == PoseRepresentation.H:
            transforms = torch.from_numpy(grasp_transforms).to(torch.float32)
            # Post-fix: Flatten 4x4 to keep ops consistent with TMRP
            return transforms.view(-1, 16)
        else:
            raise NotImplementedError(
                f"Rotation representation {self._pose_representation} not implemented."
            )

    def _log_message(self, msg):
        """Log message to logger or print if logger is None

        Args:
            msg (str): Message to log
        """

        if self.logger is None:
            print(msg)
        else:
            self.logger.log(msg)

        return
