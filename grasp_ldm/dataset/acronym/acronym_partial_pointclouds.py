import glob
import os
from typing import Tuple
import warnings

try:
    import cv2
except:
    pass
import numpy as np
import torch
import tqdm
from numpy import ma
from termcolor import colored
from torch.utils.data import Dataset
import trimesh

from grasp_ldm.utils.camera import Camera
from grasp_ldm.utils.pointcloud_helpers import PointCloudHelpers
from grasp_ldm.utils.rotations import H_to_tmrp, tmrp_to_H

from ..augmentations import Augmentations
from ...utils.gripper import SimplePandaGripper


class AcronymPartialPointclouds(Dataset):
    """
    AcronymDepthDataset:

    Loads depth images and grasp annotations pre-emptively into memory

    __getitem__(): retrieves them from memory

    This is done to overcome large latencies resulting from repeated disk reads
    and computationally heavy farthest point sampling for pointclouds.
    For latter, torch CUDA ops are used to make it ~30x faster on gpu than with numpy.
    However, to be able to move tensors to cuda device, it needs to be done before initializing the dataloader.
    Otherwise, this results in a CUDA Runtime error. https://discuss.pytorch.org/t/cuda-initialization-error-when-dataloader-with-cuda-tensor/43390
    """

    VALID_ROTATION_REPRESENTATIONS = ["mrp"]

    # Different scale factors because of imbalance of mrp values vs translation values
    _TRANSLATION_SCALE_FACTOR = 0.05
    _ROTATION_SCALE_FACTOR = 0.5

    def __init__(
        self,
        data_root_dir: str,
        camera_json: str,
        num_points_per_pc: int = 1024,
        num_grasps_per_obj: int = 50,
        rotation_repr: str = "mrp",
        max_scenes: int = None,
        augs_config: dict = None,
        split: str = "train",
        depth_px_scale=10000,
        scene_prefix: str = "scene_",
        preempt_load_data: bool = True,
        use_failed_grasps: bool = True,
        failed_grasp_ratio: float = 0.3,
        load_fixed_grasp_transforms=None,
        is_input_dataset_normalized=False,
        num_pc_per_scene=20,
        num_repeat_dataset=2,
        **kwargs,
    ) -> None:
        """_summary_

        Args:
            depth_data_dir (str): _description_
            camera (Camera): _description_
            use_obj_pcs (bool): Whether to use object pcs for dense detection or sparse detection for scene
            num_points_per_pc (int): number of points to provide in __getitem__
            num_grasps_per_obj (int): number of grasps per object in a batch
            augs_config (dict): _description_
                Example:
                    augmentations:{
                        occlusion_nclusters: 0,
                        occlusion_dropout_rate: 0.0,
                        sigma: 0.000,
                        clip: 0.005,
                    }
            split (str, optional): _description_. Defaults to "train".
            depth_px_scale (int, optional): _description_. Defaults to 10000.
            scene_prefix (str, optional): _description_. Defaults to "scene_".

        Raises:
            NotImplementedError: _description_
        """
        super().__init__()

        assert split in ["train", "test"], "Invalid split name"

        # if split == "test":
        #     # TODO: Test set
        #     raise NotImplementedError

        self.data_dir = os.path.join(data_root_dir, split)
        self.scene_prefix = scene_prefix
        self.max_scenes = max_scenes
        self.num_pc_per_scene = num_pc_per_scene

        self.depth_px_scale = depth_px_scale

        self.camera = Camera(camera_json)

        self.num_points_per_pc = num_points_per_pc
        self.num_grasps_per_obj = num_grasps_per_obj

        # Patch: Avail these properties to commonalize v
        # TODO: Refactor
        self.batch_num_points_per_pc = num_points_per_pc
        self.batch_num_grasps_per_pc = num_grasps_per_obj

        if augs_config is None:
            warnings.warn("AcronymDepthDataset: No augmentations used")

        self.augs_config = augs_config

        valid_rotation_representations = ["mrp"]
        assert rotation_repr in valid_rotation_representations

        self.rotation_representation = rotation_repr
        _failed_grasps_ratio = failed_grasp_ratio if use_failed_grasps else None

        self.use_failed_grasps = use_failed_grasps
        self.load_fixed_grasp_transforms = load_fixed_grasp_transforms
        self.n_failed_grasps_per_obj = (
            int(num_grasps_per_obj * _failed_grasps_ratio)
            if use_failed_grasps
            else None
        )

        self.n_success_grasps_per_obj = (
            (num_grasps_per_obj - self.n_failed_grasps_per_obj)
            if use_failed_grasps
            else None
        )

        self.depth_data = None

        if split == "train" or preempt_load_data:
            self.depth_data, self.scene_ids = self.collect_data()
            self.num_images = len(self.depth_data)

        self._preempt_load_data = preempt_load_data
        self._set_normalization_params(is_input_dataset_normalized)
        self._augs_config = augs_config
        self.augmentations = (
            Augmentations.build_augmentations_from_cfg(augs_cfg=augs_config)
            if augs_config is not None
            else None
        )

        self.num_repeat_dataset = num_repeat_dataset

    def __len__(self) -> int:
        return self.num_repeat_dataset * len(self.depth_data)

    def _map_to_data_index(self, idx: int) -> int:
        """Map index to orginal index for repeated dataset"""
        return idx % len(self.depth_data)

    def pre_load(self) -> None:
        """Pre-load dataset to memory if __getitem__ depends on it"""
        if self.depth_data is None:
            self.depth_data, self.scene_ids = self.collect_data()
        return

    def load_grasp_data(self):
        # dummy method to make inference code consistent across different data loaders
        # could be improved. no time right now
        if self.depth_data is None:
            if not self._preempt_load_data:
                warnings.warn("Preemptive loading was set to False. No data loaded")
            else:
                raise NotImplementedError
        else:
            pass

    def __getitem__(self, index):
        """Get data item (pc/grasp pair) when called by the data loader

        NOTE: The batch item when retrieved is a [G,N,3] pointcloud
              and the grasps tensor is of shape [G,6].
              The scene pc is repeated G times, where G is the number of grasps to consider in a batch
              The idea is to not use 1 random grasp per object in a batch sample and increase dataloading overhead.

        Args:
            index (_type_): _description_

        Returns:
            _type_: _description_
        """
        if not hasattr(self, "depth_data"):
            raise RuntimeError("The dataset was not loaded")

        # Map query index to data index, if dataset is repeated
        mapped_idx = self._map_to_data_index(index)

        pc, grasps, meta_data = self.get_pc_grasps(mapped_idx)

        pc = pc.to(torch.float32)
        grasps = grasps.to(torch.float32)
        grasp_qualities = []  # grasp_qualities.to(torch.float32)

        # Pre-proc and augmentations
        pc, grasps, metas = self.preprocess_data(pc.clone(), grasps)
        # metas["quality_order"] = quality_order
        metas = {**metas, **meta_data}

        if grasps.shape[0] < self.num_grasps_per_obj:
            raise RuntimeError(
                "`grasps` was empty. This should not happen in data loading"
            )

        # self.visualize(pc, grasps, metas)
        return dict(pc=pc, grasps=grasps, qualities=grasp_qualities, metas=metas)

    def get_pc_grasps(self, index):
        dataitem = self.depth_data[index]
        metas = dict(depth_file=dataitem[0], npz_file=dataitem[1], scene_id=dataitem[2])
        data_dict = dataitem[-1].copy()

        # Pre-proc
        grasps = data_dict["grasps"].clone()
        failed_grasps = data_dict["failed_grasps"].clone()
        pc = data_dict["pc"].clone()

        ## Grasp Qualities [num_grasps_per_obj, 4]
        grasp_qualities = data_dict["grasp_qualities"]
        failed_grasp_qualities = data_dict["failed_grasp_qualities"]
        quality_order = data_dict["quality_order"]

        # Check if filtering indices for grasps on visible points are used in the data
        # Indirect. to improve. Grasps are already filtered elsewhere
        is_collision_check_online = (
            False if "cam_filtered_grasp_indices" in data_dict else True
        )

        if self.use_failed_grasps:
            if is_collision_check_online:
                raise NotImplementedError(
                    "Failed grasps does not work with online collision filtering yet"
                )

            else:
                grasps, grasp_qualities = self._mix_good_and_bad_grasps(
                    good_grasps=grasps,
                    good_grasp_qualities=grasp_qualities,
                    failed_grasps=failed_grasps,
                    failed_grasp_qualities=failed_grasp_qualities,
                )

        else:
            if is_collision_check_online:
                raise NotImplementedError
                # Randomly select 1.5*n grasps per obj/scene
                # So, that enough positive samples remain after
                # collision checking filters out the free-space grasps
                random_idcs = torch.randperm(grasps.shape[0])[
                    : int(1.5 * self.num_grasps_per_obj)
                ]
                grasps = grasps[random_idcs]

                # TODO: grasp qualities with collision filtering
                # grasp_qualities = grasp_qualities[random_idcs]

                # Filter free-space grasps
                grasps = self.filter_grasps_by_collision(pc, grasps)
            else:
                # If already filtered, sample random n grasps
                grasps = grasps[torch.randperm(grasps.shape[0])][
                    : self.num_grasps_per_obj
                ]

            # Take another data entry if still grasps are not filled
            if grasps is None:
                pc, grasps, metas = self.get_pc_grasps(np.random.randint(0, len(self)))
            else:
                grasps = H_to_tmrp(grasps)

                # grasps: [num_grasps_per_obj, 7]
                grasps = torch.concatenate(
                    (grasps, torch.ones((grasps.shape[0], 1))), dim=-1
                )

        return pc, grasps, metas

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

    def _mix_good_and_bad_grasps(
        self, good_grasps, good_grasp_qualities, failed_grasps, failed_grasp_qualities
    ):
        # Randomize the number of bad grasps up to a max of n_failed_grasps_per_obj
        _num_failed_grasps = np.random.randint(0, self.n_failed_grasps_per_obj)

        # Shuffle good and bad grasps
        good_grasp_idcs = torch.randperm(good_grasps.shape[0])
        bad_grasp_idcs = torch.randperm(failed_grasps.shape[0])

        # Select random bad grasp indices first and fill the rest with good grasps
        bad_grasp_idcs = bad_grasp_idcs[:_num_failed_grasps]
        good_grasp_idcs = good_grasp_idcs[
            : (self.num_grasps_per_obj - _num_failed_grasps)
        ]

        # Filter good grasps by idcs
        good_grasps = H_to_tmrp(good_grasps[good_grasp_idcs])
        good_grasp_qualities = good_grasp_qualities[good_grasp_idcs]

        # Filter bad grasps by idcs
        failed_grasps = H_to_tmrp(failed_grasps[bad_grasp_idcs])
        failed_grasp_qualities = failed_grasp_qualities[bad_grasp_idcs]

        # Concat success=1 or 0
        good_grasps = torch.concatenate(
            (good_grasps, torch.ones((good_grasps.shape[0], 1))), dim=-1
        )
        failed_grasps = torch.concatenate(
            (failed_grasps, torch.zeros((failed_grasps.shape[0], 1))), dim=-1
        )

        # Concat good and bad grasps and shuffle
        shuffle_idcs = torch.randperm(good_grasps.shape[0] + failed_grasps.shape[0])

        grasps = torch.concatenate(
            (
                good_grasps,
                failed_grasps,
            ),
            dim=0,
        )[shuffle_idcs]

        grasp_qualities = torch.concatenate(
            (
                good_grasp_qualities,
                failed_grasp_qualities,
            ),
            dim=0,
        )[shuffle_idcs]

        return grasps, grasp_qualities

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

    def apply_augmentations(self, pc, grasps):
        for aug in self.augmentations:
            if aug._transforms_pc and aug._transforms_grasps:
                pc, grasps[..., :6] = aug(pc, grasps[..., :6])
            elif aug._transforms_pc and not aug._transforms_grasps:
                pc = aug(pc)
            else:
                raise NotImplementedError
        return pc, grasps

    def load_scene_npz(self, npz_fp):
        npz = np.load(npz_fp, allow_pickle=True)

        grasp_dict = npz["grasps"][()]
        obj_path = str(npz["obj_path"])
        render_data = npz["renders"][()]

        if grasp_dict["transforms"].ndim == 0:
            return None
        else:
            return grasp_dict, render_data, obj_path

    def unit_normalize(self, t, dim):
        """min-max normalization in [0,1]

        Args:
            t (Tensor): tensor [B, D1, D2 ... Dn]
            dim (int): dimension to normalize on

        Returns:
            Tensor: [B, D1, D2 ... Dn]
        """
        t -= t.min(dim, keepdim=True)[0]
        t /= t.max(dim, keepdim=True)[0]
        return t

    def prepare_grasps_and_qualities(self, grasp_dict):
        grasp_tansforms = grasp_dict["transforms"]
        grasp_success = grasp_dict["success"]
        grasp_qualities = grasp_dict["qualities"]

        if "visible_grasp_indices" in grasp_dict:
            # If grasps are filtered on visible points, use those indices
            good_indices = grasp_dict["visible_grasp_indices"]

            good_mask = np.zeros(grasp_tansforms.shape[0], dtype=bool)
            good_mask[good_indices] = True
            bad_indices = np.where(~good_mask)[0]

        else:
            # Else, use grasp success indices
            good_indices = np.where(grasp_success > 0)[0]
            bad_indices = np.where(grasp_success == 0)[0]

        good_grasps = grasp_tansforms[good_indices]
        bad_grasps = grasp_tansforms[bad_indices]

        if good_grasps.shape[0] < self.num_grasps_per_obj:
            return None

        good_grasp_qualities = []
        bad_grasp_qualities = []
        grasp_quality_order = []

        for q_key, q_vals in grasp_qualities.items():
            if q_key == "object_in_gripper":
                # Already accounted for in successful/unsuccessful
                continue
            good_grasp_qualities.append(np.array(q_vals)[good_indices])
            bad_grasp_qualities.append(np.array(q_vals)[bad_indices])
            grasp_quality_order.append(q_key)

        good_grasp_qualities = torch.from_numpy(np.array(good_grasp_qualities).T).to(
            torch.float32
        )
        bad_grasp_qualities = torch.from_numpy(np.array(bad_grasp_qualities).T).to(
            torch.float32
        )

        good_grasp_qualities = self.unit_normalize(-good_grasp_qualities, dim=0)
        bad_grasp_qualities = self.unit_normalize(-bad_grasp_qualities, dim=0)

        return dict(
            grasps=torch.from_numpy(good_grasps).to(torch.float32),
            grasp_qualities=good_grasp_qualities,
            failed_grasps=torch.from_numpy(bad_grasps).to(torch.float32),
            failed_grasp_qualities=bad_grasp_qualities,
            quality_order=grasp_quality_order,
        )

    def get_depth_image_infos(self, depth_fp):
        # Load depth render from a random pose
        depth_filename = os.path.basename(depth_fp).split(".")[0]

        # TODO: Improve the hardcoded string usage
        cam_idx = depth_filename.split("cam_")[-1]

        # Read depth file and convert to pc
        depth_img = cv2.imread(depth_fp, -cv2.IMREAD_ANYDEPTH) / self.depth_px_scale

        return depth_img, cam_idx

    def get_object_pc_grasps(self, npz_file, depth_file):
        """Loads data from depth and npz archive files

        Args:
            depth_file (str): depth image file path
            npz_file (str): npz numpy archive file path

        Returns:
            Union[dict, None]: dict with keys ["pc", "grasps", "cam_pc_mean", "cam_pose"], if valid data retreived
                        None, if no valid data retreived

            if using scene wide pc:
                ["pc": torch.Tensor, "grasp": torch.Tensor, "cam_pc_mean":torch.Tensor, "cam_pose":torch.Tensor]
            else:
                ["pc": list(torch.Tensor), "grasp": list(torch.Tensor), "cam_pc_mean":torch.Tensor, "cam_pose":torch.Tensor]

            "pc" : [N,3] Tensor or list(Tensor)
                    zero-centered pointcloud(s)
            "grasps": [N,6] Tensor or list(Tensor) -
                    cam grasp transforms (per pointcloud) in (t(3), mrp(3)) format
            "cam_pc_mean": [1,3] Tensor or list(Tensor) -
                    Pointcloud mean(s) in cam frame
            "cam_pose": [4,4] Tensor-
                    Rendering camera pose
        """
        grasp_dicts, render_data, _ = self.load_scene_npz(npz_file)

        depth_img, cam_idx = self.get_depth_image_infos(depth_file)

        # If pre-filtered grasps for partial pointclouds are available in the npz
        if "visible_grasp_indices" in render_data:
            cam_filtered_grasp_indices = render_data["visible_grasp_indices"][cam_idx]
            grasp_dicts["visible_grasp_indices"] = cam_filtered_grasp_indices

            filter_grasps_dict = dict(
                cam_filtered_grasp_indices=cam_filtered_grasp_indices
            )
        else:
            filter_grasps_dict = {}

        grasps_and_qualities = self.prepare_grasps_and_qualities(grasp_dict=grasp_dicts)

        if grasps_and_qualities is None:
            return None

        cam_pose = render_data["cam_poses"][cam_idx]
        cam_pose = torch.from_numpy(cam_pose).to(torch.float32).unsqueeze(0)

        # Camera relative transforms
        pc_cam = self.camera.depth_to_pointcloud(depth_img)
        pc_cam = pc_cam[torch.randperm(pc_cam.shape[0])]
        pc_cam = pc_cam[: self.num_points_per_pc]

        grasp_cam_transforms = torch.matmul(cam_pose, grasps_and_qualities["grasps"])
        cam_bad_transforms = torch.matmul(
            cam_pose, grasps_and_qualities["failed_grasps"]
        )

        grasps_and_qualities["grasps"] = grasp_cam_transforms  # [grasps_visible_idcs]
        grasps_and_qualities["failed_grasps"] = cam_bad_transforms

        if grasps_and_qualities is None or pc_cam.shape[0] < self.num_points_per_pc:
            data = None
        else:
            data = dict(
                pc=torch.from_numpy(pc_cam).to(torch.float32),
                cam_pose=cam_pose,
                **grasps_and_qualities,
                **filter_grasps_dict,
            )

        return data

    def center_on_pc_mean(self, pc, grasps):
        pc_mean = torch.mean(pc, dim=-2)
        pc -= pc_mean
        grasps[..., :3] -= pc_mean
        return pc, grasps, pc_mean

    def center_grasps_on_camera(self, grasps, pc_mean):
        grasps[..., :3] += pc_mean
        return grasps

    def filter_grasps_by_collision(self, pc, grasps):
        raise NotImplementedError
        collision_checker = GripperCollision()
        pc_batched = pc.clone().repeat(grasps.shape[0], 1, 1)
        collisions = collision_checker.compute_collisions(pc_batched, grasps.clone())

        grasps = grasps[collisions == True]

        num_good_grasps = grasps.shape[0]
        if num_good_grasps > self.num_grasps_per_obj:
            grasps = grasps[: self.num_grasps_per_obj]
        elif num_good_grasps == 0:
            grasps = None
        else:
            # Fill random existing grasps
            n_repeat = self.num_grasps_per_obj // num_good_grasps
            grasps = grasps.repeat(n_repeat, 1, 1)
            grasps = torch.cat(
                [
                    grasps,
                    grasps[
                        torch.randperm(grasps.shape[0])[
                            : self.num_grasps_per_obj - grasps.shape[0]
                        ]
                    ],
                ]
            )
        return grasps

    def prepare_pc_tensor_numpy(self, pc: np.ndarray) -> torch.Tensor:
        """Regularize to fixed input npoints
        # data["pc"] = self.prepare_pc_tensor_numpy(pc_cam).repeat(
        #     [self.num_grasps_per_obj, 1, 1]
        # )

        Args:
            pc (np.ndarray): raw pointcloud

        Returns:
            torch.Tensor: regularized pc tensor

        """
        pc_reg = PointCloudHelpers.regularize_pc_point_count(
            pc, self.num_points_per_pc, use_farthest_point=False
        )
        pc = torch.from_numpy(pc_reg).to(torch.float32).unsqueeze(0)
        return pc

    def collect_data(self):
        """Collect all data from the dataset

        Collects all scenes or self.max_scenes
        """
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(
                f"{self.data_dir} does not exist or is not a directory."
            )

        scene_dirs = glob.glob(os.path.join(self.data_dir, f"{self.scene_prefix}*"))

        # TODO: Add a flag
        # random.shuffle(scene_dirs)

        num_scenes = len(scene_dirs)

        print(
            f"Collecting data from {self.data_dir}"
            f"Found {len(scene_dirs)} scene data directories. Reading max_scenes={self.max_scenes} ... "
        )

        data_tuples = []

        num_max_scenes = (
            min(self.max_scenes, num_scenes)
            if self.max_scenes is not None
            else num_scenes
        )
        scene_ids = []
        scene_count = 0
        for scene_dir in tqdm.tqdm(scene_dirs[:num_max_scenes]):
            scene_image_npz_pairs = self.get_scene_depth_and_npz(scene_dir)

            if scene_image_npz_pairs is None:
                continue

            for _, (depth_fp, npz_path, scene_id) in enumerate(scene_image_npz_pairs):
                data = self.get_object_pc_grasps(depth_file=depth_fp, npz_file=npz_path)

                if data is None:
                    continue

                if "cam_filtered_grasp_indices" in data:
                    if data["cam_filtered_grasp_indices"].size == 0:
                        continue

                data_tuples += [(depth_fp, npz_path, scene_id, data)]

                scene_ids.append(scene_id)

            scene_count += 1
            if scene_count >= num_max_scenes:
                break

        scene_ids = list(dict.fromkeys(scene_ids))
        images_count = len(data_tuples)

        print(
            f"Finished: Collected {images_count} depth images from {len(scene_ids)} scenes"
        )

        return data_tuples, scene_ids

    def scene_id_from_name(self, name: str) -> int:
        return name.split(self.scene_prefix)[-1]

    def scene_name_from_id(self, id: int) -> str:
        return f"{self.scene_prefix}{id}"

    def get_scene_npz_path(self, id: int) -> str:
        return os.path.join(self.data_dir, f"{self.scene_prefix}{id}", f"{id}.npz")

    def get_scene_depth_and_npz(self, scene_dir: str) -> list:
        """Get depth images and npz archive from scene dir

        Args:
            scene_dir (str): scene directory

        Returns:
            list: list of (image, npz, scene_id) tuples
        """
        scene_id = self.scene_id_from_name(os.path.basename(scene_dir))
        depth_img_path_str = os.path.join(
            scene_dir, f"{self.scene_prefix}{scene_id}_cam_*.png"
        )
        depth_img_paths = glob.glob(depth_img_path_str)[: self.num_pc_per_scene]
        npz_path = self.get_scene_npz_path(scene_id)

        # Check depth images exist in path
        if len(depth_img_paths) == 0:
            print(
                colored(
                    f"\033[93m could not find depth images in paths {depth_img_path_str}. Skipping this scene",
                    "red",
                )
            )
            return None

        # Check npz file exists in path
        if not os.path.isfile(npz_path):
            print(
                colored(
                    f"Scene npz for scene_id = {scene_id} Not found. \n {npz_path} is not a file. Skipping ...",
                    "red",
                )
            )
            return None

        image_npz_pairs = [
            (depth_path, npz_path, scene_id) for depth_path in depth_img_paths
        ]

        return image_npz_pairs

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
            raise NotImplementedError
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

    def visualize(self, pc, grasps, metas):
        from grasp_ldm.utils.vis import visualize_pc_grasps
        from utils.rotations import tmrp_to_H

        pc = pc * metas["pc_std"] + metas["pc_mean"]
        grasp_poses = grasps[..., :6] * metas["grasp_std"] + metas["grasp_mean"]

        pc = pc.numpy().squeeze()
        grasp_poses = tmrp_to_H(grasp_poses).numpy().squeeze()
        confidences = grasps[..., 6].numpy().squeeze()

        visualize_pc_grasps(pc, grasp_poses, c=confidences).show()
        return
