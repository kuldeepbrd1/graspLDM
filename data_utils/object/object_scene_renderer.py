"""Object Scene Renderer for ACRONYM dataset"""
# Adapted from https://github.com/NVlabs/acronym/blob/main/scripts/acronym_render_observations.py

import copy
import os
import random
from typing import Sequence, Tuple
import pyrender
import numpy as np
import trimesh
import trimesh.transformations as tra
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utils.camera import Camera
from utils.meshes import Object
from acronym_tools import Scene, load_mesh
import tqdm
import glob
import json
import h5py

import torch
from grasp_ldm.utils.collision import GripperCollision


def load_data_splits(root_dir: str) -> dict:
    """Load train/test splits for each category

    Args:
        root_dir (str): path to acronym data

    Returns:
        dict -- dict of category-wise train/test object grasp files
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


def filter_grasps_by_sweep_collision(
    pc: torch.Tensor, grasps: torch.Tensor, device="cuda:0"
):
    """Filters grasps whose finger sweep collides with a pointcloud

    Args:
        pc (torch.Tensor): pointcloud [N,3]
        grasps (torch.Tensor): grasp poses [M, 4, 4]
        device (str, optional): tensor device. Defaults to "cuda:0".

    Returns:
        torch.Tensor, torch.Tensor: filtered_grasps [P, 4, 4] , filtering idxs [P]
    """
    pc = pc.to(dtype=torch.float32, device=device)
    grasps = grasps.to(dtype=torch.float32, device=device)

    collision_checker = GripperCollision(device=device)
    pc_batched = pc.clone().repeat(grasps.shape[0], 1, 1)
    collisions = collision_checker.compute_collisions(pc_batched, grasps)

    colliding_idxs = torch.where(collisions == True)[0]
    grasps = grasps[colliding_idxs]

    return grasps, colliding_idxs


class AcronymObjectRenderer(Scene):
    def __init__(
        self,
        camera_model: Camera,
        root_folder: str,
        splits,
        filter_categories: list = [],
        elevation_limits: Sequence = (20, 360),
        distance_limits: Sequence = (0.5, 1),
    ) -> None:
        """Object Scene Renderer
        Args:
            camera_model (Camera): Camera model
            root_folder (str): Root folder of acronym dataset
            splits (list): List of splits to use
            filter_categories (list, optional): List of categories to filter. Defaults to [].
            elevation_limits (Sequence, optional): Elevation limits. Defaults to (20, 360).
            distance_limits (Sequence, optional): Distance limits. Defaults to (0.4, 1.2).

        """
        super().__init__()

        # Directories
        self.root_dir = root_folder
        self.acronym_grasps_dir = os.path.join(self.root_dir, "grasps")
        self.mesh_dir = os.path.join(self.root_dir, "meshes")

        # Splits
        self.splits = splits
        self.data_splits = load_data_splits(root_folder)

        # Cameras
        self._camera_model = camera_model
        self.render_camera = self._camera_model.to_pyrender_camera()

        # Scene
        self.scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3, 1.0])

        # Filter categories
        self._full_category_list = list(self.data_splits.keys())
        self.category_list = None
        if filter_categories:
            self._set_filtered_category_list(filter_categories)
        else:
            self.category_list = self._full_category_list

        # Load all grasps from h5 files
        self.grasp_infos = self._load_all_grasps()

        # Camera node
        self._camera_node = self.scene.add(
            self.render_camera, pose=np.eye(4), name="camera"
        )

        # Renderer
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self._camera_model.width,
            viewport_height=self._camera_model.height,
            point_size=1.0,
        )

        # Camera pose sampling for random viewpoints
        self._distance_limits = np.array(distance_limits)
        self._elevation_limits = np.array(elevation_limits)
        self._orient_angle_res = 20  # deg

        (
            self._orientation_samples,
            self._worldcam_to_cam_transform,
        ) = self._sample_uniform_overlooking_orientations(elevation_limits)

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

    def _get_meshname_from_acronym_file(self, acronym_file: str) -> Tuple[str, str]:
        """Get mesh name from acronym file

        Args:
            acronym_file (str): Acronym file

        Returns:
            (str, str): (category, mesh name)
        """
        filename = acronym_file.split("_")[1] + ".obj"
        cat = acronym_file.split("_")[0]
        return cat, filename

    def _load_all_grasps(self) -> dict:
        """Load grasps from h5 files for all splits

        Returns:
            dict: dict of grasp infos per object (filename)
        """

        grasp_infos = {}
        for category_paths in tqdm.tqdm(
            self.data_splits.values(),
            desc=f"Loading grasps for {len(self.data_splits.values())} ACRONYM categories ",
        ):
            for split in self.splits:
                for grasp_filename in category_paths[split]:
                    grasp_fp = os.path.join(self.acronym_grasps_dir, grasp_filename)
                    if os.path.exists(grasp_fp):
                        data = h5py.File(grasp_fp, "r")
                        quality_group = data["grasps/qualities/flex"]

                        grasp_infos[grasp_filename] = dict(
                            transforms=np.array(data["grasps/transforms"]),
                            success=np.array(quality_group["object_in_gripper"]),
                            qualities={
                                quality_key: np.array(quality_group[quality_key])
                                for quality_key in quality_group.keys()
                            },
                        )

        return grasp_infos

    def _sample_uniform_overlooking_orientations(
        self, elevation_limits: Sequence
    ) -> Tuple[list, np.ndarray]:
        """Sample camera orientations in a hemisphere with elevation limits
            Samples azimuth uniformly in [0, 2pi]

        Args:
            elevation_limits (Sequence): Elevation limits

        Returns:
            (list, np.ndarray): (list of camera orientations, camera axes transform)
        """

        cam_orientations = []
        elevation_limits = np.array(elevation_limits) / 180.0

        # Create orientation grid along a hemisphere within elevation limits
        for az in np.linspace(0, np.pi * 2, self._orient_angle_res):
            for el in np.linspace(0, np.pi * 2, self._orient_angle_res):
                cam_orientations.append(tra.euler_matrix(0, -el, az))

        # Transforms camera axes to align z along boresight and y -up
        camera_axes_transform = tra.euler_matrix(np.pi / 2, 0, 0).dot(
            tra.euler_matrix(0, np.pi / 2, 0)
        )

        return cam_orientations, camera_axes_transform

    def _load_object(self, path: str, scale: float) -> dict:
        """Load a mesh, scale and center it

        Args:
            path (str): path to mesh
            scale (float): scale of the mesh

        Returns:
            dict: context with loaded mesh info
        """

        obj = Object(path)
        obj.apply_scale(scale)

        tmesh = obj.mesh
        tmesh_mean = np.mean(tmesh.vertices, 0)
        tmesh.vertices -= np.expand_dims(tmesh_mean, 0)

        lbs = np.min(tmesh.vertices, 0)
        ubs = np.max(tmesh.vertices, 0)
        object_distance = np.max(ubs - lbs) * 5

        mesh = pyrender.Mesh.from_trimesh(tmesh)

        context = {
            "name": path + "_" + str(scale),
            "tmesh": copy.deepcopy(tmesh),
            "distance": object_distance,
            "node": pyrender.Node(mesh=mesh, name=path + "_" + str(scale)),
            "mesh_mean": np.expand_dims(tmesh_mean, 0),
        }

        return context

    def _load_obj_grasps(
        self, grasp_path: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load grasps for an object

        Args:
            grasp_path (str): path to grasp file

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: ( [Mx4x4] grasp transforms, [M,] success, [M,] qualities)
        """
        grasp_infos = self.grasp_infos[grasp_path]

        grasp_transforms = grasp_infos["transforms"].reshape(-1, 4, 4)

        return grasp_transforms, grasp_infos["success"], grasp_infos["qualities"]

    def change_scene(
        self, obj_path: str, obj_scale: float, obj_transform: np.ndarray
    ) -> None:
        """Change scene to a new object

        Updates self.scene

        Args:
            obj_path (str): path to object mesh
            obj_scale (float): scale of the object
            obj_transform (np.ndarray): 4x4 object transform

        Returns:
            None
        """

        for n in self.scene.get_nodes():
            if n.name not in ["camera", "parent"]:
                self.scene.remove_node(n)

        object_context = self._load_object(obj_path, obj_scale)
        object_context = copy.deepcopy(object_context)

        self.scene.add_node(object_context["node"])
        self.scene.set_pose(object_context["node"], obj_transform)
        return object_context

    def _get_random_object(self) -> Tuple[trimesh.Trimesh, str]:
        """Return random object
            Object mesh is scaled but not yet centered

        Returns:
            [trimesh.Trimesh, str]: ShapeNet mesh from a random category, h5 file path
        """

        while True:
            random_category = random.choice(self.category_list)

            # Get all object paths for a random category
            cat_obj_paths = [
                obj_p
                for split in self.splits
                for obj_p in self.data_splits[random_category][split]
            ]

            # Get a random object path from the category
            if cat_obj_paths:
                random_grasp_path = random.choice(cat_obj_paths)
                mesh_cat, mesh_file = self._get_meshname_from_acronym_file(
                    random_grasp_path
                )
                mesh_path = os.path.join(self.mesh_dir, mesh_cat, mesh_file)

                # Check if the mesh exists and grasp info exists for the object
                if random_grasp_path in self.grasp_infos and os.path.isfile(mesh_path):
                    break

        obj_mesh = load_mesh(
            os.path.join(self.acronym_grasps_dir, random_grasp_path), self.root_dir
        )

        return obj_mesh, random_grasp_path

    def get_random_object_infos(self) -> Tuple[dict, str, np.ndarray, float]:
        """Return random object infos

        Returns:
            [dict, str, np.ndarray, float]: filtered grasp dict, h5 file path, object transform, object scale
        """

        # get random object
        obj_mesh, obj_grasp_path = self._get_random_object()

        # object transform
        placement_T = np.eye(4, 4)

        # place object
        self.add_object(obj_grasp_path, obj_mesh, placement_T)

        # record object infos
        obj_scale = float(obj_grasp_path.split("_")[-1].split(".h5")[0])
        obj_path = os.path.join(
            "meshes", "/".join(obj_grasp_path.split("_")[:2]) + ".obj"
        )
        obj_transform = placement_T

        # get grasp infos
        grasp_transforms, grasp_success, grasp_qualities = self._load_obj_grasps(
            obj_grasp_path
        )

        # transform grasps per object transforms
        grasp_transforms = self._transform_grasps(grasp_transforms, obj_transform)

        grasps = dict(
            transforms=grasp_transforms,
            success=grasp_success,
            qualities=grasp_qualities,
        )
        return (grasps, obj_path, obj_transform, obj_scale)

    def _transform_grasps(self, grasps, obj_transform):
        """Transform grasps into given object transform

        Args:
            grasps (np.ndarray): Nx4x4 grasps
            obj_transform (np.ndarray): 4x4 mesh pose

        Returns:
            np.ndarray: transformed grasps
        """
        transformed_grasps = np.matmul(obj_transform, grasps)
        return transformed_grasps

    def render(
        self, cam_pose: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Render object or scene in camera pose

        Args:
            cam_pose (np.ndarray, optional): 4x4 camera pose. Defaults to None.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: HxWx3 color, HxW depth, Nx4 point cloud
        """
        if cam_pose is None:
            viewing_index = np.random.randint(0, high=len(self._orientation_samples))
            camera_orientation = self._orientation_samples[viewing_index]
            cam_pose = self.get_random_cam_pose_from_orientation(camera_orientation)

        out_pose = cam_pose.copy()
        self.scene.set_pose(self._camera_node, out_pose)

        color, depth = self.renderer.render(self.scene)

        # Uncomment to debug TODO: Add a debug flag?
        # pyrender.Viewer(self.scene)

        return color, depth, out_pose

    def render_labels(
        self, full_depth: np.ndarray, obj_path: str, obj_scale: float
    ) -> Tuple[np.ndarray, list]:
        """Render instance segmentation map
        Args:
            full_depth (np.ndarray): HxW depth map
            obj_path (str): object path in scene
            obj_scale (float): object scale in scene
        Returns:
            Tuple[np.ndarray, list]: integer segmap, list of index corresponding object names
        """

        # Collect all object nodes and set everything to invisible except camera and parent
        scene_object_nodes = []
        for n in self.scene.get_nodes():
            if n.name not in ["camera", "parent"]:
                n.mesh.is_visible = False
                if n.name != "table":
                    scene_object_nodes.append(n)

        obj_name = [obj_path + "_" + str(obj_scale)]

        output = np.zeros(full_depth.shape, np.uint8)

        # create mask per object and collect in output
        for n in scene_object_nodes:
            # Set this object mesh to visible
            n.mesh.is_visible = True

            # render depth
            depth = self.renderer.render(self.scene)[1]

            # Create this object mask from two depth images
            mask = np.logical_and(
                (np.abs(depth - full_depth) < 1e-6), np.abs(full_depth) > 0
            )
            if not np.any(mask):
                continue
            if np.any(output[mask] != 0):
                raise ValueError("wrong label")

            # Mask the depth image (output) with index of the object
            # NOTE: object with index =2 will masked with values=2
            indices = [i + 1 for i, x in enumerate(obj_name) if x == n.name]
            for i in indices:
                if not np.any(output == i):
                    output[mask] = i
                    break

            # Set this object visibility to false
            n.mesh.is_visible = False

        # Restore visibility of all objects
        for n in self.scene.get_nodes():
            if n.name not in ["camera", "parent"]:
                n.mesh.is_visible = True

        return output, ["BACKGROUND"] + obj_name

    def get_random_cam_pose_from_orientation(
        self, cam_orientation, orient_perturb_limits=[20, 10, 10]
    ):
        """Samples camera pose on shell around table center given an orientation

        Args:
            cam_orientation (_type_): 3x3 camera orientation matrix
            orient_perturb_limits (list, optional): Perturbation to radial viewpoint
                                        in x,y,z in degrees. Defaults to [20, 10, 10].

        Returns:
            [np.ndarray]: 4x4 homogeneous camera pose
        """
        distance = self._distance_limits[0] + np.random.rand() * (
            self._distance_limits[1] - self._distance_limits[0]
        )

        # Aligned to world
        extrinsics = np.eye(4)
        extrinsics[0, 3] += distance

        # Rotated by azimuth/elevation
        extrinsics = cam_orientation.dot(extrinsics)

        # # Randomly Perturb viewing angle
        # rng = np.random.default_rng()
        # rand_signs = [1 if r < 0.5 else -1 for r in rng.random(3)]
        # rand_angles = np.radians(orient_perturb_limits) * rng.random(3) * rand_signs
        # extrinsics = extrinsics.dot(tra.euler_matrix(*rand_angles))

        # Aligned to camera axes
        cam_pose = extrinsics.dot(self._worldcam_to_cam_transform)

        return cam_pose

    def filter_visible_grasps(
        self, pc, grasps, num_points_regularize=2048, device="cuda:0"
    ):
        """Filter grasps that are not on visible pointcloud

        Args:
            pc_cam (np.ndarray): Nx3 point cloud in camera frame
            grasps (np.ndarray): Nx4x4 grasp transforms

        Returns:
            np.ndarray: Nx4x4 filtered grasp transforms
        """

        pc = torch.from_numpy(pc)
        pc = pc[torch.randperm(pc.shape[0])][:num_points_regularize]
        grasps = torch.from_numpy(grasps)

        grasps_on_pc, grasp_idxs = filter_grasps_by_sweep_collision(pc, grasps)

        return grasps_on_pc.detach().cpu().numpy(), grasp_idxs.tolist()
