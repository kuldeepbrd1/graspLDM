"""Script to render depth images and segmentation maps from random viewpoints w.r.t objects

    Usage:
        python generate_object_depth_dataset.py <data_root_dir> --splits <splits> --num_scenes <num_scenes> --num_renders_per_scene <num_renders_per_scene> --render_output_dir <render_output_dir> --gripper_path <gripper_path> --camera_json <camera_json> --debug

"""

import argparse
import multiprocessing
import os
import sys
from typing import List

import cv2
import numpy as np
import torch
import tqdm
from object_scene_renderer import (
    AcronymObjectRenderer,
    filter_grasps_by_sweep_collision,
)

# Use global utils module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from grasp_ldm.utils.camera import Camera
from grasp_ldm.utils.utils import spawn_multiple_processes

# from utils.pointcloud_helpers import PointCloudHelpers

## GLOBALS
# Store depth image as datatype
DEPTH_IMAGE_DATA_TYPE = np.uint16

# Scale the depth value in m by this factor to store as a depth image
# Note: Max distance represented by this scale is max_uint16/10000 = 6.5536m
DEPTH_IMAGE_PIXEL_SCALE = 10000


def parse_args():
    parser = argparse.ArgumentParser(description="Grasp data reader")
    parser.add_argument(
        "root_dir",
        help="Root dir with grasps, meshes, scene_generations and splits",
        type=str,
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=multiprocessing.cpu_count() - 1,
        help="Number of processes to spawn",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="scene id to start from",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--camera_json",
        type=str,
        default="data/cameras/camera_d435i_dummy.json",
        help="JSON file containing camera models",
    )
    parser.add_argument(
        "--render_output_dir",
        type=str,
        default="renders/objects/",
        help="Output directory inside data-root to store depth data",
    )
    parser.add_argument(
        "--num_renders_per_scene",
        type=int,
        default=20,
        help="Viewpoints to render per scene",
    )
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=1000,
        help="Number of objects",
    )
    parser.add_argument(
        "-debug",
        action="store_true",
        default=False,
        help="Setting this to true will run a single process that can be debug properly. \
            False will run multiprocessing code",
    )
    parser.add_argument(
        "--filter_categories",
        nargs="+",
        default=None,
        help="List of categories to filter",
    )

    return parser.parse_args()


def process_render_scene(
    batch_object_ids: list,
    data_root_dir: str,
    camera: Camera,
    out_dir: str,
    splits: list,
    num_renders: int = 10,
    pid: int = None,
    filter_categories: List[str] = None,
):
    """Process to render obj depth images and segmentation

    Uses pyrender to render random viewpoints around object meshes

    Args:
        num_scenes (List[str]): number of scenes to generate
        data_root_dir (str): root directory of the ACRONYM data containing meshes and grasps folders
        camera (Camera): loaded camera instance
        out_dir (str): output directory inside data_root_dir
        gripper_path (str): path to gripper mesh
        splits (list): list of splits
        num_renders (int, optional): number of renders per scene. Defaults to 10.
        pid (int, optional): Process ID. Defaults to None.

    Returns:
        None
    """

    # Instantiate object renderer for this process
    scene_renderer = AcronymObjectRenderer(
        camera_model=camera,
        root_folder=data_root_dir,
        splits=splits,
        filter_categories=filter_categories,
    )

    # Progress bar
    pbar_desc = (
        f"Generating scenes ( Process ID:{pid})"
        if pid is not None
        else "Generating Scenes"
    )

    # TODO: Make tqdm pbar proper for multiprocessing
    for scene_idx in tqdm.tqdm(batch_object_ids, desc=pbar_desc):
        (
            grasps,
            obj_path,
            obj_transform,
            obj_scale,
        ) = scene_renderer.get_random_object_infos()

        obj_path = os.path.join(data_root_dir, obj_path)

        # Output
        render_out_dir = os.path.join(out_dir, f"scene_{scene_idx}")
        os.makedirs(render_out_dir, exist_ok=True)

        # Make the scene
        obj_context = scene_renderer.change_scene(obj_path, obj_scale, obj_transform)
        grasps["transforms"][..., :3, 3] -= obj_context["mesh_mean"]
        # npz to be saved
        npz = dict(
            grasps=grasps,
            obj_path=obj_path,
            obj_scale=obj_scale,
            obj_transform=obj_transform,
        )

        rendered_segmaps = {}
        rendered_cams = {}
        objects_per_scene = {}
        grasp_idxs_per_render = {}

        for render_idx in range(num_renders):
            # Render scene
            _, depth, cam_pose = scene_renderer.render()

            segmaps, obj_names = scene_renderer.render_labels(
                depth, obj_path, obj_scale
            )

            # Save depth image
            depth_out_file = os.path.join(
                render_out_dir, f"scene_{scene_idx}_cam_{render_idx}.png"
            )

            # Scale to pixel values and quantize depth value
            depth_img = (depth.copy() * DEPTH_IMAGE_PIXEL_SCALE).astype(
                DEPTH_IMAGE_DATA_TYPE
            )
            import matplotlib.pyplot as plt

            # plt.imshow(depth)
            # plt.show()

            cv2.imwrite(depth_out_file, depth_img)

            # OpenGL to opencv type camera frame
            cam_pose[:3, 1] = -cam_pose[:3, 1]
            cam_pose[:3, 2] = -cam_pose[:3, 2]
            cam_pose = np.linalg.inv(cam_pose)

            # Filter grasps in empty regions- Not working yet
            # TODO: Speed up. Naive MP doesnt speed up. Many inefficient copy/cpu-gpu transfers.
            # following lines could be batched per scene
            grasp_transforms = grasps["transforms"].copy()
            good_grasp_idxs = np.where(grasps["success"] > 0)[0]
            good_grasp_transforms = grasp_transforms[good_grasp_idxs]
            good_grasp_transforms = cam_pose @ good_grasp_transforms

            pc_cam = scene_renderer._camera_model.depth_to_pointcloud(depth)

            if pc_cam.size > 0 and good_grasp_transforms.size > 0:
                _, collision_idxs = scene_renderer.filter_visible_grasps(
                    pc_cam, good_grasp_transforms
                )
                grasp_idxs_per_render[f"{render_idx}"] = good_grasp_idxs[collision_idxs]
            else:
                grasp_idxs_per_render[f"{render_idx}"] = []

            # Save segmap and cam poses
            rendered_segmaps[f"{render_idx}"] = segmaps
            rendered_cams[f"{render_idx}"] = cam_pose
            objects_per_scene[f"{render_idx}"] = obj_names

        npz["renders"] = dict(
            segmentation=rendered_segmaps,
            cam_poses=rendered_cams,
            objects=objects_per_scene,
            visible_grasp_indices=grasp_idxs_per_render,
        )

        np.savez(os.path.join(render_out_dir, f"{scene_idx}"), **npz)


if __name__ == "__main__":
    # args
    args = parse_args()

    start_idx = args.start_idx
    num_scenes = args.num_scenes
    splits = [args.split]
    num_renders_per_scene = args.num_renders_per_scene
    filter_categories = args.filter_categories
    n_proc = args.num_processes if not args.debug else 1

    # Directories
    root_dir = args.root_dir
    out_dir = os.path.join(root_dir, args.render_output_dir, args.split)

    print("Using root dir", args.root_dir)
    print(f"Using output directory: {out_dir}")
    print(f"Using camera model: {args.camera_json}")

    # Load camera model
    if not os.path.isfile(args.camera_json):
        raise FileNotFoundError(os.path.abspath(args.camera_model_json))

    camera = Camera(camera_json_path=args.camera_json)

    if not args.debug:
        # Offload to multiple processes
        batches = np.linspace(start_idx, num_scenes, n_proc + 1, dtype=int)
        batch_args = []

        for idx in range(n_proc):
            batch_object_ids = list(range(batches[idx], batches[idx + 1]))
            batch_args.append(
                dict(
                    batch_object_ids=batch_object_ids,
                    data_root_dir=root_dir,
                    splits=splits,
                    camera=camera,
                    out_dir=out_dir,
                    num_renders=num_renders_per_scene,
                    filter_categories=filter_categories,
                )
            )

        # Spawn processes
        spawn_multiple_processes(
            n_proc=n_proc, target_fn=process_render_scene, process_args=batch_args
        )
    else:
        # Debug mode: single process
        process_render_scene(
            batch_object_ids=list(range(start_idx, num_scenes)),
            data_root_dir=root_dir,
            splits=splits,
            camera=camera,
            out_dir=out_dir,
            num_renders=num_renders_per_scene,
            filter_categories=filter_categories,
        )
