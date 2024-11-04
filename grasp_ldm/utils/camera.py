import csv
import json
import os
import warnings

import numpy as np
import torch

from .utils import load_json

try:
    import pyrender
except:
    warnings.warn("pyrender was not found. Rendering modules will not work.")


def read_csv_realsense(csv_file_path):
    with open(csv_file_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        data = {row[0]: row[1] for row in csv_reader if len(row) > 1}

    frame_info = {
        key: data[key]
        for key in [
            "Type",
            "Depth",
            "Format",
            "Frame Number",
            "Timestamp (ms)",
            "Resolution x",
            "Resolution y",
            "Bytes per pixel",
        ]
    }
    intrinsic_info = {
        key: data[key] for key in ["Fx", "Fy", "PPx", "PPy", "Distorsion"]
    }

    cam_json = {
        "hfov": 2 * np.arctan2(data["Resolution_x"] / (2 * data["Fx"])) * 180 / np.pi,
        "vfov": 2 * np.arctan2(data["Resolution_y"] / (2 * data["Fy"])) * 180 / np.pi,
        "width": int(data["Resolution_x"]),
        "height": int(data["Resolution_y"]),
        "cameraMatrix": [
            [float(data["Fx"]), 0, float(data["PPx"])],
            [0, float(data["Fy"]), float(data["PPy"])],
            [0, 0, 1],
        ],
        "distCoeffs": [],
    }
    return cam_json


def calculate_view_frustum(start_point, end_point, fov):
    """
    Calculate the coordinates of the view frustum given the boresight line and FOV.

    Args:
    start_point (tuple): The starting point of the boresight line.
    end_point (tuple): The ending point of the boresight line.
    fov (float): The field of view of the camera in degrees.

    Returns:
    view_frustum (list): A list of tuples containing the coordinates of the view frustum.
    """

    # Convert the FOV from degrees to radians
    fov_rad = np.radians(fov)

    # Calculate the distance between the two points
    distance = np.sqrt(
        sum([(end - start) ** 2 for start, end in zip(start_point, end_point)])
    )

    # Calculate the half-angle of the FOV
    half_angle = np.tan(fov_rad / 2)

    # Calculate the coordinates of the view frustum
    view_frustum = []
    for i in range(-1, 2, 2):  # Iterate twice: -1 for near plane, +1 for far plane
        x = start_point[0] + i * distance * half_angle
        y = start_point[1] + i * distance * half_angle
        z = start_point[2] + i * distance
        view_frustum.append((x, y, z))

    return view_frustum


class Camera:
    """Camera model using a user json file"""

    def __init__(
        self,
        camera_json_path: str,
        z_near: float = 0.05,
        z_far: float = 20,
    ) -> None:
        """
        Args:
            camera_json_path (str): camera json file path
            camera_name (str): camera name from the json.
        """
        self.name = os.path.basename(camera_json_path)
        self.data = load_json(camera_json_path)

        # Intrinsics and distortion matrix
        self.K = np.array(self.data["cameraMatrix"])
        self.dists = np.array(self.data["distCoeffs"])

        # Focal Length in px
        self._fx = self.K[0, 0]
        self._fy = self.K[1, 1]

        # Principal centers
        self._cx = self.K[0, 2]
        self._cy = self.K[1, 2]

        # Near/Far limits in boresight
        self.z_near = z_near
        self.z_far = z_far

        # Image size in px
        self.width = self.data["width"]
        self.height = self.data["height"]

        # FOV
        self.xfov = self.data["hfov"]  # HFOV
        self.yfov = self.data["vfov"]  # VFOV

    def to_pyrender_camera(self):
        return pyrender.IntrinsicsCamera(
            self._fx, self._fy, self._cx, self._cy, self.z_near, self.z_far
        )

    def depth_to_pointcloud(
        self, depth: np.ndarray, rgb: np.ndarray = None
    ) -> np.ndarray:
        """Convert depth image to pointcloud given camera intrinsics.
        Args:
            depth (np.ndarray): Depth image.
        Returns:
            np.ndarray: [nx4] (x, y, z, 1) Point cloud.
        """

        height = depth.shape[0]
        width = depth.shape[1]

        assert (
            height == self.height
        ), "Something went wrong. height of the depth image does not match the camera model."
        assert (
            width == self.width
        ), "Something went wrong. width of the depth image does not match the camera model."

        mask = np.where(depth > 0)
        x, y = mask[1], mask[0]

        normalized_x = x.astype(np.float32) - self._cx
        normalized_y = y.astype(np.float32) - self._cy

        world_x = normalized_x * depth[y, x] / self._fx
        world_y = normalized_y * depth[y, x] / self._fy
        world_z = depth[y, x]

        if rgb is not None:
            rgb = rgb[y, x, :]

        pc = np.vstack((world_x, world_y, world_z)).T

        if rgb is not None:
            rgb = rgb[y, x, :]
            return pc, rgb
        else:
            return pc

    def depth_to_pointcloud_torch(
        self, depth: torch.Tensor, rgb: torch.Tensor = None
    ) -> torch.Tensor:
        """Convert depth image to pointcloud given camera intrinsics.
        Args:
            depth (torch.Tensor): Depth image.
        Returns:
            torch.Tensor: [nx4] (x, y, z, 1) Point cloud.
        """

        height = depth.shape[0]
        width = depth.shape[1]

        assert (
            height == self.height
        ), "Something went wrong. height of the depth image does not match the camera model."
        assert (
            width == self.width
        ), "Something went wrong. width of the depth image does not match the camera model."

        mask = torch.where(depth > 0)
        x, y = mask[1], mask[0]

        normalized_x = x.to(torch.float32) - self._cx
        normalized_y = y.to(torch.float32) - self._cy

        world_x = normalized_x * depth[y, x] / self._fx
        world_y = normalized_y * depth[y, x] / self._fy
        world_z = depth[y, x]

        if rgb is not None:
            rgb = rgb[y, x, :]

        pc = torch.vstack((world_x, world_y, world_z)).T

        if rgb is not None:
            rgb = rgb[y, x, :]
            return pc, rgb
        else:
            return pc

    def write_to_dir(self, out_dir):
        json_fp = os.path.join(out_dir, f"camera_{self.name}.json")

        print(f"Writing camera model {self.name} to {json_fp}.")
        with json_fp as fileobj:
            json.dump(self.data, fileobj)
        return

    # def get_trimesh_camera(self):
    #     """Get a trimesh object representing the camera intrinsics.
    #     Returns:
    #         trimesh.scene.cameras.Camera: Intrinsic parameters of the camera model
    #     """
    #     return trimesh.scene.cameras.Camera(
    #         fov=(np.rad2deg(self._fov), np.rad2deg(self._fov)),
    #         resolution=(self._height, self._width),
    #         z_near=self._z_near,
    #     )
