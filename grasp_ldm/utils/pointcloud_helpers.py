from typing import List

import numpy as np
import torch
import trimesh
from scipy.spatial import Delaunay, cKDTree


class PointCloudHelpers:
    def meshify_delaunay(point_clouds: List[torch.Tensor]) -> List[trimesh.Trimesh]:
        """Make meshes out of pointclouds based on Delaunay triangulation.

        Args:
            point_clouds (List[torch.Tensor]): list of point clouds of shape (N, 3)

        Returns:
            List[trimesh.Trimesh]: list of meshes
        """
        meshes = []

        if isinstance(point_clouds, torch.Tensor):
            point_clouds = [point_clouds]

        for pc in point_clouds:
            if pc.ndim != 2 or pc.shape[1] != 3:
                raise ValueError(
                    f"Expected point cloud to have shape (N, 3), got {pc.shape}."
                )
            pc = pc.detach().cpu().numpy() if pc.is_cuda else pc.detach().numpy()
            faces = Delaunay(pc, qhull_options="QJ Pp").simplices
            meshes.append(trimesh.Trimesh(vertices=pc, faces=faces))

        return meshes

    def regularize_pointcloud(pc, num_points):
        """Regularize pointcloud by repeating random points or downsampling.

        Args:
            pc (torch.Tensor): Pointcloud of shape (N, 3)
            num_points (int): Number of points to regularize to

        Returns:
            torch.Tensor: Regularized pointcloud of shape (1, num_points, 3)
        """
        if pc.shape[0] < num_points:
            multiplier = max(num_points // pc.shape[0], 1)
            pc = pc.repeat(multiplier, 1)
            num_extra_points = num_points - pc.shape[0]
            extra_points = pc[torch.randperm(pc.shape[0])[:num_extra_points]]
            pc = torch.cat((pc, extra_points), dim=0)
        elif pc.shape[0] > num_points:
            pc = pc[torch.randperm(pc.shape[0])[:num_points]]

        return pc.unsqueeze(0)

    # Modified from https://github.com/NVlabs/contact_graspnet
    def estimate_normals_cam_from_pc(pc_cam, max_radius=0.05, k=12):
        """Estimate normals in camera coords from given point cloud.

        Args:
            pc_cam (np.ndarray): Nx3 point cloud in camera coordinates
            max_radius (float): maximum radius for normal computation
            k (int): number of neighbors for normal computation

        Returns:
            np.ndarray: Nx3 point cloud normals
        """
        tree = cKDTree(pc_cam, leafsize=pc_cam.shape[0] + 1)
        _, ndx = tree.query(pc_cam, k=k, distance_upper_bound=max_radius, n_jobs=-1)

        for c, idcs in enumerate(ndx):
            idcs[idcs == pc_cam.shape[0]] = c
            ndx[c, :] = idcs
        neighbors = np.array([pc_cam[ndx[:, n], :] for n in range(k)]).transpose((1, 0, 2))
        return PointCloudHelpers.vectorized_normal_computation(pc_cam, neighbors)

    def vectorized_normal_computation(pc, neighbors):
        """Vectorized normal computation with numpy.

        Args:
            pc (np.ndarray): Nx3 point cloud
            neighbors (np.ndarray): Nxkx3 neighbours

        Returns:
            np.ndarray: Nx3 normal directions
        """
        diffs = neighbors - np.expand_dims(pc, 1)
        covs = np.matmul(np.transpose(diffs, (0, 2, 1)), diffs)
        covs /= diffs.shape[1] ** 2
        eigen_values, eigen_vectors = np.linalg.eig(covs)
        orders = np.argsort(-eigen_values, axis=1)
        orders_third = orders[:, 2]
        directions = eigen_vectors[np.arange(pc.shape[0]), :, orders_third]
        dots = np.sum(directions * pc, axis=1)
        directions[dots >= 0] = -directions[dots >= 0]
        return directions
