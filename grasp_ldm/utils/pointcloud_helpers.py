import warnings
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
            # Check input shape
            if pc.ndim != 2 or pc.shape[1] != 3:
                raise ValueError(
                    f"Expected point cloud to have shape (N, 3), got {pc.shape}."
                )

            # Convert to numpy array
            pc = pc.detach().cpu().numpy() if pc.is_cuda else pc.detach().numpy()

            faces = Delaunay(pc, qhull_options="QJ Pp").simplices
            meshes.append(trimesh.Trimesh(vertices=pc, faces=faces))

        return meshes

    def regularize_pointcloud(pc, num_points):
        """Regularize pointcloud by repeating random points or downsampling

        Args:
            pc (torch.Tensor): Pointcloud of shape (N, 3)
            num_points (int): Number of points to regularize to

        Returns:
            torch.Tensor: Regularized pointcloud of shape (num_points, 3)
        """

        if pc.shape[0] < num_points:
            multiplier = max(num_points // pc.shape[0], 1)

            pc = pc.repeat(multiplier, 1)

            # repeat random points
            num_extra_points = num_points - pc.shape[0]

            # Randomly sample from existing points
            extra_points = pc[torch.randperm(pc.shape[0])[:num_extra_points]]

            # Concatenate to existing points
            pc = torch.cat((pc, extra_points), dim=0)

        elif pc.shape[0] > num_points:
            # Random Downsample
            pc = pc[torch.randperm(pc.shape[0])[:num_points]]
        else:
            pass

        return pc.unsqueeze(0)

    # Modified from https://github.com/NVlabs/contact_graspnet/blob/a12b7e3a1d7adf38762bfae1c8b84b4550059a6f/contact_graspnet/data.py#L239
    def estimate_normals_cam_from_pc(pc_cam, max_radius=0.05, k=12):
        """
        Estimates normals in camera coords from given point cloud.
        Arguments:
            pc_cam {np.ndarray} -- Nx3 point cloud in camera coordinates
        Keyword Arguments:
            max_radius {float} -- maximum radius for normal computation (default: {0.05})
            k {int} -- Number of neighbors for normal computation (default: {12})
        Returns:
            [np.ndarray] -- Nx3 point cloud normals
        """
        tree = cKDTree(pc_cam, leafsize=pc_cam.shape[0] + 1)
        _, ndx = tree.query(
            pc_cam, k=k, distance_upper_bound=max_radius, n_jobs=-1
        )  # num_points x k

        for c, idcs in enumerate(ndx):
            idcs[idcs == pc_cam.shape[0]] = c
            ndx[c, :] = idcs
        neighbors = np.array([pc_cam[ndx[:, n], :] for n in range(k)]).transpose(
            (1, 0, 2)
        )
        pc_normals = PointCloudHelpers.vectorized_normal_computation(pc_cam, neighbors)
        return pc_normals

    def vectorized_normal_computation(pc, neighbors):
        """
        Vectorized normal computation with numpy
        Arguments:
            pc {np.ndarray} -- Nx3 point cloud
            neighbors {np.ndarray} -- Nxkx3 neigbours
        Returns:
            [np.ndarray] -- Nx3 normal directions
        """
        diffs = neighbors - np.expand_dims(pc, 1)  # num_point x k x 3
        covs = np.matmul(np.transpose(diffs, (0, 2, 1)), diffs)  # num_point x 3 x 3
        covs /= diffs.shape[1] ** 2
        # takes most time: 6-7ms
        eigen_values, eigen_vectors = np.linalg.eig(
            covs
        )  # num_point x 3, num_point x 3 x 3
        orders = np.argsort(-eigen_values, axis=1)  # num_point x 3
        orders_third = orders[:, 2]  # num_point
        directions = eigen_vectors[
            np.arange(pc.shape[0]), :, orders_third
        ]  # num_point x 3
        dots = np.sum(directions * pc, axis=1)  # num_point
        directions[dots >= 0] = -directions[dots >= 0]
        return directions

    def regularize_pc_point_count(pc, npoints, use_farthest_point=False):
        """
        NOTE: Deprecated in favor of
            pointnet2_utils.gather_operation(
                pc_flipped,
                pointnet2_utils.furthest_point_sample(pc, self.num_points_per_pc),
            )

        If point cloud pc has less points than npoints, it oversamples existing points.
        Otherwise, it downsample the input pc to have npoint points.
        use_farthest_point: indicates

        :param pc: Nx3 point cloud
        :param npoints: number of points the regularized point cloud should have
        :param use_farthest_point: use farthest point sampling to downsample the points, runs slower.
        :returns: npointsx3 regularized point cloud
        """

        if pc.shape[0] > npoints:
            if use_farthest_point:
                _, center_indexes = PointCloudHelpers.farthest_points(
                    pc,
                    npoints,
                    PointCloudHelpers.distance_by_translation_point,
                    return_center_indexes=True,
                )
            else:
                center_indexes = np.random.choice(
                    range(pc.shape[0]), size=npoints, replace=False
                )
            pc = pc[center_indexes, :]
        else:
            required = npoints - pc.shape[0]
            if required > 0:
                index = np.random.choice(range(pc.shape[0]), size=required)
                pc = np.concatenate((pc, pc[index, :]), axis=0)
        return pc

    def farthest_points(
        data,
        nclusters,
        dist_func,
        return_center_indexes=False,
        return_distances=False,
        verbose=False,
    ):
        """
        Performs farthest point sampling on data points.
        Args:
            data: numpy array of the data points.
            nclusters: int, number of clusters.
            dist_dunc: distance function that is used to compare two data points.
            return_center_indexes: bool, If True, returns the indexes of the center of
            clusters.
            return_distances: bool, If True, return distances of each point from centers.

        Returns clusters, [centers, distances]:
            clusters: numpy array containing the cluster index for each element in
            data.
            centers: numpy array containing the integer index of each center.
            distances: numpy array of [npoints] that contains the closest distance of
            each point to any of the cluster centers.
        """
        if nclusters >= data.shape[0]:
            if return_center_indexes:
                return np.arange(data.shape[0], dtype=np.int32), np.arange(
                    data.shape[0], dtype=np.int32
                )

            return np.arange(data.shape[0], dtype=np.int32)

        clusters = np.ones((data.shape[0],), dtype=np.int32) * -1
        distances = np.ones((data.shape[0],), dtype=np.float32) * 1e7
        centers = []
        for iter in range(nclusters):
            index = np.argmax(distances)
            centers.append(index)
            shape = list(data.shape)
            for i in range(1, len(shape)):
                shape[i] = 1

            broadcasted_data = np.tile(np.expand_dims(data[index], 0), shape)
            new_distances = dist_func(broadcasted_data, data)
            distances = np.minimum(distances, new_distances)
            clusters[distances == new_distances] = iter
            if verbose:
                print("farthest points max distance : {}".format(np.max(distances)))

        if return_center_indexes:
            if return_distances:
                return clusters, np.asarray(centers, dtype=np.int32), distances
            return clusters, np.asarray(centers, dtype=np.int32)

        return clusters

    def distance_by_translation_point(p1, p2):
        """
        Gets two nx3 points and computes the distance between point p1 and p2.
        """
        return np.sqrt(np.sum(np.square(p1 - p2), axis=-1))


def depth_image_to_point_cloud_GPU(
    depth,
    camera_matrix,
    u,
    v,
    width: float,
    height: float,
    depth_bar: float,
    device: torch.device,
):
    depth_buffer = depth.to(device)

    fu = camera_matrix[0, 0]
    fv = camera_matrix[1, 1]
    centerU = width / 2
    centerV = height / 2

    assert centerU == camera_matrix[0, 2], "centerU != camera_matrix[0, 2]"
    assert centerV == camera_matrix[1, 2], "centerV != camera_matrix[1, 2]"

    Z = depth_buffer
    X = -(u - centerU) / width * Z * fu
    Y = (v - centerV) / height * Z * fv

    Z = Z.view(-1)
    valid = Z > -depth_bar
    X = X.view(-1)
    Y = Y.view(-1)

    position = torch.vstack((X, Y, Z, torch.ones(len(X), device=device)))[:, valid]
    position = position.permute(1, 0)
    position = position @ vinv

    points = position[:, 0:3]

    return points
