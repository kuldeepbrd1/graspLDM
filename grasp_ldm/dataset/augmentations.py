from abc import abstractmethod

import numpy as np
import torch
from torch import nn

from grasp_ldm.utils.rotations import (
    H_to_tmrp,
    get_random_rotations_in_angle_limit,
    tmrp_to_H,
)

## ------------ Base Classes ------------


class BaseAugmentation(nn.Module):
    """Old Base class for augmentations that can be applied to pointclouds and grasps

    BaseGeneralAugmentation is better
    """

    def __init__(self, transforms_pc, transforms_grasps) -> None:
        super().__init__()
        self._transforms_pc: bool = transforms_pc
        self._transforms_grasps: bool = transforms_grasps

    @abstractmethod
    def forward(self, pc, grasps=None):
        raise NotImplementedError


class BaseGeneralAugmentation(nn.Module):
    """Base class for augmentations that can be applied to any input

    The idea is to preserve the augmentation transformation until reset is called.
    This is useful when the same transformation needs to apply to multple objects.
    """

    def __init__(self, transforms_pc, transforms_grasps) -> None:
        super().__init__()

    @abstractmethod
    def reset(self, input):
        raise NotImplementedError

    @abstractmethod
    def apply(self, input):
        raise NotImplementedError

    forward = apply


## ------------ New Augmentations ------------


class RandomRotationTransform(BaseGeneralAugmentation):
    def __init__(self, p=0.5, max_angle=180, *, is_degree=True) -> None:
        """Random rotation within angle limit. Keeps the rotation constant until reset.
        So it can be repeatedly applied

        Args:
            p (float, optional): probability of applying. Defaults to 0.5.
            max_angle (int, optional): max angle. Defaults to 180.
            is_degree (bool, optional): whether in degrees. Defaults to True.
        """
        super().__init__(transforms_grasps=True, transforms_pc=True)
        self.max_angle = np.radians(max_angle) if is_degree else max_angle
        self.p = p
        self.transform = None

    def reset(self):
        """Apply Random Rotation
        Args:
            pc (Tensor): (B,N,3)
            grasps (Tensor): (B,6)

        Returns:
            _type_: _description_
        """
        H = torch.eye(4)

        if torch.rand((1,)) < self.p:
            rotmat = get_random_rotations_in_angle_limit(self.max_angle, batch_size=1)
            H[:3, :3] = rotmat.squeeze(0)
            self.transform = H
        else:
            self.transform = H

        return

    def apply(self, x):
        """Apply Random Rotation

        Args:
            x (Tensor): Homogenous coordinates (B,N,4) or (B,4)
        """
        assert x.shape[-1] in (
            3,
            4,
        ), "Input expects point coordinates of 3 or 4 dimensions"

        if x.shape[-1] == 3:
            x = torch.concatenate((x, torch.ones(x.shape[0], x.shape[1], 1)), dim=-1)
            x = x @ self.transform
            x = x[..., :3]
        else:
            x = x @ self.transform

        return x


## ------------ Augmentations ------------
# Old augmentations that can be applied to pointclouds and grasps
class RandomRotation(BaseAugmentation):
    def __init__(self, p=0.5, max_angle=180, *, is_degree=True) -> None:
        """Random rotation within angle limit

        Args:
            p (float, optional): probability of applying. Defaults to 0.5.
            max_angle (int, optional): max angle. Defaults to 180.
            is_degree (bool, optional): whether in degrees. Defaults to True.
        """
        super().__init__(transforms_grasps=True, transforms_pc=True)
        self.max_angle = np.radians(max_angle) if is_degree else max_angle
        self.p = p

    def forward(self, pc, grasps):
        """Apply Random Rotation
        Args:
            pc (Tensor): (B,N,3)
            grasps (Tensor): (B,6)

        Returns:
            _type_: _description_
        """
        assert pc.shape[-1] == 3, "Pointcloud should be of 3 or 4 dimensions"
        assert grasps.shape[-1] == 6, "Pointcloud should be of 3 or 4 dimensions"

        if torch.rand((1,)) < self.p:
            B, N, D = pc.shape
            pc = torch.concatenate((pc, torch.ones(B, N, 1)), dim=-1)
            H = torch.eye(4)

            rotmat = get_random_rotations_in_angle_limit(self.max_angle, batch_size=1)
            H[:3, :3] = rotmat.squeeze(0)
            H_grasps = tmrp_to_H(grasps)
            pc = pc @ H.T
            H_grasps = H @ H_grasps

            pc = pc[..., :-1]
            grasps = H_to_tmrp(H_grasps)

        return pc, grasps


class RandomTinyPosePerturbation(BaseAugmentation):
    def __init__(self, max_perturb=0.005) -> None:
        super().__init__(transforms_pc=False, transforms_grasps=True)
        self.max_perturb = max_perturb

    def forward(self, grasps, pc=None):
        assert (
            grasps.shape[-1] == 6
        ), "Only tmrp perturbation supported. Input shape should be [..., 6]"
        perturb = (2 * self.max_perturb) * torch.rand(
            6,
        ) - self.max_perturb
        return grasps + perturb


class RandomRotationPerGrasp(BaseAugmentation):
    def __init__(self, p=0.5, max_angle=180, *, is_degree=True) -> None:
        """Random rotation within angle limit

        Args:
            p (float, optional): probability of applying. Defaults to 0.5.
            max_angle (int, optional): max angle. Defaults to 180.
            is_degree (bool, optional): whether in degrees. Defaults to True.
        """
        super().__init__(transforms_grasps=True, transforms_pc=True)
        self.max_angle = np.radians(max_angle) if is_degree else max_angle
        self.p = p

    def forward(self, pc, grasps):
        b, n, d = pc.shape[0]
        assert d == 3, "Pointcloud should be of 3 or 4 dimensions"
        assert grasps.shape[-1] == 6, "Pointcloud should be of 3 or 4 dimensions"
        assert b == grasps.shape[0], "Mismatch in batch sizes between pc and grasps"

        pc = torch.concatenate(pc, torch.ones((b, n, 1)), dim=-1)

        H = torch.eye(4).unsqueeze(0).repeat((b, 1, 1))

        num_perturbs = int(self.p * b)
        random_indices = torch.randperm(b)[:num_perturbs]
        rotmats = get_random_rotations_in_angle_limit(
            self.max_angle, batch_size=num_perturbs
        )
        H[[random_indices]] = rotmats
        H_grasps = tmrp_to_H(grasps)

        pc = pc @ H
        H_grasps = H_grasps @ H

        pc = pc[..., :-1]
        grasps = H_to_tmrp(H_grasps)

        return pc, grasps


## Pointcloud Augmentations


class PointcloudJitter(BaseAugmentation):
    # Adapted from https://github.com/charlesq34/pointnet2/blob/master/utils/provider.py
    def __init__(self, p=0.5, sigma=0.01, clip=0.05) -> None:
        super().__init__(transforms_grasps=False, transforms_pc=True)
        self.sigma = abs(sigma)
        self.clip = clip
        self.p = p

    def forward(self, pc):
        """Apply Jitter
        Args:
            pc (Tensor): pointcloud [B,N,3]

        Returns:
            Tensor : Jittered pointcloud [B,N,3]
        """
        if torch.rand((1,)) < self.p:
            B, N, C = pc.shape

            jitter = torch.clip(
                self.sigma * torch.randn((B, N, C)), -1 * self.clip, self.clip
            ).to(pc.device)
            pc += jitter
        return pc


class RandomPointcloudDropout(BaseAugmentation):
    # Adapted from https://github.com/charlesq34/pointnet2/blob/master/utils/provider.py
    def __init__(self, p=0.7, max_dropout_ratio=0.6) -> None:
        super().__init__(transforms_grasps=False, transforms_pc=True)
        self.max_dropout_ratio = max_dropout_ratio
        self.p = p

    def forward(self, pc):
        """Forward

        Args:
            pc (Tensor): pointcloud [B, N, 3]

        Returns:
            Tensor: pc with dropout [ B, N, 3]
        """

        """pc: BxNx3"""
        if torch.rand((1,)) < self.p:
            b, n, _ = pc.shape
            for b_i in range(b):
                dropout_ratio = torch.rand((1)) * self.max_dropout_ratio
                num_dropout_pts = int(dropout_ratio * n)
                drop_idx = torch.randperm(n)[:num_dropout_pts]

                if len(drop_idx) > 0:
                    # Replace dropout points with first point repeated
                    pc[b_i, drop_idx, :] = pc[b_i, 0, :].clone()

        return pc


# def shift_point_cloud(batch_data, shift_range=0.1):
#     """Randomly shift point cloud. Shift is per point cloud.
#     Input:
#       BxNx3 array, original batch of point clouds
#     Return:
#       BxNx3 array, shifted batch of point clouds
#     """
#     B, N, C = batch_data.shape
#     shifts = np.random.uniform(-shift_range, shift_range, (B, 3))
#     for batch_index in range(B):
#         batch_data[batch_index, :, :] += shifts[batch_index, :]
#     return batch_data


# def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
#     """Randomly scale the point cloud. Scale is per point cloud.
#     Input:
#         BxNx3 array, original batch of point clouds
#     Return:
#         BxNx3 array, scaled batch of point clouds
#     """
#     B, N, C = batch_data.shape
#     scales = np.random.uniform(scale_low, scale_high, B)
#     for batch_index in range(B):
#         batch_data[batch_index, :, :] *= scales[batch_index]
#     return batch_data


class Augmentations:
    POINTCLOUD_AUGMENTATIONS = {
        "PointcloudJitter": PointcloudJitter,
        "PointcloudDropout": RandomPointcloudDropout,
    }
    GRASP_AUGMENTATIONS = {
        "RandomRotation": RandomRotation,
        "RandomPointcloudDropout": RandomPointcloudDropout,
        "RandomRotationPerGrasp": RandomRotationPerGrasp,
    }

    GENERAL_AUGMENTATIONS = {
        "RandomRotationTransform": RandomRotationTransform,
    }

    ALL = {**POINTCLOUD_AUGMENTATIONS, **GRASP_AUGMENTATIONS, **GENERAL_AUGMENTATIONS}

    @staticmethod
    def build_augmentations_from_cfg(augs_cfg):
        """augs

        Args:
            augs_cfg (dict): _description_

        Returns:
            list: list of augmentation objects/instances
        """
        assert isinstance(augs_cfg, dict) or isinstance(
            augs_cfg, list
        ), "Augs config should be a dict or list of dicts"

        augs_cfg = [augs_cfg] if isinstance(augs_cfg, dict) else augs_cfg

        augs = []
        for aug in augs_cfg:
            if aug.type not in Augmentations.ALL:
                raise NotImplementedError(
                    f"augmentation of type {aug.type} is not implemented"
                )

            aug_instance = Augmentations.ALL[aug.type](**aug.args)

            if isinstance(aug_instance, BaseAugmentation):
                assert None not in (
                    aug_instance._transforms_pc,
                    aug_instance._transforms_grasps,
                ), f"`_transforms_pc` and `_transforms_grasp` static properties must be set from the derived class {aug.type}. Found None"
            elif isinstance(aug_instance, BaseGeneralAugmentation):
                pass
            else:
                raise Exception("Unexpected Augmentation base type")
            augs.append(aug_instance)

        return augs
