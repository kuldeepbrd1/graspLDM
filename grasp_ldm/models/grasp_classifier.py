from typing import Optional

import torch
from torch import Tensor, nn

from ..losses import ClassificationLoss as BCELogitLoss
from ..utils.gripper import SimplePandaGripper
from .modules.base_network import BaseGraspClassifier
from .modules.ext.pvcnn.utils import create_mlp_components
from .modules.pc_encoders import PVCNN, PVCNN2


class PointsBasedGraspClassifier(BaseGraspClassifier):
    SUPPORTED_BASE_NETWORKS = {"PVCNN": PVCNN, "PVCNN2": PVCNN2}

    SUPPORTED_LOSSES = {"BCEClassificationLoss": BCELogitLoss}

    def __init__(self, num_pc_points, points_backbone_config: dict, loss_config: dict):
        super().__init__()

        # Loss
        self._loss_config = loss_config
        _classification_loss_cfg = loss_config.classification_loss
        self.loss = self.SUPPORTED_LOSSES[_classification_loss_cfg["type"]](
            **_classification_loss_cfg["args"]
        )

        # Object point cloud
        self.num_pc_points = num_pc_points

        # Base Point cloud network
        self.base_network = self.SUPPORTED_BASE_NETWORKS[
            points_backbone_config["type"]
        ](**points_backbone_config["args"])

        # Cls sub-network
        self._cls_out_dim = 1
        self._width_multiplier = 1

        cls_mlp_layers, _ = create_mlp_components(
            in_channels=self.base_network.out_channels,
            out_channels=[128, 0.5, 1],
            classifier=True,
            dim=2,
            width_multiplier=self._width_multiplier,
        )
        logit_layer = nn.Linear(self.num_pc_points, 1)

        self.classifier = nn.Sequential(*cls_mlp_layers, logit_layer)

        # Classifier outputs binary logits. We use sigmoid to get psuedo-probability
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        pc: Tensor,
        grasp_points: Tensor,
        *,
        cls_target: Tensor = None,
        compute_loss: bool = True
    ) -> Tensor:
        """
        Args:
            pc (Tensor): [B, NP, 3] Point cloud
            grasp_points (Tensor): [B, NG, 3] Grasp pose (t(3), mrp(3))
        Returns:
            Tensor: [B, 1] Grasp success pred logit or loss
        """

        # Add feature label. 0 for pc points and 1 for gripper points
        obj_pc = torch.cat((pc, torch.zeros_like(pc[..., :1])), dim=-1)
        grasp_points = torch.cat(
            (grasp_points, torch.ones_like(grasp_points[..., :1])), dim=-1
        )

        # Concat object and gripper point cloud : [B, Np, 3] -> [B, Np+Ng, 3]
        pc_in = torch.cat((obj_pc, grasp_points), dim=-2)

        # [B, N, 3] -> [B, 3, N]
        pc_in = torch.transpose(pc_in, 1, 2).contiguous()

        # Pass through PVCNN modules
        x = self.base_network(pc_in)

        # [B, 1]
        cls_logit = self.classifier(x).squeeze()

        # # Sanity check
        # assert (
        #     cls_logit.ndim == 1 and cls_logit.shape[0] == pc.shape[0]
        # ), "Something went wrong in classifier shape broadcasting"
        preds = self.sigmoid(cls_logit)
        if compute_loss:
            if cls_target is None:
                raise ValueError("cls_target must be provided if compute_loss is True")

            if cls_target.shape[0] != cls_logit.shape[0]:
                raise ValueError("cls_target and cls_logit size mismatch")

            # Note: Loss is BCE with logits, so we don't apply sigmoid here
            loss = self.loss(cls_logit, cls_target)
            return loss, preds
        else:
            return None, preds

    # def merge_pc_gripper_points(self, pc: Tensor, grasp_pose: Tensor) -> Tensor:
    #     """Merge point cloud and gripper points for PVCNN input

    #     B: Batch size
    #     Np: Number of points in point cloud
    #     Ng: Number of gripper points

    #     Args:
    #         pc (Tensor): [B, Np, 3] Point cloud
    #         grasp_pose (Tensor): [B, 6] Grasp pose (t(3), mrp(3))

    #     Returns:
    #         Tensor: [B, 3, Np+Ng] Point cloud with gripper points
    #     """

    #     # Get projected gripper points per grasp pose: [Ng, 3] -> [Bp, Ng, 3]
    #     grasp_points = self.gripper_points @ grasp_pose
    #     grasp_points = grasp_points[..., :3]

    #     # Transpose for valid input to PVCNN: [Bp, Np, 3] -> [Bp, 3, Np]
    #     pc = pc.transpose(-1, -2).contiguous()
    #     grasp_points = grasp_points.transpose(-1, -2).contiguous()

    #     # Concat point cloud and features: [B, 3, Np+Ng]
    #     pc = torch.cat((pc, grasp_points), dim=-1)

    #     # Construct feature label tensor that is 0 for pc points and 1 for gripper points
    #     feats = torch.zeros_like(pc[..., :1, :])
    #     feats[..., : -self.num_gripper_points, :] = 1

    #     # point-features
    #     pc_with_features = torch.cat((pc, feats), dim=-2)

    #     return pc_with_features

    def classify_grasps(self, pc: Tensor, grasp_pose: Tensor) -> Tensor:
        _, preds = self.forward(pc, grasp_pose, compute_loss=False)
        return preds
