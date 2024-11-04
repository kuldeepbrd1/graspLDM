import json
import numpy as np
import random
import torch
from torch import nn
from typing import Any
from grasp_ldm.utils.rotations import tmrp_to_H

__all__ = [
    "VAEReconstructionLoss",
    "VAELatentLoss",
    "ClassificationLoss",
    "QualityLoss",
    "GraspReconstructionLoss",
    "GraspControlPointsReconstructionLoss",
]


# From: https://github.com/haofuml/cyclical_annealing
def linear_cyclical_anneling(n_iter, start=0.0, stop=1.0, n_cycle=4, ratio=0.5):
    L = np.ones(n_iter) * stop
    period = n_iter / n_cycle
    step = (stop - start) / (period * ratio)  # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i + c * period) < n_iter):
            L[int(i + c * period)] = v
            v += step
            i += 1
    return L


class VAEReconstructionLoss(nn.Module):
    def __init__(self, weight=1, name="reconstruction_loss") -> None:
        super().__init__()
        self.name = name
        self.criterion = nn.MSELoss()
        self.weight = weight

    def forward(self, input, output):
        return self.weight * self.criterion(input, output)


class GraspReconstructionLoss(VAEReconstructionLoss):
    def __init__(
        self, translation_weight=10, rotation_weight=1, name="reconstruction_loss"
    ) -> None:
        super().__init__(weight=1, name=name)

        self.translation_weight = translation_weight
        self.rotation_weight = rotation_weight

    def forward(self, x_out, x_in, **kwargs):
        """Forward

        Args:
            x_out (Tensor): [B, 6] Predicted pose- (t(3), mrp(3))
            x_in (Tensor): [B, 6] Ground truth pose- (t(3), mrp(3))

        Returns:
            _type_: _description_
        """
        x_pred = x_out.clone()
        x_pred[..., :3] = x_pred[..., :3] * self.translation_weight
        x_pred[..., 3:] = x_pred[..., 3:] * self.rotation_weight

        x_gt = x_in.clone()
        x_gt[..., :3] = x_gt[..., :3] * self.translation_weight
        x_gt[..., 3:] = x_gt[..., 3:] * self.rotation_weight

        return super().forward(x_gt, x_pred)


class GraspControlPointsReconstructionLoss(VAEReconstructionLoss):
    def __init__(
        self,
        weight=1,
        name="reconstruction_loss",
        control_pts_file="grasp_ldm/dataset/acronym/gripper_ctrl_pts.json",
    ) -> None:
        super().__init__(weight=1, name=name)

        with open(control_pts_file) as f:
            _control_pts = np.array(json.load(f))

        # append 1 to the end of each control point
        self.control_pts = torch.from_numpy(
            np.concatenate(
                [_control_pts, np.ones((_control_pts.shape[0], 1))],
                axis=1,
            )
        )
        self.criterion = nn.MSELoss()
        self.weight = weight

    def forward(self, x_target, x_pred, **kwargs):
        """Forward

        Args:
            x_out (Tensor): [B, 6] Predicted pose- (t(3), mrp(3))
            x_in (Tensor): [B, 6] Ground truth pose- (t(3), mrp(3))

        Returns:
            _type_: _description_
        """
        metas = kwargs["metas"]
        pc_batch_size = metas["grasp_std"].shape[0]
        h_target = x_target.view((pc_batch_size, -1, 6)) * metas["grasp_std"].unsqueeze(
            1
        ) + metas["grasp_mean"].unsqueeze(1)
        h_pred = x_pred.view((pc_batch_size, -1, 6)) * metas["grasp_std"].unsqueeze(
            1
        ) + metas["grasp_mean"].unsqueeze(1)

        ctrl_pts = self.control_pts.clone().to(h_target.device, h_target.dtype)

        H_target = tmrp_to_H(h_target.view((-1, 6)))
        H_pred = tmrp_to_H(h_pred.view((-1, 6)))

        # Get the control points
        control_pts_target = (H_target @ ctrl_pts.T).transpose(1, 2)
        control_pts_pred = (H_pred @ ctrl_pts.T).transpose(1, 2)

        return self.weight * self.criterion(control_pts_target, control_pts_pred)


class VAELatentLoss(nn.Module):
    def __init__(
        self,
        weight=1,
        name="kl_loss",
        cyclical_annealing=False,
        num_steps=None,
        num_cycles=None,
        start=1e-7,
        stop=0.2,
        ratio=0.25,
    ) -> None:
        super().__init__()
        self.name = name

        if not cyclical_annealing:
            self.weight = weight
            self.schedule = None
        else:
            assert num_cycles is not None and num_steps is not None
            self.weight = None
            self.schedule = linear_cyclical_anneling(
                num_steps,
                start=start,
                stop=stop,
                n_cycle=num_cycles,
                ratio=ratio,
            )
        self.is_annealed = cyclical_annealing

    def forward(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        return_unweighted: bool = False,
        **kwargs,
    ):
        """Forward
            B: Batch size
            D: Dimensions of the latent

        Args:
            mu (torch.Tensor): latent means [B, D]
            logvar (torch.Tensor): latent logvars [B, D]
            step (int, optional): step number for weight schedule.
                        None, if no schedule. i.e. Constant weight
            return_unweighted (bool, optional): Whether to also return unweighted loss
                            Defaults to False
        Returns:
            torch.Tensor:  weighted kl loss [1,]  (if return_unweighted is False)
            tuple(torch.Tensor, torch.Tensor): weighted_loss[1,], unweighted_kld[1,]
        """
        kl_d = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1)
        kl_d = torch.mean(kl_d, dim=0)

        if return_unweighted:
            return self.weight * kl_d, kl_d
        else:
            return self.weight * kl_d

    def set_weight_from_schedule(self, step):
        assert (
            hasattr(self, "schedule") and self.schedule is not None
        ), "No member schedule found in self, to set the loss weight from schedule."
        f"Weight annealing was set to {self.is_annealed}"

        self.weight = (
            self.schedule[step] if step < len(self.schedule) else self.schedule[-1]
        )
        return


class ClassificationLoss(nn.Module):
    def __init__(self, weight=1, name="classfication_loss") -> None:
        super().__init__()
        self.name = name
        self.weight = weight
        self.class_criterion = nn.BCEWithLogitsLoss(reduction="mean")
        self.class_weight = weight

    def forward(self, output, targets, **kwargs):
        classification_loss = self.class_criterion(output, targets)
        return self.weight * classification_loss


class QualityLoss(nn.Module):
    def __init__(self, weight=1, name="quality_loss") -> None:
        super().__init__()
        self.name = name
        self.weight = weight
        self.criterion = nn.SmoothL1Loss()

    def forward(self, quals_in, quals_target, **kwargs):
        confidence_loss = self.criterion(quals_in, quals_target)

        return self.weight * confidence_loss
