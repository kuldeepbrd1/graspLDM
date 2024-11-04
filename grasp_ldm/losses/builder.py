from .loss import *

ALL_LOSSES = {
    "VAEReconstructionLoss": VAEReconstructionLoss,
    "VAELatentLoss": VAELatentLoss,
    "GraspReconstructionLoss": GraspReconstructionLoss,
    "QualityLoss": QualityLoss,
    "ClassificationLoss": ClassificationLoss,
    "GraspControlPointsReconstructionLoss": GraspControlPointsReconstructionLoss,
}


def build_loss_from_cfg(loss_cfg):
    return ALL_LOSSES[loss_cfg.type](**loss_cfg.args)
