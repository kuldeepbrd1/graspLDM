from .acronym.acronym_grasp_points import (
    AcronymFullPcGraspPointsClassification,
    AcronymPartialPcGraspPointsClassification,
)
from .acronym.acronym_partial_pointclouds import AcronymPartialPointclouds
from .acronym.acronym_pointclouds import AcronymShapenetPointclouds

POINTCLOUD_GRASP_DATASETS = {
    "AcronymShapenetPointclouds": AcronymShapenetPointclouds,
    "AcronymPartialPointclouds": AcronymPartialPointclouds,
}


POINTCLOUD_GRASP_CLASIFICATION_DATASETS = {
    "AcronymFullPcGraspPointsClassification": AcronymFullPcGraspPointsClassification,
    "AcronymPartialPcGraspPointsClassification": AcronymPartialPcGraspPointsClassification,
}

ALL_DATASETS = {
    **POINTCLOUD_GRASP_DATASETS,
    **POINTCLOUD_GRASP_CLASIFICATION_DATASETS,
}


def build_dataset_from_cfg(data_cfg, split):
    """Build dataset from config

    Args:
        data_cfg (dict): data config
        split (str): split name

    Raises:
        KeyError: if split not found in data config

    Returns:
        Dataset: dataset
    """
    if split not in data_cfg:
        raise KeyError(f"Could not find split:`{split}` in the data config dict")

    split_cfg = data_cfg[split]
    return ALL_DATASETS[split_cfg.type](**split_cfg.args)
