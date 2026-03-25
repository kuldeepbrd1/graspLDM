from .acronym.acronym_partial_pointclouds import AcronymPartialPointclouds
from .acronym.acronym_pointclouds import AcronymShapenetPointclouds

ALL_DATASETS = {
    "AcronymShapenetPointclouds": AcronymShapenetPointclouds,
    "AcronymPartialPointclouds": AcronymPartialPointclouds,
}


def build_dataset_from_cfg(data_cfg, split):
    """Build dataset from config.

    Args:
        data_cfg (dict): data config with split keys
        split (str): split name (e.g. "train", "val")

    Raises:
        KeyError: if split not found in data config

    Returns:
        Dataset: dataset instance
    """
    if split not in data_cfg:
        raise KeyError(f"Could not find split:`{split}` in the data config dict")

    split_cfg = data_cfg[split]
    return ALL_DATASETS[split_cfg.type](**split_cfg.args)
