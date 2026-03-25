from torch import nn

from grasp_ldm.utils.config import Config

from . import GraspCVAE, GraspLatentDDM
from .diffusion import GaussianDiffusion1D
from .modules.class_conditioned_resnet import ClassTimeConditionedResNet1D
from .modules.resnets import ResNet1D, TimeConditionedResNet1D, Unet1D

ALL_MODELS = {
    "GraspCVAE": GraspCVAE,
    "GraspLatentDDM": GraspLatentDDM,
    "ResNet1D": ResNet1D,
    "Unet1D": Unet1D,
    "GaussianDiffusion1D": GaussianDiffusion1D,
    "TimeConditionedResNet1D": TimeConditionedResNet1D,
    "ClassTimeConditionedResNet1D": ClassTimeConditionedResNet1D,
}


def build_model(model_cfg: Config) -> nn.Module:
    """Build a single model from config with `type` and `args` keys."""
    if model_cfg.type not in ALL_MODELS:
        raise KeyError(
            f"`{model_cfg.type}` not in the model registry. "
            f"Supported models: {list(ALL_MODELS)}"
        )
    return ALL_MODELS[model_cfg.type](**model_cfg.args)


def build_model_configs_recursive(model_cfg: Config) -> Config:
    """Recursively build nested model configs.

    Allows configs like::

        model = dict(
            type="SomeModel",
            args=dict(model=dict(type="SomeOtherModel", args=dict(...)))
        )
    """
    if isinstance(model_cfg, (dict, Config)):
        for k, v in model_cfg.items():
            if k in ("args", "model") and isinstance(v, dict):
                model_cfg[k] = build_model_configs_recursive(v)
            if k == "model":
                model_cfg[k] = build_model(model_cfg[k])
    return model_cfg


def build_model_from_cfg(model_cfg: Config) -> nn.Module:
    """Build model(s) from a composite config that may contain nested models."""
    built = build_model_configs_recursive(model_cfg)
    return built.model if hasattr(built, "model") else built
