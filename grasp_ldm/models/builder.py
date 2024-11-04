from torch import nn

from grasp_ldm.utils.config import Config

from . import GraspCVAE, GraspLatentDDM
from .diffusion import GaussianDiffusion1D
from .grasp_classifier import PointsBasedGraspClassifier
from .modules.class_conditioned_resnet import (
    ClassTimeConditionedResNet1D,
)
from .modules.resnets import ResNet1D, TimeConditionedResNet1D, Unet1D

## ----------------- Makeshift Model Registry ----------------- ##
DIFFUSION_MODELS = {
    "GaussianDiffusion1D": GaussianDiffusion1D,
    "TimeConditionedResNet1D": TimeConditionedResNet1D,
    "ClassTimeConditionedResNet1D": ClassTimeConditionedResNet1D,
}

STANDARD_MODULES = {
    "ResNet1D": ResNet1D,
    "Unet1D": Unet1D,
}

CLASSIFIERS = {
    "PointsBasedGraspClassifier": PointsBasedGraspClassifier,
}


ALL_MODELS = {
    "GraspCVAE": GraspCVAE,
    "GraspLatentDDM": GraspLatentDDM,
    **CLASSIFIERS,
    **STANDARD_MODULES,
    **DIFFUSION_MODELS,
}


## ----------------- Model Build methods ----------------- ##


### For now, user `build_model` for single model and `build_model_from_cfg` for multiple models specified in a composite model config
def build_model(model_cfg: Config) -> nn.Module:
    """Build model from config

    Args:
        model_cfg (Config): model config

    Returns:
        (nn.Module): built model
    """
    if model_cfg.type not in ALL_MODELS:
        raise KeyError(
            f"`{model_cfg.type}` in the model_registry. \n Supported models are: {list(ALL_MODELS)}"
        )
    return ALL_MODELS[model_cfg.type](**model_cfg.args)


def build_model_configs_recursive(model_cfg: Config) -> Config:
    """Build model configs recursively

        This allows building of nested models. For example, if we have a model that takes in a model as an argument,
        this can be handled in the config as in the example below:
            model = dict(
                type="SomeModel",
                args=dict(
                    model=dict(
                        type="SomeOtherModel",
                        args=dict(
                            ...
                        )
                    )
                )
            )

        Returns a dict with values for all "model" keys replaced with the built model.

    Args:
        model_cfg (Config): model config

    Returns:
        Config: model config
    """
    # new_model_cfg = copy.deepcopy(cfg)
    if isinstance(model_cfg, dict) or isinstance(model_cfg, Config):
        for k, v in model_cfg.items():
            if k == "args":
                if isinstance(v, dict):
                    model_cfg[k] = build_model_configs_recursive(v)
            if k == "model":
                if isinstance(v, dict):
                    model_cfg[k] = build_model_configs_recursive(v)
                model_cfg[k] = build_model(model_cfg[k])

    return model_cfg


def build_model_from_cfg(model_cfg: Config) -> nn.Module:
    """Build model from config
    # TODO: Rename this to indicate multiple models building

        This relies on a hacky model registry specified by ALL_MODELS and the `type` key in the config.
        The `type` key is used to look up the model class and the `args` key is used to pass in the
        arguments to the model class.

    Args:
        model_cfg (Config): model config

    Returns:
        (nn.Module): model
    """

    # recursively build model configs for nested model configs
    built_model_cfg = build_model_configs_recursive(model_cfg)

    return (
        built_model_cfg.model if hasattr(built_model_cfg, "model") else built_model_cfg
    )
