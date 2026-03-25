"""Unit tests for factory builder modules."""
import pytest
from addict import Dict

from grasp_ldm.losses.builder import ALL_LOSSES, build_loss_from_cfg
from grasp_ldm.models.builder import ALL_MODELS, build_model


# ---------------------------------------------------------------------------
# Loss builder
# ---------------------------------------------------------------------------

class TestBuildLossFromCfg:
    @pytest.mark.parametrize("loss_type", list(ALL_LOSSES.keys()))
    def test_all_losses_in_registry(self, loss_type):
        """Every entry in ALL_LOSSES should be a class, not None."""
        assert ALL_LOSSES[loss_type] is not None

    def test_build_vae_reconstruction_loss(self):
        cfg = Dict(type="VAEReconstructionLoss", args=dict(weight=1.0))
        loss = build_loss_from_cfg(cfg)
        assert loss is not None
        assert loss.weight == pytest.approx(1.0)

    def test_build_grasp_reconstruction_loss(self):
        cfg = Dict(
            type="GraspReconstructionLoss",
            args=dict(translation_weight=5.0, rotation_weight=2.0),
        )
        loss = build_loss_from_cfg(cfg)
        assert loss.translation_weight == pytest.approx(5.0)
        assert loss.rotation_weight == pytest.approx(2.0)

    def test_build_vae_latent_loss_constant(self):
        cfg = Dict(type="VAELatentLoss", args=dict(weight=0.5))
        loss = build_loss_from_cfg(cfg)
        assert loss.weight == pytest.approx(0.5)
        assert loss.schedule is None

    def test_build_vae_latent_loss_annealed(self):
        cfg = Dict(
            type="VAELatentLoss",
            args=dict(
                cyclical_annealing=True,
                num_steps=100,
                num_cycles=2,
                start=0.0,
                stop=1.0,
            ),
        )
        loss = build_loss_from_cfg(cfg)
        assert loss.schedule is not None
        assert len(loss.schedule) == 100

    def test_build_classification_loss(self):
        cfg = Dict(type="ClassificationLoss", args=dict(weight=0.1))
        loss = build_loss_from_cfg(cfg)
        assert loss.weight == pytest.approx(0.1)

    def test_build_quality_loss(self):
        cfg = Dict(type="QualityLoss", args=dict(weight=0.2))
        loss = build_loss_from_cfg(cfg)
        assert loss.weight == pytest.approx(0.2)

    def test_unknown_type_raises_key_error(self):
        cfg = Dict(type="NonExistentLoss", args=dict())
        with pytest.raises(KeyError):
            build_loss_from_cfg(cfg)


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

class TestModelRegistry:
    @pytest.mark.parametrize("model_type", list(ALL_MODELS.keys()))
    def test_all_models_in_registry(self, model_type):
        """Every entry in ALL_MODELS should be a class, not None."""
        assert ALL_MODELS[model_type] is not None

    def test_build_resnet1d(self):
        from addict import Dict as ADict
        cfg = ADict(
            type="ResNet1D",
            args=dict(
                dim=16,
                dim_mults=(1, 2),
                in_features=16,
            ),
        )
        model = build_model(cfg)
        assert model is not None

    def test_unknown_model_raises_key_error(self):
        cfg = Dict(type="NonExistentModel", args=dict())
        with pytest.raises(KeyError):
            build_model(cfg)


# ---------------------------------------------------------------------------
# Dataset builder (registry only — no actual file loading)
# ---------------------------------------------------------------------------

class TestDatasetRegistry:
    def test_dataset_registry_non_empty(self):
        from grasp_ldm.dataset.builder import ALL_DATASETS
        assert len(ALL_DATASETS) > 0

    def test_all_dataset_types_are_classes(self):
        from grasp_ldm.dataset.builder import ALL_DATASETS
        for name, cls in ALL_DATASETS.items():
            assert cls is not None, f"Dataset {name} is None in registry"

    def test_missing_split_raises_key_error(self):
        from addict import Dict as ADict
        from grasp_ldm.dataset.builder import build_dataset_from_cfg
        data_cfg = ADict(
            train=ADict(type="AcronymShapenetPointclouds", args=dict())
        )
        with pytest.raises(KeyError):
            build_dataset_from_cfg(data_cfg, split="val")
