"""Shared pytest fixtures for grasp_ldm tests."""
import pytest
import torch
from addict import Dict


# ---------------------------------------------------------------------------
# Tensor fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def num_points():
    return 64


@pytest.fixture
def grasp_latent_size():
    return 4


@pytest.fixture
def pc_latent_size():
    return 16


@pytest.fixture
def dummy_pointcloud(batch_size, num_points):
    """[B, N, 3] point cloud tensor on CPU."""
    return torch.randn(batch_size, num_points, 3)


@pytest.fixture
def dummy_grasps(batch_size):
    """[B, 6] grasp pose tensor (translation + MRP) on CPU."""
    return torch.randn(batch_size, 6)


@pytest.fixture
def dummy_grasps_with_class(batch_size):
    """[B, 7] grasp pose + binary class label on CPU."""
    grasps = torch.randn(batch_size, 6)
    cls = torch.randint(0, 2, (batch_size, 1)).float()
    return torch.cat([grasps, cls], dim=-1)


@pytest.fixture
def dummy_mu(batch_size, grasp_latent_size):
    """[B, D] latent mean tensor."""
    return torch.randn(batch_size, grasp_latent_size)


@pytest.fixture
def dummy_logvar(batch_size, grasp_latent_size):
    """[B, D] latent log-variance tensor (negative to keep variance < 1)."""
    return -torch.abs(torch.randn(batch_size, grasp_latent_size))


# ---------------------------------------------------------------------------
# Config fixtures (dict-based, matching builder expectations)
# ---------------------------------------------------------------------------

@pytest.fixture
def vae_reconstruction_loss_cfg():
    return Dict(type="VAEReconstructionLoss", args=dict(weight=1.0))


@pytest.fixture
def grasp_reconstruction_loss_cfg():
    return Dict(
        type="GraspReconstructionLoss",
        args=dict(translation_weight=1.0, rotation_weight=1.0),
    )


@pytest.fixture
def vae_latent_loss_cfg():
    return Dict(type="VAELatentLoss", args=dict(weight=1.0))


@pytest.fixture
def vae_latent_loss_annealed_cfg():
    return Dict(
        type="VAELatentLoss",
        args=dict(
            cyclical_annealing=True,
            num_steps=100,
            num_cycles=2,
            start=1e-7,
            stop=0.1,
            ratio=0.25,
        ),
    )


@pytest.fixture
def classification_loss_cfg():
    return Dict(type="ClassificationLoss", args=dict(weight=1.0))


@pytest.fixture
def quality_loss_cfg():
    return Dict(type="QualityLoss", args=dict(weight=1.0))
