"""Integration tests for GraspLatentDDM forward pass.

Uses the same mock PC encoder as the VAE tests. The LDM wraps a VAE, so
both are tested together through the LDM's forward pass.
"""
import pytest
import torch
import torch.nn as nn
from addict import Dict

# ---------------------------------------------------------------------------
# Shared mock PC encoder (same as in test_vae_forward.py)
# ---------------------------------------------------------------------------

class _MockPCEncoder(nn.Module):
    def __init__(self, out_features=16, **kwargs):
        super().__init__()
        self.out_features = out_features
        self.pool_proj = nn.Linear(3, out_features)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        return self.pool_proj(xyz.mean(dim=1))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRASP_LATENT = 4
PC_LATENT = 16
BATCH = 3
N_POINTS = 32
FEATURE_RES = 8


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def patch_pc_encoder(monkeypatch):
    from grasp_ldm.models.grasp_vae import PcConditionedGraspEncoder
    monkeypatch.setitem(
        PcConditionedGraspEncoder.PC_ENCODERS,
        "MockPCEncoder",
        _MockPCEncoder,
    )


def _make_vae():
    from grasp_ldm.models.grasp_vae import GraspCVAE

    return GraspCVAE(
        grasp_latent_size=GRASP_LATENT,
        pc_latent_size=PC_LATENT,
        grasp_encoder_config=Dict(
            type="ResNet1D",
            args=dict(
                in_features=7,
                block_channels=(8, 16),
                input_conditioning_dims=PC_LATENT,
                resnet_block_groups=2,
            ),
        ),
        pc_encoder_config=Dict(type="MockPCEncoder", args=dict()),
        decoder_config=Dict(
            type="ResNet1D",
            args=dict(
                block_channels=(8, 16),
                input_conditioning_dims=PC_LATENT,
                resnet_block_groups=2,
            ),
        ),
        loss_config=Dict(
            reconstruction_loss=Dict(
                type="GraspReconstructionLoss",
                args=dict(translation_weight=1, rotation_weight=1),
            ),
            latent_loss=Dict(type="VAELatentLoss", args=dict(weight=1.0)),
            classification_loss=Dict(
                type="ClassificationLoss", args=dict(weight=0.1)
            ),
        ),
        intermediate_feature_resolution=FEATURE_RES,
    )


def _make_ldm(vae):
    from grasp_ldm.models.grasp_ldm import GraspLatentDDM
    from addict import Dict as ADict

    denoiser_cfg = ADict(
        type="TimeConditionedResNet1D",
        args=dict(
            dim=GRASP_LATENT,
            channels=1,
            block_channels=(8, 16),
            input_conditioning_dims=PC_LATENT,
            resnet_block_groups=2,
            is_time_conditioned=True,
            learned_variance=False,
            learned_sinusoidal_cond=False,
            random_fourier_features=True,
        ),
    )
    from grasp_ldm.models.builder import build_model
    denoiser = build_model(denoiser_cfg)

    ldm = GraspLatentDDM(
        model=denoiser,
        latent_in_features=GRASP_LATENT,
        diffusion_timesteps=10,  # very few steps for speed
        diffusion_loss="l2",
        beta_schedule="linear",
        noise_scheduler_type="ddpm",
        is_conditioned=True,
        joint_training=False,
        denoising_loss_weight=1,
        variance_type="fixed_small",
        elucidated_diffusion=False,
        beta_start=1e-4,
        beta_end=0.02,
    )
    ldm.set_vae_model(vae)
    ldm.freeze_vae_model()
    return ldm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGraspLatentDDMForward:
    def test_instantiation(self):
        vae = _make_vae()
        ldm = _make_ldm(vae)
        assert ldm is not None
        assert ldm.vae_model is not None
        assert ldm.is_vae_frozen

    def test_forward_returns_loss(self):
        vae = _make_vae()
        ldm = _make_ldm(vae).train()

        xyz = torch.randn(BATCH, N_POINTS, 3)
        grasps = torch.randn(BATCH, 7)

        _, loss_dict = ldm(xyz, grasps)
        assert "loss" in loss_dict

    def test_denoising_loss_is_scalar(self):
        vae = _make_vae()
        ldm = _make_ldm(vae).train()

        xyz = torch.randn(BATCH, N_POINTS, 3)
        grasps = torch.randn(BATCH, 7)

        _, loss_dict = ldm(xyz, grasps)
        loss = loss_dict.loss
        assert loss.ndim == 0, f"Expected scalar loss, got shape {loss.shape}"

    def test_denoising_loss_is_finite(self):
        vae = _make_vae()
        ldm = _make_ldm(vae).train()

        xyz = torch.randn(BATCH, N_POINTS, 3)
        grasps = torch.randn(BATCH, 7)

        _, loss_dict = ldm(xyz, grasps)
        loss = loss_dict.loss
        assert torch.isfinite(loss), "Denoising loss is not finite"

    def test_vae_params_frozen_during_ldm_training(self):
        vae = _make_vae()
        ldm = _make_ldm(vae)

        for param in ldm.vae_model.parameters():
            assert not param.requires_grad, "VAE param should be frozen"

    def test_diffusion_model_params_trainable(self):
        vae = _make_vae()
        ldm = _make_ldm(vae)
        trainable = [p for p in ldm.diffusion_model.parameters() if p.requires_grad]
        assert len(trainable) > 0, "Diffusion model should have trainable parameters"

    def test_generate_grasps_shape(self):
        """generate_grasps should return poses of shape [num_grasps, 6]."""
        vae = _make_vae()
        ldm = _make_ldm(vae).eval()
        ldm.set_inference_timesteps(5)

        xyz = torch.randn(1, N_POINTS, 3)  # single object
        num_grasps = 4

        out = ldm.generate_grasps(xyz, num_grasps=num_grasps)
        # generate_grasps returns (decoder_output, intermediates)
        # decoder_output is (tmrp, cls_logits[, quals])
        decoder_out = out[0] if isinstance(out, (tuple, list)) else out
        tmrp = decoder_out[0] if isinstance(decoder_out, (tuple, list)) else decoder_out
        assert tmrp.shape[0] == num_grasps, (
            f"Expected {num_grasps} grasps, got {tmrp.shape[0]}"
        )
