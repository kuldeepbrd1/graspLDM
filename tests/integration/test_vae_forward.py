"""Integration tests for GraspCVAE forward pass.

Uses a mock PC encoder (simple nn.Linear) to avoid the ninja-compiled PVCNN
CUDA extensions, so these tests run on CPU in CI without a GPU.
"""
import pytest
import torch
import torch.nn as nn
from addict import Dict


# ---------------------------------------------------------------------------
# Minimal mock PC encoder (replaces PVCNNEncoder)
# ---------------------------------------------------------------------------

class _MockPCEncoder(nn.Module):
    """Drop-in CPU-compatible replacement for PVCNNEncoder.

    Accepts xyz [B, N, 3] and returns a global latent [B, out_features].
    """

    def __init__(self, out_features=16, **kwargs):
        super().__init__()
        self.out_features = out_features
        self.pool_proj = nn.Linear(3, out_features)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        # xyz: [B, N, 3]  →  mean pool → [B, 3]  →  linear → [B, out_features]
        return self.pool_proj(xyz.mean(dim=1))


# ---------------------------------------------------------------------------
# Config factories
# ---------------------------------------------------------------------------

GRASP_LATENT = 4
PC_LATENT = 16
BATCH = 3
N_POINTS = 32
FEATURE_RES = 8


def _make_loss_config(max_steps=10):
    return Dict(
        reconstruction_loss=Dict(
            type="GraspReconstructionLoss",
            args=dict(translation_weight=1, rotation_weight=1),
        ),
        latent_loss=Dict(
            type="VAELatentLoss",
            args=dict(
                cyclical_annealing=True,
                num_steps=max_steps,
                num_cycles=1,
                ratio=0.5,
                start=1e-7,
                stop=0.1,
            ),
        ),
        classification_loss=Dict(
            type="ClassificationLoss", args=dict(weight=0.1)
        ),
    )


def _make_grasp_encoder_config():
    return Dict(
        type="ResNet1D",
        args=dict(
            in_features=7,  # 6 (tmrp) + 1 (class label)
            block_channels=(8, 16),
            input_conditioning_dims=PC_LATENT,
            resnet_block_groups=2,
        ),
    )


def _make_decoder_config():
    return Dict(
        type="ResNet1D",
        args=dict(
            block_channels=(8, 16),
            input_conditioning_dims=PC_LATENT,
            resnet_block_groups=2,
        ),
    )


def _make_pc_encoder_config():
    # type = "MockPCEncoder" will be patched into PC_ENCODERS
    return Dict(type="MockPCEncoder", args=dict())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def patch_pc_encoder(monkeypatch):
    """Replace PcConditionedGraspEncoder.PC_ENCODERS with mock before each test."""
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
        grasp_encoder_config=_make_grasp_encoder_config(),
        pc_encoder_config=_make_pc_encoder_config(),
        decoder_config=_make_decoder_config(),
        loss_config=_make_loss_config(),
        intermediate_feature_resolution=FEATURE_RES,
        num_output_qualities=None,
    ).eval()


class TestGraspCVAEForward:
    def test_instantiation(self):
        model = _make_vae()
        assert model is not None

    def test_forward_train_returns_loss_dict(self):
        model = _make_vae()
        xyz = torch.randn(BATCH, N_POINTS, 3)
        # grasps: [B*n_grasps, 7] where 7 = 6 (pose) + 1 (class)
        grasps = torch.randn(BATCH, 7)
        grasps[:, 6] = (grasps[:, 6] > 0).float()  # binary class

        out, loss_dict = model(xyz, grasps, compute_loss=True)
        assert "loss" in loss_dict
        assert "reconstruction_loss" in loss_dict
        assert "latent_loss" in loss_dict
        assert loss_dict.loss.ndim == 0  # scalar

    def test_forward_no_loss_returns_tuple(self):
        model = _make_vae()
        xyz = torch.randn(BATCH, N_POINTS, 3)
        grasps = torch.randn(BATCH, 7)

        out = model(xyz, grasps, compute_loss=False)
        assert isinstance(out, (tuple, list))

    def test_tmrp_output_shape(self):
        model = _make_vae()
        xyz = torch.randn(BATCH, N_POINTS, 3)
        grasps = torch.randn(BATCH, 7)

        out, _ = model(xyz, grasps, compute_loss=True)
        tmrp = out[0]
        assert tmrp.shape == (BATCH, 6), f"Expected [{BATCH}, 6], got {tmrp.shape}"

    def test_class_logits_output_shape(self):
        model = _make_vae()
        xyz = torch.randn(BATCH, N_POINTS, 3)
        grasps = torch.randn(BATCH, 7)

        out, _ = model(xyz, grasps, compute_loss=True)
        cls_logits = out[1]
        assert cls_logits.shape == (BATCH, 1), f"Expected [{BATCH}, 1], got {cls_logits.shape}"

    def test_encode_pc_shape(self):
        model = _make_vae()
        xyz = torch.randn(BATCH, N_POINTS, 3)
        z_pc = model.encode_pc(xyz)
        assert z_pc.shape == (BATCH, PC_LATENT)

    def test_generate_grasps_shape(self):
        model = _make_vae()
        xyz = torch.randn(BATCH, N_POINTS, 3)
        num_grasps = 5

        out = model.generate_grasps(xyz, num_grasps=num_grasps)
        tmrp = out[0]
        assert tmrp.shape[0] == BATCH * num_grasps

    def test_loss_is_finite(self):
        model = _make_vae()
        xyz = torch.randn(BATCH, N_POINTS, 3)
        grasps = torch.randn(BATCH, 7)

        _, loss_dict = model(xyz, grasps, compute_loss=True)
        assert torch.isfinite(loss_dict.loss), "Total loss is not finite"

    def test_gradients_flow(self):
        """Backward pass should not raise and gradients should be non-None."""
        model = _make_vae().train()
        xyz = torch.randn(BATCH, N_POINTS, 3)
        grasps = torch.randn(BATCH, 7)

        _, loss_dict = model(xyz, grasps, compute_loss=True)
        loss_dict.loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for param: {name}"


class TestGraspCVAEWithQualities:
    def test_instantiation_with_qualities(self):
        from grasp_ldm.models.grasp_vae import GraspCVAE

        model = GraspCVAE(
            grasp_latent_size=GRASP_LATENT,
            pc_latent_size=PC_LATENT,
            grasp_encoder_config=Dict(
                type="ResNet1D",
                args=dict(
                    in_features=11,  # 6 + 1 (class) + 4 (qualities)
                    block_channels=(8, 16),
                    input_conditioning_dims=PC_LATENT,
                    resnet_block_groups=2,
                ),
            ),
            pc_encoder_config=_make_pc_encoder_config(),
            decoder_config=_make_decoder_config(),
            loss_config=Dict(
                reconstruction_loss=Dict(
                    type="GraspReconstructionLoss",
                    args=dict(translation_weight=1, rotation_weight=1),
                ),
                latent_loss=Dict(
                    type="VAELatentLoss",
                    args=dict(weight=1.0),
                ),
                quality_loss=Dict(type="QualityLoss", args=dict(weight=0.1)),
            ),
            intermediate_feature_resolution=FEATURE_RES,
            num_output_qualities=4,
        ).eval()

        assert model.use_grasp_qualities is True
        xyz = torch.randn(BATCH, N_POINTS, 3)
        grasps = torch.randn(BATCH, 11)
        out, loss_dict = model(xyz, grasps, compute_loss=True)
        # Should have 3 outputs: tmrp, cls_logits, qualities
        assert len(out) == 3
        assert out[2].shape == (BATCH, 4)
