"""Unit tests for grasp_ldm/losses/loss.py"""
import pytest
import torch

from grasp_ldm.losses.loss import (
    ClassificationLoss,
    GraspReconstructionLoss,
    QualityLoss,
    VAELatentLoss,
    VAEReconstructionLoss,
    linear_cyclical_annealing,
)


# ---------------------------------------------------------------------------
# linear_cyclical_annealing
# ---------------------------------------------------------------------------

class TestLinearCyclicalAnnealing:
    def test_output_length(self):
        schedule = linear_cyclical_annealing(100, start=0.0, stop=1.0, n_cycle=2)
        assert len(schedule) == 100

    def test_starts_at_start_value(self):
        schedule = linear_cyclical_annealing(100, start=0.0, stop=1.0, n_cycle=2)
        assert schedule[0] == pytest.approx(0.0)

    def test_ends_at_stop_value(self):
        schedule = linear_cyclical_annealing(100, start=0.0, stop=1.0, n_cycle=2)
        assert schedule[-1] == pytest.approx(1.0)

    def test_values_within_bounds(self):
        schedule = linear_cyclical_annealing(200, start=0.0, stop=0.5, n_cycle=4)
        assert all(0.0 <= v <= 0.5 for v in schedule)


# ---------------------------------------------------------------------------
# VAEReconstructionLoss
# ---------------------------------------------------------------------------

class TestVAEReconstructionLoss:
    def test_output_is_scalar(self):
        loss_fn = VAEReconstructionLoss(weight=1.0)
        x = torch.randn(4, 6)
        y = torch.randn(4, 6)
        out = loss_fn(x, y)
        assert out.ndim == 0

    def test_identical_inputs_give_zero_loss(self):
        loss_fn = VAEReconstructionLoss(weight=1.0)
        x = torch.randn(4, 6)
        out = loss_fn(x, x)
        assert out.item() == pytest.approx(0.0, abs=1e-6)

    def test_weight_scales_loss(self):
        x = torch.randn(4, 6)
        y = torch.randn(4, 6)
        loss_w1 = VAEReconstructionLoss(weight=1.0)(x, y)
        loss_w2 = VAEReconstructionLoss(weight=2.0)(x, y)
        assert loss_w2.item() == pytest.approx(2.0 * loss_w1.item(), rel=1e-5)

    def test_non_negative(self):
        loss_fn = VAEReconstructionLoss()
        x, y = torch.randn(4, 6), torch.randn(4, 6)
        assert loss_fn(x, y).item() >= 0.0


# ---------------------------------------------------------------------------
# GraspReconstructionLoss
# ---------------------------------------------------------------------------

class TestGraspReconstructionLoss:
    def test_output_is_scalar(self):
        loss_fn = GraspReconstructionLoss()
        x_in = torch.randn(4, 6)
        x_out = torch.randn(4, 6)
        out = loss_fn(x_in, x_out)
        assert out.ndim == 0

    def test_identical_inputs_zero(self):
        loss_fn = GraspReconstructionLoss()
        x = torch.randn(4, 6)
        assert loss_fn(x, x).item() == pytest.approx(0.0, abs=1e-6)

    def test_translation_weight_affects_loss(self):
        """Higher translation weight should give higher loss when translation differs."""
        torch.manual_seed(0)
        x_in = torch.zeros(4, 6)
        x_out = torch.ones(4, 6)
        # Only translation differs (first 3 dims), rotation is same
        x_out_trans_only = torch.zeros(4, 6)
        x_out_trans_only[:, :3] = 1.0

        loss_low_t = GraspReconstructionLoss(translation_weight=1, rotation_weight=0)(
            x_in, x_out_trans_only
        )
        loss_high_t = GraspReconstructionLoss(translation_weight=10, rotation_weight=0)(
            x_in, x_out_trans_only
        )
        assert loss_high_t.item() > loss_low_t.item()

    def test_argument_order_convention(self):
        """x_in is ground truth (first arg), x_out is prediction (second arg).
        Loss should be symmetric for MSE, so this tests naming only."""
        loss_fn = GraspReconstructionLoss(translation_weight=1, rotation_weight=1)
        x = torch.randn(4, 6)
        y = torch.randn(4, 6)
        # MSE is symmetric: loss(x,y) == loss(y,x)
        assert loss_fn(x, y).item() == pytest.approx(loss_fn(y, x).item(), rel=1e-5)


# ---------------------------------------------------------------------------
# VAELatentLoss (KL divergence)
# ---------------------------------------------------------------------------

class TestVAELatentLoss:
    def test_output_is_scalar(self):
        loss_fn = VAELatentLoss(weight=1.0)
        mu = torch.randn(4, 4)
        logvar = torch.zeros(4, 4)
        out = loss_fn(mu, logvar)
        assert out.ndim == 0

    def test_standard_normal_gives_zero_kl(self):
        """KL(N(0,I) || N(0,I)) = 0"""
        loss_fn = VAELatentLoss(weight=1.0)
        mu = torch.zeros(8, 4)
        logvar = torch.zeros(8, 4)  # log(1) = 0
        out = loss_fn(mu, logvar)
        assert out.item() == pytest.approx(0.0, abs=1e-5)

    def test_kl_non_negative(self):
        loss_fn = VAELatentLoss(weight=1.0)
        mu = torch.randn(4, 8)
        logvar = torch.randn(4, 8)
        out = loss_fn(mu, logvar)
        assert out.item() >= -1e-6  # allow tiny floating point error

    def test_weight_scales_loss(self):
        mu = torch.randn(4, 4)
        logvar = torch.randn(4, 4)
        loss_w1 = VAELatentLoss(weight=1.0)(mu, logvar)
        loss_w2 = VAELatentLoss(weight=2.0)(mu, logvar)
        assert loss_w2.item() == pytest.approx(2.0 * loss_w1.item(), rel=1e-5)

    def test_return_unweighted(self):
        loss_fn = VAELatentLoss(weight=0.5)
        mu = torch.randn(4, 4)
        logvar = torch.randn(4, 4)
        weighted, unweighted = loss_fn(mu, logvar, return_unweighted=True)
        assert weighted.item() == pytest.approx(0.5 * unweighted.item(), rel=1e-5)

    def test_annealing_schedule_set_weight(self):
        loss_fn = VAELatentLoss(
            cyclical_annealing=True,
            num_steps=100,
            num_cycles=2,
            start=0.0,
            stop=1.0,
        )
        loss_fn.set_weight_from_schedule(step=0)
        assert loss_fn.weight == pytest.approx(0.0, abs=1e-6)
        loss_fn.set_weight_from_schedule(step=99)
        assert loss_fn.weight == pytest.approx(1.0, abs=0.05)

    def test_annealing_schedule_out_of_bounds_uses_last(self):
        loss_fn = VAELatentLoss(
            cyclical_annealing=True,
            num_steps=10,
            num_cycles=1,
            start=0.0,
            stop=1.0,
        )
        loss_fn.set_weight_from_schedule(step=9999)
        assert loss_fn.weight == pytest.approx(1.0, abs=1e-6)

    def test_no_schedule_raises_on_set_weight(self):
        loss_fn = VAELatentLoss(weight=1.0)
        with pytest.raises(AssertionError):
            loss_fn.set_weight_from_schedule(step=0)


# ---------------------------------------------------------------------------
# ClassificationLoss
# ---------------------------------------------------------------------------

class TestClassificationLoss:
    def test_output_is_scalar(self):
        loss_fn = ClassificationLoss(weight=1.0)
        logits = torch.randn(8)
        targets = torch.randint(0, 2, (8,)).float()
        out = loss_fn(output=logits, targets=targets)
        assert out.ndim == 0

    def test_non_negative(self):
        loss_fn = ClassificationLoss(weight=1.0)
        logits = torch.randn(8)
        targets = torch.randint(0, 2, (8,)).float()
        assert loss_fn(output=logits, targets=targets).item() >= 0.0

    def test_weight_scales_loss(self):
        logits = torch.randn(8)
        targets = torch.randint(0, 2, (8,)).float()
        loss_w1 = ClassificationLoss(weight=1.0)(output=logits, targets=targets)
        loss_w2 = ClassificationLoss(weight=2.0)(output=logits, targets=targets)
        assert loss_w2.item() == pytest.approx(2.0 * loss_w1.item(), rel=1e-5)


# ---------------------------------------------------------------------------
# QualityLoss
# ---------------------------------------------------------------------------

class TestQualityLoss:
    def test_output_is_scalar(self):
        loss_fn = QualityLoss(weight=1.0)
        pred = torch.randn(4, 3)
        target = torch.randn(4, 3)
        out = loss_fn(pred, target)
        assert out.ndim == 0

    def test_identical_inputs_near_zero(self):
        loss_fn = QualityLoss(weight=1.0)
        x = torch.randn(4, 3)
        assert loss_fn(x, x).item() == pytest.approx(0.0, abs=1e-6)

    def test_non_negative(self):
        loss_fn = QualityLoss(weight=1.0)
        pred = torch.randn(4, 3)
        target = torch.randn(4, 3)
        assert loss_fn(pred, target).item() >= 0.0
