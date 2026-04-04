"""Unit tests for grasp_ldm/utils/rotations.py

Focuses on round-trip consistency (H → tmrp → H and tmrp → H → tmrp)
and shape correctness for all public conversion functions.
"""
import pytest
import torch

from grasp_ldm.utils.rotations import (
    H_to_Rt,
    H_to_qt,
    H_to_tmrp,
    Rt_to_H,
    get_random_rotations_in_angle_limit,
    mrp_to_quat,
    mrp_to_rotmat,
    quat_to_rotmat,
    quat_wxyz_to_xyzw,
    quat_xyzw_to_wxyz,
    qt_to_H,
    rotmat_to_mrp,
    rotmat_to_quat,
    tmrp_to_H,
)

ATOL = 1e-5
BATCH = 8


def _identity_H(batch=1):
    return torch.eye(4).unsqueeze(0).repeat(batch, 1, 1)


def _random_H(batch=BATCH, angle_limit=0.5):
    """Random valid 4x4 homogeneous transforms (small rotations to stay away from MRP singularity)."""
    rotmats = get_random_rotations_in_angle_limit(angle_limit, batch_size=batch)
    t = torch.randn(batch, 3) * 0.5
    return Rt_to_H(rotmats, t)


# ---------------------------------------------------------------------------
# Quaternion convention helpers
# ---------------------------------------------------------------------------

class TestQuatConventionHelpers:
    def test_xyzw_to_wxyz_roundtrip(self):
        q = torch.randn(8, 4)
        q = q / q.norm(dim=-1, keepdim=True)
        assert torch.allclose(quat_wxyz_to_xyzw(quat_xyzw_to_wxyz(q)), q, atol=ATOL)

    def test_wxyz_to_xyzw_roundtrip(self):
        q = torch.randn(8, 4)
        q = q / q.norm(dim=-1, keepdim=True)
        assert torch.allclose(quat_xyzw_to_wxyz(quat_wxyz_to_xyzw(q)), q, atol=ATOL)


# ---------------------------------------------------------------------------
# rotmat ↔ quaternion
# ---------------------------------------------------------------------------

class TestRotmatQuat:
    def test_identity_rotmat_to_quat(self):
        """Identity rotation → quaternion [0,0,0,1] (xyzw)."""
        R = torch.eye(3).unsqueeze(0)
        q = rotmat_to_quat(R)  # xyzw
        expected = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        assert torch.allclose(q.abs(), expected.abs(), atol=ATOL)

    def test_quat_to_rotmat_identity(self):
        q = torch.tensor([[0.0, 0.0, 0.0, 1.0]])  # xyzw identity
        R = quat_to_rotmat(q, is_xyzw=True)
        assert torch.allclose(R, torch.eye(3).unsqueeze(0), atol=ATOL)

    def test_rotmat_quat_roundtrip(self):
        """R → q → R should recover original rotation matrix."""
        rotmats = get_random_rotations_in_angle_limit(1.0, batch_size=BATCH)
        q = rotmat_to_quat(rotmats)
        R_recovered = quat_to_rotmat(q, is_xyzw=True)
        assert torch.allclose(rotmats, R_recovered, atol=ATOL)

    def test_output_shapes(self):
        rotmats = get_random_rotations_in_angle_limit(0.5, batch_size=BATCH)
        q = rotmat_to_quat(rotmats)
        assert q.shape == (BATCH, 4)
        R = quat_to_rotmat(q, is_xyzw=True)
        assert R.shape == (BATCH, 3, 3)


# ---------------------------------------------------------------------------
# MRP ↔ quaternion ↔ rotmat
# ---------------------------------------------------------------------------

class TestMRP:
    def test_mrp_to_quat_shape(self):
        mrp = torch.randn(BATCH, 3) * 0.1  # small MRPs
        q = mrp_to_quat(mrp)
        assert q.shape == (BATCH, 4)

    def test_mrp_to_quat_unit_norm(self):
        mrp = torch.randn(BATCH, 3) * 0.1
        q = mrp_to_quat(mrp)
        norms = q.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(BATCH), atol=ATOL)

    def test_zero_mrp_is_identity(self):
        """MRP = [0,0,0] → identity quaternion [0,0,0,1] (xyzw)."""
        mrp = torch.zeros(1, 3)
        q = mrp_to_quat(mrp)
        expected = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        assert torch.allclose(q.abs(), expected.abs(), atol=ATOL)

    def test_mrp_rotmat_roundtrip(self):
        """Small MRPs: mrp → R → mrp should recover original."""
        mrp = torch.randn(BATCH, 3) * 0.1
        R = mrp_to_rotmat(mrp)
        mrp_recovered = rotmat_to_mrp(R)
        assert torch.allclose(mrp, mrp_recovered, atol=ATOL)

    def test_mrp_rotmat_shape(self):
        mrp = torch.randn(BATCH, 3) * 0.1
        R = mrp_to_rotmat(mrp)
        assert R.shape == (BATCH, 3, 3)


# ---------------------------------------------------------------------------
# tmrp ↔ H (the critical end-to-end conversion)
# ---------------------------------------------------------------------------

class TestTmrpH:
    def test_tmrp_to_H_shape(self):
        tmrp = torch.randn(BATCH, 6) * 0.1
        tmrp[:, 3:] *= 0.1  # keep MRP small
        H = tmrp_to_H(tmrp)
        assert H.shape == (BATCH, 4, 4)

    def test_H_to_tmrp_shape(self):
        H = _random_H(BATCH)
        tmrp = H_to_tmrp(H)
        assert tmrp.shape == (BATCH, 6)

    def test_tmrp_to_H_roundtrip(self):
        """tmrp → H → tmrp should recover original (small rotations)."""
        tmrp = torch.randn(BATCH, 6) * 0.3
        tmrp[:, 3:] *= 0.1  # keep MRP small to avoid singularity
        H = tmrp_to_H(tmrp)
        tmrp_recovered = H_to_tmrp(H)
        assert torch.allclose(tmrp, tmrp_recovered, atol=ATOL)

    def test_H_to_tmrp_roundtrip(self):
        """H → tmrp → H should recover original transform."""
        H = _random_H(BATCH, angle_limit=0.3)
        tmrp = H_to_tmrp(H)
        H_recovered = tmrp_to_H(tmrp)
        assert torch.allclose(H, H_recovered, atol=ATOL)

    def test_identity_H_gives_zero_tmrp(self):
        """Identity H → tmrp should have zero translation and zero MRP."""
        H = _identity_H(batch=1)
        tmrp = H_to_tmrp(H)
        assert torch.allclose(tmrp, torch.zeros(1, 6), atol=ATOL)

    def test_translation_preserved(self):
        """Translation component of tmrp should match H[:, :3, 3]."""
        H = _random_H(BATCH)
        tmrp = H_to_tmrp(H)
        t_from_H = H[:, :3, 3]
        t_from_tmrp = tmrp[:, :3]
        assert torch.allclose(t_from_H, t_from_tmrp, atol=ATOL)


# ---------------------------------------------------------------------------
# Rt_to_H / H_to_Rt
# ---------------------------------------------------------------------------

class TestRtH:
    def test_roundtrip(self):
        rotmats = get_random_rotations_in_angle_limit(0.5, batch_size=BATCH)
        t = torch.randn(BATCH, 3)
        H = Rt_to_H(rotmats, t)
        R_out, t_out = H_to_Rt(H)
        assert torch.allclose(rotmats, R_out, atol=ATOL)
        assert torch.allclose(t, t_out, atol=ATOL)

    def test_H_shape(self):
        rotmats = get_random_rotations_in_angle_limit(0.5, batch_size=BATCH)
        t = torch.randn(BATCH, 3)
        H = Rt_to_H(rotmats, t)
        assert H.shape == (BATCH, 4, 4)

    def test_bottom_row_is_0001(self):
        rotmats = get_random_rotations_in_angle_limit(0.5, batch_size=BATCH)
        t = torch.randn(BATCH, 3)
        H = Rt_to_H(rotmats, t)
        bottom = H[:, 3, :]
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(BATCH, -1)
        assert torch.allclose(bottom, expected, atol=ATOL)


# ---------------------------------------------------------------------------
# qt_to_H / H_to_qt
# ---------------------------------------------------------------------------

class TestQtH:
    def test_roundtrip(self):
        H = _random_H(BATCH, angle_limit=0.5)
        q, t = H_to_qt(H)
        H_recovered = qt_to_H(q, t, is_xyzw=True)
        assert torch.allclose(H, H_recovered, atol=ATOL)

    def test_output_shapes(self):
        H = _random_H(BATCH)
        q, t = H_to_qt(H)
        assert q.shape == (BATCH, 4)
        assert t.shape == (BATCH, 3)


# ---------------------------------------------------------------------------
# get_random_rotations_in_angle_limit
# ---------------------------------------------------------------------------

class TestRandomRotations:
    def test_output_shape(self):
        R = get_random_rotations_in_angle_limit(1.0, batch_size=BATCH)
        assert R.shape == (BATCH, 3, 3)

    def test_is_valid_rotation_matrix(self):
        """R @ R.T ≈ I and det(R) ≈ 1."""
        R = get_random_rotations_in_angle_limit(1.0, batch_size=BATCH)
        I = torch.bmm(R, R.transpose(-1, -2))
        assert torch.allclose(I, torch.eye(3).expand(BATCH, -1, -1), atol=1e-4)
        det = torch.linalg.det(R)
        assert torch.allclose(det, torch.ones(BATCH), atol=1e-4)


# ---------------------------------------------------------------------------
# 6D rotation representation — Zhou et al. CVPR 2019
# ---------------------------------------------------------------------------

from grasp_ldm.utils.rotations import H_to_t6d, matrix_to_rot6d, rot6d_to_matrix, t6d_to_H


class TestRot6D:
    def test_rot6d_shape(self):
        R = get_random_rotations_in_angle_limit(1.0, batch_size=BATCH)
        r6 = matrix_to_rot6d(R)
        assert r6.shape == (BATCH, 6)

    def test_rot6d_roundtrip(self):
        """matrix → 6D → matrix should be identity."""
        R = get_random_rotations_in_angle_limit(1.0, batch_size=BATCH)
        R_recovered = rot6d_to_matrix(matrix_to_rot6d(R))
        assert torch.allclose(R, R_recovered, atol=1e-5)

    def test_rot6d_to_matrix_is_valid_rotation(self):
        """Recovered matrix should satisfy R @ R.T ≈ I and det ≈ 1."""
        R = get_random_rotations_in_angle_limit(1.0, batch_size=BATCH)
        R2 = rot6d_to_matrix(matrix_to_rot6d(R))
        I = torch.bmm(R2, R2.transpose(-1, -2))
        assert torch.allclose(I, torch.eye(3).expand(BATCH, -1, -1), atol=1e-4)
        det = torch.linalg.det(R2)
        assert torch.allclose(det, torch.ones(BATCH), atol=1e-4)

    def test_rot6d_works_for_arbitrary_6d_input(self):
        """rot6d_to_matrix must produce valid SO(3) even for non-orthonormal input."""
        r6 = torch.randn(BATCH, 6)
        R = rot6d_to_matrix(r6)
        I = torch.bmm(R, R.transpose(-1, -2))
        assert torch.allclose(I, torch.eye(3).expand(BATCH, -1, -1), atol=1e-4)

    def test_t6d_shape(self):
        H = _random_H(BATCH)
        t6d = H_to_t6d(H)
        assert t6d.shape == (BATCH, 9)

    def test_t6d_roundtrip(self):
        """H → T6D → H should recover the original transform."""
        H = _random_H(BATCH, angle_limit=0.5)
        H2 = t6d_to_H(H_to_t6d(H))
        assert torch.allclose(H, H2, atol=1e-5)


class TestPointcloudDropoutBias:
    """Verify that RandomPointcloudDropout no longer concentrates replacements at index 0."""

    def test_replacement_not_biased_to_index_zero(self):
        from grasp_ldm.dataset.augmentations import RandomPointcloudDropout

        aug = RandomPointcloudDropout(p=1.0, max_dropout_ratio=0.5)
        torch.manual_seed(42)
        # Distinct points so we can count how often index-0 appears after dropout
        B, N = 1, 100
        pc = torch.zeros(B, N, 3)
        pc[0, :, 0] = torch.arange(N, dtype=torch.float)  # x = point index

        pc_out = aug(pc.clone())

        # Count how many of the surviving values equal original index-0 value (x=0)
        zero_count = (pc_out[0, :, 0] == 0).sum().item()
        # With unbiased sampling, index-0 should appear at most ~5% of the time (1/N)
        # With biased (old) sampling it would dominate. We check it's < 30%.
        assert zero_count < N * 0.3, (
            f"Too many replacements from index 0: {zero_count}/{N}. "
            "Dropout replacement may still be biased."
        )
