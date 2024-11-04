import enum

import numpy as np
import torch


## Add enum for pose representation
class PoseRepresentation(enum.Enum):
    TMRP = enum.auto()
    TQUAT = enum.auto()
    H = enum.auto()


def quat_xyzw_to_wxyz(q):
    return q[..., [3, 0, 1, 2]]


def quat_wxyz_to_xyzw(q):
    return q[..., [1, 2, 3, 0]]


def H_to_pyt3d(H):
    """Convert homogeneous transform matrix to PyTorch3D transform matrix

    Note that Pytorch3D uses the following convention for the transform matrix:
    M = [
        [Rxx, Ryx, Rzx, 0],
        [Rxy, Ryy, Rzy, 0],
        [Rxz, Ryz, Rzz, 0],
        [Tx,  Ty,  Tz,  1],
    ]

    which is transposed in rows and columns
    """
    assert (
        H.shape[-1] == 4 and H.shape[-2] == 4
    ), "Invalid shape of homogeneous transform matrix "
    return torch.transpose(H, -1, -2)


# Adapted from roma: https://github.com/naver/roma/blob/master/roma/internal.py


def flatten_batch_dims(tensor, end_dim):
    """
    Utility function: flatten multiple batch dimensions into a single one, or add a batch dimension if there is none.
    """
    batch_shape = tensor.shape[: end_dim + 1]
    flattened = (
        tensor.flatten(end_dim=end_dim) if len(batch_shape) > 0 else tensor.unsqueeze(0)
    )
    return flattened, batch_shape


def unflatten_batch_dims(tensor, batch_shape):
    """
    Revert flattening of a tensor.
    """
    # Note: alternative to tensor.unflatten(dim=0, sizes=batch_shape) that was not supported by PyTorch 1.6.0.
    return (
        tensor.reshape(batch_shape + tensor.shape[1:])
        if len(batch_shape) > 0
        else tensor.squeeze(0)
    )


def rotmat_to_quat(R, return_wxyz=False):
    """
    Converts rotation matrix to unit quaternion representation.
    Args:
        R (...x3x3 tensor): batch of rotation matrices.
    Returns:
        batch of quat (...x3 tensor).
    """
    matrix, batch_shape = flatten_batch_dims(R, end_dim=-3)
    num_rotations, D1, D2 = matrix.shape
    assert (D1, D2) == (3, 3), "Input should be a Bx3x3 tensor."

    # Adapted from SciPy:
    # https://github.com/scipy/scipy/blob/7cb3d751756907238996502b92709dc45e1c6596/scipy/spatial/transform/rotation.py#L480

    decision_matrix = torch.empty(
        (num_rotations, 4), dtype=matrix.dtype, device=matrix.device
    )
    decision_matrix[:, :3] = matrix.diagonal(dim1=1, dim2=2)
    decision_matrix[:, -1] = decision_matrix[:, :3].sum(axis=1)
    choices = decision_matrix.argmax(axis=1)

    quat = torch.empty((num_rotations, 4), dtype=matrix.dtype, device=matrix.device)

    ind = torch.nonzero(choices != 3, as_tuple=True)[0]
    i = choices[ind]
    j = (i + 1) % 3
    k = (j + 1) % 3

    quat[ind, i] = 1 - decision_matrix[ind, -1] + 2 * matrix[ind, i, i]
    quat[ind, j] = matrix[ind, j, i] + matrix[ind, i, j]
    quat[ind, k] = matrix[ind, k, i] + matrix[ind, i, k]
    quat[ind, 3] = matrix[ind, k, j] - matrix[ind, j, k]

    ind = torch.nonzero(choices == 3, as_tuple=True)[0]
    quat[ind, 0] = matrix[ind, 2, 1] - matrix[ind, 1, 2]
    quat[ind, 1] = matrix[ind, 0, 2] - matrix[ind, 2, 0]
    quat[ind, 2] = matrix[ind, 1, 0] - matrix[ind, 0, 1]
    quat[ind, 3] = 1 + decision_matrix[ind, -1]

    # xyzw
    quat = quat / torch.norm(quat, dim=1)[:, None]

    quat = quat_xyzw_to_wxyz(quat) if return_wxyz else quat

    return unflatten_batch_dims(quat, batch_shape)


def rotmat_to_mrp(R):
    """

    TODO: Use rotmat_to_quat and quat_to_mrp instead of this function.

    Converts rotation matrix to unit quaternion representation.
    Args:
        R (...x3x3 tensor): batch of rotation matrices.
    Returns:
        batch of quat (...x3 tensor).
    """
    matrix, batch_shape = flatten_batch_dims(R, end_dim=-3)
    num_rotations, D1, D2 = matrix.shape
    assert (D1, D2) == (3, 3), "Input should be a Bx3x3 tensor."

    # Adapted from SciPy:
    # https://github.com/scipy/scipy/blob/7cb3d751756907238996502b92709dc45e1c6596/scipy/spatial/transform/rotation.py#L480

    decision_matrix = torch.empty(
        (num_rotations, 4), dtype=matrix.dtype, device=matrix.device
    )
    decision_matrix[:, :3] = matrix.diagonal(dim1=1, dim2=2)
    decision_matrix[:, -1] = decision_matrix[:, :3].sum(axis=1)
    choices = decision_matrix.argmax(axis=1)

    quat = torch.empty((num_rotations, 4), dtype=matrix.dtype, device=matrix.device)

    ind = torch.nonzero(choices != 3, as_tuple=True)[0]
    i = choices[ind]
    j = (i + 1) % 3
    k = (j + 1) % 3

    quat[ind, i] = 1 - decision_matrix[ind, -1] + 2 * matrix[ind, i, i]
    quat[ind, j] = matrix[ind, j, i] + matrix[ind, i, j]
    quat[ind, k] = matrix[ind, k, i] + matrix[ind, i, k]
    quat[ind, 3] = matrix[ind, k, j] - matrix[ind, j, k]

    ind = torch.nonzero(choices == 3, as_tuple=True)[0]
    quat[ind, 0] = matrix[ind, 2, 1] - matrix[ind, 1, 2]
    quat[ind, 1] = matrix[ind, 0, 2] - matrix[ind, 2, 0]
    quat[ind, 2] = matrix[ind, 1, 0] - matrix[ind, 0, 1]
    quat[ind, 3] = 1 + decision_matrix[ind, -1]

    # xyzw
    quat = quat / torch.norm(quat, dim=1)[:, None]

    # quat to mrp
    mrp = quat[:, :3] / (1 + quat[:, 3].unsqueeze(1))
    return unflatten_batch_dims(mrp, batch_shape)


def mrp_to_rotmat(mrp):
    quat = mrp_to_quat(mrp, return_wxyz=False)
    return quat_to_rotmat(quat, is_xyzw=True)


def quat_to_rotmat(quat, is_xyzw=True):
    """
    Converts unit quaternion into rotation matrix representation.
    Args:
        quat (...x4 tensor, XYZW convention): batch of unit quaternions.
                No normalization is applied before computation.
    Returns:
        batch of rotation matrices (...x3x3 tensor).
    """
    if not is_xyzw:
        quat = quat_wxyz_to_xyzw(quat)

    # Adapted from SciPy:
    # https://github.com/scipy/scipy/blob/adc4f4f7bab120ccfab9383aba272954a0a12fb0/scipy/spatial/transform/rotation.py#L912
    x = quat[..., 0]
    y = quat[..., 1]
    z = quat[..., 2]
    w = quat[..., 3]

    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w

    xy = x * y
    zw = z * w
    xz = x * z
    yw = y * w
    yz = y * z
    xw = x * w

    matrix = torch.empty(quat.shape[:-1] + (3, 3), dtype=quat.dtype, device=quat.device)

    matrix[..., 0, 0] = x2 - y2 - z2 + w2
    matrix[..., 1, 0] = 2 * (xy + zw)
    matrix[..., 2, 0] = 2 * (xz - yw)

    matrix[..., 0, 1] = 2 * (xy - zw)
    matrix[..., 1, 1] = -x2 + y2 - z2 + w2
    matrix[..., 2, 1] = 2 * (yz + xw)

    matrix[..., 0, 2] = 2 * (xz + yw)
    matrix[..., 1, 2] = 2 * (yz - xw)
    matrix[..., 2, 2] = -x2 - y2 + z2 + w2
    return matrix


def mrp_to_quat(mrp, return_wxyz=False):
    """modified rodrigues parameters to quaternion

    Args:
        mrp (torch.Tensor): (...,3)

    Returns:
        torch.Tensor: (...,4) quat
    """

    if isinstance(mrp, list):
        mrp = torch.Tensor(mrp)
    elif isinstance(mrp, np.ndarray):
        mrp = torch.from_numpy(mrp)

    mrp_vec, batch_shape = flatten_batch_dims(mrp, end_dim=-2)
    num_rotations, D = mrp_vec.shape

    if D != 3:
        raise ValueError("Input tensor must be of size (3,) or (...,3) ")

    quat = torch.zeros(num_rotations, 4).to(device=mrp.device, dtype=mrp.dtype)

    # Unsqueeze at -2 to get vector multiplication in next step
    mrp_vec = mrp_vec.unsqueeze(-2)
    magsq = mrp_vec @ mrp_vec.transpose(-2, -1)
    qvec = (2 * mrp_vec) / (1 + magsq)
    qw = (1 - magsq) / (1 + magsq)

    quat[..., :3] = qvec.squeeze(1)
    quat[..., 3] = qw.squeeze()

    quat = quat if not return_wxyz else quat_xyzw_to_wxyz(quat)
    quat = unflatten_batch_dims(quat, batch_shape)
    return quat


def Rt_to_H(R, t):
    matrix, Rbatch_shape = flatten_batch_dims(R, end_dim=-3)
    num_rotations, D1, D2 = matrix.shape
    assert (D1, D2) == (3, 3), "Input should be a Bx3x3 tensor."

    vector, tbatch_shape = flatten_batch_dims(t, end_dim=-2)
    num_translations, D = vector.shape

    assert D == 3, "t should be a Bx1x3 tensor or Bx3x1 tensor."

    assert (
        num_rotations == num_translations
    ), "Found unequal number of rotations and translation vectors"

    H = torch.eye(4).repeat((num_rotations, 1, 1)).to(dtype=R.dtype, device=R.device)
    H[..., :3, :3] = matrix
    H[..., :3, 3] = vector

    H = unflatten_batch_dims(H, Rbatch_shape)
    return H


def H_to_Rt(H):
    rotmats = H[..., :3, :3]
    t = H[..., :3, 3]

    return rotmats, t


def H_to_qt(H, return_wxyz=False):
    rotmats = H[..., :3, :3]
    t = H[..., :3, 3]

    quat = rotmat_to_quat(rotmats, return_wxyz=return_wxyz)
    return (quat, t)


def qt_to_H(quat, t, is_xyzw=True):
    rotmats = quat_to_rotmat(quat, is_xyzw=is_xyzw)

    return Rt_to_H(rotmats, t)


def tmrp_to_H(tmrp):
    t = tmrp[..., :3]
    rotmats = mrp_to_rotmat(tmrp[..., 3:6])

    return Rt_to_H(rotmats, t)


def H_to_tmrp(H):
    rotmats, t = H_to_Rt(H)

    mrps = rotmat_to_mrp(rotmats)
    return torch.concatenate((t, mrps), -1)


def get_random_rotations_in_angle_limit(angle_limit, batch_size=1):
    """Get  random rotation matrices in angle limit
        Uses the axis-angle representation to generate random rotations
        First generate random (uniform) axis and rotate around this axis by a random (uniform) angle
    Args:
        angle_limit (float): Angle limit in degrees
        batch_size (int, optional): Batch size. Defaults to 1.

    Returns:
        torch.Tensor: [batch_size, 3, 3] Rotation matrices
    """
    random_axis = torch.rand((batch_size, 3))
    random_axis = random_axis / torch.linalg.norm(random_axis, dim=-1, keepdim=True)

    assert torch.norm(random_axis, dim=-1).allclose(
        torch.ones((batch_size,)), atol=1e-4
    ), "Unexpected. Found random axis was not normalized"

    random_angle = torch.rand((batch_size,)) * angle_limit
    random_angle = random_angle.unsqueeze(-1)

    q_vec = random_axis * (torch.sin(random_angle / 2))
    q = torch.concatenate((q_vec, torch.cos(random_angle / 2)), dim=-1)

    rotmats = quat_to_rotmat(q, is_xyzw=True)
    return rotmats
