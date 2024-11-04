import numpy as np
import torch
import trimesh


# TODO: To be removed in favor of gripper.py
def create_gripper_marker(color=[0, 0, 255], tube_radius=0.001, sections=6):
    """Create a 3D mesh visualizing a parallel yaw gripper. It consists of four cylinders.

    From: https://github.com/NVlabs/acronym/blob/main/acronym_tools/acronym.py

    Args:
        color (list, optional): RGB values of marker. Defaults to [0, 0, 255].
        tube_radius (float, optional): Radius of cylinders. Defaults to 0.001.
        sections (int, optional): Number of sections of each cylinder. Defaults to 6.

    Returns:
        trimesh.Trimesh: A mesh that represents a simple parallel yaw gripper.
    """
    cfl = trimesh.creation.cylinder(
        radius=0.002,
        sections=sections,
        segment=[
            [4.10000000e-02, -7.27595772e-12, 6.59999996e-02],
            [4.10000000e-02, -7.27595772e-12, 1.12169998e-01],
        ],
    )
    cfr = trimesh.creation.cylinder(
        radius=0.002,
        sections=sections,
        segment=[
            [-4.100000e-02, -7.27595772e-12, 6.59999996e-02],
            [-4.100000e-02, -7.27595772e-12, 1.12169998e-01],
        ],
    )
    cb1 = trimesh.creation.cylinder(
        radius=0.002, sections=sections, segment=[[0, 0, 0], [0, 0, 6.59999996e-02]]
    )
    cb2 = trimesh.creation.cylinder(
        radius=0.002,
        sections=sections,
        segment=[[-4.100000e-02, 0, 6.59999996e-02], [4.100000e-02, 0, 6.59999996e-02]],
    )

    tmp = trimesh.util.concatenate([cb1, cb2, cfr, cfl])
    tmp.visual.face_colors = color

    return tmp


def visualize_pc(pc):
    if isinstance(pc, torch.Tensor):
        pc = pc.squeeze().numpy()
    r = pc[..., 0] * 255 / max(pc[..., 0])
    g = pc[..., 1] * 200 / max(pc[..., 1])
    b = pc[..., 2] * 175 / max(pc[..., 2])
    a = np.ones(pc.shape[0]) * 200

    colors = np.clip(np.vstack((r, g, b, a)).T, 0, 255)

    colors = colors if colors is not None else np.ones((pc.shape[0], 3)) * 85
    pc_trimesh = trimesh.points.PointCloud(pc, colors=colors)
    scene = trimesh.Scene(pc_trimesh).show(line_settings={"point_size": 5})
    return scene


def visualize_pc_grasps(
    pc: np.ndarray, grasps: np.ndarray, c: np.ndarray = None
) -> trimesh.Scene:
    # scene = visualize_pc(pc)
    r = pc[..., 0] * 255 / max(pc[..., 0])
    g = pc[..., 1] * 200 / max(pc[..., 1])
    b = pc[..., 2] * 175 / max(pc[..., 2])
    a = np.ones(pc.shape[0]) * 200

    pc_colors = np.clip(np.vstack((r, g, b, a)).T, 0, 255)

    if c is not None:
        c = c.squeeze(1) if c.ndim == 2 else c

    if c is not None:
        gripper_marker = [
            create_gripper_marker(
                color=[150, np.clip(255 * ci, 0, 255), 0, np.clip(255 * ci, 150, 255)]
            )
            for ci in c
        ]
    else:
        gripper_marker = [create_gripper_marker(color=[0, 255, 0, 255])] * grasps.shape[
            0
        ]

    gripper_markers = [
        gripper_marker[i].copy().apply_transform(t) for i, t in enumerate(grasps)
    ]

    scene = trimesh.Scene(
        [trimesh.points.PointCloud(pc, colors=pc_colors)] + gripper_markers
    )
    return scene
