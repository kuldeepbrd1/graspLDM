import numpy as np
import torch
import trimesh


class SimplePandaGripper:
    # Gripper : top is origin
    #
    #              ==========
    #             ||
    #             ||
    # TOP ========||       =| BOTTOM
    #       CENTER||
    #             ||
    #              ==========

    ## Key points
    TOP = [0, 0, 0]
    CENTER = [0, 0, 0.0659999996]
    CENTER_RIGHT = [-4.100000e-02, 0, 6.59999996e-02]
    CENTER_LEFT = [4.100000e-02, 0, 6.59999996e-02]
    BOTTOM_RIGHT = [-4.100000e-02, 0, 1.12169998e-01]
    BOTTOM_LEFT = [4.100000e-02, 0, 1.12169998e-01]
    BOTTOM_CENTER = [0, 0, 1.12169998e-01]

    ## Segments: Open gripper
    CFL_SEGMENT = [CENTER_LEFT, BOTTOM_LEFT]
    CFR_SEGMENT = [CENTER_RIGHT, BOTTOM_RIGHT]
    CB1_SEGMENT = [TOP, CENTER]
    CB2_SEGMENT = [CENTER_RIGHT, CENTER_LEFT]
    OPEN_SEGMENTS = (CFL_SEGMENT, CFR_SEGMENT, CB1_SEGMENT, CB2_SEGMENT)

    # Segments: Finger sweep for Collision
    CFC1_SEGMENT = [
        [4.10000000e-02, -7.27595772e-12, 1.08169998e-01],
        [-4.100000e-02, -7.27595772e-12, 1.08169998e-01],
    ]

    CFC2_SEGMENT = [
        [4.10000000e-02, -7.27595772e-12, 0.98169998e-01],
        [-4.100000e-02, -7.27595772e-12, 0.98169998e-01],
    ]
    COLLISION_SEGMENTS = (
        # *OPEN_SEGMENTS,
        CFC1_SEGMENT,
        CFC2_SEGMENT,
    )

    def create_gripper_marker(color=[0, 0, 255], tube_radius=0.002, sections=6):
        """Create a 3D mesh visualizing a parallel yaw gripper. It consists of four cylinders.

        From: https://github.com/NVlabs/acronym/blob/main/acronym_tools/acronym.py

        Args:
            color (list, optional): RGB values of marker. Defaults to [0, 0, 255].
            tube_radius (float, optional): Radius of cylinders. Defaults to 0.001.
            sections (int, optional): Number of sections of each cylinder. Defaults to 6.
            collision_marker (bool, optional) : Whether to include collision marker (cylinder) between finger tips
                                                For e.g. Used to check if something is between fingers.
        Returns:
            trimesh.Trimesh: A mesh that represents a simple parallel yaw gripper.
        """

        gripper_segments = SimplePandaGripper.OPEN_SEGMENTS

        gripper_markers = [
            trimesh.creation.cylinder(
                radius=tube_radius,
                sections=sections,
                segment=segment,
            )
            for segment in gripper_segments
        ]

        tmp = trimesh.util.concatenate(gripper_markers)
        tmp.visual.face_colors = color

        return tmp

    def create_grasp_collision_marker(
        tube_radius=0.006, sections=6, subdivisions=None, color=[0, 255, 0]
    ):
        """For checking collisions in the grasp area

        Args:
            tube_radius (float, optional): _description_. Defaults to 0.002.
            sections (int, optional): _description_. Defaults to 3.
            subdivisions (int, optional): _description_. Defaults to 4.
            color (list, optional): _description_. Defaults to [0, 255, 0].

        Returns:
            _type_: _description_
        """
        gripper_segments = (
            SimplePandaGripper.COLLISION_SEGMENTS  # + SimplePandaGripper.OPEN_SEGMENTS
        )

        mesh = SimplePandaGripper.create_subdivisions(
            gripper_segments, tube_radius, sections, subdivisions
        )
        mesh.visual.face_colors = color

        return mesh

    def create_grasp_body_marker(
        tube_radius=0.006, sections=6, subdivisions=None, color=[0, 255, 0]
    ):
        """For checking collisions in the grasp area

        Args:
            tube_radius (float, optional): _description_. Defaults to 0.002.
            sections (int, optional): _description_. Defaults to 3.
            subdivisions (int, optional): _description_. Defaults to 4.
            color (list, optional): _description_. Defaults to [0, 255, 0].

        Returns:
            _type_: _description_
        """
        gripper_segments = (
            SimplePandaGripper.OPEN_SEGMENTS  # + SimplePandaGripper.OPEN_SEGMENTS
        )

        mesh = SimplePandaGripper.create_subdivisions(
            gripper_segments, tube_radius, sections, subdivisions
        )
        mesh.visual.face_colors = color

        return mesh

    def create_subdivisions(gripper_segments, tube_radius, sections, subdivisions):
        gripper_markers = []
        is_subdivided = True if subdivisions is not None else False
        for segment in gripper_segments:
            gripper_marker = trimesh.creation.cylinder(
                radius=tube_radius,
                sections=sections,
                segment=segment,
            )
            if is_subdivided:
                for _ in range(subdivisions):
                    vertices, faces = trimesh.remesh.subdivide(
                        gripper_marker.vertices, gripper_marker.faces
                    )
                    gripper_marker = trimesh.Trimesh(vertices=vertices, faces=faces)

            gripper_markers.append(gripper_marker)

        return trimesh.util.concatenate(gripper_markers)

    def subdivide_segment_points(segments, subdivisions=2):
        """Hacky subdivision to get more points for a simple gripper marker"""
        subdivided_segments = []
        for segment in segments:
            p1 = np.array(segment[0])
            p2 = np.array(segment[1])
            direction = p2 - p1
            n = subdivisions**2

            step_size = direction / n

            points = (
                [tuple((p1 + i * step_size).tolist()) for i in range(n + 1)]
                + [p1]
                + [p2]
            )

            subdivided_segments.append(points)
        return subdivided_segments

    def transform_to_fingertip_frame(grasps):
        """
        Transform grasps to fingertip frame

        Args:
            grasps (torch.Tensor or np.ndarray): (N, 4, 4) grasp poses
                        H = [R | t]
                            [0 | 1]
        """
        assert isinstance(grasps, torch.Tensor) or isinstance(
            grasps, np.ndarray
        ), "Expected torch.Tensor or np.ndarray"

        transform = (
            np.eye(4)
            if isinstance(grasps, np.ndarray)
            else torch.eye(4, dtype=grasps.dtype, device=grasps.device)
        )

        transform[..., :3, 3] += (
            np.array(SimplePandaGripper.BOTTOM_CENTER)
            if isinstance(grasps, np.ndarray)
            else torch.tensor(
                SimplePandaGripper.BOTTOM_CENTER,
                dtype=grasps.dtype,
                device=grasps.device,
            )
        )
        ## Uncomment for visualization
        # old_grasps = [
        #     SimplePandaGripper.create_gripper_marker(color=[255, 0, 0])
        #     .copy()
        #     .apply_transform(grasp_pose)
        #     for grasp_pose in grasps[0].detach().clone().cpu().numpy()
        # ]

        grasps = grasps @ transform

        # new_grasps = [
        #     SimplePandaGripper.create_gripper_marker(color=[0, 255, 0])
        #     .copy()
        #     .apply_transform(grasp_pose)
        #     for grasp_pose in grasps[0].detach().clone().cpu().numpy()
        # ]
        # import trimesh

        # trimesh.Scene([*old_grasps, *new_grasps]).show()

        return grasps

    def transform_to_gripper_wrist(self, grasps):
        """
        Transform grasps to wrist/center frame

        Args:
            grasps (torch.Tensor or np.ndarray): (N, 4, 4) grasp poses
                        H = [R | t]
                            [0 | 1]
        """
        assert isinstance(grasps, torch.Tensor) or isinstance(
            grasps, np.ndarray
        ), "Expected torch.Tensor or np.ndarray"

        transform = (
            np.eye(4)
            if isinstance(grasps, np.ndarray)
            else torch.eye(4, dtype=grasps.dtype, device=grasps.device)
        )

        transform[..., :3, 3] += (
            np.array(SimplePandaGripper.CENTER)
            if isinstance(grasps, np.ndarray)
            else torch.tensor(
                SimplePandaGripper.CENTER,
                dtype=grasps.dtype,
                device=grasps.device,
            )
        )

        grasps = grasps @ transform

        return grasps
