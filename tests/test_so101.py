"""Offline checks for the SO-101 kinematics, color perception and task registry."""

import numpy as np
import pytest

from cap_harness.contracts import CameraObservation, Pose
from cap_harness.so101.camera import synthetic_depth
from cap_harness.so101.kinematics import (
    HOME_JOINT_POSITIONS,
    So101Kinematics,
    top_down_quaternion,
    top_down_rotation,
)
from cap_harness.so101.perception import (
    So101ColorSegmentationProvider,
    green_mask,
    red_mask,
)
from cap_harness.so101.registry import So101RegistryError, So101TaskRegistry

pytest.importorskip("mujoco")


@pytest.fixture(scope="module")
def kinematics():
    return So101Kinematics()


def test_top_down_rotation_points_tool_axis_down():
    rotation = top_down_rotation(0.0)
    assert np.allclose(rotation[:, 0], (0.0, 0.0, -1.0))
    assert np.allclose(rotation.T @ rotation, np.eye(3))
    assert top_down_quaternion(0.0) == pytest.approx((0.70710678, 0.0, 0.70710678, 0.0), abs=1e-6)


def test_forward_kinematics_matches_runtime_home(kinematics):
    position, quaternion = kinematics.forward(HOME_JOINT_POSITIONS)
    assert position == pytest.approx((0.207334, -0.000979, 0.116334), abs=1e-4)
    assert quaternion == pytest.approx((0.780476, 0.019000, 0.624712, -0.015208), abs=1e-5)


def test_ik_round_trips_a_top_down_grasp(kinematics):
    target = np.array([0.20, -0.10, 0.05])
    solution = kinematics.solve(
        target, top_down_quaternion(0.0), seed_joints=HOME_JOINT_POSITIONS
    )
    assert solution is not None
    joints, position_error, orientation_error = solution
    assert position_error < 1e-3
    assert orientation_error < 1e-2
    position, _ = kinematics.forward(joints)
    assert position == pytest.approx(target, abs=1e-3)


def test_camera_intrinsics_follow_the_runtime_aspect_ratio(kinematics):
    top = kinematics.camera_intrinsics("top_cam", 320, 320)
    assert top[0, 0] == pytest.approx(328.0486, abs=1e-3)
    assert top[1, 1] == pytest.approx(328.0486, abs=1e-3)
    assert top[:2, 2] == pytest.approx((160.0, 160.0))
    wrist = kinematics.camera_intrinsics("wrist_cam", 320, 320)
    assert wrist[0, 0] == pytest.approx(200.0, abs=1e-3)
    assert wrist[1, 1] == pytest.approx(355.5556, abs=1e-3)


def _scaled_rgb(width=320, height=320):
    rgb = np.full((height, width, 3), 140, dtype=np.uint8)
    rgb[100:140, 80:120] = (210, 40, 35)
    rgb[180:220, 160:200] = (30, 200, 60)
    return rgb


def test_color_masks_and_segmentation_select_the_objects():
    rgb = _scaled_rgb()
    red, green = red_mask(rgb), green_mask(rgb)
    assert np.count_nonzero(red) == 40 * 40
    assert np.count_nonzero(green) == 40 * 40
    assert not np.any(red & green)

    observation = CameraObservation(
        rgb=rgb,
        depth_m=np.zeros(rgb.shape[:2]),
        intrinsics=np.array([[300.0, 0.0, 160.0], [0.0, 300.0, 160.0], [0.0, 0.0, 1.0]]),
        frame="camera/top_cam",
        camera_pose=Pose(position=(0.0, 0.0, 1.0), quaternion_wxyz=(1.0, 0.0, 0.0, 0.0), frame="robot_base"),
    )
    provider = So101ColorSegmentationProvider()
    cube = provider.segment_text(observation, "red cube")
    assert cube.ok
    assert cube.segmentations[0].label == "red cube"
    assert cube.segmentations[0].camera_name == "top_cam"
    assert cube.segmentations[0].box_xyxy == pytest.approx((80.0, 100.0, 120.0, 140.0))
    pad = provider.segment_text(observation, "green target pad")
    assert pad.ok
    assert pad.segmentations[0].label == "green target pad"
    assert not provider.segment_text(observation, "blue sphere").ok


def test_synthetic_depth_puts_cube_pixels_above_the_table(kinematics):
    rgb = _scaled_rgb()
    table_only = np.full_like(rgb, 140)
    position, rotation = kinematics.camera_pose("top_cam", HOME_JOINT_POSITIONS)
    intrinsics = kinematics.camera_intrinsics("top_cam", 320, 320)
    depth = synthetic_depth(rgb, intrinsics, position, rotation)
    table_depth = synthetic_depth(table_only, intrinsics, position, rotation)
    red = red_mask(rgb)
    assert depth.shape == rgb.shape[:2]
    assert np.all(depth[red] > 0.0)
    assert np.all(table_depth[red] > 0.0)
    assert depth[red].mean() < table_depth[red].mean() - 0.01


def test_task_registry_resolves_the_benchmark_alias():
    registry = So101TaskRegistry()
    task = registry.resolve("so101")
    assert task.task_name == "pick_cube"
    assert task.task_id == 0
    assert registry.resolve(("pick_cube", 0)) is task
    with pytest.raises(So101RegistryError):
        registry.resolve("pick_cube", task_id=1)
    with pytest.raises(So101RegistryError):
        registry.resolve("unknown")
