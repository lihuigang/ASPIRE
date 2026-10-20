"""SO-101 simulation embodiment: runtime client, kinematics, and adapter."""

from .adapter import So101Adapter
from .camera import camera_transform, synthetic_depth
from .client import DEFAULT_SO101_RUNTIME_URL, So101RuntimeClient, So101RuntimeError
from .ik_provider import So101IKProvider
from .kinematics import (
    HOME_JOINT_POSITIONS,
    JOINT_NAMES,
    So101Kinematics,
    top_down_quaternion,
)
from .perception import So101ColorSegmentationProvider, green_mask, red_mask
from .registry import So101TaskMetadata, So101TaskRegistry

__all__ = [
    "DEFAULT_SO101_RUNTIME_URL",
    "HOME_JOINT_POSITIONS",
    "JOINT_NAMES",
    "So101Adapter",
    "So101ColorSegmentationProvider",
    "So101IKProvider",
    "So101Kinematics",
    "So101RuntimeClient",
    "So101RuntimeError",
    "So101TaskMetadata",
    "So101TaskRegistry",
    "camera_transform",
    "green_mask",
    "red_mask",
    "synthetic_depth",
    "top_down_quaternion",
]
