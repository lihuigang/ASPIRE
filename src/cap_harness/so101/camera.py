"""Camera calibration and synthetic tabletop depth for the SO-101 runtime."""

from __future__ import annotations

import numpy as np

from cap_harness.so101.perception import green_mask, red_mask

#: Object mid-planes used to turn a color mask into a metric point cloud.  The
#: runtime publishes no depth buffer, so a mask pixel is intersected with the
#: plane a localization query is about: the cube's mid-height and the pad's top.
CUBE_PLANE_Z_M = 0.016
PAD_PLANE_Z_M = 0.008

_PINHOLE_FLIP = np.diag([1.0, -1.0, -1.0])


def pinhole_rotation(camera_rotation_mujoco: np.ndarray) -> np.ndarray:
    """Convert a MuJoCo camera frame (x right, y up, z back) to pinhole axes."""
    return np.asarray(camera_rotation_mujoco, dtype=np.float64).reshape(3, 3) @ _PINHOLE_FLIP


def synthetic_depth(
    rgb: np.ndarray,
    intrinsics: np.ndarray,
    camera_position: np.ndarray,
    camera_rotation_mujoco: np.ndarray,
    *,
    cube_plane_z_m: float = CUBE_PLANE_Z_M,
    pad_plane_z_m: float = PAD_PLANE_Z_M,
) -> np.ndarray:
    """Intersect every pixel with a tabletop plane selected by image color."""
    image = np.asarray(rgb)
    height, width = image.shape[:2]
    plane_z = np.zeros((height, width), dtype=np.float64)
    red = red_mask(image)
    green = green_mask(image) & ~red
    plane_z[red] = cube_plane_z_m
    plane_z[green] = pad_plane_z_m
    # A perfectly planar mask has zero thickness; a sub-millimeter ripple keeps
    # the PCA orientation of a localized object well conditioned.
    rows, columns = np.nonzero(red | green)
    if rows.size:
        plane_z[rows, columns] += 0.0006 * np.sin(columns * 0.7 + rows * 1.3)

    rotation = pinhole_rotation(camera_rotation_mujoco)
    focal_x, focal_y = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    center_x, center_y = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    columns, rows = np.meshgrid(np.arange(width), np.arange(height))
    normalized = np.stack(
        [
            (columns - center_x) / focal_x,
            (rows - center_y) / focal_y,
            np.ones_like(columns, dtype=np.float64),
        ],
        axis=-1,
    )
    directions = normalized @ rotation.T
    origin_z = float(camera_position[2])
    forward = directions[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = (plane_z - origin_z) / forward
    depth = np.where(np.isfinite(depth) & (depth > 0.0), depth, 0.0)
    return np.ascontiguousarray(depth)


def camera_transform(
    camera_position: np.ndarray,
    camera_rotation_mujoco: np.ndarray,
) -> np.ndarray:
    """Homogeneous transform from pinhole camera coords to the robot base frame."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = pinhole_rotation(camera_rotation_mujoco)
    transform[:3, 3] = np.asarray(camera_position, dtype=np.float64)
    return transform


__all__ = [
    "CUBE_PLANE_Z_M",
    "PAD_PLANE_Z_M",
    "camera_transform",
    "pinhole_rotation",
    "synthetic_depth",
]
