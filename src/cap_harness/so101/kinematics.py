"""SO-101 forward/inverse kinematics from the runtime's own MJCF.

The simulation runtime exposes joint targets and measured joint state, not an
IK endpoint.  This module compiles the same kinematic tree the runtime simulates
(meshes stripped, cameras kept) so a Cartesian pose can be resolved locally and
sent as a joint command.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
HOME_JOINT_POSITIONS = np.array([0.0, -0.565, 0.284, 1.631, 0.0], dtype=np.float64)
GRIPPER_CLOSED_RAD = -0.17453
GRIPPER_OPEN_RAD = 1.74533
GRIPPER_SITE = "gripperframe"
BASE_FRAME = "robot_base"
CAMERA_NAMES = ("top_cam", "wrist_cam")

ASSET_PATH = Path(__file__).resolve().parent / "assets" / "so101_kinematics.xml"

#: The site frame has X along the tool approach axis and Z across the jaws.
#: A top-down grasp therefore points site X along world -Z: R = Rz(yaw) @ Ry(90).
_TOPDOWN_Y = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64)


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def quaternion_conjugate(quaternion: np.ndarray) -> np.ndarray:
    return np.array(
        [quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]],
        dtype=np.float64,
    )


def quaternion_error(current: np.ndarray, desired: np.ndarray) -> np.ndarray:
    """World-frame rotation vector taking ``current`` to ``desired``."""
    error = quaternion_multiply(desired, quaternion_conjugate(current))
    if error[0] < 0.0:
        error = -error
    vector = error[1:]
    sine = float(np.linalg.norm(vector))
    if sine < 1e-9:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * float(np.arctan2(sine, np.clip(error[0], -1.0, 1.0)))
    return vector * (angle / sine)


def top_down_rotation(yaw_deg: float = 0.0) -> np.ndarray:
    """Rotation whose site X axis points down, yawed about world Z."""
    yaw = np.deg2rad(float(yaw_deg))
    rz = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return rz @ _TOPDOWN_Y


def matrix_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Convert an orthonormal rotation matrix to a canonical WXYZ quaternion."""
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
        quaternion = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
            ]
        )
    elif matrix[1, 1] > matrix[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
        quaternion = np.array(
            [
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
            ]
        )
    else:
        scale = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
        quaternion = np.array(
            [
                (matrix[1, 0] - matrix[0, 1]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
            ]
        )
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return quaternion


def top_down_quaternion(yaw_deg: float = 0.0) -> np.ndarray:
    return matrix_to_quaternion_wxyz(top_down_rotation(yaw_deg))


class So101Kinematics:
    """Compiled SO-101 model with deterministic damped-least-squares IK."""

    def __init__(self, model_path: Path | str | None = None) -> None:
        try:
            import mujoco
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "MuJoCo is not installed; install the harness with the 'so101' extra"
            ) from exc
        self._mujoco = mujoco
        path = Path(model_path) if model_path is not None else ASSET_PATH
        if not path.is_file():
            raise RuntimeError(f"SO-101 kinematic model is missing: {path}")
        self._model = mujoco.MjModel.from_xml_path(str(path))
        self._data = mujoco.MjData(self._model)
        self._site_id = int(mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, GRIPPER_SITE))
        self._joint_ids = np.array(
            [
                int(mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name))
                for name in JOINT_NAMES
            ]
        )
        self._qpos_addresses = np.asarray(
            [int(self._model.jnt_qposadr[joint]) for joint in self._joint_ids]
        )
        self._dof_addresses = np.asarray(
            [int(self._model.jnt_dofadr[joint]) for joint in self._joint_ids]
        )
        self._gripper_qpos = int(
            self._model.jnt_qposadr[
                int(mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "gripper"))
            ]
        )
        ranges = np.asarray([self._model.jnt_range[joint] for joint in self._joint_ids])
        self.joint_lower = ranges[:, 0].copy()
        self.joint_upper = ranges[:, 1].copy()
        self._camera_ids = {
            name: int(mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, name))
            for name in CAMERA_NAMES
        }
        self._gripper_open()

    def _gripper_open(self) -> None:
        self._data.qpos[self._gripper_qpos] = GRIPPER_OPEN_RAD

    def _load(self, data: object, joints: np.ndarray, gripper: float | None) -> None:
        values = np.asarray(joints, dtype=np.float64)
        if values.shape != (len(JOINT_NAMES),) or not bool(np.all(np.isfinite(values))):
            raise ValueError("joints must be a finite 5-joint vector")
        data.qpos[self._qpos_addresses] = values
        data.qpos[self._gripper_qpos] = (
            GRIPPER_OPEN_RAD if gripper is None else float(gripper)
        )
        self._mujoco.mj_forward(self._model, data)

    def forward(
        self, joints: np.ndarray, gripper: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the gripper-site position and WXYZ quaternion for ``joints``."""
        self._load(self._data, joints, gripper)
        position = np.array(self._data.site_xpos[self._site_id], copy=True)
        quaternion = np.empty(4, dtype=np.float64)
        self._mujoco.mju_mat2Quat(
            quaternion, self._data.site_xmat[self._site_id].ravel()
        )
        return position, quaternion

    def solve(
        self,
        target_position: object,
        target_quaternion_wxyz: object,
        *,
        seed_joints: object | None = None,
        max_iterations: int = 400,
        position_tolerance_m: float = 1e-3,
        orientation_tolerance_rad: float = 1e-2,
        max_step_rad: float = 0.08,
    ) -> tuple[np.ndarray, float, float] | None:
        """Solve site pose IK, returning joints plus residual position/orientation."""
        target = np.asarray(target_position, dtype=np.float64)
        quaternion = np.asarray(target_quaternion_wxyz, dtype=np.float64)
        if target.shape != (3,) or not bool(np.all(np.isfinite(target))):
            raise ValueError("target_position must be a finite 3-vector")
        if quaternion.shape != (4,) or not bool(np.all(np.isfinite(quaternion))):
            raise ValueError("target_quaternion_wxyz must be a finite 4-vector")
        quaternion = quaternion / np.linalg.norm(quaternion)
        current = (
            np.asarray(seed_joints, dtype=np.float64).copy()
            if seed_joints is not None
            else self._current_joints()
        )
        if current.shape != (len(JOINT_NAMES),) or not bool(np.all(np.isfinite(current))):
            raise ValueError("seed_joints must be a finite 5-joint vector")

        seeds = (current, HOME_JOINT_POSITIONS.astype(np.float64))
        best: tuple[np.ndarray, float, float] | None = None
        for index, seed in enumerate(seeds):
            solution = self._solve_from(
                target,
                quaternion,
                seed,
                max_iterations=max_iterations if index == 0 else max_iterations // 2,
                position_tolerance_m=position_tolerance_m,
                orientation_tolerance_rad=orientation_tolerance_rad,
                max_step_rad=max_step_rad,
            )
            if solution is None:
                continue
            if best is None or solution[1] + solution[2] < best[1] + best[2]:
                best = solution
            if solution[1] <= position_tolerance_m and solution[2] <= orientation_tolerance_rad:
                return solution
        return None

    def _current_joints(self) -> np.ndarray:
        return np.array(self._data.qpos[self._qpos_addresses], copy=True)

    def _solve_from(
        self,
        target: np.ndarray,
        target_quaternion: np.ndarray,
        seed: np.ndarray,
        *,
        max_iterations: int,
        position_tolerance_m: float,
        orientation_tolerance_rad: float,
        max_step_rad: float,
    ) -> tuple[np.ndarray, float, float] | None:
        mujoco = self._mujoco
        model = self._model
        trial = mujoco.MjData(model)
        trial.qpos[self._gripper_qpos] = GRIPPER_OPEN_RAD
        joints = np.clip(seed, self.joint_lower, self.joint_upper)
        jacobian_position = np.zeros((3, model.nv))
        jacobian_rotation = np.zeros((3, model.nv))
        current_quaternion = np.empty(4, dtype=np.float64)
        best: tuple[np.ndarray, float, float] | None = None
        for _ in range(max(1, max_iterations)):
            trial.qpos[self._qpos_addresses] = joints
            mujoco.mj_forward(model, trial)
            position_error = target - trial.site_xpos[self._site_id]
            rotation = trial.site_xmat[self._site_id].reshape(3, 3)
            mujoco.mju_mat2Quat(current_quaternion, rotation.ravel())
            orientation_error = quaternion_error(current_quaternion, target_quaternion)
            position_norm = float(np.linalg.norm(position_error))
            orientation_norm = float(np.linalg.norm(orientation_error))
            if best is None or position_norm + orientation_norm < best[1] + best[2]:
                best = (joints.copy(), position_norm, orientation_norm)
            if (
                position_norm < position_tolerance_m
                and orientation_norm < orientation_tolerance_rad
            ):
                return joints.copy(), position_norm, orientation_norm
            mujoco.mj_jacSite(model, trial, jacobian_position, jacobian_rotation, self._site_id)
            jacobian = np.vstack(
                (
                    jacobian_position[:, self._dof_addresses],
                    0.5 * jacobian_rotation[:, self._dof_addresses],
                )
            )
            error = np.concatenate((position_error, 0.5 * orientation_error))
            normal = jacobian.T @ jacobian + 1e-4 * np.eye(len(JOINT_NAMES))
            try:
                delta = np.linalg.solve(normal, jacobian.T @ error)
            except np.linalg.LinAlgError:
                break
            joints = np.clip(
                joints + np.clip(delta, -max_step_rad, max_step_rad),
                self.joint_lower,
                self.joint_upper,
            )
        return best

    def camera_pose(
        self, name: str, joints: np.ndarray, gripper: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return a camera's MuJoCo-frame position and rotation for ``joints``."""
        if name not in self._camera_ids:
            raise ValueError(f"unknown SO-101 camera {name!r}")
        self._load(self._data, joints, gripper)
        camera_id = self._camera_ids[name]
        position = np.array(self._data.cam_xpos[camera_id], copy=True)
        rotation = np.array(self._data.cam_xmat[camera_id], copy=True).reshape(3, 3)
        return position, rotation

    def camera_intrinsics(self, name: str, width: int, height: int) -> np.ndarray:
        """Pinhole intrinsics for a runtime-rendered camera at ``width`` x ``height``."""
        if name not in self._camera_ids:
            raise ValueError(f"unknown SO-101 camera {name!r}")
        camera_id = self._camera_ids[name]
        fovy = float(self._model.cam_fovy[camera_id])
        if not np.isfinite(fovy) or fovy <= 0.0:
            raise ValueError(f"camera {name!r} has no vertical field of view")
        tan_y = np.tan(np.deg2rad(fovy) / 2.0)
        sensor = np.asarray(self._model.cam_sensorsize[camera_id], dtype=np.float64)
        if sensor[0] > 0.0 and sensor[1] > 0.0:
            tan_x = tan_y * (sensor[0] / sensor[1])
        else:
            tan_x = tan_y * (width / height)
        return np.array(
            [
                [0.5 * width / tan_x, 0.0, 0.5 * width],
                [0.0, 0.5 * height / tan_y, 0.5 * height],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


__all__ = [
    "ASSET_PATH",
    "BASE_FRAME",
    "CAMERA_NAMES",
    "GRIPPER_CLOSED_RAD",
    "GRIPPER_OPEN_RAD",
    "GRIPPER_SITE",
    "HOME_JOINT_POSITIONS",
    "JOINT_NAMES",
    "So101Kinematics",
    "matrix_to_quaternion_wxyz",
    "quaternion_error",
    "top_down_quaternion",
    "top_down_rotation",
]
