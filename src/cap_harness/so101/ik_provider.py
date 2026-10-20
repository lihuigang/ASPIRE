"""Local MuJoCo IK provider for the SO-101 embodiment."""

from __future__ import annotations

import numpy as np

from cap_harness.contracts import IKResult, Pose, RobotState
from cap_harness.errors import ApiError, ErrorCode
from cap_harness.so101.kinematics import So101Kinematics


class So101IKProvider:
    """Resolve site poses with the same kinematic tree the runtime simulates."""

    def __init__(self, kinematics: So101Kinematics) -> None:
        if not isinstance(kinematics, So101Kinematics):
            raise ValueError("kinematics must be a So101Kinematics instance")
        self.kinematics = kinematics

    def solve_ik(
        self,
        target_pose: Pose,
        robot_state: RobotState,
        *,
        arm: str = "primary",
    ) -> IKResult:
        if not isinstance(target_pose, Pose):
            return IKResult(
                ok=False,
                arm=arm,
                embodiment="so101",
                error=ApiError(ErrorCode.INVALID_REQUEST, "target_pose must be a Pose"),
            )
        if not isinstance(robot_state, RobotState):
            return IKResult(
                ok=False,
                arm=arm,
                embodiment="so101",
                error=ApiError(ErrorCode.INVALID_REQUEST, "robot_state must be a RobotState"),
            )
        if arm not in robot_state.joint_positions:
            return IKResult(
                ok=False,
                arm=arm,
                embodiment="so101",
                error=ApiError(ErrorCode.NOT_FOUND, f"robot state has no arm {arm!r}"),
            )
        if target_pose.frame != robot_state.base_frame:
            return IKResult(
                ok=False,
                arm=arm,
                embodiment="so101",
                error=ApiError(
                    ErrorCode.INVALID_REQUEST,
                    f"target pose is in frame {target_pose.frame!r}, "
                    f"expected {robot_state.base_frame!r}",
                ),
            )
        try:
            solution = self.kinematics.solve(
                target_pose.position,
                target_pose.quaternion_wxyz,
                seed_joints=robot_state.joint_positions[arm],
            )
        except (TypeError, ValueError) as exc:
            return IKResult(
                ok=False,
                arm=arm,
                embodiment="so101",
                error=ApiError(ErrorCode.INVALID_REQUEST, f"SO-101 IK rejected the request: {exc}"),
            )
        except Exception as exc:
            return IKResult(
                ok=False,
                arm=arm,
                embodiment="so101",
                error=ApiError(
                    ErrorCode.IK_FAILED,
                    f"SO-101 IK failed: {exc}",
                    details={"exception_type": type(exc).__name__},
                ),
            )
        if solution is None:
            return IKResult(
                ok=False,
                arm=arm,
                embodiment="so101",
                error=ApiError(
                    ErrorCode.IK_UNREACHABLE,
                    "no collision-free SO-101 joint solution reaches the target pose",
                ),
            )
        joints, position_error, orientation_error = solution
        position, quaternion = self.kinematics.forward(joints)
        verified_position = float(np.linalg.norm(position - target_pose.position))
        dot = float(
            np.clip(abs(np.dot(quaternion, target_pose.quaternion_wxyz)), 0.0, 1.0)
        )
        verified_orientation = float(2.0 * np.arccos(dot))
        return IKResult(
            ok=True,
            arm=arm,
            embodiment="so101",
            joint_positions=joints,
            diagnostics={
                "solver": "mujoco-dls",
                "solution_position_error_m": position_error,
                "solution_orientation_error_rad": orientation_error,
                "verified_position_error_m": verified_position,
                "verified_orientation_error_rad": verified_orientation,
            },
        )


__all__ = ["So101IKProvider"]
