"""Environment adapter driving the remote SO-101 MuJoCo simulation."""

from __future__ import annotations

from io import BytesIO
import os
from typing import Any

import numpy as np

from cap_harness.contracts import (
    ArmCommand,
    CameraObservation,
    ExecutionResult,
    Observation,
    Pose,
    RobotAction,
    RobotPlanningContext,
    RobotState,
    StepResult,
    SynchronizedTrajectory,
    TaskContext,
    Trajectory,
)
from cap_harness.errors import ApiError, ErrorCode
from cap_harness.geometry import matrix_to_pose
from cap_harness.so101.camera import camera_transform, synthetic_depth
from cap_harness.so101.client import So101RuntimeClient, So101RuntimeError
from cap_harness.so101.kinematics import (
    BASE_FRAME,
    GRIPPER_CLOSED_RAD,
    GRIPPER_OPEN_RAD,
    HOME_JOINT_POSITIONS,
    JOINT_NAMES,
    So101Kinematics,
)
from cap_harness.so101.registry import So101TaskMetadata, So101TaskRegistry

TRAJECTORY_START_TOLERANCE_RAD = 0.02
TRAJECTORY_SETTLE_STEPS = 5
TRAJECTORY_SETTLE_TOLERANCE_RAD = 0.01
GRIPPER_RAMP_STEPS = 12
GRIPPER_HOLD_STEPS = 2


def _gripper_radians(normalized: float) -> float:
    return GRIPPER_CLOSED_RAD + float(normalized) * (GRIPPER_OPEN_RAD - GRIPPER_CLOSED_RAD)


def _decode_rgb(payload: bytes) -> np.ndarray:
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Pillow is not installed; install the harness with the 'so101' extra"
        ) from exc
    with Image.open(BytesIO(payload)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


class So101Adapter:
    """Protocol-facing adapter for the simulation-runtime SO-101 session."""

    embodiment = "so101"
    gripper_holds_intermediate = True

    def __init__(
        self,
        *,
        registry: So101TaskRegistry | None = None,
        run_observer: Any | None = None,
        camera_width: int = 320,
        camera_height: int = 320,
        horizon: int = 300,
        control_frequency: float = 20.0,
        client: So101RuntimeClient | None = None,
        kinematics: So101Kinematics | None = None,
        phase: str | None = None,
    ) -> None:
        del camera_width, camera_height  # The runtime renders fixed 320x320 frames.
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        self.registry = registry or So101TaskRegistry()
        self._client = client or So101RuntimeClient()
        self._kinematics = kinematics or So101Kinematics()
        self.control_frequency = float(control_frequency)
        self.control_period_s = 1.0 / self.control_frequency
        self.horizon = int(horizon)
        self._phase = phase or os.environ.get("CAP_HARNESS_SO101_PHASE", "playground")
        self._run_observer = run_observer
        self._metadata: So101TaskMetadata | None = None
        self._task_context: TaskContext | None = None
        self._session_id: str | None = None
        self._latest: dict[str, Any] | None = None
        self._previous_joints: np.ndarray | None = None
        self._joint_velocities = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        self._joint_targets = HOME_JOINT_POSITIONS.copy()
        self._gripper_target = 1.0
        self._elapsed_steps = 0
        self._last_success = False
        self._last_terminated = False
        self._last_truncated = False
        self._closed = False
        self._rgb_cache: dict[str, np.ndarray] = {}

    # -- identity ----------------------------------------------------------

    @property
    def kinematics(self) -> So101Kinematics:
        return self._kinematics

    @property
    def native_env(self) -> So101Kinematics:
        """Runtime-only handle for host tooling; never registered with programs."""
        return self._kinematics

    @property
    def current_time_s(self) -> float:
        return self._elapsed_steps * self.control_period_s

    @property
    def arms(self) -> tuple[str, ...]:
        return ("primary",)

    def resolve_arm(self, arm: str) -> str:
        name = str(arm)
        return "primary" if name in {"primary", "so101", "arm", "robot"} else name

    @property
    def metadata(self) -> So101TaskMetadata:
        if self._metadata is None:
            raise RuntimeError("reset() must be called before task metadata is available")
        return self._metadata

    @property
    def session_id(self) -> str | None:
        return self._session_id

    # -- lifecycle ---------------------------------------------------------

    def reset(
        self,
        task_ref: str | tuple[str, int] | So101TaskMetadata,
        seed: int,
    ) -> Observation:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        metadata = self.registry.resolve(task_ref)
        self._close_session()
        response = self._client.create_session(
            seed=seed, task=metadata.task_name, phase=self._phase
        )
        session = response["session"]
        self._session_id = str(session["session_id"])
        self._metadata = metadata
        self._closed = False
        self._elapsed_steps = 0
        self._last_success = False
        self._last_terminated = False
        self._last_truncated = False
        self._rgb_cache.clear()
        self._previous_joints = None
        self._joint_velocities = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        observation = response["observation"]
        self._ingest(observation)
        self._joint_targets = np.asarray(
            observation["action"]["arm_joint_state"], dtype=np.float64
        ).copy()
        self._gripper_target = float(observation["action"]["ee_joint_state"][0])
        instruction = str(observation.get("instruction") or metadata.language)
        self._task_context = TaskContext(
            suite=metadata.suite_name,
            task_id=metadata.task_id,
            task_name=metadata.task_name,
            language=instruction,
            family=metadata.family,
            metadata={"task_ref": metadata.task_ref, "runtime": "simulation-runtime"},
        )
        self._elapsed_steps = int(observation.get("sequence", 0))
        public = self.get_observation()
        if self._run_observer is not None:
            self._run_observer.on_reset(public, metadata.to_manifest_record())
        return public

    def _ingest(self, observation: dict[str, Any]) -> None:
        state = observation["state"]
        joints = np.asarray(state["arm_joint_state"], dtype=np.float64)
        if joints.shape != (len(JOINT_NAMES),) or not bool(np.all(np.isfinite(joints))):
            raise So101RuntimeError("runtime returned an invalid arm_joint_state")
        if self._previous_joints is not None:
            self._joint_velocities = (joints - self._previous_joints) / self.control_period_s
        else:
            self._joint_velocities = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        self._previous_joints = joints
        self._latest = observation

    def _require_session(self) -> tuple[str, dict[str, Any]]:
        if self._session_id is None or self._latest is None:
            raise RuntimeError("reset() must be called before using the SO-101 adapter")
        return self._session_id, self._latest

    def _close_session(self) -> None:
        if self._session_id is None:
            return
        session_id = self._session_id
        self._session_id = None
        try:
            self._client.close(session_id)
        except So101RuntimeError:
            pass

    def close(self) -> None:
        if self._closed:
            return
        self._close_session()
        self._closed = True

    # -- observation -------------------------------------------------------

    def get_task_context(self) -> TaskContext:
        if self._task_context is None:
            raise RuntimeError("reset() must be called before reading task context")
        return self._task_context

    def get_robot_state(self) -> RobotState:
        _, observation = self._require_session()
        state = observation["state"]
        joints = np.asarray(state["arm_joint_state"], dtype=np.float64)
        raw_pose = np.asarray(state["ee_pose"], dtype=np.float64)
        quaternion = raw_pose[3:7]
        quaternion = quaternion / np.linalg.norm(quaternion)
        gripper = float(np.clip(state["ee_joint_state"][0], 0.0, 1.0))
        return RobotState(
            joint_positions={"primary": joints},
            joint_velocities={"primary": self._joint_velocities.copy()},
            end_effector_poses={
                "primary": Pose(
                    position=raw_pose[:3],
                    quaternion_wxyz=quaternion,
                    frame=BASE_FRAME,
                )
            },
            gripper_positions={"primary": gripper},
            base_frame=BASE_FRAME,
            joint_names={"primary": JOINT_NAMES},
            timestamp_s=self.current_time_s,
            embodiment=self.embodiment,
        )

    def _camera_rgb(self, name: str, specification: dict[str, Any]) -> np.ndarray:
        artifact_id = str(specification["artifact_id"])
        cached = self._rgb_cache.get(artifact_id)
        if cached is not None:
            return cached
        payload = self._client.fetch_artifact(str(specification["href"]))
        image = _decode_rgb(payload)
        self._rgb_cache = {artifact_id: image}
        return image

    def _camera_observation(self, name: str) -> CameraObservation:
        _, observation = self._require_session()
        specification = observation["cameras"].get(name)
        if specification is None:
            raise So101RuntimeError(f"runtime observation has no camera {name!r}")
        rgb = self._camera_rgb(name, specification)
        height, width = rgb.shape[:2]
        _, current = self._require_session()
        joints = np.asarray(current["state"]["arm_joint_state"], dtype=np.float64)
        gripper = float(np.clip(current["state"]["ee_joint_state"][0], 0.0, 1.0))
        position, rotation = self._kinematics.camera_pose(name, joints, _gripper_radians(gripper))
        intrinsics = self._kinematics.camera_intrinsics(name, width, height)
        if name == "top_cam":
            depth = synthetic_depth(rgb, intrinsics, position, rotation)
        else:
            # The wrist camera has no calibrated tabletop plane model; report
            # invalid depth rather than a fabricated one.
            depth = np.zeros((height, width), dtype=np.float64)
        return CameraObservation(
            rgb=rgb,
            depth_m=depth,
            intrinsics=intrinsics,
            frame=f"camera/{name}",
            camera_pose=matrix_to_pose(
                camera_transform(position, rotation), frame=BASE_FRAME
            ),
            timestamp_s=self.current_time_s,
        )

    def get_observation(self) -> Observation:
        if self._metadata is None:
            raise RuntimeError("reset() must be called before reading observations")
        return Observation(
            cameras={
                name: self._camera_observation(name)
                for name in self._metadata.camera_names
            },
            robot_state=self.get_robot_state(),
            task_context=self.get_task_context(),
            timestamp_s=self.current_time_s,
        )

    # -- control -----------------------------------------------------------

    def _terminal_step(self) -> StepResult:
        code = ErrorCode.TERMINATED if self._last_terminated else ErrorCode.TRUNCATED
        return StepResult(
            ok=False,
            terminated=self._last_terminated,
            truncated=self._last_truncated,
            error=ApiError(code, "the SO-101 episode is already terminal"),
        )

    def step(self, action: RobotAction) -> StepResult:
        if not isinstance(action, RobotAction):
            raise ValueError("action must be a RobotAction")
        if action.embodiment != self.embodiment:
            raise ValueError(
                f"action embodiment {action.embodiment!r} does not match {self.embodiment!r}"
            )
        session_id, _ = self._require_session()
        if self._closed:
            raise RuntimeError("the SO-101 adapter is closed")
        if self._last_terminated or self._last_truncated:
            return self._terminal_step()
        unknown = set(action.arms) - set(self.arms)
        if unknown:
            return StepResult(
                ok=False,
                error=ApiError(ErrorCode.NOT_FOUND, f"unknown SO-101 arms: {sorted(unknown)}"),
            )
        if not action.arms:
            return StepResult(
                ok=False,
                error=ApiError(ErrorCode.INVALID_REQUEST, "action must command the primary arm"),
            )
        command = action.arms.get("primary")
        assert command is not None
        try:
            target = np.asarray(command.target, dtype=np.float64)
        except (TypeError, ValueError):
            target = np.empty(0)
        if target.shape != (len(JOINT_NAMES),) or not bool(np.all(np.isfinite(target))):
            return StepResult(
                ok=False,
                error=ApiError(
                    ErrorCode.INVALID_REQUEST,
                    f"SO-101 joint targets must be finite {len(JOINT_NAMES)}-joint vectors",
                ),
            )
        target = np.clip(target, self._kinematics.joint_lower, self._kinematics.joint_upper)
        gripper = self._gripper_target
        if command.gripper_position is not None:
            gripper = float(np.clip(command.gripper_position, 0.0, 1.0))
        if self._run_observer is not None:
            self._run_observer.before_step(action)
        try:
            result = self._client.step(session_id, target, gripper)
            self._ingest(result["observation"])
            self._elapsed_steps += 1
            self._joint_targets = target.copy()
            self._gripper_target = gripper
            self._last_success = bool(result.get("success", False))
            self._last_terminated = bool(result.get("terminated", False))
            self._last_truncated = bool(result.get("truncated", False))
            outcome = StepResult(
                ok=True,
                observation=self.get_observation(),
                terminated=self._last_terminated,
                truncated=self._last_truncated,
                diagnostics={
                    "native_steps_used": int(result.get("native_steps_used", 0)),
                    "native_steps_remaining": int(result.get("native_steps_remaining", 0)),
                    "control_period_s": self.control_period_s,
                },
            )
        # Surface runtime failures through the public typed error result.
        except Exception as exc:
            outcome = StepResult(
                ok=False,
                error=ApiError(
                    ErrorCode.ADAPTER_FAILED,
                    f"SO-101 runtime step failed: {exc}",
                    details={"exception_type": type(exc).__name__},
                ),
            )
        if self._run_observer is not None:
            self._run_observer.after_step(action, outcome)
        return outcome

    def _hold_steps(
        self, target: np.ndarray, gripper: float, count: int
    ) -> tuple[StepResult | None, int]:
        result: StepResult | None = None
        executed = 0
        for _ in range(count):
            result = self.step(
                RobotAction(
                    arms={
                        "primary": ArmCommand(
                            mode="joint_position",
                            target=target,
                            gripper_position=gripper,
                            embodiment=self.embodiment,
                        )
                    }
                )
            )
            if not result.ok:
                return result, executed
            executed += 1
        return result, executed

    def execute_trajectory(
        self, trajectory: Trajectory | SynchronizedTrajectory
    ) -> ExecutionResult:
        if not isinstance(trajectory, (Trajectory, SynchronizedTrajectory)):
            # Preserve the invalid-data ValueError contract.
            raise ValueError("trajectory must be a Trajectory or SynchronizedTrajectory")
        if trajectory.embodiment != self.embodiment:
            raise ValueError(
                f"trajectory embodiment {trajectory.embodiment!r} "
                f"does not match {self.embodiment!r}"
            )
        if not np.isclose(trajectory.dt_s, self.control_period_s, atol=1e-9, rtol=0.0):
            raise ValueError("trajectory dt_s must equal the SO-101 control period")
        state = self.get_robot_state()
        if isinstance(trajectory, Trajectory):
            if trajectory.arm != "primary":
                return ExecutionResult(
                    ok=False,
                    steps_executed=0,
                    error=ApiError(
                        ErrorCode.NOT_FOUND, f"unknown SO-101 arm {trajectory.arm!r}"
                    ),
                )
            positions = [np.asarray(trajectory.joint_positions, dtype=np.float64)]
            names = trajectory.joint_names
            grippers = (
                trajectory.gripper_positions
                if trajectory.gripper_positions is not None
                else np.full(len(trajectory.joint_positions), self._gripper_target)
            )
            if tuple(names) != tuple(state.joint_names["primary"]):
                raise ValueError("trajectory joint_names do not match the SO-101 arm")
        else:
            if set(trajectory.joint_positions) != {"primary"}:
                raise ValueError("synchronized SO-101 trajectories must name the primary arm")
            positions = [
                np.asarray(trajectory.joint_positions["primary"], dtype=np.float64)
            ]
            if tuple(trajectory.joint_names["primary"]) != tuple(state.joint_names["primary"]):
                raise ValueError("trajectory joint_names do not match the SO-101 arm")
            grippers = (
                trajectory.gripper_positions["primary"]
                if trajectory.gripper_positions is not None
                else np.full(trajectory.waypoint_count, self._gripper_target)
            )
        waypoints = positions[0]
        if waypoints.ndim != 2 or waypoints.shape[1] != len(JOINT_NAMES):
            raise ValueError("trajectory joint positions must be an (N, 5) array")
        expected = trajectory.expected_start
        if expected is not None:
            start = np.asarray(expected.joint_positions["primary"], dtype=np.float64)
            if float(np.max(np.abs(start - state.joint_positions["primary"]))) > (
                TRAJECTORY_START_TOLERANCE_RAD
            ):
                return ExecutionResult(
                    ok=False,
                    steps_executed=0,
                    error=ApiError(
                        ErrorCode.STALE_PLAN,
                        "SO-101 trajectory start state no longer matches the robot",
                    ),
                )
        final: Observation | None = None
        steps = 0
        last_target = waypoints[-1]
        last_gripper = float(self._gripper_target)
        for index in range(len(waypoints)):
            target = waypoints[index]
            gripper = float(grippers[index])
            result = self.step(
                RobotAction(
                    arms={
                        "primary": ArmCommand(
                            mode="joint_position",
                            target=target,
                            gripper_position=gripper,
                            embodiment=self.embodiment,
                        )
                    }
                )
            )
            if not result.ok or result.observation is None:
                return ExecutionResult(
                    ok=False,
                    steps_executed=steps,
                    final_observation=final,
                    terminated=result.terminated,
                    truncated=result.truncated,
                    error=result.error,
                )
            steps += 1
            final = result.observation
            last_target = target
            last_gripper = gripper
            if result.terminated or result.truncated:
                return ExecutionResult(
                    ok=False,
                    steps_executed=steps,
                    final_observation=final,
                    terminated=result.terminated,
                    truncated=result.truncated,
                    error=ApiError(
                        ErrorCode.TERMINATED if result.terminated else ErrorCode.TRUNCATED,
                        "the SO-101 episode ended during trajectory execution",
                    ),
                )
        # A position-controlled joint lags its target by a few ticks.  Hold the
        # final waypoint so pose verification sees the settled state, not the
        # transient one.
        error_before = float(
            np.max(np.abs(self.get_robot_state().joint_positions["primary"] - last_target))
        )
        if error_before > TRAJECTORY_SETTLE_TOLERANCE_RAD:
            hold, held_steps = self._hold_steps(
                last_target, last_gripper, TRAJECTORY_SETTLE_STEPS
            )
            if hold is not None:
                if not hold.ok or hold.observation is None:
                    return ExecutionResult(
                        ok=False,
                        steps_executed=steps + held_steps,
                        final_observation=final,
                        terminated=hold.terminated,
                        truncated=hold.truncated,
                        error=hold.error,
                    )
                steps += held_steps
                final = hold.observation
        joint_error = float(
            np.max(np.abs(self.get_robot_state().joint_positions["primary"] - last_target))
        )
        return ExecutionResult(
            ok=True,
            steps_executed=steps,
            final_observation=final,
            final_errors={"primary": joint_error},
            diagnostics={
                "control_period_s": self.control_period_s,
                "final_joint_error_rad": joint_error,
            },
        )

    def set_gripper(self, position: float, *, arm: str = "primary") -> ExecutionResult:
        if arm != "primary":
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(ErrorCode.NOT_FOUND, f"unknown SO-101 arm {arm!r}"),
            )
        try:
            target = float(position)
        except (TypeError, ValueError):
            target = float("nan")
        if not np.isfinite(target) or not 0.0 <= target <= 1.0:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    ErrorCode.INVALID_REQUEST,
                    "gripper position must be finite and in [0, 1]",
                ),
            )
        start = self.get_robot_state().gripper_positions["primary"]
        final: Observation | None = None
        steps = 0
        for index in range(GRIPPER_RAMP_STEPS):
            fraction = float(index + 1) / GRIPPER_RAMP_STEPS
            # Multiplying then dividing by the step count can land one ulp outside
            # [0, 1] and the contract rejects the command before it reaches the runtime.
            value = float(np.clip(start + (target - start) * fraction, 0.0, 1.0))
            result = self.step(
                RobotAction(
                    arms={
                        "primary": ArmCommand(
                            mode="joint_position",
                            target=self._joint_targets,
                            gripper_position=value,
                            embodiment=self.embodiment,
                        )
                    }
                )
            )
            if not result.ok or result.observation is None:
                return ExecutionResult(
                    ok=False,
                    steps_executed=steps,
                    final_observation=final,
                    terminated=result.terminated,
                    truncated=result.truncated,
                    error=result.error,
                )
            steps += 1
            final = result.observation
            if result.terminated or result.truncated:
                return ExecutionResult(
                    ok=False,
                    steps_executed=steps,
                    final_observation=final,
                    terminated=result.terminated,
                    truncated=result.truncated,
                    error=ApiError(
                        ErrorCode.TERMINATED if result.terminated else ErrorCode.TRUNCATED,
                        "the SO-101 episode ended while moving the gripper",
                    ),
                )
        hold, held_steps = self._hold_steps(self._joint_targets, target, GRIPPER_HOLD_STEPS)
        if hold is not None:
            if not hold.ok or hold.observation is None:
                return ExecutionResult(
                    ok=False,
                    steps_executed=steps + held_steps,
                    final_observation=final,
                    terminated=hold.terminated,
                    truncated=hold.truncated,
                    error=hold.error,
                )
            steps += held_steps
            final = hold.observation
        return ExecutionResult(
            ok=True,
            steps_executed=steps,
            final_observation=final,
            diagnostics={"gripper_target": target},
        )

    def set_grippers(self, positions: dict[str, float]) -> ExecutionResult:
        if set(positions) != {"primary"}:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    ErrorCode.INVALID_REQUEST,
                    "positions must name the primary SO-101 arm",
                ),
            )
        return self.set_gripper(float(positions["primary"]), arm="primary")

    # -- host-side metadata ------------------------------------------------

    def check_success(self) -> bool:
        return bool(self._last_success)

    def get_controller_metadata(self) -> dict[str, object]:
        return {
            "action_lower_bounds": tuple(float(value) for value in self._kinematics.joint_lower),
            "action_upper_bounds": tuple(float(value) for value in self._kinematics.joint_upper),
            "control_frequency_hz": self.control_frequency,
            "control_period_s": self.control_period_s,
            "arm_names": ("primary",),
            "joint_names": JOINT_NAMES,
            "home_joint_positions": tuple(float(value) for value in HOME_JOINT_POSITIONS),
            "gripper_name": "gripper",
            "gripper_range_rad": (GRIPPER_CLOSED_RAD, GRIPPER_OPEN_RAD),
        }

    def get_task_metadata(self) -> dict[str, object]:
        return self.metadata.to_manifest_record()

    def get_planning_context(self) -> RobotPlanningContext:
        return RobotPlanningContext(
            embodiment=self.embodiment,
            model="so101",
            joint_names={"primary": JOINT_NAMES},
            base_transforms={"primary": np.eye(4, dtype=np.float64)},
            end_effector_links={"primary": "gripperframe"},
        )


__all__ = ["So101Adapter"]
