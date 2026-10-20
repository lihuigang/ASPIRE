"""Task discovery for the SO-101 simulation benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class So101RegistryError(ValueError):
    """Raised when an SO-101 task reference is invalid."""


@dataclass(frozen=True, slots=True)
class So101TaskMetadata:
    task_name: str = "pick_cube"
    task_id: int = 0
    language: str = "Pick up the red cube and place it stably on the green target pad."
    family: str = "so101"
    horizon: int | None = 300
    camera_names: tuple[str, ...] = ("top_cam", "wrist_cam")

    @property
    def suite_name(self) -> str:
        return self.task_name

    @property
    def task_ref(self) -> str:
        return f"{self.task_name}:{self.task_id}"

    @property
    def arms(self) -> tuple[str, ...]:
        return ("primary",)

    def to_manifest_record(self) -> dict[str, Any]:
        return {
            "arms": list(self.arms),
            "camera_names": list(self.camera_names),
            "environment": "mujoco_so101",
            "family": self.family,
            "language": self.language,
            "robots": ["so101"],
            "suite_name": self.suite_name,
            "task_id": self.task_id,
            "task_name": self.task_name,
            "task_ref": self.task_ref,
        }


SO101_TASKS: tuple[So101TaskMetadata, ...] = (So101TaskMetadata(),)


class So101TaskRegistry:
    """Resolve only the audited SO-101 task set."""

    def __init__(self, tasks: tuple[So101TaskMetadata, ...] = SO101_TASKS) -> None:
        self._tasks = {task.task_name: task for task in tasks}
        if len(self._tasks) != len(tasks):
            raise So101RegistryError("SO-101 task names must be unique")

    @property
    def available_tasks(self) -> tuple[str, ...]:
        return tuple(self._tasks)

    def enumerate_tasks(self) -> tuple[So101TaskMetadata, ...]:
        return tuple(self._tasks.values())

    def resolve(
        self, task_ref: str | tuple[str, int] | So101TaskMetadata, task_id: int | None = None
    ) -> So101TaskMetadata:
        if isinstance(task_ref, So101TaskMetadata):
            return task_ref
        if isinstance(task_ref, tuple):
            if len(task_ref) != 2:
                raise So101RegistryError("task tuple must contain (task_name, task_id)")
            task_name, tuple_id = task_ref
            if task_id is not None and task_id != tuple_id:
                raise So101RegistryError("conflicting SO-101 task ids")
            task_id = int(tuple_id)
        else:
            task_name = str(task_ref)
            if ":" in task_name:
                task_name, encoded_id = task_name.rsplit(":", 1)
                if task_id is not None and task_id != int(encoded_id):
                    raise So101RegistryError("conflicting SO-101 task ids")
                task_id = int(encoded_id)
        if task_id is None:
            task_id = 0
        if task_id != 0:
            raise So101RegistryError("each SO-101 task currently has task_id 0")
        if task_name == "so101":
            # The benchmark name doubles as the suite when a caller is explicit.
            task_name = "pick_cube"
        try:
            return self._tasks[str(task_name)]
        except KeyError as exc:
            raise So101RegistryError(
                f"unknown SO-101 task {task_name!r}; available: "
                f"{', '.join(self.available_tasks)}"
            ) from exc


__all__ = [
    "SO101_TASKS",
    "So101RegistryError",
    "So101TaskMetadata",
    "So101TaskRegistry",
]
