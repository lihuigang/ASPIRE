"""Minimal HTTP client for the simulation-runtime session API."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

DEFAULT_SO101_RUNTIME_URL = "http://192.168.111.40:18080"
DEFAULT_SO101_BACKEND = "mujoco_so101"
DEFAULT_SO101_TASK = "pick_cube"


class So101RuntimeError(RuntimeError):
    """The simulation runtime refused a request or returned a failed operation."""


class So101RuntimeClient:
    """Drive one SO-101 simulation session over the runtime's v1 HTTP API."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        token: str | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self.base_url = (
            base_url or os.environ.get("CAP_HARNESS_SO101_URL") or DEFAULT_SO101_RUNTIME_URL
        ).rstrip("/")
        self.token = token or os.environ.get("CAP_HARNESS_SO101_TOKEN")
        self.timeout_s = float(timeout_s)

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, object] | None = None,
        *,
        timeout_s: float | None = None,
        raw: bool = False,
        retries: int = 0,
    ) -> object:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        for attempt in range(retries + 1):
            request = urllib.request.Request(self.base_url + path, data=body, method=method)
            if body is not None:
                request.add_header("Content-Type", "application/json")
            if self.token:
                request.add_header("Authorization", f"Bearer {self.token}")
            try:
                with urllib.request.urlopen(
                    request, timeout=timeout_s or self.timeout_s
                ) as response:
                    content = response.read()
                break
            except urllib.error.HTTPError as exc:
                if exc.code >= 500 and attempt < retries:
                    time.sleep(0.15 * (attempt + 1))
                    continue
                detail = exc.read().decode("utf-8", "replace")[:800]
                raise So101RuntimeError(
                    f"{method} {path} failed with HTTP {exc.code}: {detail}"
                ) from exc
            except urllib.error.URLError as exc:
                if attempt < retries:
                    time.sleep(0.15 * (attempt + 1))
                    continue
                raise So101RuntimeError(
                    f"cannot reach SO-101 runtime at {self.base_url}: {exc.reason}"
                ) from exc
        if raw:
            return content
        return json.loads(content) if content else {}

    def health(self) -> dict[str, object]:
        return self._request("GET", "/readyz", retries=2)  # type: ignore[return-value]

    def create_session(
        self,
        seed: int,
        *,
        task: str = DEFAULT_SO101_TASK,
        backend: str = DEFAULT_SO101_BACKEND,
        phase: str = "playground",
        mode: str = "typed",
    ) -> dict[str, object]:
        response = self._request(
            "POST",
            "/v1/sessions",
            {"backend": backend, "task": task, "phase": phase, "mode": mode, "seed": seed},
        )
        if not isinstance(response, dict) or "session" not in response or "observation" not in response:
            raise So101RuntimeError("create_session returned an unexpected payload")
        return response

    def step(
        self,
        session_id: str,
        arm_joint_state: object,
        gripper: float,
        *,
        timeout_s: float = 60.0,
    ) -> dict[str, object]:
        response = self._request(
            "POST",
            f"/v1/sessions/{session_id}/steps?wait=true",
            {
                "action": {
                    "arm_joint_state": [float(value) for value in arm_joint_state],
                    "ee_joint_state": [float(gripper)],
                }
            },
            timeout_s=timeout_s,
        )
        if not isinstance(response, dict):
            raise So101RuntimeError("step returned an unexpected payload")
        if response.get("status") != "succeeded" or not isinstance(response.get("result"), dict):
            raise So101RuntimeError(
                f"step operation failed: status={response.get('status')!r} "
                f"error={response.get('error')!r}"
            )
        return response["result"]

    def observation(self, session_id: str) -> dict[str, object]:
        response = self._request("GET", f"/v1/sessions/{session_id}/observation", retries=2)
        if not isinstance(response, dict):
            raise So101RuntimeError("observation returned an unexpected payload")
        return response

    def reset(self, session_id: str) -> dict[str, object]:
        response = self._request("POST", f"/v1/sessions/{session_id}/reset", {})
        if not isinstance(response, dict):
            raise So101RuntimeError("reset returned an unexpected payload")
        return response

    def close(self, session_id: str) -> None:
        self._request("POST", f"/v1/sessions/{session_id}/close", {})

    def fetch_artifact(self, href: str) -> bytes:
        path = href if href.startswith("/") else f"/{href}"
        payload = self._request("GET", path, raw=True, retries=2)
        if not isinstance(payload, bytes):
            raise So101RuntimeError("artifact download returned a non-binary payload")
        return payload


__all__ = [
    "DEFAULT_SO101_BACKEND",
    "DEFAULT_SO101_RUNTIME_URL",
    "DEFAULT_SO101_TASK",
    "So101RuntimeClient",
    "So101RuntimeError",
]
