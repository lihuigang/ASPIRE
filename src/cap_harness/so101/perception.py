"""Color-based segmentation for the SO-101 tabletop scene.

The runtime publishes RGB renders only; the benchmark's objects are the only
saturated red and green surfaces on an otherwise neutral table, so thresholding
is both sufficient and deterministic.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from cap_harness.contracts import CameraObservation, Segmentation, SegmentationSet
from cap_harness.errors import ApiError, ErrorCode

RED_TERMS = ("red", "cube", "block", "红", "立方", "方块")
GREEN_TERMS = ("green", "pad", "target", "plate", "绿", "目标", "垫")

_MIN_PIXELS = 24


def red_mask(rgb: np.ndarray) -> np.ndarray:
    pixels = np.asarray(rgb, dtype=np.float64)
    red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    return (red > 90.0) & (red > 1.6 * green) & (red > 1.6 * blue)


def green_mask(rgb: np.ndarray) -> np.ndarray:
    pixels = np.asarray(rgb, dtype=np.float64)
    red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    return (green > 70.0) & (green > 1.3 * red) & (green > 1.3 * blue)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def _segmentation(
    mask: np.ndarray,
    observation: CameraObservation,
    label: str,
) -> Segmentation | None:
    component = _largest_component(mask)
    pixels = int(np.count_nonzero(component))
    if pixels < _MIN_PIXELS:
        return None
    rows, columns = np.nonzero(component)
    box = np.array(
        [columns.min(), rows.min(), columns.max() + 1, rows.max() + 1], dtype=np.float64
    )
    return Segmentation(
        mask=component,
        label=label,
        score=min(1.0, pixels / 400.0),
        camera_name=observation.frame.removeprefix("camera/"),
        frame=observation.frame,
        box_xyxy=box,
    )


class So101ColorSegmentationProvider:
    """Segment the benchmark's red cube and green pad by material color."""

    def segment_text(
        self,
        observation: CameraObservation,
        query: str,
    ) -> SegmentationSet:
        if not isinstance(observation, CameraObservation):
            return SegmentationSet(
                ok=False,
                error=ApiError(ErrorCode.INVALID_REQUEST, "observation must be a CameraObservation"),
            )
        text = str(query).strip().lower()
        if not text:
            return SegmentationSet(
                ok=False,
                error=ApiError(ErrorCode.INVALID_REQUEST, "query must be non-empty"),
            )
        if any(term in text for term in RED_TERMS):
            mask, label = red_mask(observation.rgb), "red cube"
        elif any(term in text for term in GREEN_TERMS):
            mask, label = green_mask(observation.rgb), "green target pad"
        else:
            return SegmentationSet(
                ok=False,
                error=ApiError(
                    ErrorCode.UNSUPPORTED,
                    f"SO-101 color segmentation cannot resolve query {query!r}; "
                    "ask for the red cube or the green target pad",
                ),
            )
        segmentation = _segmentation(mask, observation, label)
        if segmentation is None:
            return SegmentationSet(
                ok=False,
                error=ApiError(
                    ErrorCode.POINT_CLOUD_FAILED,
                    f"no {label} pixels found in {observation.frame}",
                ),
            )
        return SegmentationSet(ok=True, segmentations=(segmentation,))

    def segment_points(
        self,
        observation: CameraObservation,
        points_px: np.ndarray,
        *,
        point_labels: np.ndarray | None = None,
    ) -> SegmentationSet:
        del observation, points_px, point_labels
        return SegmentationSet(
            ok=False,
            error=ApiError(
                ErrorCode.UNSUPPORTED,
                "SO-101 color segmentation supports text queries only",
            ),
        )


__all__ = [
    "So101ColorSegmentationProvider",
    "green_mask",
    "red_mask",
]
