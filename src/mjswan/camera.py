"""Camera configuration for mjswan scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CameraConfig:
    """Camera configuration for a scene."""

    position: tuple[float, float, float] | None = None
    """Initial camera position in MuJoCo coordinates (x forward, y left, z up)."""

    target: tuple[float, float, float] | None = None
    """Initial look-at target in MuJoCo coordinates (x forward, y left, z up)."""

    fov: float | None = None
    """Camera field of view in degrees. Defaults to 45."""

    track_body_name: str | None = None
    """Body name to track. The orbit camera target follows this body each frame."""

    mujoco_camera: str | None = None
    """Name of a MuJoCo camera defined in the scene XML/spec.
    The viewer uses its position and orientation directly, disabling free orbit."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict for config.json."""
        d: dict[str, Any] = {}
        if self.position is not None:
            d["position"] = list(self.position)
        if self.target is not None:
            d["target"] = list(self.target)
        if self.fov is not None:
            d["fov"] = self.fov
        if self.track_body_name is not None:
            d["trackBodyName"] = self.track_body_name
        if self.mujoco_camera is not None:
            d["mujocoCamera"] = self.mujoco_camera
        return d


__all__ = ["CameraConfig"]
