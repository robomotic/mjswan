"""Mjlab-specific custom observation registrations for mjswan examples.

Import this module before calling ``builder.build()`` to register the
custom observation classes used in mjlab tasks.
"""

import os
from typing import Any

from mjswan import ObsFunc, register_obs_func

_OBS_DIR = os.path.dirname(os.path.abspath(__file__))

register_obs_func(
    "ee_to_object_distance",
    ObsFunc(
        ts_name="EeToObjectDistance",
        ts_src=os.path.join(_OBS_DIR, "EeToObjectDistance.ts"),
    ),
)

register_obs_func(
    "object_to_goal_distance",
    ObsFunc(
        ts_name="ObjectToGoalDistance",
        ts_src=os.path.join(_OBS_DIR, "ObjectToGoalDistance.ts"),
    ),
)


def register_custom_observations(env_cfg: Any) -> None:
    """Register env_cfg-dependent observations (e.g. height_scan)."""
    terrain_scan = next(
        (
            sensor
            for sensor in (env_cfg.scene.sensors or ())
            if getattr(sensor, "name", None) == "terrain_scan"
        ),
        None,
    )
    if terrain_scan is None:
        return

    frame = terrain_scan.frame
    frame_ref_name = (
        f"{frame.entity}/{frame.name}" if getattr(frame, "entity", None) else frame.name
    )
    pattern = terrain_scan.pattern
    register_obs_func(
        "height_scan",
        ObsFunc(
            ts_name="HeightScan",
            ts_src=os.path.join(_OBS_DIR, "HeightScan.ts"),
            defaults={
                "frame_type": frame.type,
                "frame_ref_name": frame_ref_name,
                "ray_alignment": terrain_scan.ray_alignment,
                "pattern_size": list(pattern.size),
                "pattern_resolution": float(pattern.resolution),
                "pattern_direction": list(pattern.direction),
                "max_distance": float(terrain_scan.max_distance),
                "terrain_body_name": "terrain",
            },
        ),
    )
