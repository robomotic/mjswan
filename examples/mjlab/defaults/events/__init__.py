"""Mjlab-specific custom event registrations for mjswan examples.

Import this module before calling ``builder.build()`` to register the
custom reset event classes used in mjlab tasks.
"""

import os
from typing import Any

from mjswan import EventFunc, register_event_func

_EVENT_DIR = os.path.dirname(os.path.abspath(__file__))


def register_custom_events(env_cfg: Any | None = None) -> None:
    """Register mjlab-specific reset events for browser execution."""
    del env_cfg  # Reserved for future env-specific defaults.

    register_event_func(
        "reset_joints_by_offset",
        EventFunc(
            ts_name="ResetJointsByOffset",
            ts_src=os.path.join(_EVENT_DIR, "ResetJointsByOffset.ts"),
        ),
    )

    register_event_func(
        "randomize_terrain",
        EventFunc(
            ts_name="RandomizeTerrain",
            ts_src=os.path.join(_EVENT_DIR, "RandomizeTerrain.ts"),
        ),
    )
