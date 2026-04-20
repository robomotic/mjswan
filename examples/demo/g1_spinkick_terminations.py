"""Custom termination registrations for the G1 spinkick demo."""

from __future__ import annotations

from pathlib import Path

from mjswan import TermFunc, register_termination_func

_TERM_DIR = Path(__file__).resolve().parent / "g1_spinkick_terminations"

register_termination_func(
    "bad_anchor_pos_z_only",
    TermFunc(
        ts_name="BadAnchorPosZOnly",
        ts_src=str(_TERM_DIR / "BadAnchorPosZOnly.ts"),
    ),
)

register_termination_func(
    "bad_anchor_ori",
    TermFunc(
        ts_name="BadAnchorOri",
        ts_src=str(_TERM_DIR / "BadAnchorOri.ts"),
    ),
)

register_termination_func(
    "bad_motion_body_pos_z_only",
    TermFunc(
        ts_name="BadMotionBodyPosZOnly",
        ts_src=str(_TERM_DIR / "BadMotionBodyPosZOnly.ts"),
    ),
)

register_termination_func(
    "base_ang_vel_exceed",
    TermFunc(
        ts_name="BaseAngVelExceed",
        ts_src=str(_TERM_DIR / "BaseAngVelExceed.ts"),
    ),
)
