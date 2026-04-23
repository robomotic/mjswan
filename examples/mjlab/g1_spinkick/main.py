"""G1 spinkick motion-tracking demo.

This example exercises mjswan's mjlab tracking playback path end-to-end:
- MuJoCo scene from mjlab's play config
- policy checkpoints exported from a W&B training run
- reference motion auto-imported from the run's motion artifact

Requirements:
- ``mjlab``, ``torch``, and ``wandb`` must be installed
- W&B must already be authenticated for the target run

Environment variables:
- ``MJSWAN_NO_LAUNCH=1``: Build without opening a browser
- ``MJSWAN_WANDB_RUN_PATH``: Override the default W&B run path
"""

from __future__ import annotations

import os
from pathlib import Path

import mjlab.tasks  # noqa: F401 - populates the mjlab task registry
from mjlab.tasks.registry import load_env_cfg

import mjswan

from . import terminations  # noqa: F401 - registers custom terminations

DEFAULT_RUN_PATH = "ttktjmt-org/mjlab/mayq0rtd"
TASK_ID = "Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation"


def setup_builder() -> mjswan.Builder:
    """Create the builder for the G1 spinkick tracking demo."""
    example_dir = Path(__file__).resolve().parent
    os.chdir(example_dir)

    run_path = os.getenv("MJSWAN_WANDB_RUN_PATH", DEFAULT_RUN_PATH)
    builder = mjswan.Builder()

    project = builder.add_project(name="mjswan Tracking Demo")
    scene = project.add_mjlab_scene(TASK_ID, play=True)

    env_cfg = load_env_cfg(TASK_ID, play=True)
    scene.add_policy_from_wandb(
        run_path,
        task_id=TASK_ID,
        observations={"policy": env_cfg.observations["actor"]},
        commands=env_cfg.commands,
        actions=env_cfg.actions,
        terminations=env_cfg.terminations,
    )

    return builder


def main() -> None:
    """Build and optionally launch the G1 spinkick demo."""
    app = setup_builder().build()
    if os.getenv("MJSWAN_NO_LAUNCH") == "1":
        return
    app.launch()


if __name__ == "__main__":
    main()
