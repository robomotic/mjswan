"""G1 spinkick motion-tracking demo.

This example exercises mjswan's mjlab tracking playback path end-to-end:
- MuJoCo scene from mjlab's play config
- policy checkpoints exported from a W&B training run
- reference motion auto-imported from the run's motion artifact

Important:
- The export environment needs a local ``MotionCommandCfg.motion_file``.
- By default this script downloads ``motion.npz`` from the configured W&B
  artifact and stages it into a temporary file before constructing the mjlab
  export environment.

Requirements:
- ``mjlab``, ``torch``, and ``wandb`` must be installed
- W&B must already be authenticated for the target run

Environment variables:
- ``MJSWAN_BASE_PATH``: Override the built app base path
- ``MJSWAN_NO_LAUNCH=1``: Build without opening a browser
- ``MJSWAN_WANDB_RUN_PATH``: Override the default W&B run path
- ``MJSWAN_SPINKICK_MOTION_ARTIFACT``: Override the W&B motion artifact URL/path
- ``MJSWAN_SPINKICK_MOTION_FILE``: Override the local reference ``.npz`` path
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import g1_spinkick_terminations  # noqa: F401 - registers custom terminations
import mjlab.tasks  # noqa: F401 - populates the mjlab task registry
from mjlab.tasks.registry import load_env_cfg

import mjswan
from mjswan.wandb_utils import (
    create_pt_onnx_export_context,
    fetch_motion_npz_from_wandb_artifact,
)

DEFAULT_RUN_PATH = "ttktjmt-org/mjlab/mayq0rtd"
DEFAULT_MOTION_ARTIFACT = (
    "https://wandb.ai/ttktjmt-org/csv_to_npz/artifacts/motions/"
    "mimickit_spinkick_safe/latest/files/motion.npz"
)
TASK_ID = "Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation"


def _resolve_motion_file(staging_dir: Path) -> Path:
    """Resolve the reference motion file used by mjlab's export env."""
    override = os.getenv("MJSWAN_SPINKICK_MOTION_FILE")
    if override:
        motion_path = Path(override).expanduser().resolve()
        if not motion_path.is_file():
            raise FileNotFoundError(
                f"MJSWAN_SPINKICK_MOTION_FILE does not exist: {motion_path}"
            )
        return motion_path

    artifact_ref = os.getenv("MJSWAN_SPINKICK_MOTION_ARTIFACT", DEFAULT_MOTION_ARTIFACT)
    motion_name, payload = fetch_motion_npz_from_wandb_artifact(artifact_ref)
    motion_path = staging_dir / f"{motion_name}.npz"
    motion_path.write_bytes(payload)
    return motion_path


def setup_builder() -> mjswan.Builder:
    """Create the builder for the G1 spinkick tracking demo."""
    example_dir = Path(__file__).resolve().parent
    os.chdir(example_dir)

    run_path = os.getenv("MJSWAN_WANDB_RUN_PATH", DEFAULT_RUN_PATH)
    base_path = os.getenv("MJSWAN_BASE_PATH", "/")
    builder = mjswan.Builder(base_path=base_path)

    with tempfile.TemporaryDirectory(prefix="mjswan-g1-spinkick-") as tmp_dir:
        motion_file = _resolve_motion_file(Path(tmp_dir))

        env_cfg = load_env_cfg(TASK_ID, play=True)
        env_cfg.commands["motion"].motion_file = str(motion_file)

        project = builder.add_project(name="mjswan Tracking Demo")
        scene = project.add_mjlab_scene(TASK_ID, play=True)

        export_context = create_pt_onnx_export_context(TASK_ID, env_cfg=env_cfg)
        try:
            scene.add_policy_from_wandb(
                run_path,
                task_id=TASK_ID,
                export_context=export_context,
                observations={"policy": env_cfg.observations["actor"]},
                commands=env_cfg.commands,
                actions=env_cfg.actions,
                terminations=env_cfg.terminations,
            )
        finally:
            export_context.close()

    return builder


def main() -> None:
    """Build and optionally launch the G1 spinkick demo."""
    app = setup_builder().build()
    if os.getenv("MJSWAN_NO_LAUNCH") == "1":
        return
    app.launch()


if __name__ == "__main__":
    main()
