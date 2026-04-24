"""Unitree RL mjlab demo."""

import importlib
import pathlib
import sys

import mjlab.tasks  # noqa: F401 - populates the mjlab task registry
import src.tasks  # noqa: F401
from mjlab.tasks.registry import load_env_cfg

import mjswan

sys.path.insert(0, str(pathlib.Path(__file__).parent))
importlib.import_module("terminations")  # noqa: F401 - registers custom terminations


def setup_builder() -> mjswan.Builder:
    """Create the builder for the unitree_rl_mjlab demo."""

    run_path = "ttktjmt-org/mjlab/l3tgm74z"
    task_id = "Unitree-G1-Tracking-No-State-Estimation"

    builder = mjswan.Builder()

    project = builder.add_project(name="Unitree RL")
    scene = project.add_mjlab_scene(task_id, play=True)

    env_cfg = load_env_cfg(task_id, play=True)
    scene.add_policy_from_wandb(
        run_path,
        task_id=task_id,
        observations={"policy": env_cfg.observations["actor"]},
        commands=env_cfg.commands,
        actions=env_cfg.actions,
        terminations=env_cfg.terminations,
    )

    return builder


def main() -> None:
    """Build and launch the unitree_rl_mjlab demo."""
    app = setup_builder().build()
    app.launch()


if __name__ == "__main__":
    main()
