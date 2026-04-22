"""mjlab Integration Example - Visualize MuJoCo scenes from all mjlab default tasks

Extracts the MuJoCo model from each mjlab default task and visualizes them
in the browser using mjswan.
"""

from __future__ import annotations

from mjlab.tasks.registry import load_env_cfg

import mjswan
from mjswan.envs.mdp import observations as obs_fns
from mjswan.wandb_utils import create_pt_onnx_export_context

if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    __package__ = "examples.mjlab"

from . import commands  # noqa: F401 - for command registrations
from .events import register_custom_events
from .observations import register_custom_observations
from .terminations import register_custom_terminations

pole_angle_cos_sin = obs_fns.joint_pos_cos_sin  # cartpole semantic alias

ENTITY = "ttktjmt-org"
PROJECT = "mjlab"
TASK_RUN_ID_MAP: dict[str, str | list[str]] = {
    # "Mjlab-Cartpole-Balance": "cartpole-balance-v2",
    # "Mjlab-Cartpole-Swingup": "cartpole-swingup",
    "Mjlab-Lift-Cube-Yam": "ajfybu8m",
    "Mjlab-Velocity-Flat-Unitree-G1": "vel-flat-g1",
    "Mjlab-Velocity-Flat-Unitree-Go1": "vel-flat-go1-v3",
    "Mjlab-Velocity-Rough-Unitree-G1": ["mowqlkd5", "sif72y3p", "rsb8tc3g", "7veqaznf"],
    "Mjlab-Velocity-Rough-Unitree-Go1": ["basgo8hx", "ad4peite"],
}


def main():
    builder = mjswan.Builder(mt=True)
    project = builder.add_project(name="mjlab Tasks")
    export_contexts = []

    for task_id, wandb_run_id in TASK_RUN_ID_MAP.items():
        env_cfg = load_env_cfg(task_id, play=True)
        export_context = create_pt_onnx_export_context(task_id)
        export_contexts.append(export_context)
        register_custom_events(env_cfg)
        register_custom_observations(env_cfg)
        register_custom_terminations(env_cfg)
        scene = project.add_mjlab_scene(task_id, play=True)
        run_ids = wandb_run_id
        if isinstance(run_ids, str):
            run_ids = [run_ids]
        wandb_paths = [f"{ENTITY}/{PROJECT}/{rid}" for rid in run_ids]
        scene.add_policy_from_wandb(
            wandb_paths,
            task_id=task_id,
            export_context=export_context,
            observations={"policy": env_cfg.observations["actor"]},
            commands=env_cfg.commands,
            actions=env_cfg.actions,
            terminations=env_cfg.terminations,
        )

    try:
        app = builder.build()
    finally:
        for export_context in export_contexts:
            export_context.close()
    app.launch()


if __name__ == "__main__":
    main()
