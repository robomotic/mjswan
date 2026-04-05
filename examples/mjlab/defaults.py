"""mjlab Integration Example - Visualize MuJoCo scenes from all mjlab default tasks

Extracts the MuJoCo model from each mjlab default task and visualizes them
in the browser using mjswan.
"""

import os

from mjlab.tasks.registry import (  # noqa: F401 - for task registrations
    _REGISTRY,
    load_env_cfg,
)

import mjswan
from mjswan import ObsFunc, ViewerConfig, register_obs_func
from mjswan.envs.mdp import observations as obs_fns

pole_angle_cos_sin = obs_fns.joint_pos_cos_sin  # cartpole semantic alias

_OBS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "observations")
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

ENTITY = "ttktjmt-org"
PROJECT = "mjlab"
TASK_CONFIG = {
    # "Mjlab-Cartpole-Balance": {
    #     "wandb_run_id": "cartpole-balance-v2",
    #     "viewer_cfg": ViewerConfig(
    #         lookat=(0.0, 0.0, 0.4),
    #         distance=5,
    #         elevation=-20,
    #         azimuth=-90,
    #     ),
    # },
    # "Mjlab-Cartpole-Swingup": {
    #     "wandb_run_id": "cartpole-swingup",
    #     "viewer_cfg": ViewerConfig(
    #         lookat=(0.0, 0.0, 0.4),
    #         distance=5,
    #         elevation=-20,
    #         azimuth=-90,
    #     ),
    # },
    "Mjlab-Lift-Cube-Yam": {
        "wandb_run_id": "ajfybu8m",
        "viewer_cfg": ViewerConfig(
            lookat=(0.0, 0.0, 0.4),
            distance=3,
            elevation=-20,
            azimuth=30,
        ),
    },
    # "Mjlab-Velocity-Flat-Unitree-G1": {
    #     "wandb_run_id": "vel-flat-g1",
    #     "viewer_cfg": ViewerConfig(
    #         lookat=(0.0, 0.0, 0.4),
    #         distance=3,
    #         elevation=-20,
    #         azimuth=30,
    #     ),
    # },
    # "Mjlab-Velocity-Flat-Unitree-Go1": {
    #     "wandb_run_id": "vel-flat-go1-v3",
    #     "viewer_cfg": ViewerConfig(
    #         lookat=(0.0, 0.0, 0.4),
    #         distance=3,
    #         elevation=-20,
    #         azimuth=30,
    #     ),
    # },
    # "Mjlab-Velocity-Rough-Unitree-G1": {
    #     "wandb_run_id": "mowqlkd5",
    #     "viewer_cfg": ViewerConfig(
    #         lookat=(0.0, 0.0, 0.4),
    #         distance=3,
    #         elevation=-20,
    #         azimuth=30,
    #     ),
    # },
    # "Mjlab-Velocity-Rough-Unitree-Go1": {
    #     "wandb_run_id": "vel-rough-go1",
    # },
}


def main():
    builder = mjswan.Builder()
    project = builder.add_project(name="mjlab Examples")

    for task_id, config in TASK_CONFIG.items():
        env_cfg = load_env_cfg(task_id)
        scene = project.add_mjlab_scene(task_id)
        scene = scene.set_viewer_config(config["viewer_cfg"])
        policies = scene.add_policy_from_wandb(
            f"{ENTITY}/{PROJECT}/{config['wandb_run_id']}",
            task_id=task_id,
            observations={"policy": env_cfg.observations["actor"]},
            actions=env_cfg.actions,
            terminations=env_cfg.terminations,
        )
        for p in policies:
            if task_id in [
                "Mjlab-Velocity-Flat-Unitree-G1",
                "Mjlab-Velocity-Flat-Unitree-Go1",
            ]:
                p.add_velocity_command(name="twist")

    app = builder.build()
    app.launch()


if __name__ == "__main__":
    main()
