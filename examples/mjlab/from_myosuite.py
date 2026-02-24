"""MJLab Integration Example with mjlab_myosuite

This example demonstrates how to build mjswan projects using mjlab_myosuite setups.
"""

import os

os.environ["MUJOCO_GL"] = "disable"

import mjlab_myosuite  # noqa: F401, E402
from mjlab.scene import Scene  # noqa: E402
from mjlab.tasks.registry import list_tasks, load_env_cfg  # noqa: E402

import mjswan  # noqa: E402


def main():
    builder = mjswan.Builder()
    project = builder.add_project(name="mjlab Examples")

    for task_id in list_tasks():
        env_cfg = load_env_cfg(task_id)
        env_cfg.scene.num_envs = 1
        scene = Scene(env_cfg.scene, device="cpu")
        project.add_scene(spec=scene.spec, name=task_id)

    app = builder.build()
    app.launch()


if __name__ == "__main__":
    main()
