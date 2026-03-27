"""mjswan Demo Application

This is a demo application showcasing the usage of mjswan.
The demo app is hosted on GitHub Pages: https://ttktjmt.github.io/mjswan/
"""

import os
import posixpath
from pathlib import Path

import gymnasium.logger as gym_logger
import mujoco
import onnx
from mujoco_playground import registry

# Suppress gymnasium logger output from myosuite
_prev_gym_level = gym_logger.min_level
gym_logger.set_level(gym_logger.DISABLED)

from myosuite import gym_registry_specs  # noqa: E402
from myosuite.envs.myo import myochallenge  # noqa: E402, F401 - for env registration

gym_logger.set_level(_prev_gym_level)

from robot_descriptions._descriptions import DESCRIPTIONS  # noqa: E402

import mjswan  # noqa: E402
from mjswan.envs.mdp import observations as obs_fns  # noqa: E402
from mjswan.managers.observation_manager import (  # noqa: E402
    ObservationGroupCfg,
    ObservationTermCfg,
)


def _fix_unitree_mujoco_macos() -> None:
    """Pre-fix the unitree_mujoco cache on macOS to avoid case-sensitivity errors.

    On macOS (case-insensitive filesystem), robot_descriptions fails to checkout
    the unitree_mujoco repo because git history contains a rename from
    terrain.STL -> terrain.stl, which macOS treats as the same file.

    Fix: clone with --no-checkout so no files exist in the working tree before
    the target commit is checked out, and set core.ignorecase=false so git
    handles the case-rename correctly.
    """
    import platform
    import shutil
    import subprocess

    if platform.system() != "Darwin":
        return

    cache_dir = Path.home() / ".cache/robot_descriptions/unitree_mujoco"

    if cache_dir.exists():
        result = subprocess.run(
            ["git", "config", "core.ignorecase"],
            cwd=cache_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.stdout.strip() == "false":
            return  # Already correctly configured
        shutil.rmtree(cache_dir)

    print("Preparing unitree_mujoco cache for macOS (one-time setup)...")
    subprocess.run(
        [
            "git",
            "clone",
            "--no-checkout",
            "https://github.com/unitreerobotics/unitree_mujoco.git",
            str(cache_dir),
        ],
        check=True,
    )
    subprocess.run(
        ["git", "config", "core.ignorecase", "false"],
        cwd=cache_dir,
        check=True,
    )


def setup_builder() -> mjswan.Builder:
    """Set up and return the builder with all demo projects configured.

    This function creates the builder and adds all projects, scenes, and policies
    but does not build or launch the application. Useful for testing.

    Returns:
        Configured Builder instance ready to be built.
    """
    _fix_unitree_mujoco_macos()
    # Ensure asset-relative paths resolve regardless of current working directory.
    os.chdir(Path(__file__).resolve().parent)
    base_path = os.getenv("MJSWAN_BASE_PATH", "/")
    builder = mjswan.Builder(base_path=base_path, gtm_id="GTM-W79HQ38W")

    # =======================
    # 1. mjswan Demo Project
    # =======================

    demo_project = builder.add_project(name="mjswan Demo")

    # 1.A. Unitree G1
    g1_scene = demo_project.add_scene(
        spec=mujoco.MjSpec.from_file("assets/unitree_g1/scene.xml"),
        name="G1",
    ).set_viewer_config(
        mjswan.ViewerConfig(
            lookat=(0.0, 0.0, 0.7),
            distance=4.3,
            elevation=-33.0,
            azimuth=-34.0,
            origin_type=mjswan.ViewerConfig.OriginType.ASSET_BODY,
            body_name="torso_link",
        )
    )
    g1_scene.add_splat(
        name="Street",
        source="assets/unitree_g1/street.spz",
        scale=3.275,
        z_offset=0.708,
        yaw=40,
        control=True,
    )
    g1_loco_policy = g1_scene.add_policy(
        policy=onnx.load("assets/unitree_g1/locomotion.onnx"),
        name="Locomotion",
        config_path="assets/unitree_g1/locomotion.json",
        observations={
            "policy": ObservationGroupCfg(
                terms={
                    "base_lin_vel": ObservationTermCfg(func=obs_fns.base_lin_vel),
                    "base_ang_vel": ObservationTermCfg(func=obs_fns.base_ang_vel),
                    "projected_gravity": ObservationTermCfg(
                        func=obs_fns.projected_gravity
                    ),
                    "joint_pos": ObservationTermCfg(
                        func=obs_fns.joint_pos_rel, params={"pos_steps": [0]}
                    ),
                    "joint_vel": ObservationTermCfg(func=obs_fns.joint_vel_rel),
                    "last_action": ObservationTermCfg(func=obs_fns.last_action),
                    "velocity_cmd": ObservationTermCfg(
                        func=obs_fns.simple_velocity_command
                    ),
                }
            )
        },
    )
    g1_loco_policy.add_velocity_command(
        lin_vel_x=(-1.5, 1.5),
        lin_vel_y=(-0.5, 0.5),
        default_lin_vel_x=0.5,
        default_lin_vel_y=0.0,
    )
    g1_scene.add_policy(
        policy=onnx.load("assets/unitree_g1/balance.onnx"),
        name="Balance",
        config_path="assets/unitree_g1/balance.json",
        observations={
            "observation": ObservationGroupCfg(
                terms={
                    "base_ang_vel": ObservationTermCfg(
                        func=obs_fns.base_ang_vel, history_length=1
                    ),
                    "projected_gravity": ObservationTermCfg(
                        func=obs_fns.projected_gravity_isaac,
                        history_length=1,
                        params={"gravity": [0, 0, -1.0]},
                    ),
                    "joint_pos": ObservationTermCfg(
                        func=obs_fns.joint_positions_isaac, history_length=1
                    ),
                    "joint_vel": ObservationTermCfg(
                        func=obs_fns.joint_vel_rel,
                        params={"joint_names": "isaac"},
                        history_length=1,
                    ),
                    "prev_actions": ObservationTermCfg(func=obs_fns.previous_actions),
                }
            )
        },
    )

    # 1.B. Unitree Go2
    go2_scene = demo_project.add_scene(
        spec=mujoco.MjSpec.from_file("assets/unitree_go2/scene.xml"),
        name="Go2",
    ).set_viewer_config(
        mjswan.ViewerConfig(
            lookat=(0.0, 0.0, 0.7),
            distance=3.8,
            elevation=-20.0,
            azimuth=34.0,
            origin_type=mjswan.ViewerConfig.OriginType.ASSET_BODY,
            body_name="base",
        )
    )
    go2_scene.add_policy(
        policy=onnx.load("assets/unitree_go2/facet.onnx"),
        name="Facet",
        config_path="assets/unitree_go2/facet.json",
        observations={
            "policy": ObservationGroupCfg(
                terms={
                    "projected_gravity": ObservationTermCfg(
                        func=obs_fns.projected_gravity_isaac, history_length=3
                    ),
                    "joint_pos": ObservationTermCfg(
                        func=obs_fns.joint_positions_isaac,
                        history_length=3,
                        params={"subtract_default": False},
                    ),
                    "joint_vel": ObservationTermCfg(
                        func=obs_fns.joint_vel_rel,
                        params={"joint_names": "isaac"},
                        history_length=3,
                    ),
                    "prev_actions": ObservationTermCfg(
                        func=obs_fns.previous_actions,
                        history_length=3,
                        params={"transpose": True},
                    ),
                }
            ),
            "command": ObservationGroupCfg(
                terms={
                    "impedance_cmd": ObservationTermCfg(func=obs_fns.impedance_command),
                }
            ),
        },
    ).add_velocity_command()

    _go2_velocity_obs = {
        "policy": ObservationGroupCfg(
            terms={
                "projected_gravity": ObservationTermCfg(
                    func=obs_fns.projected_gravity_isaac, history_length=3
                ),
                "joint_pos": ObservationTermCfg(
                    func=obs_fns.joint_positions_isaac, history_length=3
                ),
                "joint_vel": ObservationTermCfg(
                    func=obs_fns.joint_vel_rel,
                    params={"joint_names": "isaac"},
                    history_length=3,
                ),
                "prev_actions": ObservationTermCfg(
                    func=obs_fns.previous_actions,
                    history_length=3,
                    params={"transpose": True},
                ),
            }
        ),
        "command_": ObservationGroupCfg(
            terms={
                "velocity_cmd": ObservationTermCfg(
                    func=obs_fns.velocity_command_with_oscillators
                ),
            }
        ),
    }
    go2_scene.add_policy(
        policy=onnx.load("assets/unitree_go2/vanilla.onnx"),
        name="Vanilla",
        config_path="assets/unitree_go2/vanilla.json",
        observations=_go2_velocity_obs,
    ).add_velocity_command()
    go2_scene.add_policy(
        policy=onnx.load("assets/unitree_go2/robust.onnx"),
        name="Robust",
        config_path="assets/unitree_go2/robust.json",
        observations=_go2_velocity_obs,
    ).add_velocity_command()

    # 1.C. Unitree Go1
    go1_scene = demo_project.add_scene(
        spec=mujoco.MjSpec.from_file("assets/unitree_go1/go1.xml"),
        name="Go1",
    ).set_viewer_config(
        mjswan.ViewerConfig(
            lookat=(0.0, 0.0, 0.2),
            distance=3.1,
            elevation=-25.0,
            azimuth=-45.0,
            origin_type=mjswan.ViewerConfig.OriginType.ASSET_BODY,
            body_name="trunk",
        )
    )
    # NOTE: himloco uses an interleaved history format (dict with "interleaved": true)
    # that is not yet expressible via ObservationGroupCfg. obs_config remains in himloco.json.
    go1_scene.add_policy(
        policy=onnx.load("assets/unitree_go1/himloco.onnx"),
        name="HiMLoco",
        config_path="assets/unitree_go1/himloco.json",
    ).add_velocity_command()
    go1_scene.add_policy(
        policy=onnx.load("assets/unitree_go1/decap.onnx"),
        name="Decap",
        config_path="assets/unitree_go1/decap.json",
        observations={
            "obs_history": ObservationGroupCfg(
                terms={
                    "projected_gravity": ObservationTermCfg(
                        func=obs_fns.projected_gravity_isaac, history_length=1
                    ),
                    "velocity_cmd": ObservationTermCfg(
                        func=obs_fns.simple_velocity_command,
                        scale=(2.0, 2.0, 0.25),
                    ),
                    "joint_pos": ObservationTermCfg(
                        func=obs_fns.joint_positions_isaac, history_length=1
                    ),
                    "joint_vel": ObservationTermCfg(
                        func=obs_fns.joint_vel_rel,
                        params={"joint_names": "isaac"},
                        scale=0.05,
                        history_length=1,
                    ),
                    "prev_actions": ObservationTermCfg(func=obs_fns.previous_actions),
                }
            )
        },
    ).add_velocity_command()

    # ==============================
    # 2. Robot Descriptions Project
    # ==============================

    robotdesc_project = builder.add_project(name="Robot Descriptions", id="robotdesc")

    # ANYmal C Velocity from https://github.com/mujocolab/anymal_c_velocity
    anymal_c_scene = robotdesc_project.add_scene(
        name="ANYmal C Velocity",
        spec=mujoco.MjSpec.from_zip("assets/anymal_c_velocity/scene.mjz"),
    )
    anymal_c_scene.add_policy(
        name="velocity 3000 iters",
        policy=onnx.load(
            "assets/anymal_c_velocity/Mjlab-Velocity-Flat-Anymal-C.3000.onnx"
        ),
        config_path="assets/anymal_c_velocity/Mjlab-Velocity-Flat-Anymal-C.3000.json",
        observations={
            "obs": ObservationGroupCfg(
                terms={
                    "base_lin_vel": ObservationTermCfg(func=obs_fns.base_lin_vel),
                    "base_ang_vel": ObservationTermCfg(func=obs_fns.base_ang_vel),
                    "projected_gravity": ObservationTermCfg(
                        func=obs_fns.projected_gravity
                    ),
                    "joint_pos": ObservationTermCfg(
                        func=obs_fns.joint_pos_rel, params={"pos_steps": [0]}
                    ),
                    "joint_vel": ObservationTermCfg(func=obs_fns.joint_vel_rel),
                    "last_action": ObservationTermCfg(func=obs_fns.last_action),
                    "velocity_cmd": ObservationTermCfg(
                        func=obs_fns.simple_velocity_command
                    ),
                }
            )
        },
    ).add_velocity_command(
        lin_vel_x=(-1.0, 1.0),
        lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-0.5, 0.5),
        default_lin_vel_x=0.5,
        default_lin_vel_y=0.0,
        default_ang_vel_z=0.0,
    )

    def _rd_spec(module_name: str) -> mujoco.MjSpec:
        from importlib import import_module

        mjcf_path = Path(import_module(f"robot_descriptions.{module_name}").MJCF_PATH)
        # Prefer scene.xml (floor + lights) over the robot-only MJCF when available.
        scene_path = mjcf_path.parent / "scene.xml"
        return mujoco.MjSpec.from_file(
            str(scene_path if scene_path.exists() else mjcf_path)
        )

    for module, desc in DESCRIPTIONS.items():
        if desc.has_mjcf:
            scene_name = module.replace("_mj_description", "")
            scene_name = " ".join([word.capitalize() for word in scene_name.split("_")])

            robotdesc_project.add_scene(name=scene_name, spec=_rd_spec(module))

    # =============================
    # 3. MuJoCo Playground Project
    # =============================

    playground_project = builder.add_project(name="MuJoCo Playground", id="playground")

    for env_name in registry.ALL_ENVS:
        if "Sparse" in env_name:
            continue

        env = registry.load(env_name)
        xml_content = open(env.xml_path).read()
        spec = mujoco.MjSpec.from_string(xml_content, env.model_assets)

        # model_assets is consumed at parse time but not stored in spec.assets.
        # Remap basename keys (as in env.model_assets) to the effective paths
        # that spec.to_xml() looks up: dir/file (or just file when dir is empty).
        mesh_dir = spec.meshdir or ""
        tex_dir = spec.texturedir or ""

        def _add(directory: str, filename: str) -> None:
            if not filename:
                return
            key = posixpath.join(directory, filename) if directory else filename
            basename = os.path.basename(key)
            if basename in env.model_assets:
                spec.assets[key] = env.model_assets[basename]

        for mesh in spec.meshes:
            _add(mesh_dir, mesh.file)
        for texture in spec.textures:
            _add(tex_dir, texture.file)
            for cf in texture.cubefiles:
                _add(tex_dir, cf)
        for hfield in spec.hfields:
            _add("", hfield.file)

        playground_project.add_scene(name=env_name, spec=spec)

    # ====================
    # 4. MyoSuite Project
    # ====================

    myosuite_project = builder.add_project(name="MyoSuite", id="myosuite")

    registry_specs = gym_registry_specs()

    # (display_name, lookat, distance, elevation, azimuth)
    target_envs = {
        "myoChallengeDieReorientP2-v0": (
            "mc22 Die Reorient",
            (-0.1, -0.5, 1.4),
            1.3,
            -9.0,
            -61.0,
        ),
        "myoChallengeBaodingP2-v1": (
            "mc22 Baoding",
            (-0.1, -0.5, 1.4),
            1.3,
            -9.0,
            -61.0,
        ),
        "myoChallengeRelocateP2-v0": (
            "mc23 Relocate",
            (0.0, -0.1, 1.4),
            1.7,
            -7.0,
            -90.0,
        ),
        "myoChallengeChaseTagP2-v0": (
            "mc23 Chase Tag",
            (0.0, 0.0, 1.4),
            10.0,
            -15.0,
            -62.0,
        ),
        "myoChallengeBimanual-v0": (
            "mc24 Bimanual",
            (-0.1, -0.5, 1.4),
            1.3,
            -9.0,
            -61.0,
        ),
        "myoChallengeOslRunRandom-v0": (
            "mc24 OSL Run",
            (0.0, 0.0, 1.4),
            10.0,
            -15.0,
            -62.0,
        ),
        "myoChallengeTableTennisP2-v0": (
            "mc25 Table Tennis",
            (0.0, -1.0, 1.4),
            3.3,
            -11.0,
            -129.0,
        ),
        "myoChallengeSoccerP2-v0": (
            "mc25 Soccer",
            (0.0, -3.0, 2.0),
            14.7,
            -16.0,
            -172.0,
        ),
    }

    for env_name, (
        display_name,
        lookat,
        distance,
        elevation,
        azimuth,
    ) in target_envs.items():
        model_path = registry_specs[env_name].kwargs["model_path"]
        mjspec = mujoco.MjSpec.from_file(model_path)
        myosuite_project.add_scene(name=display_name, spec=mjspec).set_viewer_config(
            mjswan.ViewerConfig(
                lookat=lookat,
                distance=distance,
                elevation=elevation,
                azimuth=azimuth,
                origin_type=mjswan.ViewerConfig.OriginType.WORLD,
            )
        )

    return builder


def _copy_licenses(output_dir: Path) -> None:
    """Copy LICENSE and NOTICE files into the built output.

    - robot_descriptions (robotdesc): copies per scene from each repo's REPOSITORY_PATH.
    - myosuite / mujoco_playground: copies to the project root from the dist-info licenses/.
    """
    import importlib.metadata
    import shutil
    from importlib import import_module

    # Per-scene for robot_descriptions (robotdesc project)
    robotdesc_assets = output_dir / "robotdesc" / "assets"
    if robotdesc_assets.exists():
        for module, desc in DESCRIPTIONS.items():
            if not desc.has_mjcf:
                continue
            scene_id = module.replace("_mj_description", "")
            scene_dir = robotdesc_assets / scene_id
            if not scene_dir.exists():
                continue
            mod = import_module(f"robot_descriptions.{module}")
            if not hasattr(mod, "REPOSITORY_PATH"):
                continue
            for fname in ["LICENSE", "NOTICE"]:
                src = Path(mod.REPOSITORY_PATH) / fname
                if src.exists():
                    shutil.copy2(src, scene_dir / fname)

    # Project-level for myosuite and mujoco_playground
    for project_id, pkg_name in [
        ("myosuite", "myosuite"),
        ("playground", "mujoco_playground"),
    ]:
        project_dir = output_dir / project_id
        if not project_dir.exists():
            continue
        try:
            dist = importlib.metadata.Distribution.from_name(pkg_name)
        except importlib.metadata.PackageNotFoundError:
            continue
        for fname in ["LICENSE", "NOTICE"]:
            src = Path(str(dist.locate_file(f"licenses/{fname}")))
            if src.exists():
                shutil.copy2(src, project_dir / fname)


def main():
    """Main entry point for the demo application.

    Environment variables:
        MJSWAN_BASE_PATH: Base path for deployment (default: '/')
        MJSWAN_NO_LAUNCH: Set to '1' to skip launching the browser
        MJSWAN_SKIP_BUILD: Set to '1' to skip build and launch the pre-built app
    """
    dist_dir = Path(__file__).resolve().parent / "dist"
    if os.getenv("MJSWAN_SKIP_BUILD") == "1":
        app = mjswan.mjswanApp(dist_dir)
    else:
        builder = setup_builder()
        app = builder.build()
        _copy_licenses(dist_dir)
    if os.getenv("MJSWAN_NO_LAUNCH") != "1":
        app.launch()


if __name__ == "__main__":
    main()
