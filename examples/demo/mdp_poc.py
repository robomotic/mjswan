"""POC: mjlab-compatible MDP observation config via mjswan.

Demonstrates the Python -> JS mapping proposed in issue #32.

The observation configuration uses the exact same API as mjlab::

    # mjlab (training)
    from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
    from mjlab.envs.mdp import observations as obs_fns

    # mjswan (browser deployment) — identical API
    from mjswan.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
    from mjswan.envs.mdp import observations as obs_fns

At build time, observation groups are serialized into obs_config in the
policy JSON. The browser-side PolicyRunner resolves each entry against
the existing TypeScript observation registry.

Usage:
    uv run python examples/demo/mdp_poc.py
"""

import mujoco
import onnx

import mjswan
from mjswan.envs.mdp import observations as obs_fns
from mjswan.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg

go2_xml = "examples/demo/assets/unitree_go2/scene.xml"
go2_onnx = "examples/demo/assets/unitree_go2/vanilla.onnx"
go2_cfg = "examples/demo/assets/unitree_go2/vanilla.json"

spec = mujoco.MjSpec.from_file(go2_xml)
model = onnx.load(go2_onnx)

builder = mjswan.Builder()
project = builder.add_project(name="Go2 MDP POC")
scene = project.add_scene(spec=spec, name="Velocity Flat")

scene.add_policy(
    name="vanilla",
    policy=model,
    config_path=go2_cfg,
    observations={
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
            },
        ),
        "command_": ObservationGroupCfg(
            terms={
                "velocity_cmd": ObservationTermCfg(
                    func=obs_fns.velocity_command_with_oscillators
                ),
            }
        ),
    },
).add_velocity_command()

app = builder.build()
app.launch()
