"""Simple mjswan Demo

A basic example demonstrating how to use mjswan to create a viewer application
with multiple robot scenes, including a dual-arm SO101 leader/follower setup.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field

import mujoco
import onnx
from dataclasses import dataclass, field, asdict
from mjlab.envs.mdp import observations as obs_fns
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg

import mjswan
from mjswan.envs.mdp.actions import JointPositionActionCfg, JointEffortActionCfg

@dataclass
class HoldOnCommandCfg:
    joint_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"joint_names": self.joint_names}

mjswan.register_command_term(
    "HoldOnCommandCfg",
    mjswan.CommandTermSpec(
        ts_name="HoldOnCommand",
        ts_src=os.path.join(os.path.dirname(__file__), "HoldOnCommand.ts"),
        serializer=asdict,
    )
)

# G1 humanoid: per-joint action scale, stiffness, damping (keyed by joint name)
_G1_SCALE = {
    "left_hip_pitch_joint": 0.5475464629911068,
    "left_hip_roll_joint": 0.35066146637882434,
    "left_hip_yaw_joint": 0.5475464629911068,
    "left_knee_joint": 0.35066146637882434,
    "left_ankle_pitch_joint": 0.43857731392336724,
    "left_ankle_roll_joint": 0.43857731392336724,
    "right_hip_pitch_joint": 0.5475464629911068,
    "right_hip_roll_joint": 0.35066146637882434,
    "right_hip_yaw_joint": 0.5475464629911068,
    "right_knee_joint": 0.35066146637882434,
    "right_ankle_pitch_joint": 0.43857731392336724,
    "right_ankle_roll_joint": 0.43857731392336724,
    "waist_yaw_joint": 0.5475464629911068,
    "waist_roll_joint": 0.43857731392336724,
    "waist_pitch_joint": 0.43857731392336724,
    "left_shoulder_pitch_joint": 0.43857731392336724,
    "left_shoulder_roll_joint": 0.43857731392336724,
    "left_shoulder_yaw_joint": 0.43857731392336724,
    "left_elbow_joint": 0.43857731392336724,
    "left_wrist_roll_joint": 0.43857731392336724,
    "left_wrist_pitch_joint": 0.07450087032950714,
    "left_wrist_yaw_joint": 0.07450087032950714,
    "right_shoulder_pitch_joint": 0.43857731392336724,
    "right_shoulder_roll_joint": 0.43857731392336724,
    "right_shoulder_yaw_joint": 0.43857731392336724,
    "right_elbow_joint": 0.43857731392336724,
    "right_wrist_roll_joint": 0.43857731392336724,
    "right_wrist_pitch_joint": 0.07450087032950714,
    "right_wrist_yaw_joint": 0.07450087032950714,
}
_G1_STIFFNESS = {
    "left_hip_pitch_joint": 40.17923863450712,
    "left_hip_roll_joint": 99.09842777666111,
    "left_hip_yaw_joint": 40.17923863450712,
    "left_knee_joint": 99.09842777666111,
    "left_ankle_pitch_joint": 28.50124619574858,
    "left_ankle_roll_joint": 28.50124619574858,
    "right_hip_pitch_joint": 40.17923863450712,
    "right_hip_roll_joint": 99.09842777666111,
    "right_hip_yaw_joint": 40.17923863450712,
    "right_knee_joint": 99.09842777666111,
    "right_ankle_pitch_joint": 28.50124619574858,
    "right_ankle_roll_joint": 28.50124619574858,
    "waist_yaw_joint": 40.17923863450712,
    "waist_roll_joint": 28.50124619574858,
    "waist_pitch_joint": 28.50124619574858,
    "left_shoulder_pitch_joint": 14.25062309787429,
    "left_shoulder_roll_joint": 14.25062309787429,
    "left_shoulder_yaw_joint": 14.25062309787429,
    "left_elbow_joint": 14.25062309787429,
    "left_wrist_roll_joint": 14.25062309787429,
    "left_wrist_pitch_joint": 16.77832748089279,
    "left_wrist_yaw_joint": 16.77832748089279,
    "right_shoulder_pitch_joint": 14.25062309787429,
    "right_shoulder_roll_joint": 14.25062309787429,
    "right_shoulder_yaw_joint": 14.25062309787429,
    "right_elbow_joint": 14.25062309787429,
    "right_wrist_roll_joint": 14.25062309787429,
    "right_wrist_pitch_joint": 16.77832748089279,
    "right_wrist_yaw_joint": 16.77832748089279,
}
_G1_DAMPING = {
    "left_hip_pitch_joint": 2.557889775413375,
    "left_hip_roll_joint": 6.308801853496639,
    "left_hip_yaw_joint": 2.557889775413375,
    "left_knee_joint": 6.308801853496639,
    "left_ankle_pitch_joint": 1.814445686584846,
    "left_ankle_roll_joint": 1.814445686584846,
    "right_hip_pitch_joint": 2.557889775413375,
    "right_hip_roll_joint": 6.308801853496639,
    "right_hip_yaw_joint": 2.557889775413375,
    "right_knee_joint": 6.308801853496639,
    "right_ankle_pitch_joint": 1.814445686584846,
    "right_ankle_roll_joint": 1.814445686584846,
    "waist_yaw_joint": 2.557889775413375,
    "waist_roll_joint": 1.814445686584846,
    "waist_pitch_joint": 1.814445686584846,
    "left_shoulder_pitch_joint": 0.907222843292423,
    "left_shoulder_roll_joint": 0.907222843292423,
    "left_shoulder_yaw_joint": 0.907222843292423,
    "left_elbow_joint": 0.907222843292423,
    "left_wrist_roll_joint": 0.907222843292423,
    "left_wrist_pitch_joint": 1.06814150219,
    "left_wrist_yaw_joint": 1.06814150219,
    "right_shoulder_pitch_joint": 0.907222843292423,
    "right_shoulder_roll_joint": 0.907222843292423,
    "right_shoulder_yaw_joint": 0.907222843292423,
    "right_elbow_joint": 0.907222843292423,
    "right_wrist_roll_joint": 0.907222843292423,
    "right_wrist_pitch_joint": 1.06814150219,
    "right_wrist_yaw_joint": 1.06814150219,
}
_G1_OFFSET = {
    "left_shoulder_pitch_joint": 0.5,
    "right_shoulder_pitch_joint": -0.5,
    "left_wrist_roll_joint": 2.0,
    "right_wrist_roll_joint": -2.0,
}

# Per-joint stiffness (kp) derived from DC motor model for each variant.
# C044 (7.4V, 1:191): kp = 998.22 * 191/345 = 552.7 Nm/rad
# C001 (7.4V, 1:345): kp = 998.22 * 345/345 = 998.22 Nm/rad (same base motor, full ratio)
# C046 (7.4V, 1:147): kp = 998.22 * 147/345 = 425.4 Nm/rad
# kv = 2.731 Nm·s/rad is gear-ratio and voltage independent (Km*Kb/R).
_SO101_LEADER_STIFFNESS = [552.7, 998.22, 552.7, 425.4, 425.4, 425.4]  # shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
_SO101_LEADER_DAMPING = [2.731] * 6

_SO101_JOINT_SPECS = (
    ("shoulder_pan", "Shoulder Pan", (-1.91986, 1.91986), 0.0),
    ("shoulder_lift", "Shoulder Lift", (-1.74533, 1.74533), 0.25),
    ("elbow_flex", "Elbow Flex", (-1.69, 1.69), 0.65),
    ("wrist_flex", "Wrist Flex", (-1.65806, 1.65806), -0.35),
    ("wrist_roll", "Wrist Roll", (-2.74385, 2.84121), 0.0),
    ("gripper", "Gripper", (-0.17453, 1.74533), 0.6),
)
_SO101_LEADER_JOINT_NAMES = [
    f"leader/{name}" for name, _, _, _ in _SO101_JOINT_SPECS
]
_SO101_FOLLOWER_JOINT_NAMES = [
    f"follower/{name}" for name, _, _, _ in _SO101_JOINT_SPECS
]
_SO101_POLICY_JOINT_NAMES = [
    *_SO101_LEADER_JOINT_NAMES,
    *_SO101_FOLLOWER_JOINT_NAMES,
]


def _build_gravity_mode_policy() -> onnx.ModelProto:
    """Build ONNX policy for gravity/compliant mode with wrist_roll force hook.

    Input:  [1, 7]  — first 6 = leader actual joint positions, last 1 = wrist_hook slider
    Output: [1, 12] — first 6 = leader target deltas (zero except wrist_roll hook),
                      last 6  = follower target = leader actual qpos (tracking)
    """
    import numpy as np

    input_info = onnx.helper.make_tensor_value_info(
        "policy", onnx.TensorProto.FLOAT, [1, 7]
    )
    output_info = onnx.helper.make_tensor_value_info(
        "action", onnx.TensorProto.FLOAT, [1, 12]
    )

    # Slice leader qpos: policy[:, 0:6]
    starts_qpos = onnx.numpy_helper.from_array(
        np.array([0, 0], dtype=np.int64), name="starts_qpos"
    )
    ends_qpos = onnx.numpy_helper.from_array(
        np.array([1, 6], dtype=np.int64), name="ends_qpos"
    )
    node_slice_qpos = onnx.helper.make_node(
        "Slice",
        inputs=["policy", "starts_qpos", "ends_qpos"],
        outputs=["leader_qpos"],
    )

    # Slice wrist_hook value: policy[:, 6:7]
    starts_hook = onnx.numpy_helper.from_array(
        np.array([0, 6], dtype=np.int64), name="starts_hook"
    )
    ends_hook = onnx.numpy_helper.from_array(
        np.array([1, 7], dtype=np.int64), name="ends_hook"
    )
    node_slice_hook = onnx.helper.make_node(
        "Slice",
        inputs=["policy", "starts_hook", "ends_hook"],
        outputs=["wrist_hook_val"],
    )

    # Constant zeros: shape [1, 4] and [1, 1] for padding
    zeros4 = onnx.numpy_helper.from_array(
        np.zeros((1, 4), dtype=np.float32), name="zeros4"
    )
    zeros1 = onnx.numpy_helper.from_array(
        np.zeros((1, 1), dtype=np.float32), name="zeros1"
    )

    # leader_acts = [zeros4, wrist_hook_val, zeros1]  -> shape [1, 6]
    # Index layout: 0:shoulder_pan, 1:shoulder_lift, 2:elbow_flex,
    #               3:wrist_flex, 4:wrist_roll (hook), 5:gripper
    node_concat_leader = onnx.helper.make_node(
        "Concat",
        inputs=["zeros4", "wrist_hook_val", "zeros1"],
        outputs=["leader_acts"],
        axis=1,
    )

    # output = [leader_acts, leader_qpos]  -> shape [1, 12]
    node_concat_out = onnx.helper.make_node(
        "Concat",
        inputs=["leader_acts", "leader_qpos"],
        outputs=["action"],
        axis=1,
    )

    graph = onnx.helper.make_graph(
        nodes=[node_slice_qpos, node_slice_hook,
               node_concat_leader, node_concat_out],
        name="so101_gravity_mode",
        inputs=[input_info],
        outputs=[output_info],
        initializer=[starts_qpos, ends_qpos, starts_hook, ends_hook, zeros4, zeros1],
    )
    model = onnx.helper.make_model(
        graph,
        producer_name="mjswan",
        opset_imports=[onnx.helper.make_operatorsetid("", 13)],
    )
    model.ir_version = min(model.ir_version, 10)
    onnx.checker.check_model(model)
    return model


def _build_so101_mirror_policy() -> onnx.ModelProto:
    """Create a tiny ONNX policy that mirrors leader commands to both arms."""
    input_size = len(_SO101_JOINT_SPECS)
    output_size = len(_SO101_POLICY_JOINT_NAMES)
    input_info = onnx.helper.make_tensor_value_info(
        "policy", onnx.TensorProto.FLOAT, [1, input_size]
    )
    output_info = onnx.helper.make_tensor_value_info(
        "action", onnx.TensorProto.FLOAT, [1, output_size]
    )
    graph = onnx.helper.make_graph(
        nodes=[
            onnx.helper.make_node(
                "Concat",
                inputs=["policy", "policy"],
                outputs=["action"],
                axis=1,
            )
        ],
        name="so101_leader_follower_mirror",
        inputs=[input_info],
        outputs=[output_info],
    )
    model = onnx.helper.make_model(graph, producer_name="mjswan")
    model.opset_import[0].version = 13
    model.ir_version = min(model.ir_version, 10)
    onnx.checker.check_model(model)
    return model


def _build_hold_mode_policy() -> onnx.ModelProto:
    """Build ONNX policy for Hold On mode.

    Input:  [1, 19] — 6 leader qpos, 6 leader qvel, 7 hold_cmd (6 held_qpos + 1 is_held)
    Output: [1, 12] — 6 leader torque (PD if is_held else 0), 6 follower target
    """
    import numpy as np

    input_info = onnx.helper.make_tensor_value_info(
        "policy", onnx.TensorProto.FLOAT, [1, 19]
    )
    output_info = onnx.helper.make_tensor_value_info(
        "action", onnx.TensorProto.FLOAT, [1, 12]
    )

    def slice_node(name: str, start: int, end: int, out: str) -> tuple[list, list]:
        s = onnx.numpy_helper.from_array(np.array([start], dtype=np.int64), name=f"{name}_start")
        e = onnx.numpy_helper.from_array(np.array([end], dtype=np.int64), name=f"{name}_end")
        a = onnx.numpy_helper.from_array(np.array([1], dtype=np.int64), name=f"{name}_axes")
        n = onnx.helper.make_node(
            "Slice",
            inputs=["policy", f"{name}_start", f"{name}_end", f"{name}_axes"],
            outputs=[out],
        )
        return [n], [s, e, a]

    nodes, inits = [], []
    
    n, i = slice_node("qpos", 0, 6, "leader_qpos")
    nodes.extend(n); inits.extend(i)
    
    n, i = slice_node("qvel", 6, 12, "leader_qvel")
    nodes.extend(n); inits.extend(i)
    
    n, i = slice_node("hold_qpos", 12, 18, "held_qpos")
    nodes.extend(n); inits.extend(i)

    n, i = slice_node("is_held", 18, 19, "is_held")
    nodes.extend(n); inits.extend(i)

    kp = onnx.numpy_helper.from_array(np.array(_SO101_LEADER_STIFFNESS, dtype=np.float32), name="kp")
    kd = onnx.numpy_helper.from_array(np.array(_SO101_LEADER_DAMPING, dtype=np.float32), name="kd")
    inits.extend([kp, kd])

    # Error = held_qpos - leader_qpos
    nodes.append(onnx.helper.make_node("Sub", inputs=["held_qpos", "leader_qpos"], outputs=["err_p"]))
    # P = kp * Error
    nodes.append(onnx.helper.make_node("Mul", inputs=["err_p", "kp"], outputs=["P_term"]))
    # D = kd * leader_qvel
    nodes.append(onnx.helper.make_node("Mul", inputs=["leader_qvel", "kd"], outputs=["D_term"]))
    # torque = P - D
    nodes.append(onnx.helper.make_node("Sub", inputs=["P_term", "D_term"], outputs=["pd_torque"]))
    
    # final_torque = pd_torque * is_held (broadcast to [1, 6])
    nodes.append(onnx.helper.make_node("Mul", inputs=["pd_torque", "is_held"], outputs=["leader_torque"]))
    
    # action = Concat(leader_torque, leader_qpos)
    nodes.append(onnx.helper.make_node("Concat", inputs=["leader_torque", "leader_qpos"], outputs=["action"], axis=1))

    graph = onnx.helper.make_graph(
        nodes=nodes,
        name="so101_hold_mode",
        inputs=[input_info],
        outputs=[output_info],
        initializer=inits,
    )
    model = onnx.helper.make_model(
        graph,
        producer_name="mjswan",
        opset_imports=[onnx.helper.make_operatorsetid("", 13)],
    )
    model.ir_version = min(model.ir_version, 10)
    onnx.checker.check_model(model)
    return model
    """Create a tiny ONNX policy that mirrors leader commands to both arms."""
    input_size = len(_SO101_JOINT_SPECS)
    output_size = len(_SO101_POLICY_JOINT_NAMES)
    input_info = onnx.helper.make_tensor_value_info(
        "policy", onnx.TensorProto.FLOAT, [1, input_size]
    )
    output_info = onnx.helper.make_tensor_value_info(
        "action", onnx.TensorProto.FLOAT, [1, output_size]
    )
    graph = onnx.helper.make_graph(
        nodes=[
            onnx.helper.make_node(
                "Concat",
                inputs=["policy", "policy"],
                outputs=["action"],
                axis=1,
            )
        ],
        name="so101_leader_follower_mirror",
        inputs=[input_info],
        outputs=[output_info],
    )
    model = onnx.helper.make_model(
        graph,
        producer_name="mjswan",
        opset_imports=[onnx.helper.make_operatorsetid("", 13)],
    )
    model.ir_version = min(model.ir_version, 10)
    onnx.checker.check_model(model)
    return model


def _add_so101_scene(project) -> None:
    """Add a dual-arm SO101 scene where the follower copies the leader joints."""
    so101_scene = project.add_scene(
        spec=mujoco.MjSpec.from_file("assets/so101_dual_arm/scene.xml"),
        name="SO101 Leader/Follower",
    ).set_viewer_config(
        mjswan.ViewerConfig(
            lookat=(-0.16, 0.0, 0.18),
            distance=1.6,
            elevation=-18.0,
            azimuth=135.0,
        )
    )

    _defaults = [default for _, _, _, default in _SO101_JOINT_SPECS]

    # ----- Mode 1: User controls leader via sliders, follower mirrors commands -----
    leader_policy = so101_scene.add_policy(
        name="Leader Joint Control",
        policy=_build_so101_mirror_policy(),
        observations={
            "policy": ObservationGroupCfg(
                terms={
                    "leader_joint_targets": ObservationTermCfg(
                        func=obs_fns.generated_commands,
                        params={"command_name": "leader_joints"},
                    ),
                }
            )
        },
        actions={
            "leader_joint_pos": JointPositionActionCfg(
                actuator_names=tuple(_SO101_LEADER_JOINT_NAMES),
                use_default_offset=False,
                stiffness=_SO101_LEADER_STIFFNESS,
                damping=_SO101_LEADER_DAMPING,
            ),
            "follower_joint_pos": JointPositionActionCfg(
                actuator_names=tuple(_SO101_FOLLOWER_JOINT_NAMES),
                use_default_offset=False,
            ),
        },
        policy_joint_names=_SO101_POLICY_JOINT_NAMES,
        default_joint_pos=_defaults * 2,
    )
    leader_policy.add_command(
        name="leader_joints",
        inputs=[
            mjswan.Slider(
                name=name,
                label=label,
                range=joint_range,
                default=default,
                step=0.01,
            )
            for name, label, joint_range, default in _SO101_JOINT_SPECS
        ],
    )

    # ----- Mode 2: Gravity + Force Hook -----
    # Leader arm is compliant (very low stiffness) — it sags under gravity.
    # A wrist_roll slider applies a spring-like "hook" force on that joint.
    # Follower tracks the leader's actual joint positions in real-time.
    gravity_policy = so101_scene.add_policy(
        name="Gravity + Force Hook",
        policy=_build_gravity_mode_policy(),
        observations={
            "policy": ObservationGroupCfg(
                terms={
                    "leader_qpos": ObservationTermCfg(
                        func=obs_fns.joint_pos_rel,
                        params={
                            "joint_names": _SO101_LEADER_JOINT_NAMES,
                            "subtract_default": False,
                        },
                    ),
                    "wrist_hook": ObservationTermCfg(
                        func=obs_fns.generated_commands,
                        params={"command_name": "wrist_hook"},
                    ),
                }
            )
        },
        actions={
            "leader_passive": JointEffortActionCfg(
                actuator_names=tuple(_SO101_LEADER_JOINT_NAMES),
            ),
            "follower_tracking": JointPositionActionCfg(
                actuator_names=tuple(_SO101_FOLLOWER_JOINT_NAMES),
                use_default_offset=False,
            ),
        },
        policy_joint_names=_SO101_POLICY_JOINT_NAMES,
        default_joint_pos=_defaults * 2,
    )
    gravity_policy.add_command(
        name="wrist_hook",
        inputs=[
            mjswan.Slider(
                name="wrist_roll",
                label="Wrist Roll Hook",
                range=(-2.0, 2.0),
                default=0.0,
                step=0.05,
            )
        ],
    )

    # ----- Mode 3: Hold On (Space bar) -----
    hold_policy = so101_scene.add_policy(
        name="Hold On Mode",
        policy=_build_hold_mode_policy(),
        observations={
            "policy": ObservationGroupCfg(
                terms={
                    "leader_qpos": ObservationTermCfg(
                        func=obs_fns.joint_pos_rel,
                        params={
                            "joint_names": _SO101_LEADER_JOINT_NAMES,
                            "subtract_default": False,
                        },
                    ),
                    "leader_qvel": ObservationTermCfg(
                        func=obs_fns.joint_vel_rel,
                        params={
                            "joint_names": _SO101_LEADER_JOINT_NAMES,
                        },
                    ),
                    "hold_cmd": ObservationTermCfg(
                        func=obs_fns.generated_commands,
                        params={"command_name": "hold_cmd"},
                    ),
                }
            )
        },
        actions={
            "leader_joint_pos": JointEffortActionCfg(
                actuator_names=tuple(_SO101_LEADER_JOINT_NAMES),
            ),
            "follower_joint_pos": JointPositionActionCfg(
                actuator_names=tuple(_SO101_FOLLOWER_JOINT_NAMES),
                use_default_offset=False,
            ),
        },
        commands={
            "hold_cmd": HoldOnCommandCfg(joint_names=_SO101_LEADER_JOINT_NAMES),
        },
        policy_joint_names=_SO101_POLICY_JOINT_NAMES,
        default_joint_pos=_defaults * 2,
    )


def setup_builder() -> mjswan.Builder:
    """Set up and return the builder with demo projects configured.

    Creates a builder and adds a project with three robot scenes.
    Does not build or launch the application.

    Returns:
        Configured Builder instance ready to be built.
    """
    # Ensure asset-relative paths resolve regardless of current working directory.
    os.chdir(Path(__file__).resolve().parent)
    base_path = os.getenv("MJSWAN_BASE_PATH", "/")
    builder = mjswan.Builder(base_path=base_path)

    demo_project = builder.add_project(
        name="mjswan Demo",
    )

    _add_so101_scene(demo_project)

    demo_project.add_scene(
        spec=mujoco.MjSpec.from_file("assets/unitree_g1/scene.xml"),
        name="G1",
    ).set_viewer_config(
        mjswan.ViewerConfig(
            lookat=(0.0, 0.0, 0.7),
            distance=3.7,
            elevation=-13.0,
            azimuth=-34.0,
            origin_type=mjswan.ViewerConfig.OriginType.ASSET_BODY,
            body_name="torso_link",
        )
    ).add_policy(
        policy=onnx.load("assets/unitree_g1/locomotion.onnx"),
        name="Locomotion",
        config_path="assets/unitree_g1/locomotion.json",
        actions={
            # mjlab's JointPositionActionCfg has no stiffness/damping fields, so
            # mjswan's JointPositionActionCfg is used directly for the actions dict.
            # Stiffness/damping are required here because G1 uses motor actuators
            # (biastype=none) that need external PD control in the browser runtime.
            "joint_pos": JointPositionActionCfg(
                entity_name="robot",
                actuator_names=(".*",),
                scale=_G1_SCALE,
                offset=_G1_OFFSET,
                stiffness=_G1_STIFFNESS,
                damping=_G1_DAMPING,
            ),
        },
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
                        func=obs_fns.generated_commands,
                        params={"command_name": "velocity"},
                    ),
                }
            )
        },
    ).add_velocity_command(
        lin_vel_x=(-2.0, 2.0),
        lin_vel_y=(-0.5, 0.5),
        default_lin_vel_x=0.5,
        default_lin_vel_y=0.0,
    )
    demo_project.add_scene(
        # model=mujoco.MjModel.from_xml_path("assets/unitree_go2/scene.xml"),
        spec=mujoco.MjSpec.from_file("assets/unitree_go2/scene.xml"),
        name="Go2",
    )

    return builder


def main():
    """Main entry point for the simple demo.

    Sets up the builder, builds the application, and launches it in a browser.

    Environment variables:
        MJSWAN_BASE_PATH: Base path for deployment (default: '/')
        MJSWAN_NO_LAUNCH: Set to '1' to skip launching the browser
    """
    builder = setup_builder()
    # Build and launch the application
    app = builder.build()
    if os.getenv("MJSWAN_NO_LAUNCH") == "1":
        return
    app.launch()


if __name__ == "__main__":
    main()
