"""Policy configuration and management.

This module defines the PolicyConfig dataclass and PolicyHandle class for
ONNX policy configuration and command management.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import onnx

from .adapters import adapt_commands
from .command import CommandInput, CommandTermConfig, ui_command, velocity_command

if TYPE_CHECKING:
    from .envs.mdp.actions.actions import ActionTermCfg
    from .managers.observation_manager import ObservationGroupCfg
    from .managers.termination_manager import TerminationTermCfg
    from .scene import SceneHandle


@dataclass
class PolicyConfig:
    """Configuration for an ONNX policy."""

    name: str
    """Name of the policy."""

    model: onnx.ModelProto
    """ONNX model for the policy."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata for the policy."""

    source_path: str | None = None
    """Optional source path for the policy ONNX file."""

    config_path: str | None = None
    """Optional source path for the policy config JSON file."""

    commands: dict[str, CommandTermConfig] = field(default_factory=dict)
    """Command terms keyed by their policy-visible names."""

    observations: dict[str, ObservationGroupCfg] | None = None
    """Observation group configurations (mjlab-compatible).

    Keys are group names (e.g. ``"policy"``, ``"critic"``).  Values are
    ``ObservationGroupCfg`` instances whose terms are serialized into
    ``observations`` in the policy JSON at build time.
    """

    actions: Mapping[str, ActionTermCfg] | None = None
    """Action term configurations (mjlab-compatible).

    Keys are term names (e.g. ``"joint_pos"``).  Values are
    ``ActionTermCfg`` subclass instances serialized into ``actions``
    in the policy JSON at build time.
    """

    terminations: dict[str, TerminationTermCfg] | None = None
    """Termination term configurations (mjlab-compatible).

    Keys are term names (e.g. ``"time_out"``, ``"fallen"``).  Values are
    ``TerminationTermCfg`` instances serialized into ``terminations``
    in the policy JSON at build time.
    """

    policy_joint_names: list[str] | None = None
    """Ordered list of joint names controlled by the policy.

    Required by the browser-side ``PolicyRunner`` to map policy outputs to
    the correct actuators in the MuJoCo model.  When set, serialized as
    ``policy_joint_names`` at the top level of the policy JSON config.
    """

    default_joint_pos: list[float] | None = None
    """Default joint positions corresponding to ``policy_joint_names``.

    Used by the browser runtime when ``use_default_offset=True``: action=0
    commands this pose.  Must be in the same order as ``policy_joint_names``.
    """


class PolicyHandle:
    """Handle for configuring a policy and its commands.

    This class provides methods for adding commands and customizing policy properties.
    Similar to viser's client handles, this allows for a fluent API pattern.

    Example:
        policy = scene.add_policy(
            policy=onnx.load("locomotion.onnx"),
            name="Locomotion",
            config_path="locomotion.json",
        )
        policy.add_command(
            name="velocity",
            inputs=[
                mjswan.Slider("lin_vel_x", "Forward Velocity", range=(-1.0, 1.0)),
                mjswan.Slider("lin_vel_y", "Lateral Velocity", range=(-0.5, 0.5)),
                mjswan.Slider("ang_vel_z", "Yaw Rate", range=(-1.0, 1.0)),
            ]
        )
    """

    def __init__(self, policy_config: PolicyConfig, scene: SceneHandle) -> None:
        self._config = policy_config
        self._scene = scene

    @property
    def name(self) -> str:
        """Name of the policy."""
        return self._config.name

    @property
    def model(self) -> onnx.ModelProto:
        """ONNX model for the policy."""
        return self._config.model

    def add_command(
        self,
        name: str,
        inputs: list[CommandInput],
    ) -> PolicyHandle:
        """Add a command group to this policy.

        A command group represents a set of related inputs (sliders, buttons)
        that are passed together to an observation. The name is used by
        observations to retrieve command values.

        Args:
            name: Identifier for this command group (e.g., "velocity").
                  This name is used by GeneratedCommands observation.
            inputs: List of command input configurations (Slider, Button, etc.)

        Returns:
            Self for method chaining.

        Example:
            policy.add_command(
                name="velocity",
                inputs=[
                    mjswan.Slider("lin_vel_x", "Forward Velocity", range=(-1.0, 1.0)),
                    mjswan.Slider("lin_vel_y", "Lateral Velocity", range=(-0.5, 0.5)),
                    mjswan.Slider("ang_vel_z", "Yaw Rate", range=(-1.0, 1.0)),
                ]
            )
        """
        self._config.commands[name] = ui_command(list(inputs))
        return self

    def add_command_term(
        self,
        name: str,
        term: CommandTermConfig | Any,
    ) -> PolicyHandle:
        """Add a command term directly.

        Accepts either an mjswan ``CommandTermConfig`` or an mjlab
        ``CommandTermCfg`` registered via ``mjswan.register_command_term()``.
        """

        adapted = adapt_commands({name: term})
        if adapted is None or name not in adapted:
            raise ValueError(f"Failed to adapt command term '{name}'.")
        self._config.commands[name] = adapted[name]
        return self

    def add_velocity_command(
        self,
        lin_vel_x: tuple[float, float] = (-1.0, 1.0),
        lin_vel_y: tuple[float, float] = (-0.5, 0.5),
        ang_vel_z: tuple[float, float] = (-1.0, 1.0),
        default_lin_vel_x: float = 0.5,
        default_lin_vel_y: float = 0.0,
        default_ang_vel_z: float = 0.0,
        name: str = "velocity",
    ) -> PolicyHandle:
        """Add a standard velocity command group.

        This is a convenience method for adding the common velocity command
        pattern used in locomotion policies.

        Args:
            lin_vel_x: Range for forward velocity (min, max)
            lin_vel_y: Range for lateral velocity (min, max)
            ang_vel_z: Range for yaw rate (min, max)
            default_lin_vel_x: Default forward velocity
            default_lin_vel_y: Default lateral velocity
            default_ang_vel_z: Default yaw rate

        Returns:
            Self for method chaining.
        """
        cmd = velocity_command(
            lin_vel_x=lin_vel_x,
            lin_vel_y=lin_vel_y,
            ang_vel_z=ang_vel_z,
            default_lin_vel_x=default_lin_vel_x,
            default_lin_vel_y=default_lin_vel_y,
            default_ang_vel_z=default_ang_vel_z,
        )
        self._config.commands[name] = cmd
        return self

    def set_metadata(self, key: str, value: Any) -> PolicyHandle:
        """Set metadata for this policy.

        Args:
            key: Metadata key.
            value: Metadata value.

        Returns:
            Self for method chaining.
        """
        self._config.metadata[key] = value
        return self


__all__ = ["PolicyConfig", "PolicyHandle"]
