"""Policy configuration and management.

This module defines the PolicyConfig dataclass and PolicyHandle class for
ONNX policy configuration and command management.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import onnx

from .command import CommandGroupConfig, CommandInput, velocity_command

if TYPE_CHECKING:
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

    commands: dict[str, CommandGroupConfig] = field(default_factory=dict)
    """Command groups for user-controlled inputs."""


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
        command_group = CommandGroupConfig(name=name, inputs=list(inputs))
        self._config.commands[name] = command_group
        return self

    def add_velocity_command(
        self,
        lin_vel_x: tuple[float, float] = (-1.0, 1.0),
        lin_vel_y: tuple[float, float] = (-0.5, 0.5),
        ang_vel_z: tuple[float, float] = (-1.0, 1.0),
        default_lin_vel_x: float = 0.5,
        default_lin_vel_y: float = 0.0,
        default_ang_vel_z: float = 0.0,
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
        self._config.commands[cmd.name] = cmd
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
