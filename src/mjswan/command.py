"""Command configuration for user-controlled inputs.

Commands are user inputs (like velocity targets) that get passed to the
ONNX policy as part of observations. This module defines the configuration
dataclasses for different command types.

Example usage:
    policy_handle.add_command(
        name="velocity",
        inputs=[
            mjswan.Slider("lin_vel_x", "Forward Velocity", range=(-1.0, 1.0), default=0.5),
            mjswan.Slider("lin_vel_y", "Lateral Velocity", range=(-0.5, 0.5), default=0.0),
            mjswan.Slider("ang_vel_z", "Yaw Rate", range=(-1.0, 1.0), default=0.0),
        ]
    )
"""

from dataclasses import dataclass, field
from typing import Any, Literal

CommandType = Literal["slider", "button"]


@dataclass
class SliderConfig:
    """Configuration for a slider command input.

    Args:
        name: Internal identifier used to reference this input (e.g., "lin_vel_x")
        label: Display label shown in the UI (e.g., "Forward Velocity")
        range: Tuple of (min, max) values for the slider
        default: Default value when initialized
        step: Step size for the slider (default: 0.01)
    """

    name: str
    label: str
    range: tuple[float, float] = (-1.0, 1.0)
    default: float = 0.0
    step: float = 0.01

    @property
    def min(self) -> float:
        return self.range[0]

    @property
    def max(self) -> float:
        return self.range[1]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "type": "slider",
            "name": self.name,
            "label": self.label,
            "min": self.min,
            "max": self.max,
            "step": self.step,
            "default": self.default,
        }


# Alias for convenience
Slider = SliderConfig


@dataclass
class ButtonConfig:
    """Configuration for a button command input.

    Args:
        name: Internal identifier (e.g., "reset")
        label: Display label shown in the UI (e.g., "Reset Simulation")
    """

    name: str
    label: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "type": "button",
            "name": self.name,
            "label": self.label,
        }


# Alias for convenience
Button = ButtonConfig


# Union type for all command input types
CommandInput = SliderConfig | ButtonConfig


@dataclass
class CommandGroupConfig:
    """Configuration for a group of command inputs.

    A command group represents a set of related inputs that are passed
    together to an observation. For example, a "velocity" command group
    might contain lin_vel_x, lin_vel_y, and ang_vel_z sliders.

    Args:
        name: Identifier for this command group (e.g., "velocity")
              This name is used by observations to retrieve command values.
        inputs: List of command input configurations (sliders, buttons, etc.)
    """

    name: str
    inputs: list[CommandInput] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {"inputs": [inp.to_dict() for inp in self.inputs]}


def velocity_command(
    lin_vel_x: tuple[float, float] = (-1.0, 1.0),
    lin_vel_y: tuple[float, float] = (-0.5, 0.5),
    ang_vel_z: tuple[float, float] = (-1.0, 1.0),
    default_lin_vel_x: float = 0.5,
    default_lin_vel_y: float = 0.0,
    default_ang_vel_z: float = 0.0,
) -> CommandGroupConfig:
    """Create a standard velocity command group.

    This is a convenience function for creating the common velocity command
    pattern used in locomotion policies.

    Args:
        lin_vel_x: Range for forward velocity (min, max)
        lin_vel_y: Range for lateral velocity (min, max)
        ang_vel_z: Range for yaw rate (min, max)
        default_lin_vel_x: Default forward velocity
        default_lin_vel_y: Default lateral velocity
        default_ang_vel_z: Default yaw rate

    Returns:
        CommandGroupConfig with velocity inputs configured
    """
    return CommandGroupConfig(
        name="velocity",
        inputs=[
            SliderConfig(
                name="lin_vel_x",
                label="Forward Velocity",
                range=lin_vel_x,
                default=default_lin_vel_x,
                step=0.05,
            ),
            SliderConfig(
                name="lin_vel_y",
                label="Lateral Velocity",
                range=lin_vel_y,
                default=default_lin_vel_y,
                step=0.05,
            ),
            SliderConfig(
                name="ang_vel_z",
                label="Yaw Rate",
                range=ang_vel_z,
                default=default_ang_vel_z,
                step=0.05,
            ),
        ],
    )
