"""Command-term configuration and registration.

Commands follow the mjlab model: each policy owns a dictionary of command
terms, and each term produces a vector consumed by observations.

The browser UI is represented as metadata on top of command terms. Manual
slider/button controls are therefore implemented as a built-in ``UiCommand``
term rather than as a separate command system.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

CommandType = Literal["slider", "button"]


@dataclass
class SliderConfig:
    """Configuration for a slider input exposed by a command term."""

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
        return {
            "type": "slider",
            "name": self.name,
            "label": self.label,
            "min": self.min,
            "max": self.max,
            "step": self.step,
            "default": self.default,
        }


Slider = SliderConfig


@dataclass
class ButtonConfig:
    """Configuration for a button input exposed by a command term."""

    name: str
    label: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "button",
            "name": self.name,
            "label": self.label,
        }


Button = ButtonConfig

CommandInput: TypeAlias = SliderConfig | ButtonConfig


@dataclass
class CommandUiConfig:
    """Optional UI metadata attached to a command term."""

    inputs: list[CommandInput] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"inputs": [inp.to_dict() for inp in self.inputs]}


@dataclass
class CommandTermConfig:
    """Serialized browser-side command-term configuration."""

    term_name: str
    params: dict[str, Any] = field(default_factory=dict)
    ui: CommandUiConfig | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {"name": self.term_name, **self.params}
        if self.ui is not None:
            data["ui"] = self.ui.to_dict()
        return data


@dataclass(frozen=True)
class CommandTermSpec:
    """Python-side registry entry for a custom command term."""

    ts_name: str
    serializer: Callable[[Any], Mapping[str, Any]]
    ts_src: str | None = None


_custom_registry: dict[str, CommandTermSpec] = {}


def register_command_term(mjlab_name: str, spec: CommandTermSpec) -> None:
    """Register a custom command term adapter.

    ``mjlab_name`` should typically be the mjlab config class name, e.g.
    ``"LiftingCommandCfg"``.
    """

    _custom_registry[mjlab_name] = spec


def ui_command(inputs: list[CommandInput]) -> CommandTermConfig:
    """Create the built-in manual UI command term."""

    return CommandTermConfig(
        term_name="UiCommand",
        ui=CommandUiConfig(inputs=list(inputs)),
    )


def velocity_command(
    lin_vel_x: tuple[float, float] = (-1.0, 1.0),
    lin_vel_y: tuple[float, float] = (-0.5, 0.5),
    ang_vel_z: tuple[float, float] = (-1.0, 1.0),
    default_lin_vel_x: float = 0.5,
    default_lin_vel_y: float = 0.0,
    default_ang_vel_z: float = 0.0,
) -> CommandTermConfig:
    """Create a built-in UI command term for planar velocity control."""

    return ui_command(
        [
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
        ]
    )


def _serialize_motion_command(cfg: Any) -> dict[str, Any]:
    """Convert mjlab's ``MotionCommandCfg`` into browser tracking metadata."""
    data: dict[str, Any] = {
        "anchor_body_name": getattr(cfg, "anchor_body_name", ""),
        "body_names": list(getattr(cfg, "body_names", ()) or ()),
        "sampling_mode": getattr(cfg, "sampling_mode", "start"),
        "pose_range": {
            key: list(value)
            for key, value in (getattr(cfg, "pose_range", None) or {}).items()
        },
        "velocity_range": {
            key: list(value)
            for key, value in (getattr(cfg, "velocity_range", None) or {}).items()
        },
        "joint_position_range": list(getattr(cfg, "joint_position_range", (0.0, 0.0))),
    }
    entity_name = getattr(cfg, "entity_name", None)
    if entity_name:
        data["entity_name"] = entity_name
    return data


# Bridges mjlab's MotionCommandCfg (e.g. isaac_lab_tasks MotionCommandCfg) to the TrackingCommand term.
register_command_term(
    "MotionCommandCfg",
    CommandTermSpec(
        ts_name="TrackingCommand",
        serializer=_serialize_motion_command,
    ),
)


__all__ = [
    "Button",
    "ButtonConfig",
    "CommandInput",
    "CommandTermConfig",
    "CommandTermSpec",
    "CommandType",
    "CommandUiConfig",
    "Slider",
    "SliderConfig",
    "_custom_registry",
    "register_command_term",
    "ui_command",
    "velocity_command",
]
