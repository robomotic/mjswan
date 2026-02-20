"""Scene configuration and management.

This module defines the SceneConfig dataclass and SceneHandle class for
managing MuJoCo scenes and their associated policies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import mujoco
import onnx

from .policy import PolicyConfig, PolicyHandle

if TYPE_CHECKING:
    from .project import ProjectHandle


@dataclass
class SceneConfig:
    """Configuration for a MuJoCo scene."""

    name: str
    """Name of the scene."""

    model: mujoco.MjModel | None = None
    """MuJoCo model for the scene (saved as .mjb)."""

    spec: mujoco.MjSpec | None = None
    """MuJoCo spec for the scene (saved as .mjz)."""

    policies: list[PolicyConfig] = field(default_factory=list)
    """List of policies available for this scene."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata for the scene."""

    @property
    def scene_filename(self) -> str:
        """Return the scene filename based on which field is set."""
        return "scene.mjz" if self.spec is not None else "scene.mjb"


class SceneHandle:
    """Handle for adding policies and configuring a scene.

    This class provides methods for adding policies and customizing scene properties.
    Similar to viser's client handles, this allows for a fluent API pattern.
    """

    def __init__(self, scene_config: SceneConfig, project: ProjectHandle) -> None:
        self._config = scene_config
        self._project = project

    @property
    def name(self) -> str:
        """Name of the scene."""
        return self._config.name

    def add_policy(
        self,
        policy: onnx.ModelProto,
        name: str,
        *,
        metadata: dict[str, Any] | None = None,
        source_path: str | None = None,
        config_path: str | None = None,
    ) -> PolicyHandle:
        """Add an ONNX policy to this scene.

        Args:
            policy: ONNX model containing the policy.
            name: Name for the policy (displayed in the UI).
            metadata: Optional metadata dictionary for the policy.
            source_path: Optional source path for the policy ONNX file.
            config_path: Optional source path for the policy config JSON file.

        Returns:
            PolicyHandle for configuring the policy (adding commands, etc.)

        Example:
            policy = scene.add_policy(
                policy=onnx.load("locomotion.onnx"),
                name="Locomotion",
                config_path="locomotion.json",
            )
            policy.add_velocity_command()
        """
        if metadata is None:
            metadata = {}

        policy_config = PolicyConfig(
            name=name,
            model=policy,
            metadata=metadata,
            source_path=source_path,
            config_path=config_path,
        )
        self._config.policies.append(policy_config)
        return PolicyHandle(policy_config, self)

    def set_metadata(self, key: str, value: Any) -> SceneHandle:
        """Set metadata for this scene.

        Args:
            key: Metadata key.
            value: Metadata value.

        Returns:
            Self for method chaining.
        """
        self._config.metadata[key] = value
        return self


__all__ = ["SceneConfig", "SceneHandle"]
