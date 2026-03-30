"""Scene configuration and management.

This module defines the SceneConfig dataclass and SceneHandle class for
managing MuJoCo scenes and their associated policies.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import mujoco
import onnx

from .adapters import adapt_actions, adapt_observations, adapt_terminations
from .policy import PolicyConfig, PolicyHandle
from .splat import SplatConfig, SplatHandle
from .viewer_config import ViewerConfig

if TYPE_CHECKING:
    from .envs.mdp.actions.actions import ActionTermCfg
    from .managers.observation_manager import ObservationGroupCfg
    from .managers.termination_manager import TerminationTermCfg
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

    splats: list[SplatConfig] = field(default_factory=list)
    """Gaussian Splat backgrounds available for this scene."""

    splat_section: bool = False
    """Show the Splat section in the control panel even when no splats are defined."""

    viewer: ViewerConfig | None = None
    """Optional viewer configuration for this scene."""

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
        name: str,
        policy: onnx.ModelProto,
        *,
        metadata: dict[str, Any] | None = None,
        source_path: str | None = None,
        config_path: str | None = None,
        observations: dict[str, ObservationGroupCfg] | dict[str, Any] | None = None,
        actions: Mapping[str, ActionTermCfg] | Mapping[str, Any] | None = None,
        terminations: dict[str, TerminationTermCfg] | dict[str, Any] | None = None,
    ) -> PolicyHandle:
        """Add an ONNX policy to this scene.

        Args:
            policy: ONNX model containing the policy.
            name: Name for the policy (displayed in the UI).
            metadata: Optional metadata dictionary for the policy.
            source_path: Optional source path for the policy ONNX file.
            config_path: Optional source path for the policy config JSON file.
            observations: Observation group configurations.  Accepts both
                mjswan and mjlab ``ObservationGroupCfg`` instances — mjlab
                types are converted automatically (mjlab is a soft dependency).
            actions: Action term configurations.  Accepts both mjswan and
                mjlab ``ActionTermCfg`` subclass instances.
            terminations: Termination term configurations.  Accepts both
                mjswan and mjlab ``TerminationTermCfg`` instances.

        Returns:
            PolicyHandle for configuring the policy (adding commands, etc.)

        Example:
            from mjswan.managers.observation_manager import (
                ObservationGroupCfg,
                ObservationTermCfg,
            )
            from mjswan.envs.mdp import observations as obs_fns

            policy = scene.add_policy(
                policy=onnx.load("locomotion.onnx"),
                name="Locomotion",
                config_path="locomotion.json",
                observations={
                    "policy": ObservationGroupCfg(
                        terms={
                            "base_lin_vel": ObservationTermCfg(
                                func=obs_fns.base_lin_vel
                            ),
                            "joint_pos": ObservationTermCfg(
                                func=obs_fns.joint_pos_rel, scale=0.5
                            ),
                        },
                    ),
                },
            )
            policy.add_velocity_command()
        """
        if metadata is None:
            metadata = {}

        # Adapt mjlab types to mjswan internals (no-op if already mjswan)
        adapted_observations = adapt_observations(observations)
        adapted_actions = adapt_actions(actions)
        adapted_terminations = adapt_terminations(terminations)

        policy_config = PolicyConfig(
            name=name,
            model=policy,
            metadata=metadata,
            source_path=source_path,
            config_path=config_path,
            observations=adapted_observations,
            actions=adapted_actions,
            terminations=adapted_terminations,
        )
        self._config.policies.append(policy_config)
        return PolicyHandle(policy_config, self)

    def add_splat(
        self,
        name: str,
        *,
        source: str | None = None,
        url: str | None = None,
        scale: float = 1.0,
        x_offset: float = 0.0,
        y_offset: float = 0.0,
        z_offset: float = 0.0,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
        collider_url: str | None = None,
        control: bool = False,
    ) -> SplatHandle:
        """Add a Gaussian Splat background to this scene.

        Provide either ``source`` (recommended) or ``url`` — not both.

        Using ``source`` copies the .spz file into the built application so it
        is served locally, giving you a fully self-contained deployment with no
        external dependencies. This is the recommended approach.

        Using ``url`` keeps the .spz file on an external server. The app stays
        smaller, but requires network access at runtime and will not work
        offline.

        Args:
            name: Display name shown in the viewer control panel.
            source: Local path to a .spz splat file to bundle into the app.
                The file is copied during :meth:`Builder.build`.
            url: URL to an external .spz splat file. The browser fetches it at
                runtime; the file is not bundled.
            scale: Metric scale factor. Use ``metric_scale_factor`` from your
                capture metadata if available.
            x_offset: X-axis position offset (in scaled splat units).
            y_offset: Y-axis position offset (in scaled splat units).
            z_offset: Vertical position offset. Use ``ground_plane_offset`` from
                your capture metadata if available.
            roll: Roll rotation in degrees applied on top of the COLMAP→Three.js
                base rotation.
            pitch: Pitch rotation in degrees applied on top of the COLMAP→Three.js
                base rotation.
            yaw: Yaw rotation in degrees applied on top of the COLMAP→Three.js
                base rotation.
            collider_url: Optional URL or local path to a .glb collision mesh.
            control: If True, shows scale and offset controls in the viewer
                control panel. Defaults to False.

        Returns:
            SplatHandle for further configuration.

        Example:
            # Recommended: bundle the .spz file into the app
            scene.add_splat(
                "Outdoor",
                source="background.spz",
                scale=1.35,
                z_offset=1.0,
            )

            # Alternative: reference an external URL
            scene.add_splat(
                "Outdoor",
                url="https://cdn.example.com/background.spz",
                scale=1.35,
                z_offset=1.0,
            )
        """
        if source is None and url is None:
            raise ValueError(
                "Provide either 'source' (local .spz file path to bundle) "
                "or 'url' (external URL)."
            )
        if source is not None and url is not None:
            raise ValueError("Provide either 'source' or 'url', not both.")

        splat_config = SplatConfig(
            name=name,
            source=source,
            url=url,
            scale=scale,
            x_offset=x_offset,
            y_offset=y_offset,
            z_offset=z_offset,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            collider_url=collider_url,
            control=control,
        )
        self._config.splats.append(splat_config)
        return SplatHandle(splat_config, self)

    def add_splat_section(self) -> SceneHandle:
        """Show the Splat section in the control panel even when no splats are defined.

        This allows users to load splats by pasting a .spz URL directly in the
        control panel, without requiring any pre-configured splats.

        Returns:
            Self for method chaining.

        Example:
            scene.add_splat_section()
        """
        self._config.splat_section = True
        return self

    def set_viewer_config(self, config: ViewerConfig) -> SceneHandle:
        """Set viewer configuration for this scene.

        Args:
            config: A :class:`ViewerConfig` instance describing the camera
                position, tracking mode, and rendering settings.

        Returns:
            Self for method chaining.

        Example::

            from mjswan import ViewerConfig
            scene.set_viewer_config(ViewerConfig(
                lookat=(0.0, 0.0, 0.7),
                distance=4.3,
                elevation=-33,
                azimuth=-34,
                origin_type=ViewerConfig.OriginType.ASSET_BODY,
                body_name="torso_link",
            ))
        """
        self._config.viewer = config
        return self

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


__all__ = ["ViewerConfig", "SceneConfig", "SceneHandle", "SplatConfig", "SplatHandle"]
