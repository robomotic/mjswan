"""Scene configuration and management.

This module defines the SceneConfig dataclass and SceneHandle class for
managing MuJoCo scenes and their associated policies.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np
import onnx

from .adapters import (
    adapt_actions,
    adapt_commands,
    adapt_observations,
    adapt_terminations,
    resolve_action_scales,
)
from .policy import PolicyConfig, PolicyHandle
from .splat import SplatConfig, SplatHandle
from .viewer_config import ViewerConfig

if TYPE_CHECKING:
    from .envs.mdp.actions.actions import ActionTermCfg
    from .managers.observation_manager import ObservationGroupCfg
    from .managers.termination_manager import TerminationTermCfg
    from .project import ProjectHandle


def _get_scene_model(scene_config: SceneConfig) -> mujoco.MjModel | None:
    if scene_config.model is not None:
        return scene_config.model
    if scene_config.spec is None:
        return None
    try:
        return scene_config.spec.compile()
    except Exception:
        return None


def _get_default_qpos(model: mujoco.MjModel) -> list[float]:
    if model.nkey > 0:
        try:
            key_qpos = np.asarray(model.key_qpos).reshape(model.nkey, model.nq)
            return [float(v) for v in key_qpos[0]]
        except Exception:
            pass
    return [float(v) for v in np.asarray(model.qpos0).reshape(model.nq)]


def _resolve_observation_joints(
    model: mujoco.MjModel,
    config: dict[str, Any],
) -> tuple[list[str], list[float]] | None:
    joint_names_cfg = config.get("joint_names")
    entity_name = config.get("entity_name")
    if joint_names_cfg is None and entity_name is None:
        return None

    default_qpos = _get_default_qpos(model)
    prefix = f"{entity_name}/" if entity_name else ""
    joints: list[tuple[str, int]] = []
    for i in range(model.njnt):
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = model.joint(i).name
        if prefix and not name.startswith(prefix):
            continue
        joints.append((name, int(model.jnt_qposadr[i])))

    if not joints:
        return None

    if joint_names_cfg in (None, "all"):
        selected = joints
    else:
        patterns = (
            list(joint_names_cfg)
            if isinstance(joint_names_cfg, (list, tuple))
            else [joint_names_cfg]
        )
        regexes = []
        for pattern in patterns:
            try:
                regexes.append(re.compile(f"^(?:{pattern})$"))
            except re.error:
                continue
        if not regexes:
            return None

        def _matches(name: str) -> bool:
            bare = name[len(prefix) :] if prefix and name.startswith(prefix) else name
            return any(rex.fullmatch(bare) or rex.fullmatch(name) for rex in regexes)

        selected = [(name, adr) for name, adr in joints if _matches(name)]

    if not selected:
        return None

    names = [name for name, _ in selected]
    defaults = [
        default_qpos[adr] if adr < len(default_qpos) else 0.0 for _, adr in selected
    ]
    return names, defaults


def _enrich_joint_observations(
    scene_config: SceneConfig,
    observations: dict[str, Any] | None,
) -> None:
    if observations is None:
        return
    model = _get_scene_model(scene_config)
    if model is None:
        return

    for group in observations.values():
        terms = getattr(group, "terms", None)
        if not isinstance(terms, dict):
            continue
        for term in terms.values():
            ts_name = getattr(getattr(term, "func", None), "ts_name", None)
            if ts_name not in {"JointPos", "JointPositions", "JointVelocities"}:
                continue
            params = dict(getattr(term, "params", {}) or {})
            merged = {**getattr(term.func, "defaults", {}), **params}
            if merged.get("joint_name") is not None:
                continue
            resolved = _resolve_observation_joints(model, merged)
            if resolved is None:
                continue
            joint_names, default_joint_pos = resolved
            params["joint_names"] = joint_names
            if ts_name in {"JointPos", "JointPositions"}:
                params["default_joint_pos"] = default_joint_pos
            term.params = params


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
        commands: Mapping[str, Any] | None = None,
        actions: Mapping[str, ActionTermCfg] | Mapping[str, Any] | None = None,
        terminations: dict[str, TerminationTermCfg] | dict[str, Any] | None = None,
        policy_joint_names: list[str] | None = None,
        default_joint_pos: list[float] | None = None,
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
            commands: Command term configurations. Accepts both mjswan and
                mjlab ``CommandTermCfg`` instances. Custom mjlab terms are
                converted through the Python command-term registry.
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
        adapted_commands = adapt_commands(commands)
        adapted_actions = adapt_actions(actions)
        adapted_terminations = adapt_terminations(terminations)
        _enrich_joint_observations(self._config, adapted_observations)
        if adapted_actions and policy_joint_names:
            resolve_action_scales(adapted_actions, policy_joint_names)

        policy_config = PolicyConfig(
            name=name,
            model=policy,
            metadata=metadata,
            source_path=source_path,
            config_path=config_path,
            commands=adapted_commands or {},
            observations=adapted_observations,
            actions=adapted_actions,
            terminations=adapted_terminations,
            policy_joint_names=policy_joint_names,
            default_joint_pos=default_joint_pos,
        )
        self._config.policies.append(policy_config)
        return PolicyHandle(policy_config, self)

    def add_policy_from_wandb(
        self,
        run_path: str | list[str],
        *,
        only_latest: bool = False,
        task_id: str | None = None,
        config_path: str | None = None,
        metadata: dict[str, Any] | None = None,
        observations: dict[str, ObservationGroupCfg] | dict[str, Any] | None = None,
        commands: Mapping[str, Any] | None = None,
        actions: Mapping[str, ActionTermCfg] | Mapping[str, Any] | None = None,
        terminations: dict[str, TerminationTermCfg] | dict[str, Any] | None = None,
    ) -> list[PolicyHandle]:
        """Add ONNX policies fetched from one or more W&B runs to this scene.

        ``config_path``, ``observations``, ``commands``, ``actions``, and
        ``terminations`` are
        applied identically to every policy fetched from every run.

        Args:
            run_path: W&B run path in the format ``"entity/project/run_id"``, or
                a list of such paths to fetch policies from multiple runs.
            only_latest: If ``False`` (default), fetches all ``model_*.pt``
                checkpoints and converts each to ONNX via mjlab — requires
                ``mjlab`` and ``torch`` to be installed and ``task_id`` to be
                provided.  If ``True``, fetches only the ``.onnx`` file from
                each run (the latest exported checkpoint).
            task_id: mjlab task identifier required when ``only_latest=False``
                (e.g. ``"go2_flat"``).  Ignored when ``only_latest=True``.
            config_path: Optional path to a policy config JSON file applied to
                all fetched policies.
            metadata: Optional metadata dictionary applied to all fetched
                policies.
            observations: Observation group configurations applied to all
                fetched policies.
            commands: Command term configurations applied to all fetched policies.
            actions: Action term configurations applied to all fetched policies.
            terminations: Termination term configurations applied to all fetched
                policies.

        Returns:
            Flat list of :class:`PolicyHandle` instances across all runs, in the
            order the runs were provided.

        Raises:
            ValueError: If ``only_latest=False`` and ``task_id`` is not provided,
                or if no matching files are found in a W&B run.
            ImportError: If ``only_latest=False`` and ``mjlab``/``torch`` are not
                installed.

        Example — all logged checkpoints from a single run (default):
            ```python
            scene.add_policy_from_wandb(
                run_path="my-org/my-project/run-id",
                task_id="go2_flat",
                config_path="assets/locomotion.json",
                actions={"joint_pos": JointPositionActionCfg(scale=1.0)},
            )
            ```

        Example — latest checkpoint only:
            ```python
            scene.add_policy_from_wandb(
                run_path="my-org/my-project/run-id",
                only_latest=True,
                config_path="assets/locomotion.json",
                actions={"joint_pos": JointPositionActionCfg(scale=1.0)},
            )
            ```

        Example — multiple runs:
            ```python
            scene.add_policy_from_wandb(
                run_path=[
                    "my-org/my-project/run-id-1",
                    "my-org/my-project/run-id-2",
                ],
                only_latest=True,
                config_path="assets/locomotion.json",
                actions={"joint_pos": JointPositionActionCfg(scale=1.0)},
            )
            ```
        """
        if not only_latest and task_id is None:
            raise ValueError(
                "task_id is required when only_latest=False. "
                "Provide the mjlab task identifier, e.g. task_id='go2_flat'."
            )

        run_paths = [run_path] if isinstance(run_path, str) else run_path

        handles = []
        seen_names: set[str] = set()
        for path in run_paths:
            if only_latest:
                from .wandb_utils import fetch_onnx_from_wandb_run

                name, model = fetch_onnx_from_wandb_run(path)
                if name not in seen_names:
                    seen_names.add(name)
                    handle = self.add_policy(
                        name=name,
                        policy=model,
                        config_path=config_path,
                        metadata=metadata,
                        observations=observations,
                        commands=commands,
                        actions=actions,
                        terminations=terminations,
                    )
                    handles.append(handle)
            else:
                assert task_id is not None
                from .wandb_utils import fetch_pt_onnx_from_wandb_run

                for name, model, joint_names, djp in fetch_pt_onnx_from_wandb_run(
                    path, task_id
                ):
                    if name in seen_names:
                        continue
                    seen_names.add(name)
                    handle = self.add_policy(
                        name=name,
                        policy=model,
                        config_path=config_path,
                        metadata=metadata,
                        observations=observations,
                        commands=commands,
                        actions=actions,
                        terminations=terminations,
                        policy_joint_names=joint_names or None,
                        default_joint_pos=djp or None,
                    )
                    handles.append(handle)
        return handles

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
