"""Utilities for loading ONNX policies from Weights & Biases."""

from __future__ import annotations

import tempfile
from pathlib import Path

import onnx


def fetch_onnx_from_wandb_run(run_path: str) -> tuple[str, onnx.ModelProto]:
    """Download the latest ONNX policy file from a W&B run.

    Finds the most recently updated ``.onnx`` file attached to the run and
    loads it into memory as an :class:`onnx.ModelProto`.  The policy name is
    the filename with its extension removed (e.g. ``"2026-02-25_04-30-08.onnx"``
    becomes ``"2026-02-25_04-30-08"``).

    Args:
        run_path: W&B run path in the format ``"entity/project/run_id"``.

    Returns:
        A ``(policy_name, onnx_model)`` tuple for the latest ``.onnx`` file.

    Raises:
        ValueError: If no ``.onnx`` files are found in the run.
    """
    import wandb

    api = wandb.Api()
    run = api.run(run_path)

    onnx_files = [f for f in run.files() if f.name.endswith(".onnx")]

    if not onnx_files:
        raise ValueError(f"No .onnx files found in W&B run: {run_path}")

    latest = max(onnx_files, key=lambda f: f.updated_at)

    with tempfile.TemporaryDirectory() as tmp_dir:
        latest.download(root=tmp_dir, replace=True)
        local_path = Path(tmp_dir) / latest.name
        name = local_path.stem
        model = onnx.load(str(local_path))

    return name, model


def fetch_pt_onnx_from_wandb_run(
    run_path: str,
    task_id: str,
) -> list[tuple[str, onnx.ModelProto, list[str], list[float]]]:
    """Download all ``model_*.pt`` checkpoints from a W&B run and convert each to ONNX.

    Checkpoints are sorted by training step before conversion, so the returned
    list is ordered from earliest to latest (e.g. ``model_0``, ``model_50``, …).

    The mjlab environment is constructed once and reused across all checkpoints
    to avoid repeated startup cost.  Requires ``mjlab`` and ``torch`` to be
    installed.

    Args:
        run_path: W&B run path in the format ``"entity/project/run_id"``.
        task_id: mjlab task identifier (e.g. ``"go2_flat"``). Used to
            reconstruct the environment and runner required for ONNX export.

    Returns:
        List of ``(policy_name, onnx_model, joint_names, default_joint_pos)``
        tuples sorted by training step, where joint_names is the ordered list
        of ACTUATED joints controlled by the policy and default_joint_pos is
        the action=0 target pose in the same order.

    Raises:
        ImportError: If ``mjlab`` or ``torch`` are not installed.
        ValueError: If no ``model_*.pt`` files are found in the run.
    """
    import re
    from dataclasses import asdict

    import wandb

    try:
        import mjlab.tasks  # noqa: F401 — populates the task registry
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
        from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    except ImportError as e:
        raise ImportError(
            "mjlab and torch are required for only_latest=False. "
            "Install them with: pip install mjlab torch"
        ) from e

    api = wandb.Api()
    run = api.run(run_path)

    pt_files = [f for f in run.files() if re.match(r"^model_\d+\.pt$", f.name)]

    if not pt_files:
        raise ValueError(f"No model_*.pt files found in W&B run: {run_path}")

    # Sort by training step number (model_0.pt < model_50.pt < model_100.pt, …)
    pt_files.sort(key=lambda f: int(re.search(r"\d+", f.name).group()))  # type: ignore[union-attr]

    results: list[tuple[str, onnx.ModelProto, list[str], list[float]]] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Build mjlab environment once — reused across all checkpoints.
        env_cfg = load_env_cfg(task_id, play=True)
        env_cfg.scene.num_envs = 1
        agent_cfg = load_rl_cfg(task_id)

        env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        try:
            runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
            runner = runner_cls(env, asdict(agent_cfg), device="cpu")

            # Collect the ACTUATED joints controlled by the policy. Observation
            # terms may use a broader joint set (for example passive joints
            # coupled via equality constraints), but control and ONNX output
            # dimensions must remain aligned with the action manager.
            joint_names: list[str] = []
            default_joint_pos: list[float] = []
            inner_env = env.env if hasattr(env, "env") else env
            action_mgr = getattr(inner_env, "action_manager", None)
            if action_mgr is not None:
                for term_name in action_mgr.active_terms:
                    term = action_mgr.get_term(term_name)
                    if hasattr(term, "target_names"):
                        entity_name = getattr(
                            getattr(term, "cfg", None), "entity_name", None
                        )
                        prefix = f"{entity_name}/" if entity_name else ""
                        joint_names = [f"{prefix}{n}" for n in term.target_names]
                        # Default joint positions (the pose action=0 commands).
                        if hasattr(term, "offset") and term.offset is not None:
                            offset = term.offset
                            if hasattr(offset, "tolist"):
                                flat = offset.flatten().tolist()
                            elif hasattr(offset, "__iter__"):
                                flat = list(offset)
                            else:
                                flat = [float(offset)] * len(joint_names)
                            default_joint_pos = flat[: len(joint_names)]
                        break

            for wandb_file in pt_files:
                wandb_file.download(root=tmp_dir, replace=True)
                pt_path = tmp_path / wandb_file.name
                name = pt_path.stem  # e.g. "model_0", "model_50"
                onnx_filename = f"{name}.onnx"

                runner.load(
                    str(pt_path),
                    load_cfg={"actor": True},
                    strict=True,
                    map_location="cpu",
                )
                runner.export_policy_to_onnx(tmp_dir, onnx_filename)

                model = onnx.load(str(tmp_path / onnx_filename))
                results.append((name, model, joint_names, default_joint_pos))
        finally:
            env.close()

    return results


__all__ = ["fetch_onnx_from_wandb_run", "fetch_pt_onnx_from_wandb_run"]
