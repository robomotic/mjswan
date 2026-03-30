"""Adapter for converting mjlab types to mjswan internal representations.

mjlab is a **soft dependency** — this module never fails at import time.
When mjlab is not installed the ``adapt_*`` functions simply return their
inputs unchanged (they are assumed to already be mjswan types).

The adapter detects mjlab types by checking the module path of the class
(``type(obj).__module__``) rather than ``isinstance``, so mjlab does not
need to be importable for mjswan to function.

Mapping strategy
----------------
Because mjswan sentinels and classes **share the same names** as their
mjlab counterparts, the adapter resolves mappings dynamically via
``getattr`` on the mjswan modules — no hardcoded registries required.

* **Observation / termination functions**: ``func.__name__`` is looked
  up directly on ``mjswan.envs.mdp.observations`` /
  ``mjswan.envs.mdp.terminations``.
* **Action configs**: ``type(cfg).__name__`` is looked up on
  ``mjswan.envs.mdp.actions``, and dataclass fields are copied
  automatically.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Mapping
from typing import Any

from ..envs.mdp import actions as _actions_module
from ..envs.mdp import observations as _obs_module
from ..envs.mdp import terminations as _term_module
from ..envs.mdp.actions.actions import (
    ActionTermCfg as MjswanActionTermCfg,
)
from ..envs.mdp.observations import ObsFunc
from ..envs.mdp.terminations import TermFunc
from ..managers.observation_manager import (
    ObservationGroupCfg as MjswanObservationGroupCfg,
)
from ..managers.observation_manager import (
    ObservationTermCfg as MjswanObservationTermCfg,
)
from ..managers.termination_manager import (
    TerminationTermCfg as MjswanTerminationTermCfg,
)


def _is_from_mjlab(obj: Any) -> bool:
    """Check whether *obj*'s class originates from the ``mjlab`` package."""
    module = getattr(type(obj), "__module__", "") or ""
    return module.startswith("mjlab")


# ---------------------------------------------------------------------------
# Observation adaptation
# ---------------------------------------------------------------------------


def _adapt_obs_func(func: Any) -> ObsFunc:
    """Convert an mjlab observation callable to an mjswan ``ObsFunc`` sentinel.

    If *func* is already an mjswan ``ObsFunc`` it is returned as-is, so
    mjswan sentinels can be passed directly inside mjlab ``ObservationTermCfg``
    for functions that have no mjlab equivalent.

    Otherwise, looks up ``func.__name__`` directly on
    ``mjswan.envs.mdp.observations``.
    """
    if isinstance(func, ObsFunc):
        return func
    name = getattr(func, "__name__", None)
    sentinel = getattr(_obs_module, name, None) if name else None
    if isinstance(sentinel, ObsFunc):
        return sentinel
    raise ValueError(
        f"No mjswan mapping for mjlab observation function '{name}'. "
        f"Ensure a matching ObsFunc sentinel exists in "
        f"mjswan.envs.mdp.observations."
    )


def _adapt_obs_term(term: Any) -> MjswanObservationTermCfg:
    """Convert a single mjlab ``ObservationTermCfg`` to mjswan."""
    return MjswanObservationTermCfg(
        func=_adapt_obs_func(term.func),
        params=dict(getattr(term, "params", None) or {}),
        scale=getattr(term, "scale", None),
        clip=getattr(term, "clip", None),
        history_length=getattr(term, "history_length", 0) or 0,
    )


def _adapt_obs_group(group: Any) -> MjswanObservationGroupCfg:
    """Convert a single mjlab ``ObservationGroupCfg`` to mjswan."""
    raw_terms = getattr(group, "terms", None) or {}
    terms = {name: _adapt_obs_term(cfg) for name, cfg in raw_terms.items()}
    return MjswanObservationGroupCfg(
        terms=terms,
        concatenate_terms=getattr(group, "concatenate_terms", True),
        enable_corruption=getattr(group, "enable_corruption", False),
        history_length=getattr(group, "history_length", None),
    )


def adapt_observations(
    observations: dict[str, Any] | None,
) -> dict[str, MjswanObservationGroupCfg] | None:
    """Adapt observation groups, converting mjlab types if detected.

    If the values are already ``mjswan.ObservationGroupCfg`` instances they
    are returned as-is.  mjlab ``ObservationGroupCfg`` instances are
    converted transparently.
    """
    if observations is None:
        return None
    return {
        key: group
        if isinstance(group, MjswanObservationGroupCfg)
        else _adapt_obs_group(group)
        if _is_from_mjlab(group)
        else group
        for key, group in observations.items()
    }


# ---------------------------------------------------------------------------
# Termination adaptation
# ---------------------------------------------------------------------------


def _adapt_term_func(func: Any) -> TermFunc:
    """Convert an mjlab termination callable to an mjswan ``TermFunc`` sentinel.

    If *func* is already an mjswan ``TermFunc`` it is returned as-is, so
    mjswan sentinels can be passed directly inside mjlab ``TerminationTermCfg``
    for functions that have no mjlab equivalent.

    Otherwise, looks up ``func.__name__`` directly on
    ``mjswan.envs.mdp.terminations``.
    """
    if isinstance(func, TermFunc):
        return func
    name = getattr(func, "__name__", None)
    sentinel = getattr(_term_module, name, None) if name else None
    if isinstance(sentinel, TermFunc):
        return sentinel
    raise ValueError(
        f"No mjswan mapping for mjlab termination function '{name}'. "
        f"Ensure a matching TermFunc sentinel exists in "
        f"mjswan.envs.mdp.terminations."
    )


def _adapt_term_cfg(term: Any) -> MjswanTerminationTermCfg:
    """Convert a single mjlab ``TerminationTermCfg`` to mjswan."""
    return MjswanTerminationTermCfg(
        func=_adapt_term_func(term.func),
        params=dict(getattr(term, "params", None) or {}),
        time_out=getattr(term, "time_out", False),
    )


def adapt_terminations(
    terminations: dict[str, Any] | None,
) -> dict[str, MjswanTerminationTermCfg] | None:
    """Adapt termination configs, converting mjlab types if detected."""
    if terminations is None:
        return None
    return {
        key: term
        if isinstance(term, MjswanTerminationTermCfg)
        else _adapt_term_cfg(term)
        if _is_from_mjlab(term)
        else term
        for key, term in terminations.items()
    }


# ---------------------------------------------------------------------------
# Action adaptation
# ---------------------------------------------------------------------------


def _adapt_action_cfg(term: Any) -> MjswanActionTermCfg | None:
    """Convert a single mjlab ``ActionTermCfg`` to mjswan.

    Looks up ``type(term).__name__`` on ``mjswan.envs.mdp.actions`` to
    find the corresponding mjswan class, then copies all matching
    dataclass fields automatically.

    Returns ``None`` if no mjswan equivalent exists; the caller is
    responsible for dropping the entry.
    """
    class_name = type(term).__name__
    mjswan_cls = getattr(_actions_module, class_name, None)

    if mjswan_cls is None or not (
        isinstance(mjswan_cls, type) and issubclass(mjswan_cls, MjswanActionTermCfg)
    ):
        warnings.warn(
            f"mjlab action type '{class_name}' has no mjswan equivalent. "
            f"It will be skipped.",
            category=RuntimeWarning,
            stacklevel=3,
        )
        return None

    # Copy all matching dataclass fields from the mjlab instance
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(mjswan_cls):
        if f.name == "unsupported_reason":
            continue
        val = getattr(term, f.name, dataclasses.MISSING)
        if val is not dataclasses.MISSING:
            kwargs[f.name] = val
    return mjswan_cls(**kwargs)


def adapt_actions(
    actions: Mapping[str, Any] | None,
) -> Mapping[str, MjswanActionTermCfg] | None:
    """Adapt action configs, converting mjlab types if detected."""
    if actions is None:
        return None
    result: dict[str, MjswanActionTermCfg] = {}
    for key, term in actions.items():
        if isinstance(term, MjswanActionTermCfg):
            result[key] = term
        elif _is_from_mjlab(term):
            adapted = _adapt_action_cfg(term)
            if adapted is not None:
                result[key] = adapted
        else:
            result[key] = term
    return result


__all__ = ["adapt_observations", "adapt_actions", "adapt_terminations"]
