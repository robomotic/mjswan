"""Mjlab-specific custom termination registrations for mjswan examples.

Import this module before calling ``builder.build()`` to register the
terrain-generator termination classes that are used in mjlab velocity tasks.
"""

import os
from typing import Any

from mjswan import TermFunc, register_termination_func

_TERM_DIR = os.path.dirname(os.path.abspath(__file__))

register_termination_func(
    "out_of_terrain_bounds",
    TermFunc(
        ts_name="OutOfTerrainBounds",
        ts_src=os.path.join(_TERM_DIR, "OutOfTerrainBounds.ts"),
    ),
)

register_termination_func(
    "terrain_edge_reached",
    TermFunc(
        ts_name="TerrainEdgeReached",
        ts_src=os.path.join(_TERM_DIR, "TerrainEdgeReached.ts"),
    ),
)


def register_custom_terminations(env_cfg: Any) -> None:
    """Re-register terminations with env_cfg-derived default params."""
    terrain = getattr(env_cfg.scene, "terrain", None)
    terrain_generator = getattr(terrain, "terrain_generator", None)
    is_generator = (
        terrain is not None
        and getattr(terrain, "terrain_type", None) == "generator"
        and terrain_generator is not None
    )

    out_defaults: dict[str, float] = {}
    terrain_edge_defaults: dict[str, float] = (
        {
            "half_x": 0.5 * terrain_generator.size[0],
            "half_y": 0.5 * terrain_generator.size[1],
        }
        if is_generator
        else {}
    )

    if is_generator:
        out_term = env_cfg.terminations.get("out_of_terrain_bounds")
        out_params = getattr(out_term, "params", {}) if out_term is not None else {}
        margin = float(out_params.get("margin", 0.3))
        half_x = 0.5 * terrain_generator.num_rows * terrain_generator.size[0]
        half_y = 0.5 * terrain_generator.num_cols * terrain_generator.size[1]
        out_defaults = {
            "limit_x": max(0.0, half_x - margin),
            "limit_y": max(0.0, half_y - margin),
        }

    register_termination_func(
        "out_of_terrain_bounds",
        TermFunc(
            ts_name="OutOfTerrainBounds",
            ts_src=os.path.join(_TERM_DIR, "OutOfTerrainBounds.ts"),
            defaults=out_defaults,
        ),
    )
    register_termination_func(
        "terrain_edge_reached",
        TermFunc(
            ts_name="TerrainEdgeReached",
            ts_src=os.path.join(_TERM_DIR, "TerrainEdgeReached.ts"),
            defaults=terrain_edge_defaults,
        ),
    )
