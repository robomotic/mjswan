"""Gaussian Splat Background Demo

Demonstrates how to use add_splat() to attach a real-world Gaussian Splat
background to a scene.

Run with:
    uv run splat
"""

import os
from pathlib import Path

import mujoco
import onnx

import mjswan

SPLAT_URLs = [
    "https://cdn.marble.worldlabs.ai/be100eec-f02e-491d-899e-d702652d424d/cb27e09c-e2ca-46c7-8abf-bcd24d2bf9ed_ceramic_500k.spz",
    "https://cdn.marble.worldlabs.ai/09eaec3b-9114-455a-b7f1-da4d037cc511/660e6ce6-959c-42fb-8a9d-66178cb84f4d_ceramic.spz",
]


def setup_builder() -> mjswan.Builder:
    """Set up the builder with a splat-backed scene.

    Returns:
        Configured Builder instance ready to be built.
    """
    # Ensure asset-relative paths resolve regardless of current working directory.
    os.chdir(Path(__file__).resolve().parent)
    base_path = os.getenv("MJSWAN_BASE_PATH", "/")
    builder = mjswan.Builder(base_path=base_path)

    project = builder.add_project(name="Splat Demo")

    scene = project.add_scene(
        spec=mujoco.MjSpec.from_file("assets/unitree_g1/scene.xml"),
        name="G1",
    )

    scene.add_policy(
        name="balance",
        policy=onnx.load("assets/unitree_g1/balance.onnx"),
        config_path="assets/unitree_g1/balance.json",
    )

    scene.add_policy(
        name="locomotion",
        policy=onnx.load("assets/unitree_g1/locomotion.onnx"),
        config_path="assets/unitree_g1/locomotion.json",
    ).add_velocity_command(
        lin_vel_x=(-1.5, 1.5),
        lin_vel_y=(-0.5, 0.5),
        default_lin_vel_x=0.5,
    )

    for splat_url in SPLAT_URLs:
        scene.add_splat(
            name=f"Splat {SPLAT_URLs.index(splat_url) + 1}",
            url=splat_url,
            scale=3.275,
            z_offset=0.708,
            control=True,
        )

    return builder


def main():
    """Build and launch the splat demo."""
    builder = setup_builder()
    app = builder.build()
    if os.getenv("MJSWAN_NO_LAUNCH") != "1":
        app.launch()


if __name__ == "__main__":
    main()
