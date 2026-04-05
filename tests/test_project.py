"""Tests for mjswan project/scene/policy data models and fluent-API handles.

Layer: L1 (no I/O beyond in-memory MuJoCo and ONNX objects).
Tests the "contract" of the builder's hierarchical configuration API:
  Builder → ProjectHandle → SceneHandle → PolicyHandle
"""

from pathlib import Path

import mujoco
import pytest

import mjswan
from mjswan.builder import Builder
from mjswan.command import CommandTermConfig, SliderConfig
from mjswan.project import _collect_mjlab_scene_assets
from mjswan.scene import SceneConfig


# ===========================================================================
# SceneConfig — scene_filename property
# ===========================================================================
class TestSceneConfig:
    def test_scene_filename_is_mjz_when_spec_provided(self, minimal_spec):
        cfg = SceneConfig(name="Test", spec=minimal_spec)
        assert cfg.scene_filename == "scene.mjz"

    def test_scene_filename_is_mjb_when_model_provided(self, minimal_model):
        cfg = SceneConfig(name="Test", model=minimal_model)
        assert cfg.scene_filename == "scene.mjb"


# ===========================================================================
# ProjectHandle — add_scene validation and return type
# ===========================================================================
class TestProjectHandle:
    def test_add_scene_with_neither_raises(self):
        project = Builder().add_project(name="P")
        with pytest.raises(ValueError):
            project.add_scene(name="S")  # neither model nor spec

    def test_add_scene_with_both_raises(self, minimal_model, minimal_spec):
        project = Builder().add_project(name="P")
        with pytest.raises(ValueError):
            project.add_scene(name="S", model=minimal_model, spec=minimal_spec)

    def test_add_scene_with_model_returns_scene_handle(self, minimal_model):
        project = Builder().add_project(name="P")
        scene = project.add_scene(name="S", model=minimal_model)
        assert isinstance(scene, mjswan.SceneHandle)

    def test_add_scene_with_spec_returns_scene_handle(self, minimal_spec):
        project = Builder().add_project(name="P")
        scene = project.add_scene(name="S", spec=minimal_spec)
        assert isinstance(scene, mjswan.SceneHandle)

    def test_add_scene_appended_to_project_scenes(self, minimal_model):
        builder = Builder()
        project = builder.add_project(name="P")
        project.add_scene(name="Scene A", model=minimal_model)
        project.add_scene(name="Scene B", model=minimal_model)
        scenes = builder.get_projects()[0].scenes
        assert len(scenes) == 2
        assert scenes[0].name == "Scene A"
        assert scenes[1].name == "Scene B"

    def test_project_name_and_id_exposed(self):
        project = Builder().add_project(name="My Project", id="my_project")
        assert project.name == "My Project"
        assert project.id == "my_project"

    def test_collect_mjlab_scene_assets_uses_terrain_and_entities(
        self, monkeypatch, tmp_path: Path
    ):
        def make_spec(label: str) -> mujoco.MjSpec:
            xml_path = tmp_path / f"{label}.xml"
            xml_path.write_text(
                f'<mujoco model="{label}">'
                '<worldbody><geom type="sphere" size="0.1"/></worldbody>'
                "</mujoco>"
            )
            return mujoco.MjSpec.from_file(str(xml_path))

        class FakeCfg:
            def __init__(self, spec: mujoco.MjSpec):
                self._spec = spec

            def spec_fn(self):
                return self._spec

        terrain_spec = make_spec("terrain")
        robot_spec = make_spec("robot")
        prop_spec = make_spec("prop")

        class FakeSceneCfg:
            terrain = FakeCfg(terrain_spec)
            entities = {
                "robot": FakeCfg(robot_spec),
                "prop": FakeCfg(prop_spec),
            }

        def fake_collect_spec_assets(spec):
            return {f"{spec.modelname}.bin": spec.modelname.encode()}

        monkeypatch.setattr(
            "mjswan.project.collect_spec_assets",
            fake_collect_spec_assets,
        )

        assets = _collect_mjlab_scene_assets(FakeSceneCfg())

        assert assets == {
            "terrain.bin": b"terrain",
            "robot.bin": b"robot",
            "prop.bin": b"prop",
        }


# ===========================================================================
# SceneHandle — add_policy, set_metadata
# ===========================================================================
class TestSceneHandle:
    def test_add_policy_appends_to_scene_policies(self, minimal_model, minimal_onnx):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(name="S", model=minimal_model)
        scene.add_policy(name="Policy A", policy=minimal_onnx)
        scene.add_policy(name="Policy B", policy=minimal_onnx)
        policies = builder.get_projects()[0].scenes[0].policies
        assert len(policies) == 2
        assert policies[0].name == "Policy A"
        assert policies[1].name == "Policy B"

    def test_add_policy_returns_policy_handle(self, minimal_model, minimal_onnx):
        scene = Builder().add_project(name="P").add_scene(name="S", model=minimal_model)
        handle = scene.add_policy(name="Policy", policy=minimal_onnx)
        assert isinstance(handle, mjswan.PolicyHandle)

    def test_set_metadata_returns_self_for_chaining(self, minimal_model):
        scene = Builder().add_project(name="P").add_scene(name="S", model=minimal_model)
        result = scene.set_metadata("key", "value")
        assert result is scene

    def test_set_metadata_stores_value(self, minimal_model):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(name="S", model=minimal_model)
        scene.set_metadata("author", "tester")
        cfg = builder.get_projects()[0].scenes[0]
        assert cfg.metadata["author"] == "tester"


# ===========================================================================
# PolicyHandle — add_command, add_velocity_command, set_metadata
# ===========================================================================
class TestPolicyHandle:
    def _make_policy(self, minimal_model, minimal_onnx):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(name="S", model=minimal_model)
        policy = scene.add_policy(name="Policy", policy=minimal_onnx)
        return builder, policy

    def test_add_velocity_command_stored_under_velocity_key(
        self, minimal_model, minimal_onnx
    ):
        builder, policy = self._make_policy(minimal_model, minimal_onnx)
        policy.add_velocity_command()
        commands = builder.get_projects()[0].scenes[0].policies[0].commands
        assert "velocity" in commands
        assert commands["velocity"].term_name == "UiCommand"

    def test_add_velocity_command_returns_self_for_chaining(
        self, minimal_model, minimal_onnx
    ):
        _, policy = self._make_policy(minimal_model, minimal_onnx)
        result = policy.add_velocity_command()
        assert result is policy

    def test_add_command_stores_inputs(self, minimal_model, minimal_onnx):
        builder, policy = self._make_policy(minimal_model, minimal_onnx)
        policy.add_command(
            name="custom",
            inputs=[SliderConfig(name="x", label="X", range=(-1.0, 1.0))],
        )
        commands = builder.get_projects()[0].scenes[0].policies[0].commands
        assert "custom" in commands
        assert commands["custom"].ui is not None
        assert len(commands["custom"].ui.inputs) == 1

    def test_add_command_returns_self_for_chaining(self, minimal_model, minimal_onnx):
        _, policy = self._make_policy(minimal_model, minimal_onnx)
        result = policy.add_command(name="cmd", inputs=[])
        assert result is policy

    def test_add_command_term_stores_serialized_term(self, minimal_model, minimal_onnx):
        builder, policy = self._make_policy(minimal_model, minimal_onnx)
        result = policy.add_command_term(
            name="goal",
            term=CommandTermConfig(term_name="DummyCommand", params={"value": 1}),
        )
        commands = builder.get_projects()[0].scenes[0].policies[0].commands
        assert result is policy
        assert commands["goal"].term_name == "DummyCommand"
        assert commands["goal"].params["value"] == 1

    def test_set_metadata_stores_value(self, minimal_model, minimal_onnx):
        builder, policy = self._make_policy(minimal_model, minimal_onnx)
        policy.set_metadata("version", "1.0")
        cfg = builder.get_projects()[0].scenes[0].policies[0]
        assert cfg.metadata["version"] == "1.0"
