"""Test suite for mjswan.utils — _strip_leading_dotdot, _rewrite_xml_paths,
to_zip_deflated, and collect_spec_assets.
"""

from __future__ import annotations

import io
import os
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import mujoco
import pytest

from mjswan.utils import (
    _make_zip_safe_path,
    _rewrite_xml_paths,
    _strip_leading_dotdot,
    collect_spec_assets,
    to_zip_deflated,
)

# ---------------------------------------------------------------------------
# Base directory for demo assets used in integration tests
# ---------------------------------------------------------------------------
DEMO_DIR = Path(__file__).parent.parent / "examples" / "demo"


# ===========================================================================
# _strip_leading_dotdot
# ===========================================================================
class TestStripLeadingDotdot:
    def test_no_dotdot(self):
        assert _strip_leading_dotdot("meshes/foo.stl") == "meshes/foo.stl"

    def test_single_dotdot(self):
        assert _strip_leading_dotdot("../meshes/foo.stl") == "meshes/foo.stl"

    def test_multiple_dotdot(self):
        assert (
            _strip_leading_dotdot("../../myo_sim/meshes/clavicle.stl")
            == "myo_sim/meshes/clavicle.stl"
        )

    def test_dotdot_resolved_by_normpath(self):
        # "a/b/../../c.stl" normalises to "c.stl", no leading ..
        assert _strip_leading_dotdot("a/b/../../c.stl") == "c.stl"

    def test_only_dotdot(self):
        assert _strip_leading_dotdot("../..") == ""

    def test_empty_string(self):
        assert _strip_leading_dotdot("") == ""

    def test_dot(self):
        assert _strip_leading_dotdot("./meshes/foo.stl") == "meshes/foo.stl"

    def test_trailing_slash_stripped(self):
        # posixpath.normpath removes trailing slashes
        assert _strip_leading_dotdot("../meshes/") == "meshes"

    def test_deep_relative_path(self):
        assert (
            _strip_leading_dotdot("../../../../simhive/myo_sim/meshes/a.stl")
            == "simhive/myo_sim/meshes/a.stl"
        )


# ===========================================================================
# _make_zip_safe_path
# ===========================================================================
class TestMakeZipSafePath:
    def test_relative_no_dotdot(self):
        assert _make_zip_safe_path("meshes/foo.stl") == "meshes/foo.stl"

    def test_relative_with_dotdot(self):
        assert _make_zip_safe_path("../meshes/foo.stl") == "meshes/foo.stl"

    def test_absolute_unix_returns_basename(self):
        assert (
            _make_zip_safe_path("/Users/alice/models/scene/skybox.png") == "skybox.png"
        )

    def test_absolute_deep_path_returns_basename(self):
        assert (
            _make_zip_safe_path(
                "/home/user/.venv/lib/python3.13/site-packages/pkg/model/scene/tex.png"
            )
            == "tex.png"
        )

    def test_windows_style_absolute(self):
        assert _make_zip_safe_path("C:/Users/bob/models/mesh.stl") == "mesh.stl"

    def test_empty_string(self):
        assert _make_zip_safe_path("") == ""

    def test_only_dotdot(self):
        assert _make_zip_safe_path("../..") == ""


# ===========================================================================
# _rewrite_xml_paths
# ===========================================================================
class TestRewriteXmlPaths:
    def test_removes_meshdir_and_texturedir(self):
        xml_in = (
            '<mujoco><compiler meshdir="../" texturedir="../" angle="radian"/></mujoco>'
        )
        result = _rewrite_xml_paths(xml_in, mesh_dir="../", texture_dir="../")
        root = ET.fromstring(result)
        compiler = root.find("compiler")
        assert compiler is not None
        assert "meshdir" not in compiler.attrib
        assert "texturedir" not in compiler.attrib
        # other attrs preserved
        assert compiler.get("angle") == "radian"

    def test_rewrites_mesh_file(self):
        xml_in = (
            "<mujoco>"
            '<compiler meshdir="../"/>'
            "<asset>"
            '<mesh name="m1" file="../myo_sim/meshes/foo.stl"/>'
            "</asset>"
            "</mujoco>"
        )
        result = _rewrite_xml_paths(xml_in, mesh_dir="../", texture_dir="")
        root = ET.fromstring(result)
        mesh = root.find(".//mesh")
        assert mesh is not None
        assert mesh.get("file") == "myo_sim/meshes/foo.stl"

    def test_rewrites_texture_file(self):
        xml_in = (
            "<mujoco>"
            '<compiler texturedir="../"/>'
            "<asset>"
            '<texture name="t1" file="../scene/tex.png"/>'
            "</asset>"
            "</mujoco>"
        )
        result = _rewrite_xml_paths(xml_in, mesh_dir="", texture_dir="../")
        root = ET.fromstring(result)
        tex = root.find(".//texture")
        assert tex is not None
        assert tex.get("file") == "scene/tex.png"

    def test_no_change_when_no_dotdot(self):
        xml_in = (
            "<mujoco>"
            '<compiler meshdir="assets"/>'
            "<asset>"
            '<mesh name="m1" file="robot.stl"/>'
            "</asset>"
            "</mujoco>"
        )
        result = _rewrite_xml_paths(xml_in, mesh_dir="assets", texture_dir="")
        root = ET.fromstring(result)
        mesh = root.find(".//mesh")
        assert mesh is not None
        assert mesh.get("file") == "assets/robot.stl"

    def test_hfield_file_rewritten(self):
        xml_in = (
            '<mujoco><asset><hfield name="h1" file="../terrain.png"/></asset></mujoco>'
        )
        result = _rewrite_xml_paths(xml_in, mesh_dir="", texture_dir="")
        root = ET.fromstring(result)
        hfield = root.find(".//hfield")
        assert hfield is not None
        assert hfield.get("file") == "terrain.png"

    def test_empty_meshdir_leaves_file_unchanged(self):
        xml_in = (
            '<mujoco><asset><mesh name="m1" file="meshes/foo.stl"/></asset></mujoco>'
        )
        result = _rewrite_xml_paths(xml_in, mesh_dir="", texture_dir="")
        root = ET.fromstring(result)
        mesh = root.find(".//mesh")
        assert mesh is not None
        assert mesh.get("file") == "meshes/foo.stl"

    def test_mesh_without_file_attr_ignored(self):
        xml_in = '<mujoco><asset><mesh name="m1"/></asset></mujoco>'
        # Should not raise
        result = _rewrite_xml_paths(xml_in, mesh_dir="../", texture_dir="")
        root = ET.fromstring(result)
        mesh = root.find(".//mesh")
        assert mesh is not None
        assert mesh.get("file") is None

    def test_absolute_texture_path_becomes_basename(self):
        """Absolute paths emitted by MjSpec.to_xml() must be reduced to basename."""
        abs_path = "/Users/alice/.venv/lib/site-packages/pkg/scene/skybox.png"
        xml_in = (
            "<mujoco>"
            '<compiler texturedir=".."/>'
            "<asset>"
            f'<texture type="skybox" name="sky" file="{abs_path}"/>'
            "</asset>"
            "</mujoco>"
        )
        result = _rewrite_xml_paths(xml_in, mesh_dir="", texture_dir="..")
        root = ET.fromstring(result)
        tex = root.find(".//texture")
        assert tex is not None
        assert tex.get("file") == "skybox.png"

    def test_absolute_mesh_path_becomes_basename(self):
        abs_path = "/home/user/.venv/lib/site-packages/pkg/meshes/robot.stl"
        xml_in = (
            "<mujoco>"
            '<compiler meshdir=".."/>'
            "<asset>"
            f'<mesh name="robot" file="{abs_path}"/>'
            "</asset>"
            "</mujoco>"
        )
        result = _rewrite_xml_paths(xml_in, mesh_dir="..", texture_dir="")
        root = ET.fromstring(result)
        mesh = root.find(".//mesh")
        assert mesh is not None
        assert mesh.get("file") == "robot.stl"


# ===========================================================================
# to_zip_deflated — unit-level (synthetic MjSpec)
# ===========================================================================
class TestToZipDeflatedUnit:
    @pytest.fixture()
    def simple_spec(self, tmp_path: Path):
        """Create a minimal MuJoCo model (no external mesh files)."""
        xml = (
            '<mujoco model="test">'
            "<worldbody>"
            "<body>"
            '<geom type="sphere" size="0.1"/>'
            "</body>"
            "</worldbody>"
            "</mujoco>"
        )
        xml_path = tmp_path / "model.xml"
        xml_path.write_text(xml)
        return mujoco.MjSpec.from_file(str(xml_path))

    def test_zip_is_deflate_compressed(self, simple_spec):
        buf = io.BytesIO()
        to_zip_deflated(simple_spec, buf)
        buf.seek(0)
        with zipfile.ZipFile(buf, "r") as zf:
            for info in zf.infolist():
                assert info.compress_type == zipfile.ZIP_DEFLATED

    def test_zip_contains_xml(self, simple_spec):
        buf = io.BytesIO()
        to_zip_deflated(simple_spec, buf)
        buf.seek(0)
        with zipfile.ZipFile(buf, "r") as zf:
            names = zf.namelist()
            assert "test.xml" in names

    def test_zip_xml_has_no_meshdir(self, simple_spec):
        buf = io.BytesIO()
        to_zip_deflated(simple_spec, buf)
        buf.seek(0)
        with zipfile.ZipFile(buf, "r") as zf:
            xml = zf.read("test.xml").decode()
            root = ET.fromstring(xml)
            for compiler in root.iter("compiler"):
                assert "meshdir" not in compiler.attrib

    def test_no_dotdot_in_zip_entries(self, simple_spec):
        buf = io.BytesIO()
        to_zip_deflated(simple_spec, buf)
        buf.seek(0)
        with zipfile.ZipFile(buf, "r") as zf:
            for name in zf.namelist():
                assert ".." not in name, f"ZIP entry contains '..': {name}"

    def test_write_to_filepath(self, simple_spec, tmp_path: Path):
        out = str(tmp_path / "sub" / "model.mjz")
        to_zip_deflated(simple_spec, out)
        assert os.path.isfile(out)
        with zipfile.ZipFile(out, "r") as zf:
            assert "test.xml" in zf.namelist()


# ===========================================================================
# to_zip_deflated — integration with real demo models
# ===========================================================================

# Models that had the .. path issue (MyoSuite uses meshdir=".." extensively)
MYOSUITE_MODELS = [
    "assets/scene/myosuite/myosuite/simhive/myo_sim/hand/myohand.xml",
    "assets/scene/myosuite/myosuite/simhive/myo_sim/arm/myoarm.xml",
    "assets/scene/myosuite/myosuite/simhive/myo_sim/elbow/myoelbow_2dof6muscles.xml",
    "assets/scene/myosuite/myosuite/simhive/myo_sim/leg/myolegs.xml",
    "assets/scene/myosuite/myosuite/simhive/myo_sim/finger/myofinger_v0.xml",
    "assets/scene/myosuite/myosuite/envs/myo/assets/arm/myoarm_relocate.xml",
]

# A model with no .. paths (sanity check)
SIMPLE_MODELS = [
    "assets/scene/mujoco_menagerie/shadow_hand/scene_left.xml",
    "assets/scene/mjswan/unitree_g1/scene.xml",
]


@pytest.mark.parametrize("model_path", MYOSUITE_MODELS + SIMPLE_MODELS)
def test_to_zip_deflated_no_dotdot_entries(model_path: str):
    """ZIP entries must not contain '..' path components."""
    spec = mujoco.MjSpec.from_file(str(DEMO_DIR / model_path))
    buf = io.BytesIO()
    to_zip_deflated(spec, buf)
    buf.seek(0)
    with zipfile.ZipFile(buf, "r") as zf:
        for name in zf.namelist():
            assert ".." not in name, (
                f"ZIP entry contains '..': {name} (model: {model_path})"
            )


@pytest.mark.parametrize("model_path", MYOSUITE_MODELS + SIMPLE_MODELS)
def test_to_zip_deflated_no_dotdot_in_xml(model_path: str):
    """The XML inside the ZIP must not reference meshdir/texturedir with '..'."""
    spec = mujoco.MjSpec.from_file(str(DEMO_DIR / model_path))
    buf = io.BytesIO()
    to_zip_deflated(spec, buf)
    buf.seek(0)
    with zipfile.ZipFile(buf, "r") as zf:
        xml_names = [n for n in zf.namelist() if n.endswith(".xml")]
        assert len(xml_names) == 1
        root = ET.fromstring(zf.read(xml_names[0]))
        for compiler in root.iter("compiler"):
            assert "meshdir" not in compiler.attrib
            assert "texturedir" not in compiler.attrib
        for mesh in root.iter("mesh"):
            f = mesh.get("file", "")
            assert not f.startswith(".."), f"mesh file starts with '..': {f}"


@pytest.mark.parametrize("model_path", MYOSUITE_MODELS + SIMPLE_MODELS)
def test_to_zip_deflated_includes_assets(model_path: str):
    """ZIP must contain asset files (meshes/textures) alongside the XML."""
    spec = mujoco.MjSpec.from_file(str(DEMO_DIR / model_path))
    buf = io.BytesIO()
    to_zip_deflated(spec, buf)
    buf.seek(0)
    with zipfile.ZipFile(buf, "r") as zf:
        non_xml = [n for n in zf.namelist() if not n.endswith(".xml")]
        # The zip should include at least one asset file for models with meshes
        if any(m.file for m in spec.meshes):
            assert len(non_xml) > 0, "Expected at least one asset file in ZIP"


@pytest.mark.parametrize("model_path", MYOSUITE_MODELS)
def test_to_zip_deflated_loadable(model_path: str):
    """The generated ZIP must produce a loadable model when extracted."""
    spec = mujoco.MjSpec.from_file(str(DEMO_DIR / model_path))
    buf = io.BytesIO()
    to_zip_deflated(spec, buf)
    buf.seek(0)

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(buf, "r") as zf:
            zf.extractall(tmpdir)
            xml_names = [n for n in zf.namelist() if n.endswith(".xml")]

        xml_path = os.path.join(tmpdir, xml_names[0])
        model = mujoco.MjModel.from_xml_path(xml_path)
        assert model.nq > 0, "Model loaded but has zero DOFs"


@pytest.mark.parametrize("model_path", MYOSUITE_MODELS + SIMPLE_MODELS)
def test_to_zip_deflated_no_absolute_paths_in_xml(model_path: str):
    """The XML inside the ZIP must not contain absolute file paths."""
    spec = mujoco.MjSpec.from_file(str(DEMO_DIR / model_path))
    buf = io.BytesIO()
    to_zip_deflated(spec, buf)
    buf.seek(0)
    with zipfile.ZipFile(buf, "r") as zf:
        xml_names = [n for n in zf.namelist() if n.endswith(".xml")]
        assert len(xml_names) == 1
        root = ET.fromstring(zf.read(xml_names[0]))
        for tag in ("mesh", "texture", "hfield", "skin"):
            _TEX_FILE_ATTRS = (
                "file",
                "fileup",
                "fileback",
                "filedown",
                "filefront",
                "fileleft",
                "fileright",
            )
            for elem in root.iter(tag):
                for attr in _TEX_FILE_ATTRS:
                    f = elem.get(attr, "")
                    assert not f.startswith("/"), (
                        f"<{tag} {attr}='{f}'> is an absolute path (model: {model_path})"
                    )


@pytest.mark.parametrize("model_path", MYOSUITE_MODELS + SIMPLE_MODELS)
def test_to_zip_deflated_no_absolute_paths_in_zip_entries(model_path: str):
    """ZIP entry names must not be absolute paths."""
    spec = mujoco.MjSpec.from_file(str(DEMO_DIR / model_path))
    buf = io.BytesIO()
    to_zip_deflated(spec, buf)
    buf.seek(0)
    with zipfile.ZipFile(buf, "r") as zf:
        for name in zf.namelist():
            assert not name.startswith("/"), (
                f"ZIP entry is an absolute path: {name} (model: {model_path})"
            )


@pytest.mark.parametrize("model_path", MYOSUITE_MODELS + SIMPLE_MODELS)
def test_to_zip_deflated_spec_unchanged(model_path: str):
    """to_zip_deflated must not mutate the input spec's file references."""
    spec = mujoco.MjSpec.from_file(str(DEMO_DIR / model_path))
    # Note: spec.to_xml() (called internally) may normalise meshdir by
    # appending a trailing slash.  We therefore call to_xml() once upfront
    # so the "before" snapshot reflects any MuJoCo normalisation, then
    # verify our code does not add further changes.
    spec.to_xml()
    orig_meshdir = spec.meshdir
    orig_texturedir = spec.texturedir
    orig_files = [(m.name, m.file) for m in spec.meshes]

    buf = io.BytesIO()
    to_zip_deflated(spec, buf)

    assert spec.meshdir == orig_meshdir
    assert spec.texturedir == orig_texturedir
    assert [(m.name, m.file) for m in spec.meshes] == orig_files


# ===========================================================================
# collect_spec_assets
# ===========================================================================
class TestCollectSpecAssets:
    def test_collects_mesh_files(self):
        spec = mujoco.MjSpec.from_file(
            str(DEMO_DIR / "assets/scene/mujoco_menagerie/shadow_hand/scene_left.xml")
        )
        assets = collect_spec_assets(spec)
        assert len(assets) > 0
        assert all(isinstance(v, bytes) for v in assets.values())

    def test_myosuite_collects_stl_files(self):
        spec = mujoco.MjSpec.from_file(
            str(
                DEMO_DIR
                / "assets/scene/myosuite/myosuite/simhive/myo_sim/hand/myohand.xml"
            )
        )
        assets = collect_spec_assets(spec)
        stl_keys = [k for k in assets if k.endswith(".stl")]
        assert len(stl_keys) > 0, "Expected STL mesh files in collected assets"
