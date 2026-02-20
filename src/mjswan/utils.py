"""Utility functions for mjswan."""

from __future__ import annotations

import os
import posixpath
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import IO, Union

import mujoco


def _strip_leading_dotdot(path: str) -> str:
    """Normalize a POSIX path and strip any leading ``..`` components."""
    normalized = posixpath.normpath(path)
    if normalized == ".":
        return ""
    parts = normalized.split("/")
    while parts and parts[0] == "..":
        parts.pop(0)
    return "/".join(parts) if parts else ""


def _make_zip_safe_path(path: str) -> str:
    """Return a ZIP-safe relative path.

    Absolute paths (e.g. those that MuJoCo resolves when loading a spec) are
    reduced to their basename so the ZIP entry and the rewritten XML reference
    only the filename.  Relative paths have leading ``..`` components stripped
    by the :func:`_strip_leading_dotdot` logic.
    """
    posix_path = path.replace("\\", "/")
    # Absolute POSIX path (/foo/...) or Windows drive path (C:/foo/...)
    is_absolute = posixpath.isabs(posix_path) or (
        len(posix_path) >= 3 and posix_path[1] == ":" and posix_path[2] == "/"
    )
    if is_absolute:
        return posixpath.basename(posix_path)
    return _strip_leading_dotdot(posix_path)


def collect_spec_assets(spec: mujoco.MjSpec) -> dict[str, bytes]:
    """Collect all asset files referenced by a MjSpec from disk.

    Uses ``spec.modelfiledir``, ``spec.meshdir``, and ``spec.texturedir``
    to resolve file paths.  Returns a dictionary mapping POSIX-style relative
    paths (suitable for ZIP entries) to file contents.
    """
    base_dir = spec.modelfiledir or ""
    mesh_dir = spec.meshdir or ""
    texture_dir = spec.texturedir or ""

    assets: dict[str, bytes] = {}

    def _read(dir_hint: str, filename: str) -> None:
        if not filename:
            return
        # POSIX-style key for ZIP entries / MuJoCo asset references
        rel = posixpath.join(dir_hint, filename) if dir_hint else filename
        # OS-native path for filesystem reads
        full = os.path.join(base_dir, rel)
        if os.path.isfile(full):
            assets[_make_zip_safe_path(rel)] = Path(full).read_bytes()

    # Meshes
    for mesh in spec.meshes:
        _read(mesh_dir, mesh.file)

    # Textures (single file and cube-map faces)
    for texture in spec.textures:
        _read(texture_dir, texture.file)
        for i in range(len(texture.cubefiles)):
            _read(texture_dir, texture.cubefiles[i])

    # Heightfields
    for hfield in spec.hfields:
        _read("", hfield.file)

    # Skins
    for skin in spec.skins:
        _read("", skin.file)

    return assets


def _rewrite_xml_paths(xml_str: str, mesh_dir: str, texture_dir: str) -> str:
    """Rewrite asset paths in MuJoCo XML so the file is self-contained.

    Resolves ``meshdir``/``texturedir`` + ``file`` into a single normalised
    path (with leading ``..`` stripped), then removes the directory hints
    from ``<compiler>`` elements.
    """
    root = ET.fromstring(xml_str)

    # Rewrite <compiler> meshdir / texturedir
    for compiler in root.iter("compiler"):
        compiler.attrib.pop("meshdir", None)
        compiler.attrib.pop("texturedir", None)

    # Rewrite <mesh file="...">
    for mesh in root.iter("mesh"):
        f = mesh.get("file")
        if f:
            rel = posixpath.join(mesh_dir, f) if mesh_dir else f
            mesh.set("file", _make_zip_safe_path(rel))

    # Rewrite <texture file/cube-map="...">
    _TEX_FILE_ATTRS = (
        "file",
        "fileup",
        "fileback",
        "filedown",
        "filefront",
        "fileleft",
        "fileright",
    )
    for tex in root.iter("texture"):
        for attr in _TEX_FILE_ATTRS:
            f = tex.get(attr)
            if f:
                rel = posixpath.join(texture_dir, f) if texture_dir else f
                tex.set(attr, _make_zip_safe_path(rel))

    # Rewrite <hfield file="..."> and <skin file="...">
    for tag in ("hfield", "skin"):
        for elem in root.iter(tag):
            f = elem.get("file")
            if f:
                elem.set("file", _make_zip_safe_path(f))

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True) + "\n"


def to_zip_deflated(spec: mujoco.MjSpec, file: Union[str, IO[bytes]]) -> None:
    """Save an MjSpec as a ZIP file with DEFLATE compression.

    This is a compressed alternative to ``mujoco.to_zip`` which stores
    entries uncompressed.  The output is a standard ZIP that JSZip (and
    other readers) can decompress transparently.

    The function collects asset files from disk, normalises all relative
    paths (stripping leading ``..`` components) so that the resulting ZIP
    is self-contained.
    """
    base_dir = spec.modelfiledir or ""
    mesh_dir = spec.meshdir or ""
    texture_dir = spec.texturedir or ""

    # Collect asset files from disk with normalised zip-entry paths
    files_to_zip: dict[str, bytes | str] = {}

    for mesh in spec.meshes:
        if not mesh.file:
            continue
        rel = posixpath.join(mesh_dir, mesh.file) if mesh_dir else mesh.file
        full = os.path.join(base_dir, rel)
        if os.path.isfile(full):
            files_to_zip[_make_zip_safe_path(rel)] = Path(full).read_bytes()

    for texture in spec.textures:
        for fname in [texture.file] + [
            texture.cubefiles[i] for i in range(len(texture.cubefiles))
        ]:
            if not fname:
                continue
            rel = posixpath.join(texture_dir, fname) if texture_dir else fname
            full = os.path.join(base_dir, rel)
            if os.path.isfile(full):
                files_to_zip[_make_zip_safe_path(rel)] = Path(full).read_bytes()

    for hfield in spec.hfields:
        if not hfield.file:
            continue
        full = os.path.join(base_dir, hfield.file)
        if os.path.isfile(full):
            files_to_zip[_make_zip_safe_path(hfield.file)] = Path(full).read_bytes()

    for skin in spec.skins:
        if not skin.file:
            continue
        full = os.path.join(base_dir, skin.file)
        if os.path.isfile(full):
            files_to_zip[_make_zip_safe_path(skin.file)] = Path(full).read_bytes()

    # Generate XML and rewrite paths
    xml_str = spec.to_xml()
    xml_str = _rewrite_xml_paths(xml_str, mesh_dir, texture_dir)
    files_to_zip[spec.modelname + ".xml"] = xml_str

    # Write the ZIP
    if isinstance(file, str):
        directory = os.path.dirname(file)
        os.makedirs(directory, exist_ok=True)
        file = open(file, "wb")
    with zipfile.ZipFile(file, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for filename, contents in files_to_zip.items():
            zf.writestr(filename, contents)


def name2id(name: str) -> str:
    """Convert a name to a URL-friendly identifier.

    This function normalizes names by converting them to lowercase
    and replacing spaces and hyphens with underscores, making them
    suitable for use in URLs, file paths, and identifiers.

    Args:
        name: The name to sanitize.

    Returns:
        A URL-friendly identifier string with lowercase letters,
        underscores instead of spaces and hyphens.

    Examples:
        >>> name2id("My Project")
        'my_project'
        >>> name2id("Test-Scene")
        'test_scene'
        >>> name2id("Complex Name-With Spaces")
        'complex_name_with_spaces'
    """
    return name.lower().replace(" ", "_").replace("-", "_")
