from __future__ import annotations

import base64
import copy
from collections import defaultdict
import os
import re
import tempfile
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Optional

import numpy as np

from .presets import get_default_catalog, workspace_root
from .scene import build_scene_from_atoms, legacy_scene, pc, scene_json, scene_metadata, scene_ops


@dataclass
class LoadedCrystal:
    name: str
    title: str
    cif_path: str
    scene: Dict[str, Any]
    raw_atoms: list[dict[str, Any]] = field(default_factory=list)
    cell: Any | None = None
    M: Any | None = None
    view_direction: list[float] = field(default_factory=list)
    up: list[float] = field(default_factory=list)
    scene_cache: dict[tuple[str, bool], Dict[str, Any]] = field(default_factory=dict)
    pymatgen_structure: Any | None = None
    crystal: Any | None = None
    fragment_table: list[dict[str, Any]] = field(default_factory=list)
    topology_fragment_table: list[dict[str, Any]] = field(default_factory=list)
    atom_fragment_labels: list[str] = field(default_factory=list)
    source: str = "catalog"

    def metadata(self) -> Dict[str, Any]:
        meta = scene_metadata(self.scene)
        meta.update({
            "source": self.source,
            "fragment_count": len(self.topology_fragment_table or self.fragment_table),
            "has_topology": bool(self.topology_fragment_table or self.fragment_table),
        })
        return meta


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "uploaded"


def _unique_name(base: str, existing: Iterable[str]) -> str:
    existing_set = set(existing)
    if base not in existing_set:
        return base
    idx = 2
    while f"{base}_{idx}" in existing_set:
        idx += 1
    return f"{base}_{idx}"


def _infer_title_from_scene(scene: Dict[str, Any]) -> str:
    title = scene.get("title")
    if title:
        return str(title)
    return scene.get("name", "Uploaded Structure")


def build_empty_bundle(
    *,
    name: str = "__upload__",
    title: str = "Upload CIF to begin",
) -> LoadedCrystal:
    cell = SimpleNamespace(
        a=1.0,
        b=1.0,
        c=1.0,
        alpha=90.0,
        beta=90.0,
        gamma=90.0,
        volume=1.0,
    )
    M = np.eye(3, dtype=float)
    R = np.eye(3, dtype=float)
    scene = {
        "name": name,
        "title": title,
        "cell": cell,
        "M": M,
        "R": R,
        "view_x": np.array([1.0, 0.0, 0.0], dtype=float),
        "view_y": np.array([0.0, 1.0, 0.0], dtype=float),
        "view_z": np.array([0.0, 0.0, 1.0], dtype=float),
        "selected_atoms": [],
        "draw_atoms": [],
        "bonds": [],
        "label_items": [],
        "bounds": {
            "center": [0.0, 0.0, 0.0],
            "ranges": [1.0, 1.0, 1.0],
            "mins": [0.0, 0.0, 0.0],
            "maxs": [1.0, 1.0, 1.0],
            "screen_ranges": [1.0, 1.0, 1.0],
        },
        "camera": {
            "position": [0.0, 0.0, 8.0],
            "focal_point": [0.0, 0.0, 0.0],
            "up": [0.0, 1.0, 0.0],
        },
        "style": {},
        "show_hydrogen": False,
        "has_minor": False,
        "preset_entry": {},
        "display_mode": "formula_unit",
        "cif_path": None,
        "view_direction": np.array([0.0, 0.0, 1.0], dtype=float),
        "up": np.array([0.0, 1.0, 0.0], dtype=float),
        "fragment_table": [],
        "atom_fragment_labels": [],
    }
    return LoadedCrystal(
        name=name,
        title=title,
        cif_path="",
        scene=scene,
        raw_atoms=[],
        cell=cell,
        M=M,
        view_direction=[0.0, 0.0, 1.0],
        up=[0.0, 1.0, 0.0],
        scene_cache={("formula_unit", False): scene},
        fragment_table=[],
        topology_fragment_table=[],
        atom_fragment_labels=[],
        source="placeholder",
    )


def _cluster_components(n_items: int, pairs: Iterable[tuple[int, int]]) -> list[list[int]]:
    parents = list(range(n_items))

    def find(idx: int) -> int:
        while parents[idx] != idx:
            parents[idx] = parents[parents[idx]]
            idx = parents[idx]
        return idx

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parents[ra] = rb

    for i, j in pairs:
        union(int(i), int(j))

    groups: dict[int, list[int]] = defaultdict(list)
    for idx in range(n_items):
        groups[find(idx)].append(idx)
    return [sorted(group) for _, group in sorted(groups.items(), key=lambda item: min(item[1]))]


def _fragment_table_from_atoms(
    bundle_name: str,
    atoms,
    cell,
    M,
    *,
    use_source_indices: bool = True,
) -> tuple[list[dict[str, Any]], list[str]]:
    ops = scene_ops()
    atom_pool = []
    source_indices = []
    for idx, atom in enumerate(atoms):
        if ops.is_minor(atom):
            continue
        atom_pool.append(dict(atom))
        source_indices.append(idx if use_source_indices else len(source_indices))
    if not atom_pool:
        return [], []

    bond_pairs = ops.find_bonds(atom_pool, cell=cell)
    components = _cluster_components(len(atom_pool), bond_pairs)
    for component in components:
        atom_pool = pc.assemble_component_p1(atom_pool, component, bond_pairs, M)

    fragments = []
    for component in components:
        site_indices = sorted(source_indices[idx] for idx in component)
        component_atoms = [atom_pool[idx] for idx in component]
        heavy_atoms = [atom for atom in component_atoms if atom["elem"] != "H"]
        center_atoms = heavy_atoms or component_atoms
        elem_set = {atom["elem"] for atom in heavy_atoms}
        if not center_atoms:
            continue
        center_cart = np.mean([atom["cart"] for atom in center_atoms], axis=0)
        center_frac = np.mean([atom["frac"] for atom in center_atoms], axis=0)
        fragments.append({
            "site_indices": site_indices,
            "center": [float(x) for x in center_cart],
            "frac_center": [float(x) for x in center_frac],
            "elem_set": sorted(elem_set),
            "heavy_atom_count": len(heavy_atoms),
            "cluster_size": len(component_atoms),
            "species": "".join(sorted(elem_set)) or "?",
        })

    x_fragments = [frag for frag in fragments if "Cl" in frag["elem_set"]]
    organic_fragments = [frag for frag in fragments if "Cl" not in frag["elem_set"] and ("C" in frag["elem_set"] or "N" in frag["elem_set"])]
    other_fragments = [frag for frag in fragments if frag not in x_fragments and frag not in organic_fragments]

    if organic_fragments:
        min_heavy = min(frag["heavy_atom_count"] for frag in organic_fragments)
        for frag in organic_fragments:
            frag["type"] = "B" if frag["heavy_atom_count"] == min_heavy else "A"
    for frag in x_fragments:
        frag["type"] = "X"
    for frag in other_fragments:
        frag["type"] = "?"

    type_order = {"B": 0, "A": 1, "X": 2, "?": 3}
    fragments.sort(
        key=lambda frag: (
            type_order.get(frag["type"], 9),
            *[float(x % 1.0) for x in frag["frac_center"]],
            frag["heavy_atom_count"],
            frag["cluster_size"],
        )
    )

    counters: dict[str, int] = defaultdict(int)
    atom_fragment_labels = ["?"] * len(atoms)
    final_table = []
    for frag_idx, frag in enumerate(fragments):
        frag_type = frag["type"]
        label_index = counters[frag_type]
        counters[frag_type] += 1
        for site_idx in frag["site_indices"]:
            atom_fragment_labels[site_idx] = frag_type
        final_table.append({
            "index": frag_idx,
            "type": frag_type,
            "label": f"{frag_type}{label_index}",
            "species": frag["species"],
            "center": frag["center"],
            "frac_center": frag["frac_center"],
            "site_indices": frag["site_indices"],
            "source": bundle_name,
            "heavy_atom_count": frag["heavy_atom_count"],
            "cluster_size": frag["cluster_size"],
        })
    return final_table, atom_fragment_labels


def build_bundle_scene(
    bundle: LoadedCrystal,
    *,
    display_mode: str = "formula_unit",
    show_hydrogen: bool = False,
    preset: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cache_key = (display_mode, bool(show_hydrogen))
    if cache_key in bundle.scene_cache:
        return bundle.scene_cache[cache_key]

    ops = scene_ops()
    view_dir = np.array(bundle.view_direction, dtype=float)
    up = np.array(bundle.up, dtype=float)
    R = ops.view_rotation(view_dir, up)
    scene = build_scene_from_atoms(
        name=bundle.name,
        title=bundle.title,
        atoms=bundle.raw_atoms,
        cell=bundle.cell,
        M=bundle.M,
        R=R,
        show_hydrogen=show_hydrogen,
        preset=preset,
        display_mode=display_mode,
        ops=ops,
    )
    scene["cif_path"] = bundle.cif_path
    scene["view_direction"] = view_dir
    scene["up"] = up
    fragment_table, atom_fragment_labels = _fragment_table_from_atoms(
        bundle.name,
        scene["draw_atoms"],
        scene["cell"],
        scene["M"],
        use_source_indices=False,
    )
    scene["fragment_table"] = fragment_table
    scene["atom_fragment_labels"] = atom_fragment_labels
    bundle.scene_cache[cache_key] = scene
    return scene


def build_loaded_crystal(
    *,
    name: str,
    cif_path: str,
    title: Optional[str] = None,
    preset: Optional[Dict[str, Any]] = None,
    source: str = "catalog",
) -> LoadedCrystal:
    ops = scene_ops()
    preset = preset or {}
    raw_atoms, cell, M = ops.parse_asu(cif_path)
    view_dir, up = legacy_scene._resolve_view(ops, name, raw_atoms, M, cell, preset)
    R = ops.view_rotation(view_dir, up)
    final_title = title or name
    initial_scene = build_scene_from_atoms(
        name=name,
        title=final_title,
        atoms=raw_atoms,
        cell=cell,
        M=M,
        R=R,
        preset=preset,
        show_hydrogen=False,
        display_mode="formula_unit",
        ops=ops,
    )
    initial_scene["cif_path"] = cif_path
    initial_scene["view_direction"] = np.array(view_dir, dtype=float)
    initial_scene["up"] = np.array(up, dtype=float)
    fragment_table, atom_fragment_labels = _fragment_table_from_atoms(
        name,
        initial_scene["draw_atoms"],
        initial_scene["cell"],
        initial_scene["M"],
        use_source_indices=False,
    )
    initial_scene["fragment_table"] = fragment_table
    initial_scene["atom_fragment_labels"] = atom_fragment_labels
    topology_fragment_table, _ = _fragment_table_from_atoms(name, raw_atoms, cell, M, use_source_indices=True)

    bundle = LoadedCrystal(
        name=name,
        title=final_title,
        cif_path=cif_path,
        scene=initial_scene,
        raw_atoms=[dict(atom) for atom in raw_atoms],
        cell=cell,
        M=M,
        view_direction=np.array(view_dir, dtype=float).tolist(),
        up=np.array(up, dtype=float).tolist(),
        scene_cache={("formula_unit", False): initial_scene},
        fragment_table=fragment_table,
        topology_fragment_table=topology_fragment_table,
        atom_fragment_labels=atom_fragment_labels,
        source=source,
    )
    return bundle


def load_default_catalog(
    *,
    root_dir: Optional[str] = None,
    names: Optional[Iterable[str]] = None,
    preset: Optional[Dict[str, Any]] = None,
) -> Dict[str, LoadedCrystal]:
    catalog = get_default_catalog(root_dir=root_dir or workspace_root())
    selected = list(names) if names else list(catalog.keys())
    loaded = {}
    for name in selected:
        entry = catalog[name]
        loaded[name] = build_loaded_crystal(
            name=name,
            cif_path=entry["cif_path"],
            title=entry["title"],
            preset=preset,
            source="catalog",
        )
    return loaded


def infer_uploaded_name(filename: str, existing_names: Iterable[str]) -> str:
    stem = os.path.splitext(os.path.basename(filename))[0]
    return _unique_name(_slugify(stem), existing_names)


def write_uploaded_cif(contents: str, filename: str, upload_dir: Optional[str] = None) -> str:
    if not contents.startswith("data:"):
        raise ValueError("Dash upload contents must be a data URL.")
    header, encoded = contents.split(",", 1)
    if "base64" not in header:
        raise ValueError("Only base64 CIF uploads are supported.")
    data = base64.b64decode(encoded)
    target_dir = upload_dir or os.path.join(tempfile.gettempdir(), "crystal_viewer_uploads")
    os.makedirs(target_dir, exist_ok=True)
    safe_name = _slugify(filename)
    path = os.path.join(target_dir, safe_name)
    with open(path, "wb") as handle:
        handle.write(data)
    return path


def load_uploaded_cif(
    *,
    contents: str,
    filename: str,
    existing_names: Iterable[str],
    preset: Optional[Dict[str, Any]] = None,
    upload_dir: Optional[str] = None,
) -> LoadedCrystal:
    cif_path = write_uploaded_cif(contents, filename, upload_dir=upload_dir)
    name = infer_uploaded_name(filename, existing_names)
    title = os.path.splitext(os.path.basename(filename))[0]
    return build_loaded_crystal(name=name, cif_path=cif_path, title=title, preset=preset, source="upload")


def bundle_json(bundle: LoadedCrystal) -> Dict[str, Any]:
    return {
        "name": bundle.name,
        "title": bundle.title,
        "cif_path": bundle.cif_path,
        "scene": scene_json(bundle.scene),
        "fragment_table": copy.deepcopy(bundle.fragment_table),
        "topology_fragment_table": copy.deepcopy(bundle.topology_fragment_table),
        "source": bundle.source,
    }
