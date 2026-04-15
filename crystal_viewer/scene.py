from __future__ import annotations

import copy
import os
from typing import Any, Dict, Optional

import numpy as np

from .presets import DEFAULT_STYLE, deep_merge, default_preset, json_safe


PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(PACKAGE_DIR)
from .legacy import crystal_scene as legacy_scene  # noqa: E402
from .legacy import plot_crystal as pc  # noqa: E402


def scene_ops():
    return pc._scene_ops()


def _to_builtin(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    return value


def scene_style(scene: Dict[str, Any], override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    style = copy.deepcopy(DEFAULT_STYLE)
    style.update(scene.get("style", {}))
    if override:
        style.update(override)
    return style


def scene_metadata(scene: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": scene["name"],
        "title": scene["title"],
        "has_minor": bool(scene.get("has_minor", False)),
        "atom_count": len(scene.get("draw_atoms", [])),
        "bond_count": len(scene.get("bonds", [])),
        "cif_path": scene.get("cif_path"),
    }


def scene_json(scene: Dict[str, Any]) -> Dict[str, Any]:
    payload = {}
    for key, value in scene.items():
        if key == "cell":
            payload[key] = {
                "a": float(value.a),
                "b": float(value.b),
                "c": float(value.c),
                "alpha": float(value.alpha),
                "beta": float(value.beta),
                "gamma": float(value.gamma),
                "volume": float(value.volume),
            }
        else:
            payload[key] = _to_builtin(value)
    return payload


def rebuild_scene_with_style(scene: Dict[str, Any], style: Dict[str, Any]) -> Dict[str, Any]:
    updated = dict(scene)
    updated["style"] = scene_style(scene, style)
    return updated


def _asymmetric_unit_atoms(atoms):
    selected = []
    seen = set()
    for atom in atoms:
        key = (
            atom.get("label"),
            atom.get("elem"),
            atom.get("dg", "").strip(),
            atom.get("da", "").strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        selected.append(dict(atom))
    return selected


def _continuous_components(ops: Any, atoms, M, cell):
    atoms_out = [dict(atom) for atom in atoms]
    bond_pairs = ops.find_bonds(atoms_out, cell=cell)
    clusters = pc.cluster_atoms(atoms_out, bonds=bond_pairs)
    ordered = [sorted(idxs) for _, idxs in sorted(clusters.items(), key=lambda item: min(item[1]))]
    for idxs in ordered:
        atoms_out = pc.assemble_component_p1(atoms_out, idxs, bond_pairs, M)
    return atoms_out, ordered


def _best_component_shift_frac(component_atoms) -> np.ndarray:
    best_shift = np.zeros(3, dtype=float)
    best_score = np.inf
    fracs = np.array([atom["frac"] for atom in component_atoms], dtype=float)
    for na in range(-2, 3):
        for nb in range(-2, 3):
            for nc in range(-2, 3):
                shift = np.array([na, nb, nc], dtype=float)
                shifted = fracs + shift[None, :]
                lower = np.clip(-shifted, 0.0, None)
                upper = np.clip(shifted - 1.0, 0.0, None)
                outside_penalty = float(np.sum(lower * lower + upper * upper))
                center_penalty = float(np.linalg.norm(shifted.mean(axis=0) - 0.5))
                score = outside_penalty * 50.0 + center_penalty
                if score < best_score:
                    best_score = score
                    best_shift = shift
    return best_shift


def _translate_component_frac(atoms, idxs, shift_frac, M):
    shift_frac = np.array(shift_frac, dtype=float)
    shift_cart = M @ shift_frac
    translated = [dict(atom) for atom in atoms]
    for idx in idxs:
        translated[idx]["frac"] = np.array(translated[idx]["frac"], dtype=float) + shift_frac
        translated[idx]["cart"] = np.array(translated[idx]["cart"], dtype=float) + shift_cart
    return translated


def _whole_components_in_box(ops: Any, atoms, M, cell):
    atoms_out, components = _continuous_components(ops, atoms, M, cell)
    for idxs in components:
        component_atoms = [atoms_out[idx] for idx in idxs]
        shift_frac = _best_component_shift_frac(component_atoms)
        atoms_out = _translate_component_frac(atoms_out, idxs, shift_frac, M)
    return atoms_out


def _selected_atoms_for_mode(ops: Any, atoms, M, cell, display_mode: str):
    if display_mode == "unit_cell":
        return _whole_components_in_box(ops, atoms, M, cell)
    if display_mode == "asymmetric_unit":
        asym_atoms = _asymmetric_unit_atoms(atoms)
        return _whole_components_in_box(ops, asym_atoms, M, cell)
    atoms_out, sel_idxs = ops.select_formula_unit(atoms, M, cell)
    return [atoms_out[idx] for idx in sel_idxs]


def _bond_endpoints(ai, aj, cell, display_mode: str):
    start = np.array(ai["cart"], dtype=float)
    if display_mode == "formula_unit":
        end = np.array(aj["cart"], dtype=float)
    else:
        end = np.array(pc._nearest_pbc_cart(ai["cart"], aj["cart"], cell), dtype=float)
    return start, end


def build_scene_from_atoms(
    *,
    name: str,
    title: str,
    atoms,
    cell,
    M,
    R,
    show_hydrogen: bool = False,
    preset: Optional[Dict[str, Any]] = None,
    display_mode: str = "formula_unit",
    ops=None,
) -> Dict[str, Any]:
    ops = scene_ops() if ops is None else ops
    preset = default_preset() if preset is None else preset
    style = deep_merge(DEFAULT_STYLE, preset.get("style"))
    entry = preset.get("structures", {}).get(name, {})
    style = deep_merge(style, entry.get("style"))
    show_h = bool(entry.get("show_hydrogen", style.get("show_hydrogen", show_hydrogen)))

    sel_atoms = _selected_atoms_for_mode(ops, atoms, M, cell, display_mode=display_mode)
    draw_atoms = [dict(atom) for atom in sel_atoms if show_h or atom["elem"] != "H"]

    view_x = np.array(R[0], dtype=float)
    view_y = np.array(R[1], dtype=float)
    view_z = np.array(R[2], dtype=float)

    if draw_atoms:
        depths = np.array([atom["cart"] @ view_z for atom in draw_atoms], dtype=float)
        z_min, z_max = depths.min(), depths.max()
        z_span = max(z_max - z_min, 1e-6)
        for atom, depth in zip(draw_atoms, depths):
            atom["_depth_t"] = float((depth - z_min) / z_span)
            atom["is_minor"] = bool(ops.is_minor(atom))
            atom["disorder_alpha"] = float(ops.disorder_alpha(atom))
            atom["color"] = ops.elem_color(atom["elem"])
            atom["color_light"] = ops.elem_color_light(atom["elem"])
            atom["atom_radius"] = float(ops.atom_r(atom["elem"]))

    bond_pairs = ops.find_bonds(draw_atoms, cell=cell)
    bonds = []
    for i, j in bond_pairs:
        ai = draw_atoms[i]
        aj = draw_atoms[j]
        start, end = _bond_endpoints(ai, aj, cell, display_mode=display_mode)
        bonds.append(
            {
                "i": i,
                "j": j,
                "start": start.copy(),
                "end": end.copy(),
                "color_i": ai["color"],
                "color_j": aj["color"],
                "alpha_i": ai["disorder_alpha"],
                "alpha_j": aj["disorder_alpha"],
                "is_minor": bool(ai["is_minor"] or aj["is_minor"]),
                "depth_t": float((ai["_depth_t"] + aj["_depth_t"]) / 2.0),
            }
        )

    label_items = legacy_scene._label_payload(ops, draw_atoms, view_x, view_y, view_z)
    bounds = legacy_scene._compute_bounds(draw_atoms or sel_atoms, view_x, view_y, view_z)
    camera = entry.get("camera") or legacy_scene._camera_from_bounds(bounds, view_y, view_z)

    return {
        "name": name,
        "title": title,
        "cell": cell,
        "M": M,
        "R": np.array(R, dtype=float),
        "view_x": view_x,
        "view_y": view_y,
        "view_z": view_z,
        "selected_atoms": sel_atoms,
        "draw_atoms": draw_atoms,
        "bonds": bonds,
        "label_items": label_items,
        "bounds": bounds,
        "camera": camera,
        "style": style,
        "show_hydrogen": show_h,
        "has_minor": any(bool(atom["is_minor"]) for atom in draw_atoms),
        "preset_entry": entry,
        "display_mode": display_mode,
    }


def build_scene_from_cif(
    *,
    name: str,
    cif_path: str,
    title: str,
    preset: Optional[Dict[str, Any]] = None,
    show_hydrogen: bool = False,
    display_mode: str = "formula_unit",
    ops=None,
) -> Dict[str, Any]:
    ops = scene_ops() if ops is None else ops
    preset = default_preset() if preset is None else preset
    atoms, cell, M = ops.parse_asu(cif_path)
    view_dir, up = legacy_scene._resolve_view(ops, name, atoms, M, cell, preset)
    R = ops.view_rotation(view_dir, up)
    scene = build_scene_from_atoms(
        name=name,
        title=title,
        atoms=atoms,
        cell=cell,
        M=M,
        R=R,
        preset=preset,
        show_hydrogen=show_hydrogen,
        display_mode=display_mode,
        ops=ops,
    )
    scene["cif_path"] = cif_path
    scene["view_direction"] = np.array(view_dir, dtype=float)
    scene["up"] = np.array(up, dtype=float)
    return scene


def merge_structure_style(preset: Dict[str, Any], name: str, style: Dict[str, Any]) -> Dict[str, Any]:
    merged = default_preset() if preset is None else copy.deepcopy(preset)
    merged["style"] = deep_merge(merged.get("style", {}), style)
    merged.setdefault("structures", {})
    merged["structures"].setdefault(name, {})
    merged["structures"][name]["style"] = json_safe(style)
    return merged
