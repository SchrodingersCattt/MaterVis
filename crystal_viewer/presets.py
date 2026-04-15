from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, Iterable, Optional

import numpy as np


DEFAULT_STYLE = {
    "display_mode": "formula_unit",
    "atom_scale": 1.0,
    "bond_radius": 0.16,
    "major_opacity": 1.0,
    "minor_opacity": 0.35,
    "minor_wireframe": False,
    "minor_bond_scale": 0.82,
    "show_labels": True,
    "show_axes": True,
    "show_hydrogen": False,
    "show_unit_cell": False,
    "show_minor_only": False,
    "depth_cue_enabled": False,
    "background": "#FFFFFF",
    "axis_scale": 0.14,
    "axis_color": "#666666",
    "axis_opacity": 0.72,
    "fast_rendering": False,
    "topology_enabled": True,
}

DEFAULT_CATALOG = {
    "SY": {
        "title": "SY  (Cmc2₁, Z=4)",
        "relative_cif": os.path.join("..", "单晶数据检查文件", "SY", "298k-SY.cif"),
    },
    "PEP": {
        "title": "PEP  (Pbca, Z=16)",
        "relative_cif": os.path.join("..", "单晶数据检查文件", "单晶数据检查PEP", "298k-PEP.cif"),
    },
    "MPEP": {
        "title": "MPEP  (P2₁/c, Z=4)",
        "relative_cif": os.path.join("..", "单晶数据检查文件", "单晶数据检查MPEP", "298K-MPEP.cif"),
    },
    "HPEP": {
        "title": "HPEP  (P2₁/c, Z=4)",
        "relative_cif": os.path.join("..", "单晶数据检查文件", "单晶数据检查HPEP", "298k-HPEP.cif"),
    },
}

LOCAL_STATE_DIRNAME = ".local"
LOCAL_PRESET_FILENAME = "crystal_view_preset.json"
LOCAL_CATALOG_FILENAMES = (
    "catalog.local.json",
    os.path.join(LOCAL_STATE_DIRNAME, "catalog.local.json"),
)

DEFAULT_STRUCTURE_PRESETS = {
    "SY": {
        "view_direction": [-0.24192189559966773, -0.9702957262759965, 0.0],
        "up": [0.0, 0.0, 1.0],
        "show_hydrogen": False,
    },
    "PEP": {
        "view_direction": [-0.2768338134060447, 0.19755746781799224, 0.9403903905636266],
        "up": [0.05579013787329076, 0.9802913071681004, -0.18951626211683326],
        "show_hydrogen": False,
    },
    "MPEP": {
        "view_direction": [0.4112857076968057, -0.43468649161027, 0.8011814530154155],
        "up": [0.15934048006252013, 0.8997166983344863, 0.4063501866020694],
        "show_hydrogen": False,
    },
    "HPEP": {
        "view_direction": [-0.9596418442823493, 1.175222312928242e-16, -0.2812250534756305],
        "up": [1.1277925078202266e-16, 1.0, 3.30501957798999e-17],
        "show_hydrogen": False,
    },
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _deep_merge(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    if not override:
        return merged
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def workspace_root(package_dir: Optional[str] = None) -> str:
    if package_dir is None:
        package_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(package_dir)


def default_preset_path(root_dir: Optional[str] = None) -> str:
    root = workspace_root() if root_dir is None else root_dir
    return os.path.join(root, LOCAL_STATE_DIRNAME, LOCAL_PRESET_FILENAME)


def _resolve_catalog_entry(base_dir: str, entry: Dict[str, Any]) -> Optional[Dict[str, str]]:
    cif_path = entry.get("cif_path")
    if not cif_path:
        return None
    resolved_path = cif_path if os.path.isabs(cif_path) else os.path.normpath(os.path.join(base_dir, cif_path))
    if not os.path.exists(resolved_path):
        return None
    title = str(entry.get("title") or os.path.splitext(os.path.basename(resolved_path))[0])
    return {
        "title": title,
        "cif_path": resolved_path,
    }


def _load_local_catalog(root: str) -> Dict[str, Dict[str, str]]:
    catalog: Dict[str, Dict[str, str]] = {}
    for relative_path in LOCAL_CATALOG_FILENAMES:
        config_path = os.path.join(root, relative_path)
        if not os.path.exists(config_path):
            continue
        with open(config_path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        raw_entries = raw.get("structures", raw) if isinstance(raw, dict) else {}
        if not isinstance(raw_entries, dict):
            continue
        for name, entry in raw_entries.items():
            if not isinstance(entry, dict):
                continue
            resolved = _resolve_catalog_entry(os.path.dirname(config_path), entry)
            if resolved:
                catalog[str(name)] = resolved
    return catalog


def get_default_catalog(root_dir: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    root = workspace_root() if root_dir is None else root_dir
    catalog = _load_local_catalog(root)
    for name, entry in DEFAULT_CATALOG.items():
        if name in catalog:
            continue
        cif_path = os.path.normpath(os.path.join(root, entry["relative_cif"]))
        if not os.path.exists(cif_path):
            continue
        catalog[name] = {
            "title": entry["title"],
            "cif_path": cif_path,
        }
    return catalog


def default_preset() -> Dict[str, Any]:
    return {
        "version": 1,
        "style": copy.deepcopy(DEFAULT_STYLE),
        "structures": copy.deepcopy(DEFAULT_STRUCTURE_PRESETS),
    }


def load_preset(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return default_preset()
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return _deep_merge(default_preset(), raw)


def save_preset(path: str, preset: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(preset), handle, indent=2, ensure_ascii=False)


def scene_from_camera(position: Iterable[float], focal_point: Iterable[float], up: Iterable[float]):
    position = np.array(position, dtype=float)
    focal_point = np.array(focal_point, dtype=float)
    up = np.array(up, dtype=float)
    view_dir = position - focal_point
    if np.linalg.norm(view_dir) < 1e-8:
        view_dir = np.array([0.0, 0.0, 1.0], dtype=float)
    view_dir /= np.linalg.norm(view_dir)
    if np.linalg.norm(up) < 1e-8:
        up = np.array([0.0, 1.0, 0.0], dtype=float)
    up /= np.linalg.norm(up)
    return view_dir, up


def scene_to_preset_entry(scene: Dict[str, Any], camera=None, style=None) -> Dict[str, Any]:
    entry = {
        "camera": _json_safe(camera or scene.get("camera", {})),
        "show_hydrogen": bool(scene.get("show_hydrogen", False)),
    }
    if style:
        entry["style"] = _json_safe(style)
    return entry


def json_safe(value: Any) -> Any:
    return _json_safe(value)


def deep_merge(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return _deep_merge(base, override)
