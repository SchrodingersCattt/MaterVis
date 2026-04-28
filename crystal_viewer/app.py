from __future__ import annotations

import argparse
import copy
import io
import os
import subprocess
import tempfile
import threading
from typing import Any, Dict, Iterable, Optional

import numpy as np
import plotly.io as pio

try:
    from dash import Dash, Input, Output, State, callback_context, dcc, html, no_update
except ImportError as exc:  # pragma: no cover - user-facing fallback
    raise SystemExit(
        "Dash is required for the browser viewer. "
        "Install it with `python -m pip install dash`."
    ) from exc

from .api import register_api
from .loader import LoadedCrystal, build_bundle_scene, build_empty_bundle, build_loaded_crystal, load_uploaded_cif
from .presets import (
    DEFAULT_CATALOG,
    DEFAULT_STYLE,
    default_preset,
    default_preset_path,
    get_default_catalog,
    load_preset,
    save_preset,
    workspace_root,
)
from .renderer import build_figure, style_from_controls, topology_histogram_figure, topology_results_markdown
from .scene import scene_json
from .topology import analyze_topology


PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = workspace_root(PACKAGE_DIR)
DEFAULT_PRESET_PATH = default_preset_path(WORKSPACE_DIR)
LEGACY_EXPORT_MODULE = "crystal_viewer.legacy.plot_crystal"
PLACEHOLDER_STRUCTURE = "__upload__"


def _structure_summary(scene: dict) -> str:
    if not scene.get("draw_atoms"):
        return "No structure loaded yet. Upload a CIF to begin."
    minor_atoms = sum(1 for atom in scene["draw_atoms"] if atom["is_minor"])
    minor_bonds = sum(1 for bond in scene["bonds"] if bond["is_minor"])
    if minor_atoms:
        return f"Disorder detected: {minor_atoms} minor atoms, {minor_bonds} minor bonds."
    return "Disorder: none detected."


def _display_options_from_style(style: dict) -> list[str]:
    return [
        token
        for enabled, token in (
            (style.get("show_labels", True), "labels"),
            (style.get("show_axes", True), "axes"),
            (style.get("show_minor_only", False), "minor_only"),
            (style.get("minor_wireframe", False), "minor_wireframe"),
            (style.get("show_hydrogen", False), "hydrogens"),
            (style.get("show_unit_cell", False), "unit_cell_box"),
        )
        if enabled
    ]


def _plotly_camera(camera: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not camera:
        return None
    if "eye" in camera:
        return camera
    position = np.array(camera.get("position", [0.0, 0.0, 1.0]), dtype=float)
    focal = np.array(camera.get("focal_point", [0.0, 0.0, 0.0]), dtype=float)
    up = np.array(camera.get("up", [0.0, 1.0, 0.0]), dtype=float)
    eye = position - focal
    norm = np.linalg.norm(eye)
    if norm < 1e-8:
        eye = np.array([0.0, 0.0, 1.8], dtype=float)
    else:
        eye = eye / norm * 1.8
    up_norm = np.linalg.norm(up)
    if up_norm < 1e-8:
        up = np.array([0.0, 1.0, 0.0], dtype=float)
    else:
        up = up / up_norm
    return {
        "eye": {"x": float(eye[0]), "y": float(eye[1]), "z": float(eye[2])},
        "center": {"x": 0.0, "y": 0.0, "z": 0.0},
        "up": {"x": float(up[0]), "y": float(up[1]), "z": float(up[2])},
    }


def _camera_vectors(camera: Optional[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cam = _plotly_camera(camera) or {
        "eye": {"x": 0.0, "y": 0.0, "z": 1.8},
        "center": {"x": 0.0, "y": 0.0, "z": 0.0},
        "up": {"x": 0.0, "y": 1.0, "z": 0.0},
    }
    eye = np.array([cam["eye"]["x"], cam["eye"]["y"], cam["eye"]["z"]], dtype=float)
    center = np.array([cam.get("center", {}).get("x", 0.0), cam.get("center", {}).get("y", 0.0), cam.get("center", {}).get("z", 0.0)], dtype=float)
    up = np.array([cam["up"]["x"], cam["up"]["y"], cam["up"]["z"]], dtype=float)
    up_norm = np.linalg.norm(up)
    if up_norm < 1e-8:
        up = np.array([0.0, 1.0, 0.0], dtype=float)
    else:
        up = up / up_norm
    return eye, center, up


def _camera_payload(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> dict[str, Any]:
    return {
        "eye": {"x": float(eye[0]), "y": float(eye[1]), "z": float(eye[2])},
        "center": {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])},
        "up": {"x": float(up[0]), "y": float(up[1]), "z": float(up[2])},
    }


def _rotate_vector(vec: np.ndarray, axis: np.ndarray, angle_deg: float) -> np.ndarray:
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-8 or abs(angle_deg) < 1e-8:
        return vec
    axis = axis / axis_norm
    theta = np.deg2rad(angle_deg)
    return (
        vec * np.cos(theta)
        + np.cross(axis, vec) * np.sin(theta)
        + axis * np.dot(axis, vec) * (1.0 - np.cos(theta))
    )


def _fallback_png(message: str) -> bytes:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return bytes.fromhex(
            "89504E470D0A1A0A0000000D4948445200000001000000010802000000907753DE"
            "0000000C49444154789C63606060000000040001F61738550000000049454E44AE426082"
        )
    image = Image.new("RGB", (960, 240), "white")
    draw = ImageDraw.Draw(image)
    draw.text((18, 18), message, fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class ViewerBackend:
    def __init__(self, preset_path: str, names: Optional[Iterable[str]] = None, root_dir: Optional[str] = None):
        self.root_dir = root_dir or WORKSPACE_DIR
        self.preset_path = preset_path
        self.preset = load_preset(preset_path) if os.path.exists(preset_path) else default_preset()
        self.catalog = get_default_catalog(root_dir=self.root_dir)
        self._lock = threading.Lock()
        self._bundle_lock = threading.Lock()
        default_names = [name for name in DEFAULT_CATALOG.keys() if name in self.catalog]
        requested_names = [name for name in (names or []) if name in self.catalog]
        self.structure_names = requested_names if requested_names else default_names
        if not self.structure_names:
            self.structure_names = list(self.catalog.keys())
        self.bundles: Dict[str, LoadedCrystal] = {}
        if not self.structure_names:
            placeholder = build_empty_bundle(name=PLACEHOLDER_STRUCTURE)
            self.bundles[placeholder.name] = placeholder
            self.structure_names = [placeholder.name]
        first_name = self.structure_names[0]
        self.current_state = self.default_state(first_name)
        self.pending_state: Optional[dict[str, Any]] = None
        self.version = 0

    def default_state(self, structure: str) -> dict[str, Any]:
        bundle = self.get_bundle(structure)
        scene = bundle.scene
        style = dict(DEFAULT_STYLE)
        style.update(scene.get("style", {}))
        preset_style = self.preset.get("style", {})
        entry_style = self.preset.get("structures", {}).get(structure, {}).get("style", {})
        style.update(preset_style)
        style.update(entry_style)
        if scene.get("has_minor") and "minor_wireframe" not in preset_style and "minor_wireframe" not in entry_style:
            style["minor_wireframe"] = True
        # Pick a fragment type that actually exists in this structure (B if a metal
        # B-site is present, otherwise A, otherwise whatever is first available).
        types_present = [
            t for t in ("B", "A", "X")
            if any(f.get("type") == t for f in scene.get("fragment_table") or [])
        ]
        default_frag_type = types_present[0] if types_present else "B"
        return {
            "structure": structure,
            "atom_scale": float(style["atom_scale"]),
            "bond_radius": float(style["bond_radius"]),
            "minor_opacity": float(style["minor_opacity"]),
            "axis_scale": float(style["axis_scale"]),
            "display_options": _display_options_from_style(style),
            "display_mode": style.get("display_mode", scene.get("display_mode", "formula_unit")),
            "topology_fragment_type": default_frag_type,
            "topology_site_index": None,
            "topology_enabled": bool(style.get("topology_enabled", True)),
            "fast_rendering": bool(style.get("fast_rendering", False)),
            "camera": scene.get("camera"),
            "cutoff": 10.0,
        }

    def _bump_version(self):
        self.version += 1

    def list_structures(self) -> list[dict[str, Any]]:
        return [self.get_bundle(name).metadata() for name in self.structure_names]

    def structure_options(self) -> list[dict[str, str]]:
        return [
            {
                "label": "Upload CIF to begin" if name == PLACEHOLDER_STRUCTURE else name,
                "value": name,
            }
            for name in self.structure_names
        ]

    def fragment_type_options(self, structure: Optional[str] = None) -> list[dict[str, str]]:
        """Dropdown options for fragment-type selector, narrowed to types
        that actually appear in this structure (so the user doesn't get to
        pick "B fragments" on a CIF with no metal centers, etc.)."""
        target = structure or (self.structure_names[0] if self.structure_names else None)
        if target is None or target not in self.bundles:
            return [
                {"label": "A fragments", "value": "A"},
                {"label": "B fragments", "value": "B"},
                {"label": "X fragments", "value": "X"},
            ]
        bundle = self.get_bundle(target)
        scene = bundle.scene
        present = {f.get("type") for f in (scene.get("fragment_table") or [])}
        labels = {"A": "A (organic cations)", "B": "B (metal centers)", "X": "X (ClO\u2084 / halide)"}
        opts = []
        for t in ("B", "A", "X"):
            if t in present:
                count = sum(1 for f in scene.get("fragment_table") or [] if f.get("type") == t)
                opts.append({"label": f"{labels[t]} \u00d7{count}", "value": t})
        return opts or [{"label": "(no fragments)", "value": "B", "disabled": True}]

    def _drop_placeholder(self) -> None:
        if PLACEHOLDER_STRUCTURE in self.structure_names and len(self.structure_names) == 1:
            self.structure_names = []
        self.bundles.pop(PLACEHOLDER_STRUCTURE, None)

    def get_bundle(self, name: str) -> LoadedCrystal:
        if name in self.bundles:
            return self.bundles[name]
        if name not in self.catalog:
            raise KeyError(name)

        entry = self.catalog[name]
        built = build_loaded_crystal(
            name=name,
            cif_path=entry["cif_path"],
            title=entry["title"],
            preset=self.preset,
            source="catalog",
        )

        with self._bundle_lock:
            existing = self.bundles.get(name)
            if existing is not None:
                return existing
            self.bundles[name] = built
            return built

    def get_scene_json(self, name: str) -> dict[str, Any]:
        state = self.get_state()
        if state["structure"] != name:
            state = self.normalize_state({"structure": name})
        bundle = self.get_bundle(name)
        scene = self.scene_for_state(state)
        return {
            "name": bundle.name,
            "title": bundle.title,
            "scene": scene_json(scene),
            "fragment_table": copy.deepcopy(scene.get("fragment_table", [])),
            "topology_fragment_table": copy.deepcopy(bundle.topology_fragment_table),
            "summary": _structure_summary(scene),
        }

    def normalize_state(self, patch: Optional[dict[str, Any]]) -> dict[str, Any]:
        state = copy.deepcopy(self.current_state)
        patch = patch or {}
        if "structure" in patch and patch["structure"] in self.structure_names:
            structure = patch["structure"]
            defaults = self.default_state(structure)
            state.update(defaults)
            state["structure"] = structure
        for key in ("atom_scale", "bond_radius", "minor_opacity", "axis_scale", "cutoff"):
            if key in patch and patch[key] is not None:
                state[key] = float(patch[key])
        if "display_options" in patch and patch["display_options"] is not None:
            state["display_options"] = list(patch["display_options"])
        if "display_mode" in patch and patch["display_mode"] is not None:
            state["display_mode"] = str(patch["display_mode"])
            if "topology_site_index" not in patch:
                state["topology_site_index"] = None
        if "topology_fragment_type" in patch:
            state["topology_fragment_type"] = patch["topology_fragment_type"] or "B"
        if "topology_site_index" in patch:
            value = patch["topology_site_index"]
            state["topology_site_index"] = None if value in ("", None) else int(value)
        if "topology_enabled" in patch:
            state["topology_enabled"] = bool(patch["topology_enabled"])
        if "fast_rendering" in patch:
            state["fast_rendering"] = bool(patch["fast_rendering"])
        if "camera" in patch:
            state["camera"] = patch["camera"]
        return state

    def get_state(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self.current_state)

    def patch_state(self, patch: Optional[dict[str, Any]]) -> dict[str, Any]:
        with self._lock:
            self.current_state = self.normalize_state(patch)
            self.pending_state = copy.deepcopy(self.current_state)
            self._bump_version()
            return copy.deepcopy(self.current_state)

    def pop_pending_state(self) -> Optional[dict[str, Any]]:
        with self._lock:
            pending = self.pending_state
            self.pending_state = None
            return copy.deepcopy(pending) if pending else None

    def record_state(self, patch: Optional[dict[str, Any]]) -> None:
        with self._lock:
            self.current_state = self.normalize_state(patch)
            self._bump_version()

    def show_hydrogen_for_state(self, state: Optional[dict[str, Any]] = None) -> bool:
        state = self.current_state if state is None else state
        return "hydrogens" in set(state.get("display_options", []))

    def scene_for_state(self, state: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        state = self.current_state if state is None else state
        bundle = self.get_bundle(state["structure"])
        scene = build_bundle_scene(
            bundle,
            display_mode=state.get("display_mode", "formula_unit"),
            show_hydrogen=self.show_hydrogen_for_state(state),
            preset=self.preset,
        )
        bundle.scene = scene
        bundle.fragment_table = scene.get("fragment_table", bundle.fragment_table)
        return scene

    def style_for_state(self, state: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        state = self.current_state if state is None else state
        scene = self.scene_for_state(state)
        style = dict(scene.get("style", {}))
        style.update(
            style_from_controls(
                state["atom_scale"],
                state["bond_radius"],
                state["minor_opacity"],
                state["axis_scale"],
                state["display_options"],
            )
        )
        style["display_mode"] = state.get("display_mode", scene.get("display_mode", "formula_unit"))
        style["fast_rendering"] = bool(state.get("fast_rendering", False))
        style["topology_enabled"] = bool(state.get("topology_enabled", True))
        return style

    def add_uploaded_bundle(self, contents: str, filename: str) -> LoadedCrystal:
        bundle = load_uploaded_cif(
            contents=contents,
            filename=filename,
            existing_names=self.structure_names,
            preset=self.preset,
        )
        self._drop_placeholder()
        self.bundles[bundle.name] = bundle
        self.structure_names.append(bundle.name)
        self.patch_state({"structure": bundle.name})
        return bundle

    def add_uploaded_file_bytes(self, data: bytes, filename: str) -> LoadedCrystal:
        upload_dir = os.path.join(tempfile.gettempdir(), "crystal_viewer_uploads")
        os.makedirs(upload_dir, exist_ok=True)
        path = os.path.join(upload_dir, filename)
        with open(path, "wb") as handle:
            handle.write(data)
        safe_name = os.path.splitext(os.path.basename(filename))[0]
        suffix = 2
        while safe_name in self.structure_names:
            safe_name = f"{os.path.splitext(os.path.basename(filename))[0]}_{suffix}"
            suffix += 1
        bundle = build_loaded_crystal(name=safe_name, cif_path=path, title=os.path.splitext(filename)[0], preset=self.preset, source="upload")
        self._drop_placeholder()
        self.bundles[bundle.name] = bundle
        self.structure_names.append(bundle.name)
        self.patch_state({"structure": bundle.name})
        return bundle

    def topology_candidates(self, structure: str, fragment_type: Optional[str] = None) -> list[dict[str, Any]]:
        state = self.get_state()
        if state["structure"] != structure:
            state = self.normalize_state({"structure": structure})
        fragments = self.scene_for_state(state).get("fragment_table", [])
        if fragment_type and fragment_type not in ("", "Any"):
            filtered = [fragment for fragment in fragments if fragment.get("type") == fragment_type]
            if filtered:
                return filtered
        return fragments

    def fragment_index_for_atom(self, scene: dict, atom_index: int) -> Optional[int]:
        for fragment in scene.get("fragment_table", []):
            if atom_index in fragment.get("site_indices", []):
                return int(fragment["index"])
        atom = scene["draw_atoms"][atom_index]
        atom_cart = np.array(atom["cart"], dtype=float)
        fragments = scene.get("fragment_table", [])
        if not fragments:
            return atom_index
        distances = [
            (float(np.linalg.norm(np.array(fragment["center"], dtype=float) - atom_cart)), int(fragment["index"]))
            for fragment in fragments
        ]
        distances.sort(key=lambda item: item[0])
        return distances[0][1]

    def _display_fragment(self, scene: dict, display_index: int | None) -> Optional[dict[str, Any]]:
        if display_index is None:
            return None
        return next((fragment for fragment in scene.get("fragment_table", []) if int(fragment["index"]) == int(display_index)), None)

    def _pbc_distance(self, bundle: LoadedCrystal, frac_a, frac_b) -> float:
        delta = np.array(frac_b, dtype=float) - np.array(frac_a, dtype=float)
        delta -= np.round(delta)
        return float(np.linalg.norm(np.array(bundle.M, dtype=float) @ delta))

    def map_display_fragment_to_topology(self, bundle: LoadedCrystal, display_fragment: dict | None) -> Optional[dict[str, Any]]:
        if display_fragment is None:
            return None
        candidates = [
            fragment
            for fragment in bundle.topology_fragment_table
            if fragment.get("type") == display_fragment.get("type")
        ] or list(bundle.topology_fragment_table)
        if not candidates:
            return None
        display_frac = np.array(display_fragment.get("frac_center", [0.0, 0.0, 0.0]), dtype=float)
        ranked = []
        for fragment in candidates:
            ranked.append((self._pbc_distance(bundle, display_frac, fragment.get("frac_center", [0.0, 0.0, 0.0])), fragment))
        ranked.sort(key=lambda item: item[0])
        return ranked[0][1]

    def resolve_topology_site(
        self,
        *,
        state: dict[str, Any],
        structure: str,
        explicit_site: Optional[int],
        fragment_type: Optional[str],
        click_data: Optional[dict[str, Any]],
    ) -> Optional[int]:
        scene = self.scene_for_state(state)
        fragments = scene.get("fragment_table", [])
        requested_type = fragment_type if fragment_type not in ("", "Any", None) else None
        if explicit_site is not None:
            chosen = self._display_fragment(scene, explicit_site)
            if chosen is not None and (requested_type is None or chosen.get("type") == requested_type):
                return int(explicit_site)
        if click_data and click_data.get("points"):
            point = click_data["points"][0]
            custom = point.get("customdata")
            if custom:
                atom_index = int(custom[0])
                return self.fragment_index_for_atom(scene, atom_index)
        if requested_type is not None:
            candidates = [fragment for fragment in fragments if fragment.get("type") == requested_type]
            if not candidates:
                # Honest about it: user asked for B, no B exists -> no overlay.
                return None
        else:
            candidates = fragments
        if candidates:
            return int(candidates[0]["index"])
        return None

    def topology_for_state(self, state: dict[str, Any], click_data: Optional[dict[str, Any]] = None):
        if not state.get("topology_enabled", True):
            return None
        structure = state["structure"]
        bundle = self.get_bundle(structure)
        scene = self.scene_for_state(state)
        site_index = self.resolve_topology_site(
            state=state,
            structure=structure,
            explicit_site=state.get("topology_site_index"),
            fragment_type=state.get("topology_fragment_type"),
            click_data=click_data,
        )
        if site_index is None:
            return None
        display_fragment = self._display_fragment(scene, site_index)
        topology_fragment = self.map_display_fragment_to_topology(bundle, display_fragment)
        if topology_fragment is None:
            return None
        return analyze_topology(
            bundle,
            center_index=int(topology_fragment["index"]),
            cutoff=float(state.get("cutoff", 10.0)),
            display_center=display_fragment.get("center") if display_fragment else None,
            display_label=display_fragment.get("label") if display_fragment else None,
            display_type=display_fragment.get("type") if display_fragment else None,
        )

    def figure_for_state(self, state: Optional[dict[str, Any]] = None, click_data: Optional[dict[str, Any]] = None):
        state = self.get_state() if state is None else state
        scene = self.scene_for_state(state)
        topology_data = self.topology_for_state(state, click_data=click_data)
        fig = build_figure(scene, self.style_for_state(state), topology_data=topology_data)
        camera = _plotly_camera(state.get("camera"))
        if camera:
            fig.update_layout(scene_camera=camera)
        return fig, topology_data

    def render_current_png(self) -> bytes:
        fig, _ = self.figure_for_state(self.get_state())
        try:
            return pio.to_image(fig, format="png", scale=2)
        except Exception as exc:  # pragma: no cover - depends on local Chrome/Kaleido state
            return _fallback_png(f"Plotly image export failed: {exc}")

    def default_camera(self, state: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        scene = self.scene_for_state(self.get_state() if state is None else state)
        return _plotly_camera(scene.get("camera")) or _plotly_camera(None)

    def get_camera(self) -> dict[str, Any]:
        state = self.get_state()
        return _plotly_camera(state.get("camera")) or self.default_camera(state)

    def set_camera(self, camera: dict[str, Any]) -> dict[str, Any]:
        self.patch_state({"camera": camera})
        return self.get_camera()

    def camera_action(self, action: str, **payload) -> dict[str, Any]:
        if action == "reset":
            return self.set_camera(self.default_camera())

        eye, center, up = _camera_vectors(self.get_camera())
        if action == "zoom":
            factor = float(payload.get("factor", 1.0))
            if abs(factor) > 1e-8:
                eye = eye / factor
        elif action == "pan":
            delta = np.array(
                [
                    float(payload.get("dx", 0.0)),
                    float(payload.get("dy", 0.0)),
                    float(payload.get("dz", 0.0)),
                ],
                dtype=float,
            )
            center = center + delta
        elif action == "orbit":
            yaw_deg = float(payload.get("yaw_deg", 0.0))
            pitch_deg = float(payload.get("pitch_deg", 0.0))
            eye = _rotate_vector(eye, up, yaw_deg)
            right = np.cross(eye, up)
            if np.linalg.norm(right) > 1e-8:
                eye = _rotate_vector(eye, right, pitch_deg)
                up = _rotate_vector(up, right, pitch_deg)
        camera = _camera_payload(eye, center, up)
        return self.set_camera(camera)

    def save_preset(self, path: Optional[str] = None) -> dict[str, Any]:
        state = self.get_state()
        bundle = self.get_bundle(state["structure"])
        scene = self.scene_for_state(state)
        target = path or self.preset_path
        preset_data = load_preset(target) if os.path.exists(target) else default_preset()
        preset_data["style"].update(self.style_for_state(state))
        preset_data.setdefault("structures", {})
        preset_data["structures"][bundle.name] = {
            "camera": state.get("camera") or scene.get("camera"),
            "show_hydrogen": self.show_hydrogen_for_state(state),
            "style": self.style_for_state(state),
        }
        save_preset(target, preset_data)
        self.preset = preset_data
        return {"path": target, "structure": bundle.name}

    def load_preset_from_path(self, path: Optional[str]) -> dict[str, Any]:
        if not path:
            raise ValueError("path is required")
        self.preset = load_preset(path)
        self.preset_path = path
        for bundle in self.bundles.values():
            bundle.scene_cache.clear()
        structure = self.get_state()["structure"]
        self.patch_state(self.default_state(structure))
        return {"path": path, "state": self.get_state()}

    def export_static(self, output_path: Optional[str] = None) -> dict[str, Any]:
        state = self.get_state()
        if state.get("structure") == PLACEHOLDER_STRUCTURE:
            return {
                "returncode": 1,
                "stdout": "",
                "stderr": "No structure is loaded yet. Upload or preload a CIF before exporting.",
            }
        self.save_preset()
        cmd = [
            os.environ.get("PYTHON", "python"),
            "-m",
            LEGACY_EXPORT_MODULE,
            "--preset",
            self.preset_path,
            "--both",
        ]
        proc = subprocess.run(cmd, cwd=self.root_dir, capture_output=True, text=True)
        payload = {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
        if output_path:
            payload["output_path"] = output_path
        return payload

    def query_topology(self, structure: str, center_index: int, cutoff: float = 10.0) -> dict[str, Any]:
        state = self.get_state()
        if state["structure"] != structure:
            state = self.normalize_state({"structure": structure})
        state["topology_site_index"] = center_index
        state["cutoff"] = cutoff
        return self.topology_for_state(state)

    def websocket_snapshot(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "state": self.get_state(),
            "structures": self.list_structures(),
        }


def create_app(
    preset_path: str = DEFAULT_PRESET_PATH,
    names=None,
    root_dir: Optional[str] = None,
    cif_paths: Optional[Iterable[str]] = None,
) -> Dash:
    backend = ViewerBackend(preset_path=preset_path, names=names, root_dir=root_dir)
    for cif_path in cif_paths or []:
        bundle = build_loaded_crystal(
            name=os.path.splitext(os.path.basename(cif_path))[0],
            cif_path=cif_path,
            title=os.path.splitext(os.path.basename(cif_path))[0],
            preset=backend.preset,
            source="cli",
        )
        backend.bundles[bundle.name] = bundle
        if bundle.name not in backend.structure_names:
            backend.structure_names.append(bundle.name)
    backend._drop_placeholder()
    if backend.structure_names and backend.current_state.get("structure") not in backend.structure_names:
        backend.current_state = backend.default_state(backend.structure_names[0])
    app = Dash(__name__, assets_folder=os.path.join(PACKAGE_DIR, "assets"))
    app.crystal_backend = backend

    first_state = backend.get_state()
    first_figure, first_topology = backend.figure_for_state(first_state)
    first_scene = backend.scene_for_state(first_state)

    app.layout = html.Div(
        [
            dcc.Store(id="agent-state-store", data=first_state),
            dcc.Interval(id="agent-state-poll", interval=800, n_intervals=0),
            html.Div(id="state-sync-sentinel", style={"display": "none"}),
            html.Div(
                [
                    html.H3("Crystal Viewer", style={"marginTop": "0"}),
                    html.Label("Structure"),
                    dcc.RadioItems(
                        id="structure-selector",
                        options=backend.structure_options(),
                        value=first_state["structure"],
                        labelStyle={"display": "block", "marginBottom": "4px"},
                    ),
                    html.Div(
                        id="structure-summary",
                        children=_structure_summary(first_scene),
                        style={"marginBottom": "12px", "fontSize": "13px", "color": "#444444"},
                    ),
                    html.Label("Upload CIF"),
                    dcc.Upload(
                        id="cif-upload",
                        children=html.Div(["Drag and drop CIF, or click to upload"]),
                        multiple=True,
                        style={
                            "border": "1px dashed #999999",
                            "padding": "10px",
                            "marginBottom": "12px",
                            "textAlign": "center",
                        },
                    ),
                    html.Div(
                        id="upload-status",
                        style={"marginBottom": "12px", "whiteSpace": "pre-wrap", "fontSize": "13px"},
                    ),
                    html.Label("Display Scope"),
                    dcc.Dropdown(
                        id="display-mode-selector",
                        options=[
                            {"label": "Formula unit cluster", "value": "formula_unit"},
                            {"label": "Unit cell", "value": "unit_cell"},
                            {"label": "Asymmetric unit", "value": "asymmetric_unit"},
                            {"label": "Isolated cluster (no PBC)", "value": "cluster"},
                        ],
                        value=first_state["display_mode"],
                        clearable=False,
                        style={"marginBottom": "12px"},
                    ),
                    html.Label("Display"),
                    dcc.Checklist(
                        id="display-options",
                        options=[
                            {"label": "Labels", "value": "labels"},
                            {"label": "Axes", "value": "axes"},
                            {"label": "Minor Only", "value": "minor_only"},
                            {"label": "Minor Wireframe", "value": "minor_wireframe"},
                            {"label": "Hydrogens", "value": "hydrogens"},
                            {"label": "Unit Cell Box", "value": "unit_cell_box"},
                        ],
                        value=first_state["display_options"],
                    ),
                    html.Div(style={"height": "10px"}),
                    dcc.Checklist(
                        id="fast-rendering-toggle",
                        options=[{"label": "Fast rendering fallback", "value": "fast"}],
                        value=["fast"] if first_state.get("fast_rendering") else [],
                    ),
                    html.Label("Atom Scale"),
                    dcc.Slider(
                        id="atom-scale-slider",
                        min=0.5, max=1.8, step=0.02,
                        value=float(first_state["atom_scale"]),
                        marks={0.5: "0.5", 1.0: "1.0", 1.5: "1.5", 1.8: "1.8"},
                        tooltip={"placement": "bottom", "always_visible": False},
                        updatemode="mouseup",
                    ),
                    html.Label("Bond Radius"),
                    dcc.Slider(
                        id="bond-radius-slider",
                        min=0.05, max=0.40, step=0.01,
                        value=float(first_state["bond_radius"]),
                        marks={0.05: "0.05", 0.20: "0.20", 0.40: "0.40"},
                        tooltip={"placement": "bottom", "always_visible": False},
                        updatemode="mouseup",
                    ),
                    html.Label("Minor Opacity"),
                    dcc.Slider(
                        id="minor-opacity-slider",
                        min=0.10, max=0.90, step=0.02,
                        value=float(first_state["minor_opacity"]),
                        marks={0.1: "0.1", 0.5: "0.5", 0.9: "0.9"},
                        tooltip={"placement": "bottom", "always_visible": False},
                        updatemode="mouseup",
                    ),
                    html.Label("Axis Scale"),
                    dcc.Slider(
                        id="axis-scale-slider",
                        min=0.05, max=0.25, step=0.01,
                        value=float(first_state["axis_scale"]),
                        marks={0.05: "0.05", 0.15: "0.15", 0.25: "0.25"},
                        tooltip={"placement": "bottom", "always_visible": False},
                        updatemode="mouseup",
                    ),
                    html.Hr(),
                    html.H4("Topology"),
                    dcc.Checklist(
                        id="topology-toggle",
                        options=[{"label": "Show topology overlay", "value": "enabled"}],
                        value=["enabled"] if first_state.get("topology_enabled", True) else [],
                    ),
                    dcc.Dropdown(
                        id="topology-fragment-type",
                        options=backend.fragment_type_options(first_state["structure"]),
                        value=first_state["topology_fragment_type"],
                        clearable=False,
                    ),
                    dcc.Input(
                        id="topology-site-index",
                        type="number",
                        placeholder="Site / fragment index",
                        value=first_state["topology_site_index"],
                        style={"width": "100%", "marginTop": "8px"},
                    ),
                    html.Div(style={"height": "12px"}),
                    html.Button("Save Preset", id="save-preset-btn", n_clicks=0),
                    html.Button("Export Static Figure", id="export-btn", n_clicks=0, style={"marginLeft": "8px"}),
                    html.Div(
                        id="status",
                        children=f"Preset: {preset_path}",
                        style={"marginTop": "12px", "whiteSpace": "pre-wrap", "fontSize": "13px"},
                    ),
                ],
                id="left-panel",
                style={
                    "width": "340px",
                    "minWidth": "260px",
                    "maxWidth": "640px",
                    "flex": "0 0 auto",
                    "padding": "16px",
                    "borderRight": "1px solid #DDDDDD",
                    "fontFamily": "Arial, sans-serif",
                    "overflowY": "auto",
                    "height": "100vh",
                },
            ),
            html.Div(id="left-splitter", className="panel-splitter"),
            html.Div(
                [dcc.Graph(id="crystal-graph", figure=first_figure, style={"height": "100vh"})],
                id="center-panel",
                style={"flex": "1", "minWidth": 0},
            ),
            html.Div(id="right-splitter", className="panel-splitter"),
            html.Div(
                [
                    html.H4("Topology Analysis"),
                    dcc.Graph(id="topology-histogram", figure=topology_histogram_figure(first_topology), style={"height": "280px"}),
                    html.Pre(
                        id="topology-results",
                        children=topology_results_markdown(first_topology),
                        style={"whiteSpace": "pre-wrap", "fontSize": "13px", "fontFamily": "Arial, sans-serif"},
                    ),
                ],
                id="right-panel",
                style={
                    "width": "320px",
                    "minWidth": "260px",
                    "maxWidth": "640px",
                    "flex": "0 0 auto",
                    "padding": "16px",
                    "borderLeft": "1px solid #DDDDDD",
                    "backgroundColor": "#FAFAFA",
                    "height": "100vh",
                    "overflowY": "auto",
                },
            ),
        ],
        id="viewer-root",
        style={"display": "flex", "height": "100vh", "backgroundColor": "#FFFFFF"},
    )

    @app.callback(
        Output("structure-selector", "options"),
        Output("structure-selector", "value"),
        Output("upload-status", "children"),
        Input("cif-upload", "contents"),
        State("cif-upload", "filename"),
        prevent_initial_call=True,
    )
    def upload_cif(contents_list, filenames):
        if not contents_list:
            return no_update, no_update, no_update
        names_out = []
        for contents, filename in zip(contents_list, filenames or []):
            bundle = backend.add_uploaded_bundle(contents, filename)
            names_out.append(bundle.name)
        return backend.structure_options(), names_out[-1], f"Uploaded CIF(s): {', '.join(names_out)}"

    @app.callback(
        Output("topology-fragment-type", "options"),
        Output("topology-fragment-type", "value", allow_duplicate=True),
        Input("structure-selector", "value"),
        State("topology-fragment-type", "value"),
        prevent_initial_call=True,
    )
    def refresh_fragment_type_options(structure, current_value):
        opts = backend.fragment_type_options(structure)
        valid_values = {opt["value"] for opt in opts if not opt.get("disabled")}
        new_value = current_value if current_value in valid_values else (
            opts[0]["value"] if opts else "B"
        )
        return opts, new_value

    @app.callback(
        Output("structure-selector", "value", allow_duplicate=True),
        Output("display-mode-selector", "value"),
        Output("display-options", "value"),
        Output("atom-scale-slider", "value"),
        Output("bond-radius-slider", "value"),
        Output("minor-opacity-slider", "value"),
        Output("axis-scale-slider", "value"),
        Output("topology-fragment-type", "value"),
        Output("topology-site-index", "value"),
        Output("topology-toggle", "value"),
        Output("fast-rendering-toggle", "value"),
        Output("agent-state-store", "data"),
        Input("agent-state-poll", "n_intervals"),
        prevent_initial_call=True,
    )
    def sync_agent_state(_):
        state = backend.pop_pending_state()
        if not state:
            return (no_update,) * 12
        return (
            state["structure"],
            state["display_mode"],
            state["display_options"],
            state["atom_scale"],
            state["bond_radius"],
            state["minor_opacity"],
            state["axis_scale"],
            state["topology_fragment_type"],
            state["topology_site_index"],
            ["enabled"] if state.get("topology_enabled", True) else [],
            ["fast"] if state.get("fast_rendering", False) else [],
            state,
        )

    @app.callback(
        Output("state-sync-sentinel", "children"),
        Input("structure-selector", "value"),
        Input("display-mode-selector", "value"),
        Input("display-options", "value"),
        Input("atom-scale-slider", "value"),
        Input("bond-radius-slider", "value"),
        Input("minor-opacity-slider", "value"),
        Input("axis-scale-slider", "value"),
        Input("topology-fragment-type", "value"),
        Input("topology-site-index", "value"),
        Input("topology-toggle", "value"),
        Input("fast-rendering-toggle", "value"),
        Input("crystal-graph", "relayoutData"),
        Input("crystal-graph", "clickData"),
    )
    def capture_state(
        structure,
        display_mode,
        display_options,
        atom_scale,
        bond_radius,
        minor_opacity,
        axis_scale,
        fragment_type,
        site_index,
        topology_toggle,
        fast_rendering_toggle,
        relayout_data,
        click_data,
    ):
        camera = None
        if relayout_data:
            camera = relayout_data.get("scene.camera") or relayout_data.get("scene", {}).get("camera")
        explicit_site = None if site_index in ("", None) else int(site_index)
        if explicit_site is not None or (click_data and click_data.get("points")):
            resolved_site = backend.resolve_topology_site(
                state=backend.normalize_state({"structure": structure, "display_mode": display_mode, "display_options": display_options}),
                structure=structure,
                explicit_site=explicit_site,
                fragment_type=fragment_type,
                click_data=click_data,
            )
        else:
            resolved_site = None
        backend.record_state(
            {
                "structure": structure,
                "display_mode": display_mode,
                "display_options": display_options,
                "atom_scale": atom_scale,
                "bond_radius": bond_radius,
                "minor_opacity": minor_opacity,
                "axis_scale": axis_scale,
                "topology_fragment_type": fragment_type,
                "topology_site_index": resolved_site,
                "topology_enabled": "enabled" in (topology_toggle or []),
                "fast_rendering": "fast" in (fast_rendering_toggle or []),
                "camera": camera,
            }
        )
        return ""

    @app.callback(
        Output("crystal-graph", "figure"),
        Output("topology-histogram", "figure"),
        Output("topology-results", "children"),
        Output("structure-summary", "children"),
        Input("structure-selector", "value"),
        Input("display-mode-selector", "value"),
        Input("display-options", "value"),
        Input("atom-scale-slider", "value"),
        Input("bond-radius-slider", "value"),
        Input("minor-opacity-slider", "value"),
        Input("axis-scale-slider", "value"),
        Input("topology-fragment-type", "value"),
        Input("topology-site-index", "value"),
        Input("topology-toggle", "value"),
        Input("fast-rendering-toggle", "value"),
        Input("crystal-graph", "clickData"),
        Input("agent-state-store", "data"),
    )
    def update_view(
        structure,
        display_mode,
        display_options,
        atom_scale,
        bond_radius,
        minor_opacity,
        axis_scale,
        fragment_type,
        site_index,
        topology_toggle,
        fast_rendering_toggle,
        click_data,
        agent_state,
    ):
        state = backend.normalize_state(
            {
                **(agent_state or {}),
                "structure": structure,
                "display_mode": display_mode,
                "display_options": display_options,
                "atom_scale": atom_scale,
                "bond_radius": bond_radius,
                "minor_opacity": minor_opacity,
                "axis_scale": axis_scale,
                "topology_fragment_type": fragment_type,
                "topology_site_index": None if site_index in ("", None) else int(site_index),
                "topology_enabled": "enabled" in (topology_toggle or []),
                "fast_rendering": "fast" in (fast_rendering_toggle or []),
            }
        )
        fig, topology_data = backend.figure_for_state(state, click_data=click_data)
        summary = _structure_summary(backend.scene_for_state(state))
        return fig, topology_histogram_figure(topology_data), topology_results_markdown(topology_data), summary

    @app.callback(
        Output("status", "children"),
        Input("save-preset-btn", "n_clicks"),
        Input("export-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def save_or_export(_, __):
        triggered = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else None
        if triggered == "export-btn":
            result = backend.export_static()
            return f"Saved preset: {backend.preset_path}\nStatic export return code: {result['returncode']}\n{result['stderr'] or result['stdout']}"
        result = backend.save_preset()
        return f"Saved preset: {result['path']}"

    register_api(app, backend)
    return app


def _build_parser():
    parser = argparse.ArgumentParser(description="Standalone crystal viewer with topology analysis.")
    parser.add_argument("--preset", default=DEFAULT_PRESET_PATH, help="Preset JSON to load and save.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=8051, help="Port to expose.")
    parser.add_argument("--structure", nargs="*", help="Serve only selected catalog structure(s).")
    parser.add_argument("--cif", nargs="*", help="Optional CIF path(s) to preload.")
    parser.add_argument("--api-only", action="store_true", help="Reserved for automation mode; still serves the same app.")
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    app = create_app(args.preset, names=args.structure, root_dir=WORKSPACE_DIR, cif_paths=args.cif or [])
    print(f"Serving crystal viewer at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
