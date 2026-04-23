from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

import numpy as np
import plotly.graph_objects as go


def _normalize(vec: Iterable[float], fallback: Iterable[float]) -> np.ndarray:
    arr = np.array(list(vec), dtype=float)
    if arr.shape != (3,) or np.linalg.norm(arr) < 1e-8:
        arr = np.array(list(fallback), dtype=float)
    norm = np.linalg.norm(arr)
    if norm < 1e-8:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return arr / norm


def _plotly_camera_from_scene(scene: dict) -> dict:
    eye = _normalize(scene.get("view_direction", [0.0, 0.0, 1.0]), [0.0, 0.0, 1.0]) * 1.8
    up = _normalize(scene.get("up", [0.0, 1.0, 0.0]), [0.0, 1.0, 0.0])
    return {
        "eye": {"x": float(eye[0]), "y": float(eye[1]), "z": float(eye[2])},
        "center": {"x": 0.0, "y": 0.0, "z": 0.0},
        "up": {"x": float(up[0]), "y": float(up[1]), "z": float(up[2])},
    }


def _visible_atoms(scene: dict, style: dict):
    atoms = scene["draw_atoms"]
    if style.get("show_minor_only", False):
        atoms = [atom for atom in atoms if atom["is_minor"]]
    return atoms or scene["draw_atoms"]


def _scene_ranges(scene: dict, style: dict, topology_data: dict | None = None):
    """Compute ``[xr, yr, zr]`` axis ranges for the Plotly scene.

    A scene-level ``viewport`` override (set by :func:`uniform_viewport`) wins
    unconditionally; this is how caller code pins several scenes to a shared
    world cube so they render at identical screen scale.

    Otherwise the bounds are inflated by each atom's **visual radius** (rather
    than a blanket 18 % fractional pad) so spheres — especially large halides
    like Cl, Br, I — are never clipped at the panel edge. Unit-cell corners and
    topology markers expand the box but do not contribute radii.
    """
    override = scene.get("viewport")
    if override:
        return [
            [float(override["x"][0]), float(override["x"][1])],
            [float(override["y"][0]), float(override["y"][1])],
            [float(override["z"][0]), float(override["z"][1])],
        ]

    atoms = _visible_atoms(scene, style)
    atom_scale = float(style.get("atom_scale", 1.0))

    atom_mins = None
    atom_maxs = None
    if atoms:
        carts = np.array([atom["cart"] for atom in atoms], dtype=float)
        radii = np.array(
            [max(float(atom.get("atom_radius", 0.18)), 0.05) for atom in atoms],
            dtype=float,
        ) * atom_scale
        atom_mins = (carts - radii[:, None]).min(axis=0)
        atom_maxs = (carts + radii[:, None]).max(axis=0)

    extras = []
    if style.get("show_unit_cell", False):
        a = np.array(scene["M"][:, 0], dtype=float)
        b = np.array(scene["M"][:, 1], dtype=float)
        c = np.array(scene["M"][:, 2], dtype=float)
        for corner in (
            np.zeros(3, dtype=float),
            a, b, c, a + b, a + c, b + c, a + b + c,
        ):
            extras.append(corner)
    if topology_data:
        center = topology_data.get("center_coords")
        if center is not None:
            extras.append(np.array(center, dtype=float))
        for point in topology_data.get("shell_coords") or []:
            extras.append(np.array(point, dtype=float))
    if extras:
        extras_arr = np.array(extras, dtype=float)
        extras_min = extras_arr.min(axis=0)
        extras_max = extras_arr.max(axis=0)
        if atom_mins is None:
            atom_mins, atom_maxs = extras_min, extras_max
        else:
            atom_mins = np.minimum(atom_mins, extras_min)
            atom_maxs = np.maximum(atom_maxs, extras_max)

    if atom_mins is None:
        return [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]

    span = np.maximum(atom_maxs - atom_mins, 0.8)
    # Small breathing-room pad layered on top of radius-aware bounds.
    pad = np.maximum(span * 0.06, 0.25)
    mins = atom_mins - pad
    maxs = atom_maxs + pad
    return [
        [float(mins[0]), float(maxs[0])],
        [float(mins[1]), float(maxs[1])],
        [float(mins[2]), float(maxs[2])],
    ]


def uniform_viewport(scenes, *, style=None, padding=0.0):
    """Stamp a shared world-cube viewport on each scene so ``build_figure``
    renders them at identical screen scale.

    For every scene the viewport becomes a cube centred on that scene's own
    atom-bounding centroid. The cube side length equals the largest
    radius-aware axis-aligned span across **all** input scenes (+ ``padding``
    in Å on every side). Callers that later draw the scenes in a grid get
    panels with a single physical length scale — no more "small molecule
    ballooning to fill the panel while the big one shrinks to pinheads".

    The ``viewport`` key is written in-place on each scene dict. Subsequent
    calls to :func:`_scene_ranges` (and therefore :func:`build_figure`) honour
    it and skip their own bounds calculation.

    Parameters
    ----------
    scenes
        Iterable of scene dicts (as returned by ``build_scene_from_cif`` /
        ``build_scene_from_atoms``).
    style
        Optional style dict used to infer ``atom_scale``. When omitted, each
        scene's own ``scene["style"]`` is consulted with a default of 1.0.
    padding
        Extra padding in Å added symmetrically to every face of the cube.

    Returns
    -------
    list[dict]
        The stamped ``viewport`` dicts, one per scene, in the order the
        scenes were provided.
    """
    scenes = list(scenes)
    if not scenes:
        return []

    radius_spans = []
    centroids = []
    for scene in scenes:
        scn_style = style if style is not None else scene.get("style") or {}
        atom_scale = float(scn_style.get("atom_scale", 1.0))
        atoms = scene.get("draw_atoms") or []
        if not atoms:
            radius_spans.append(1.0)
            centroids.append(np.zeros(3, dtype=float))
            continue
        carts = np.array([atom["cart"] for atom in atoms], dtype=float)
        radii = np.array(
            [max(float(atom.get("atom_radius", 0.18)), 0.05) for atom in atoms],
            dtype=float,
        ) * atom_scale
        mins = (carts - radii[:, None]).min(axis=0)
        maxs = (carts + radii[:, None]).max(axis=0)
        radius_spans.append(float((maxs - mins).max()))
        centroids.append(0.5 * (mins + maxs))

    half = 0.5 * max(radius_spans) + float(padding)
    viewports = []
    for scene, center in zip(scenes, centroids):
        viewport = {
            "x": [float(center[0] - half), float(center[0] + half)],
            "y": [float(center[1] - half), float(center[1] + half)],
            "z": [float(center[2] - half), float(center[2] + half)],
            "center": [float(center[0]), float(center[1]), float(center[2])],
            "half_span": float(half),
        }
        scene["viewport"] = viewport
        viewports.append(viewport)
    return viewports


def _style_bool(style: dict, key: str, default: bool = False) -> bool:
    return bool(style.get(key, default))


def style_from_controls(atom_scale, bond_radius, minor_opacity, axis_scale, options) -> dict:
    options = set(options or [])
    return {
        "atom_scale": float(atom_scale),
        "bond_radius": float(bond_radius),
        "minor_opacity": float(minor_opacity),
        "axis_scale": float(axis_scale),
        "show_labels": "labels" in options,
        "show_axes": "axes" in options,
        "show_minor_only": "minor_only" in options,
        "minor_wireframe": "minor_wireframe" in options,
        "show_hydrogen": "hydrogens" in options,
        "show_unit_cell": "unit_cell_box" in options,
        "fast_rendering": "fast_rendering" in options,
        "topology_enabled": "topology" in options,
    }


def _unit_sphere(lat_steps: int = 9, lon_steps: int = 14) -> Tuple[np.ndarray, np.ndarray]:
    vertices = []
    for lat_idx in range(lat_steps + 1):
        theta = math.pi * lat_idx / lat_steps
        for lon_idx in range(lon_steps):
            phi = 2.0 * math.pi * lon_idx / lon_steps
            vertices.append(
                [
                    math.sin(theta) * math.cos(phi),
                    math.sin(theta) * math.sin(phi),
                    math.cos(theta),
                ]
            )
    triangles = []
    for lat_idx in range(lat_steps):
        for lon_idx in range(lon_steps):
            next_lon = (lon_idx + 1) % lon_steps
            a = lat_idx * lon_steps + lon_idx
            b = lat_idx * lon_steps + next_lon
            c = (lat_idx + 1) * lon_steps + lon_idx
            d = (lat_idx + 1) * lon_steps + next_lon
            triangles.append([a, c, b])
            triangles.append([b, c, d])
    return np.array(vertices, dtype=float), np.array(triangles, dtype=int)


def _append_mesh(mesh: dict, vertices: np.ndarray, triangles: np.ndarray):
    base = len(mesh["x"])
    mesh["x"].extend(vertices[:, 0].tolist())
    mesh["y"].extend(vertices[:, 1].tolist())
    mesh["z"].extend(vertices[:, 2].tolist())
    mesh["i"].extend((triangles[:, 0] + base).tolist())
    mesh["j"].extend((triangles[:, 1] + base).tolist())
    mesh["k"].extend((triangles[:, 2] + base).tolist())


def _sphere_mesh(center: Iterable[float], radius: float, lat_steps: int = 9, lon_steps: int = 14):
    unit_vertices, unit_triangles = _unit_sphere(lat_steps=lat_steps, lon_steps=lon_steps)
    center = np.array(center, dtype=float)
    vertices = unit_vertices * float(radius) + center[None, :]
    return vertices, unit_triangles


def _cylinder_mesh(p0: Iterable[float], p1: Iterable[float], radius: float, sides: int = 8):
    start = np.array(p0, dtype=float)
    end = np.array(p1, dtype=float)
    axis = end - start
    length = np.linalg.norm(axis)
    if length < 1e-8:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=int)
    axis /= length
    ref = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(np.dot(axis, ref)) > 0.92:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
    u = np.cross(axis, ref)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)

    ring0 = []
    ring1 = []
    for idx in range(sides):
        ang = 2.0 * math.pi * idx / sides
        offset = math.cos(ang) * u * radius + math.sin(ang) * v * radius
        ring0.append(start + offset)
        ring1.append(end + offset)
    vertices = np.array(ring0 + ring1 + [start, end], dtype=float)
    cap0 = len(vertices) - 2
    cap1 = len(vertices) - 1
    triangles = []
    for idx in range(sides):
        nxt = (idx + 1) % sides
        a0 = idx
        a1 = nxt
        b0 = idx + sides
        b1 = nxt + sides
        triangles.extend([[a0, b0, a1], [a1, b0, b1], [cap0, a1, a0], [cap1, b0, b1]])
    return vertices, np.array(triangles, dtype=int)


def _atom_selection_trace(scene: dict, style: dict):
    xs, ys, zs, sizes, labels, customdata = [], [], [], [], [], []
    for idx, atom in enumerate(scene["draw_atoms"]):
        if style.get("show_minor_only", False) and not atom["is_minor"]:
            continue
        xs.append(float(atom["cart"][0]))
        ys.append(float(atom["cart"][1]))
        zs.append(float(atom["cart"][2]))
        sizes.append(max(6.0, 48.0 * atom["atom_radius"] * float(style["atom_scale"])))
        labels.append(atom["label"])
        customdata.append([idx, atom["label"], atom["elem"], int(atom["is_minor"])])
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="markers",
        marker=dict(size=sizes, color="rgba(0,0,0,0)", opacity=0.02),
        customdata=customdata,
        hovertemplate="%{customdata[1]} (%{customdata[2]})<extra></extra>",
        showlegend=False,
        name="atom-selection",
    )


def _bond_segments(scene: dict, style: dict):
    for bond in scene["bonds"]:
        if style.get("show_minor_only", False) and not bond["is_minor"]:
            continue
        start = np.array(bond["start"], dtype=float)
        end = np.array(bond["end"], dtype=float)
        mid = (start + end) / 2.0
        yield bond["color_i"], bond["is_minor"], start, mid
        yield bond["color_j"], bond["is_minor"], mid, end


def _bond_mesh_traces(scene: dict, style: dict):
    groups: Dict[Tuple[str, bool], dict] = {}
    radius = max(0.04, float(style["bond_radius"]))
    for color, is_minor, start, end in _bond_segments(scene, style):
        key = (color, is_minor)
        groups.setdefault(key, {"x": [], "y": [], "z": [], "i": [], "j": [], "k": []})
        vertices, triangles = _cylinder_mesh(
            start,
            end,
            radius * (float(style["minor_bond_scale"]) if is_minor else 1.0),
            sides=7,
        )
        if len(vertices):
            _append_mesh(groups[key], vertices, triangles)

    traces = []
    for (color, is_minor), payload in groups.items():
        traces.append(
            go.Mesh3d(
                x=payload["x"],
                y=payload["y"],
                z=payload["z"],
                i=payload["i"],
                j=payload["j"],
                k=payload["k"],
                color=color,
                opacity=float(style["minor_opacity"]) if is_minor else 1.0,
                hoverinfo="skip",
                showlegend=False,
                flatshading=False,
            )
        )
    return traces


def _atom_mesh_traces(scene: dict, style: dict):
    groups: Dict[Tuple[str, bool], dict] = {}
    for atom in scene["draw_atoms"]:
        if style.get("show_minor_only", False) and not atom["is_minor"]:
            continue
        key = (atom["color"], atom["is_minor"])
        groups.setdefault(key, {"x": [], "y": [], "z": [], "i": [], "j": [], "k": []})
        radius = float(atom["atom_radius"]) * float(style["atom_scale"])
        if atom["is_minor"]:
            radius *= 1.12
        vertices, triangles = _sphere_mesh(atom["cart"], radius, lat_steps=8, lon_steps=12)
        _append_mesh(groups[key], vertices, triangles)

    traces = []
    for (color, is_minor), payload in groups.items():
        traces.append(
            go.Mesh3d(
                x=payload["x"],
                y=payload["y"],
                z=payload["z"],
                i=payload["i"],
                j=payload["j"],
                k=payload["k"],
                color=color,
                opacity=max(0.48, float(style["minor_opacity"])) if is_minor else float(style.get("major_opacity", 1.0)),
                hoverinfo="skip",
                showlegend=False,
                flatshading=False,
            )
        )
    return traces


def _bond_scatter_traces(scene: dict, style: dict):
    groups: Dict[Tuple[str, bool], list[list[float]]] = {}
    for color, is_minor, start, end in _bond_segments(scene, style):
        groups.setdefault((color, is_minor), []).append([start, end])

    traces = []
    base_width = max(4.0, 24.0 * float(style["bond_radius"]))
    for (color, is_minor), segments in groups.items():
        xs, ys, zs = [], [], []
        for start, end in segments:
            xs.extend([float(start[0]), float(end[0]), None])
            ys.extend([float(start[1]), float(end[1]), None])
            zs.extend([float(start[2]), float(end[2]), None])
        traces.append(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line=dict(color=color, width=base_width * (float(style["minor_bond_scale"]) if is_minor else 1.0)),
                opacity=float(style["minor_opacity"]) if is_minor else 1.0,
                hoverinfo="skip",
                showlegend=False,
            )
        )
    return traces


def _atom_scatter_traces(scene: dict, style: dict):
    groups: Dict[Tuple[str, bool], dict] = {}
    for idx, atom in enumerate(scene["draw_atoms"]):
        if style.get("show_minor_only", False) and not atom["is_minor"]:
            continue
        key = (atom["elem"], atom["is_minor"])
        groups.setdefault(
            key,
            {"x": [], "y": [], "z": [], "size": [], "text": [], "color": atom["color"], "customdata": []},
        )
        base_size = max(10.0, 95.0 * atom["atom_radius"] * float(style["atom_scale"]))
        groups[key]["x"].append(float(atom["cart"][0]))
        groups[key]["y"].append(float(atom["cart"][1]))
        groups[key]["z"].append(float(atom["cart"][2]))
        groups[key]["size"].append(base_size * (1.12 if atom["is_minor"] else 1.0))
        groups[key]["text"].append(atom["label"])
        groups[key]["customdata"].append([idx, atom["label"], atom["elem"], int(atom["is_minor"])])

    traces = []
    for (elem, is_minor), payload in groups.items():
        traces.append(
            go.Scatter3d(
                x=payload["x"],
                y=payload["y"],
                z=payload["z"],
                mode="markers",
                text=payload["text"],
                customdata=payload["customdata"],
                hovertemplate="%{text}<extra></extra>",
                marker=dict(
                    size=payload["size"],
                    color=payload["color"],
                    opacity=max(0.48, float(style["minor_opacity"])) if is_minor else float(style.get("major_opacity", 1.0)),
                    line=dict(color="#444444" if is_minor else payload["color"], width=3.5 if is_minor else 0),
                ),
                showlegend=False,
                name=f"{elem}{' minor' if is_minor else ''}",
            )
        )
    return traces


def _minor_bond_wireframe_traces(scene: dict, style: dict):
    if not style.get("minor_wireframe", False):
        return []
    xs, ys, zs = [], [], []
    for bond in scene["bonds"]:
        if not bond["is_minor"]:
            continue
        start = np.array(bond["start"], dtype=float)
        end = np.array(bond["end"], dtype=float)
        xs.extend([float(start[0]), float(end[0]), None])
        ys.extend([float(start[1]), float(end[1]), None])
        zs.extend([float(start[2]), float(end[2]), None])
    if not xs:
        return []
    base_width = max(3.0, 22.0 * float(style["bond_radius"]))
    return [
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(color="#202020", width=base_width),
            opacity=0.9,
            hoverinfo="skip",
            showlegend=False,
        )
    ]


def _minor_outline_traces(scene: dict, style: dict):
    payload = {"x": [], "y": [], "z": [], "size": []}
    for atom in scene["draw_atoms"]:
        if not atom["is_minor"]:
            continue
        if style.get("show_minor_only", False) and not atom["is_minor"]:
            continue
        base_size = max(10.0, 95.0 * atom["atom_radius"] * float(style["atom_scale"]))
        ring_scale = 1.34 if style.get("minor_wireframe", False) else 1.20
        payload["x"].append(float(atom["cart"][0]))
        payload["y"].append(float(atom["cart"][1]))
        payload["z"].append(float(atom["cart"][2]))
        payload["size"].append(base_size * ring_scale)
    if not payload["x"]:
        return []
    line_color = "#111111" if style.get("minor_wireframe", False) else "#555555"
    line_width = 7.0 if style.get("minor_wireframe", False) else 4.5
    return [
        go.Scatter3d(
            x=payload["x"],
            y=payload["y"],
            z=payload["z"],
            mode="markers",
            marker=dict(
                size=payload["size"],
                color="rgba(255,255,255,0.0)",
                opacity=1.0,
                line=dict(color=line_color, width=line_width),
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    ]


def _highlight_traces(scene: dict, style: dict):
    if style.get("show_minor_only", False):
        return []
    light_dir = (
        -0.28 * np.array(scene["view_x"], dtype=float)
        + 0.34 * np.array(scene["view_y"], dtype=float)
        + 0.72 * np.array(scene["view_z"], dtype=float)
    )
    norm = np.linalg.norm(light_dir)
    if norm < 1e-8:
        return []
    light_dir /= norm

    groups: Dict[str, dict] = {}
    for atom in scene["draw_atoms"]:
        if atom["is_minor"] or atom["elem"] == "H":
            continue
        size = max(5.0, 55.0 * atom["atom_radius"] * float(style["atom_scale"]))
        center = np.array(atom["cart"], dtype=float) + light_dir * (atom["atom_radius"] * float(style["atom_scale"]) * 0.25)
        key = atom["color_light"]
        groups.setdefault(key, {"x": [], "y": [], "z": [], "size": []})
        groups[key]["x"].append(float(center[0]))
        groups[key]["y"].append(float(center[1]))
        groups[key]["z"].append(float(center[2]))
        groups[key]["size"].append(size)

    traces = []
    for color, payload in groups.items():
        traces.append(
            go.Scatter3d(
                x=payload["x"],
                y=payload["y"],
                z=payload["z"],
                mode="markers",
                marker=dict(
                    size=payload["size"],
                    color=color,
                    opacity=0.65,
                    line=dict(color="rgba(255,255,255,0.6)", width=1.5),
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    return traces


def _label_traces(scene: dict, style: dict):
    if not style.get("show_labels", True):
        return []
    buckets = {
        False: {"x": [], "y": [], "z": [], "text": [], "color": "#111111"},
        True: {"x": [], "y": [], "z": [], "text": [], "color": "#777777"},
    }
    for item in scene["label_items"]:
        if style.get("show_minor_only", False) and not item["is_minor"]:
            continue
        bucket = buckets[item["is_minor"]]
        bucket["x"].append(float(item["label_cart"][0]))
        bucket["y"].append(float(item["label_cart"][1]))
        bucket["z"].append(float(item["label_cart"][2]))
        bucket["text"].append(item["text"])

    traces = []
    for is_minor, bucket in buckets.items():
        if not bucket["x"]:
            continue
        traces.append(
            go.Scatter3d(
                x=bucket["x"],
                y=bucket["y"],
                z=bucket["z"],
                mode="text",
                text=bucket["text"],
                textfont=dict(size=10 if is_minor else 11, color=bucket["color"]),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    return traces


def _axis_traces(scene: dict, style: dict):
    if not style.get("show_axes", True):
        return []
    mins = np.array(scene["bounds"]["mins"], dtype=float)
    screen_span = max(scene["bounds"]["screen_ranges"])
    offset = 0.10 * screen_span
    origin = mins - offset * np.array(scene["view_x"], dtype=float)
    origin -= offset * np.array(scene["view_y"], dtype=float)
    scale = float(style["axis_scale"]) * screen_span
    color = style.get("axis_color", "#666666")
    opacity = float(style.get("axis_opacity", 0.72))
    labels = style.get("axes_labels") or ["a", "b", "c"]
    labels = list(labels) + ["", "", ""]  # pad defensively

    traces = []
    for vec, label in zip(
        [scene["M"][:, 0], scene["M"][:, 1], scene["M"][:, 2]],
        labels[:3],
    ):
        v = _normalize(vec, [1.0, 0.0, 0.0])
        end = origin + v * scale
        traces.append(
            go.Scatter3d(
                x=[float(origin[0]), float(end[0])],
                y=[float(origin[1]), float(end[1])],
                z=[float(origin[2]), float(end[2])],
                mode="lines",
                line=dict(color=color, width=5),
                opacity=opacity,
                hoverinfo="skip",
                showlegend=False,
            )
        )
        traces.append(
            go.Scatter3d(
                x=[float(end[0])],
                y=[float(end[1])],
                z=[float(end[2])],
                mode="text",
                text=[label],
                textfont=dict(size=12, color=color),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    return traces


def _unit_cell_traces(scene: dict, style: dict):
    if not style.get("show_unit_cell", False):
        return []
    origin = np.zeros(3, dtype=float)
    a = np.array(scene["M"][:, 0], dtype=float)
    b = np.array(scene["M"][:, 1], dtype=float)
    c = np.array(scene["M"][:, 2], dtype=float)
    corners = {
        "000": origin,
        "100": a,
        "010": b,
        "001": c,
        "110": a + b,
        "101": a + c,
        "011": b + c,
        "111": a + b + c,
    }
    edges = [
        ("000", "100"), ("000", "010"), ("000", "001"),
        ("100", "110"), ("100", "101"),
        ("010", "110"), ("010", "011"),
        ("001", "101"), ("001", "011"),
        ("110", "111"), ("101", "111"), ("011", "111"),
    ]
    xs, ys, zs = [], [], []
    for start_key, end_key in edges:
        start = corners[start_key]
        end = corners[end_key]
        xs.extend([float(start[0]), float(end[0]), None])
        ys.extend([float(start[1]), float(end[1]), None])
        zs.extend([float(start[2]), float(end[2]), None])
    return [
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(color="#777777", width=4),
            opacity=0.8,
            hoverinfo="skip",
            showlegend=False,
            name="unit-cell-box",
        )
    ]


def hull_mesh_trace(shell_coords, color: str, opacity: float = 0.15):
    coords = np.array(shell_coords, dtype=float)
    if len(coords) < 4:
        return None
    try:
        from scipy.spatial import ConvexHull
    except Exception:  # pragma: no cover - optional dependency
        return None
    hull = ConvexHull(coords)
    return go.Mesh3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        i=hull.simplices[:, 0],
        j=hull.simplices[:, 1],
        k=hull.simplices[:, 2],
        color=color,
        opacity=opacity,
        flatshading=True,
        hoverinfo="skip",
        showlegend=False,
        name="coordination-hull",
    )


def hull_edge_traces(shell_coords, color: str):
    coords = np.array(shell_coords, dtype=float)
    if len(coords) < 4:
        return []
    try:
        from scipy.spatial import ConvexHull
    except Exception:  # pragma: no cover - optional dependency
        return []
    hull = ConvexHull(coords)
    edges = set()
    for simplex in hull.simplices:
        a, b, c = simplex
        edges.add(tuple(sorted((int(a), int(b)))))
        edges.add(tuple(sorted((int(b), int(c)))))
        edges.add(tuple(sorted((int(a), int(c)))))

    xs, ys, zs = [], [], []
    for i, j in sorted(edges):
        p0 = coords[i]
        p1 = coords[j]
        xs.extend([float(p0[0]), float(p1[0]), None])
        ys.extend([float(p0[1]), float(p1[1]), None])
        zs.extend([float(p0[2]), float(p1[2]), None])
    return [
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(color=color, width=6),
            opacity=0.95,
            hoverinfo="skip",
            showlegend=False,
            name="coordination-edges",
        )
    ]


def shell_center_lines(center, shell_coords):
    center = np.array(center, dtype=float)
    coords = np.array(shell_coords, dtype=float)
    if len(coords) == 0:
        return []
    xs, ys, zs = [], [], []
    for point in coords:
        xs.extend([float(center[0]), float(point[0]), None])
        ys.extend([float(center[1]), float(point[1]), None])
        zs.extend([float(center[2]), float(point[2]), None])
    return [
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(color="#6A5ACD", width=4, dash="dash"),
            opacity=0.85,
            hoverinfo="skip",
            showlegend=False,
            name="coordination-lines",
        )
    ]


def shell_atom_traces(shell_coords, distances, color="#7C5CBF"):
    coords = np.array(shell_coords, dtype=float)
    if len(coords) == 0:
        return []
    dists = np.array(distances, dtype=float)
    if len(dists) == 0:
        dists = np.ones(len(coords))
    size = 12.0 + (dists.max() - dists + 0.1) * 5.0
    return [
        go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode="markers",
            marker=dict(size=size.tolist(), color=color, opacity=0.9, line=dict(color="#FFFFFF", width=1.5)),
            hovertemplate="d=%{text:.3f} Å<extra></extra>",
            text=dists.tolist(),
            showlegend=False,
            name="coordination-shell",
        )
    ]


def topology_traces(topology_data: dict | None):
    if not topology_data:
        return []
    traces = []
    shell_coords = topology_data.get("shell_coords") or []
    center = topology_data.get("center_coords")
    distances = topology_data.get("distances") or []
    hull_trace = hull_mesh_trace(shell_coords, color="#7C5CBF", opacity=0.16)
    if hull_trace is not None:
        traces.append(hull_trace)
    traces.extend(hull_edge_traces(shell_coords, color="#7C5CBF"))
    if center is not None:
        traces.extend(shell_center_lines(center, shell_coords))
        traces.append(
            go.Scatter3d(
                x=[float(center[0])],
                y=[float(center[1])],
                z=[float(center[2])],
                mode="markers",
                marker=dict(size=14, color="#E07C24", opacity=0.95, line=dict(color="#FFFFFF", width=1.5)),
                hovertemplate=f"{topology_data.get('center_label', 'center')}<extra></extra>",
                showlegend=False,
            )
        )
    traces.extend(shell_atom_traces(shell_coords, distances))
    return traces


def topology_histogram_figure(topology_data: dict | None) -> go.Figure:
    fig = go.Figure()
    distances = (topology_data or {}).get("all_distances", [])
    shell = set((topology_data or {}).get("distances", []))
    if distances:
        colors = ["#7C5CBF" if dist in shell else "#C9C9E8" for dist in distances]
        fig.add_trace(go.Bar(x=list(range(1, len(distances) + 1)), y=distances, marker_color=colors))
    fig.update_layout(
        margin=dict(l=18, r=18, t=28, b=28),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis_title="Neighbor rank",
        yaxis_title="Distance (Å)",
        showlegend=False,
        title="Distance Histogram",
    )
    return fig


def topology_results_markdown(topology_data: dict | None) -> str:
    if not topology_data:
        return "Topology analysis inactive."
    angular = topology_data.get("angular", {})
    best = angular.get("best_match")
    planarity = topology_data.get("planarity", {})
    prism = topology_data.get("prism_analysis", {})
    lines = [
        f"Center: {topology_data.get('center_label', '?')} ({topology_data.get('center_type', '?')})",
        f"CN: {topology_data.get('coordination_number', 0)}",
    ]
    if best:
        lines.append(f"Best ideal: {best['name']} (angular RMSD {best['angular_rmsd']:.2f}°)")
    if planarity.get("best_rms") is not None:
        lines.append(f"Best planarity RMS: {planarity['best_rms']:.3f} Å")
    if prism.get("classification"):
        lines.append(f"Prism test: {prism['classification']} ({prism['twist_deg']:.1f}°)")
    return "\n".join(lines)


def build_figure(scene: dict, style: dict, topology_data: dict | None = None) -> go.Figure:
    fig = go.Figure()
    xr, yr, zr = _scene_ranges(scene, style, topology_data=topology_data if style.get("topology_enabled", True) else None)
    use_fast = bool(style.get("fast_rendering", False)) or len(scene.get("draw_atoms", [])) > 200

    bond_traces = _bond_scatter_traces(scene, style) if use_fast else _bond_mesh_traces(scene, style)
    atom_traces = _atom_scatter_traces(scene, style) if use_fast else _atom_mesh_traces(scene, style)

    for trace in bond_traces:
        fig.add_trace(trace)
    for trace in _minor_bond_wireframe_traces(scene, style):
        fig.add_trace(trace)
    for trace in atom_traces:
        fig.add_trace(trace)
    for trace in _minor_outline_traces(scene, style):
        fig.add_trace(trace)
    for trace in _highlight_traces(scene, style):
        fig.add_trace(trace)
    for trace in _label_traces(scene, style):
        fig.add_trace(trace)
    for trace in _axis_traces(scene, style):
        fig.add_trace(trace)
    for trace in _unit_cell_traces(scene, style):
        fig.add_trace(trace)
    if style.get("topology_enabled", True):
        for trace in topology_traces(topology_data):
            fig.add_trace(trace)
    fig.add_trace(_atom_selection_trace(scene, style))

    show_title = bool(style.get("show_title", True))
    title_arg = dict(text=scene["title"], x=0.5) if show_title else None
    top_margin = 50 if show_title else 0

    # If all three axis ranges share a side (i.e. a caller stamped a cube via
    # uniform_viewport), lock the aspect ratio to ``cube`` so the camera does
    # not stretch when Plotly renders to a non-square viewport.
    xr_span = xr[1] - xr[0]
    yr_span = yr[1] - yr[0]
    zr_span = zr[1] - zr[0]
    is_cube = max(
        abs(xr_span - yr_span),
        abs(yr_span - zr_span),
        abs(xr_span - zr_span),
    ) < 1e-6
    aspectmode = "cube" if is_cube else "data"

    fig.update_layout(
        title=title_arg,
        showlegend=False,
        paper_bgcolor=style.get("background", "#FFFFFF"),
        plot_bgcolor=style.get("background", "#FFFFFF"),
        margin=dict(l=0, r=0, t=top_margin, b=0),
        scene=dict(
            xaxis=dict(visible=False, range=xr),
            yaxis=dict(visible=False, range=yr),
            zaxis=dict(visible=False, range=zr),
            aspectmode=aspectmode,
            camera=_plotly_camera_from_scene(scene),
            bgcolor=style.get("background", "#FFFFFF"),
        ),
    )
    return fig
