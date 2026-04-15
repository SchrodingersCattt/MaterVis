from __future__ import annotations

import itertools
import math
from typing import Any, Dict, Iterable, Sequence

import numpy as np

from .ideal_polyhedra import ideal_polyhedra_for_cn

try:
    from scipy.spatial import ConvexHull
except Exception:  # pragma: no cover - optional dependency
    ConvexHull = None


def _array(points: Iterable[Iterable[float]]) -> np.ndarray:
    arr = np.array(list(points), dtype=float)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def classify_fragments(bundle) -> list[dict[str, Any]]:
    return list(getattr(bundle, "topology_fragment_table", None) or bundle.fragment_table)


def _lattice_vectors(bundle) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    M = np.array(bundle.M if getattr(bundle, "M", None) is not None else bundle.scene["M"], dtype=float)
    return M[:, 0], M[:, 1], M[:, 2]


def _neighbor_types(fragments: list[dict[str, Any]], center_type: str) -> list[str]:
    available = {frag.get("type", "?") for frag in fragments}
    if center_type in ("A", "B") and "X" in available:
        return ["X"]
    if center_type == "X":
        if "B" in available:
            return ["B"]
        if "A" in available:
            return ["A"]
    return [frag_type for frag_type in ("B", "A", "X", "?") if frag_type in available and frag_type != center_type]


def _translation_grid(bundle, cutoff: float) -> list[tuple[int, int, int, np.ndarray]]:
    lattice = _lattice_vectors(bundle)
    ranges = []
    for vec in lattice:
        length = max(np.linalg.norm(vec), 1e-6)
        span = max(1, int(math.ceil((cutoff + 1.0) / length)))
        ranges.append(range(-span, span + 1))
    translations = []
    for na, nb, nc in itertools.product(*ranges):
        shift_vec = na * lattice[0] + nb * lattice[1] + nc * lattice[2]
        translations.append((na, nb, nc, shift_vec))
    return translations


def _neighbor_pool(bundle, center_fragment: dict, cutoff: float) -> list[dict[str, Any]]:
    fragments = classify_fragments(bundle)
    center_type = center_fragment.get("type", "?")
    allowed_types = set(_neighbor_types(fragments, center_type))
    center = np.array(center_fragment["center"], dtype=float)
    translations = _translation_grid(bundle, cutoff)
    candidates = []
    for fragment in fragments:
        if fragment["index"] == center_fragment["index"] and center_type not in {"X"}:
            continue
        if allowed_types and fragment.get("type", "?") not in allowed_types:
            continue
        base_center = np.array(fragment["center"], dtype=float)
        for na, nb, nc, shift_vec in translations:
            if fragment["index"] == center_fragment["index"] and (na, nb, nc) == (0, 0, 0):
                continue
            point = base_center + shift_vec
            distance = float(np.linalg.norm(point - center))
            if 1e-8 < distance <= cutoff:
                item = dict(fragment)
                item["image_shift"] = [na, nb, nc]
                item["center"] = [float(x) for x in point]
                item["distance"] = distance
                candidates.append(item)
    candidates.sort(key=lambda item: item["distance"])
    return candidates


def detect_coordination_number(distances: Sequence[float], fallback_max: int | None = None) -> dict[str, Any]:
    sorted_distances = np.sort(np.array(distances, dtype=float))
    if len(sorted_distances) == 0:
        return {"coordination_number": 0, "gap_index": None, "gap_value": None}
    if len(sorted_distances) == 1:
        return {"coordination_number": 1, "gap_index": 0, "gap_value": 0.0}

    gaps = np.diff(sorted_distances)
    cn = int(np.argmax(gaps) + 1)
    if fallback_max is not None:
        cn = min(cn, int(fallback_max))
    return {
        "coordination_number": max(1, cn),
        "gap_index": cn - 1,
        "gap_value": float(gaps[cn - 1]),
        "sorted_distances": sorted_distances.tolist(),
        "gaps": gaps.tolist(),
    }


def extract_coordination_shell(
    bundle,
    center_index: int,
    cutoff: float = 10.0,
    *,
    display_center: Iterable[float] | None = None,
    display_label: str | None = None,
    display_type: str | None = None,
) -> dict[str, Any]:
    fragments = classify_fragments(bundle)
    center_fragment = next((frag for frag in fragments if int(frag["index"]) == int(center_index)), None)
    if center_fragment is None:
        raise IndexError(f"Unknown fragment index: {center_index}")

    source_center = np.array(center_fragment["center"], dtype=float)
    plot_center = source_center if display_center is None else np.array(display_center, dtype=float)
    candidates = _neighbor_pool(bundle, center_fragment, cutoff=cutoff)

    cn_info = detect_coordination_number([item["distance"] for item in candidates])
    cn = int(cn_info["coordination_number"])
    shell = candidates[:cn]
    source_shell_coords = np.array([item["center"] for item in shell], dtype=float) if shell else np.zeros((0, 3), dtype=float)
    shell_coords = (
        plot_center[None, :] + (source_shell_coords - source_center[None, :])
        if len(source_shell_coords)
        else np.zeros((0, 3), dtype=float)
    )
    shell_distances = [float(item["distance"]) for item in shell]

    return {
        "center_index": int(center_index),
        "center_label": display_label or center_fragment.get("label", f"site-{center_index}"),
        "center_type": display_type or center_fragment.get("type", "?"),
        "center_coords": plot_center.tolist(),
        "source_center_coords": source_center.tolist(),
        "cutoff": float(cutoff),
        "neighbor_pool_size": len(candidates),
        "coordination_number": cn,
        "gap_info": cn_info,
        "shell": shell,
        "shell_coords": shell_coords.tolist(),
        "source_shell_coords": source_shell_coords.tolist(),
        "distances": shell_distances,
        "all_distances": [float(item["distance"]) for item in candidates],
    }


def compute_angular_signature(shell_coords: Iterable[Iterable[float]], center: Iterable[float] | None = None) -> dict[str, Any]:
    coords = _array(shell_coords)
    if len(coords) == 0:
        return {"angles": [], "sorted_angles": [], "count": 0}
    center_vec = np.zeros(3, dtype=float) if center is None else np.array(center, dtype=float)
    vectors = coords - center_vec
    norms = np.linalg.norm(vectors, axis=1)
    angles = []
    for i, j in itertools.combinations(range(len(vectors)), 2):
        if norms[i] < 1e-8 or norms[j] < 1e-8:
            continue
        cosang = np.clip(np.dot(vectors[i], vectors[j]) / (norms[i] * norms[j]), -1.0, 1.0)
        angles.append(float(np.degrees(np.arccos(cosang))))
    angles.sort()
    return {"angles": angles, "sorted_angles": angles, "count": len(angles)}


def angular_rmsd_vs_ideals(shell_coords: Iterable[Iterable[float]], center: Iterable[float] | None = None) -> dict[str, Any]:
    coords = _array(shell_coords)
    cn = int(len(coords))
    signature = compute_angular_signature(coords, center=center)
    actual = np.array(signature["sorted_angles"], dtype=float)
    results = []
    for name, ideal in ideal_polyhedra_for_cn(cn).items():
        ideal_signature = np.array(compute_angular_signature(ideal)["sorted_angles"], dtype=float)
        size = min(len(actual), len(ideal_signature))
        if size == 0:
            rmsd = float("inf")
        else:
            diff = actual[:size] - ideal_signature[:size]
            rmsd = float(np.sqrt(np.mean(diff * diff)))
        results.append({"name": name, "angular_rmsd": rmsd})
    results.sort(key=lambda item: item["angular_rmsd"])
    return {
        "coordination_number": cn,
        "results": results,
        "best_match": results[0] if results else None,
    }


def planarity_analysis(shell_coords: Iterable[Iterable[float]], group_size: int = 5) -> dict[str, Any]:
    coords = _array(shell_coords)
    if len(coords) < group_size:
        return {"best_rms": None, "best_indices": [], "group_size": group_size}
    best_rms = float("inf")
    best_indices = None
    for combo in itertools.combinations(range(len(coords)), group_size):
        subset = coords[list(combo)]
        centered = subset - subset.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[-1]
        distances = centered @ normal
        rms = float(np.sqrt(np.mean(distances * distances)))
        if rms < best_rms:
            best_rms = rms
            best_indices = combo
    return {
        "best_rms": best_rms if best_indices is not None else None,
        "best_indices": list(best_indices or []),
        "group_size": group_size,
    }


def detect_prism_vs_antiprism(shell_coords: Iterable[Iterable[float]]) -> dict[str, Any]:
    coords = _array(shell_coords)
    if len(coords) < 10:
        return {"classification": None, "twist_deg": None}
    z_sorted = np.argsort(coords[:, 2])
    bottom = coords[z_sorted[:5]]
    top = coords[z_sorted[-5:]]
    top_angles = np.sort(np.degrees(np.arctan2(top[:, 1], top[:, 0])) % 360.0)
    bottom_angles = np.sort(np.degrees(np.arctan2(bottom[:, 1], bottom[:, 0])) % 360.0)
    shifts = []
    for angle_top, angle_bottom in zip(top_angles, bottom_angles):
        delta = (angle_top - angle_bottom + 180.0) % 360.0 - 180.0
        shifts.append(abs(delta))
    twist = float(np.mean(shifts))
    classification = "antiprism" if twist > 18.0 else "prism"
    return {"classification": classification, "twist_deg": twist}


def convex_hull_payload(shell_coords: Iterable[Iterable[float]]) -> dict[str, Any]:
    coords = _array(shell_coords)
    if len(coords) < 4 or ConvexHull is None:
        return {"vertices": coords.tolist(), "simplices": [], "edges": []}
    hull = ConvexHull(coords)
    edges = set()
    for simplex in hull.simplices:
        simplex = list(simplex)
        for i, j in itertools.combinations(simplex, 2):
            edges.add(tuple(sorted((int(i), int(j)))))
    return {
        "vertices": coords.tolist(),
        "simplices": hull.simplices.tolist(),
        "edges": [list(edge) for edge in sorted(edges)],
    }


def analyze_topology(
    bundle,
    center_index: int,
    cutoff: float = 10.0,
    *,
    display_center: Iterable[float] | None = None,
    display_label: str | None = None,
    display_type: str | None = None,
) -> dict[str, Any]:
    shell = extract_coordination_shell(
        bundle,
        center_index=center_index,
        cutoff=cutoff,
        display_center=display_center,
        display_label=display_label,
        display_type=display_type,
    )
    center = shell["center_coords"]
    shell_coords = shell["shell_coords"]
    angular = angular_rmsd_vs_ideals(shell_coords, center=center)
    planarity = planarity_analysis(shell_coords, group_size=min(5, len(shell_coords)) if shell_coords else 5)
    prism = detect_prism_vs_antiprism(shell_coords)
    hull = convex_hull_payload(shell_coords)
    return {
        **shell,
        "angular": angular,
        "planarity": planarity,
        "prism_analysis": prism,
        "hull": hull,
    }
