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
    """Pick which fragment types should populate the neighbour pool.

    XYn perovskite-style chemistry: cations (A or B) are coordinated by
    anions (X), and X is coordinated by cations. We treat A and B as a
    *single class* of cation when X is the centre; otherwise the classifier's
    A/B size split would arbitrarily exclude half of the surrounding cage
    just because half the cations happen to be heavier than the others.
    """
    available = {frag.get("type", "?") for frag in fragments}
    if center_type in ("A", "B") and "X" in available:
        return ["X"]
    if center_type == "X":
        cations = [t for t in ("A", "B") if t in available]
        if cations:
            return cations
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


def _neighbor_pool_uncached(bundle, center_fragment: dict, cutoff: float) -> list[dict[str, Any]]:
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


def _neighbor_pool(bundle, center_fragment: dict, cutoff: float) -> list[dict[str, Any]]:
    """Cached PBC neighbour search. Bundle topology is immutable after load,
    so the (center_index, cutoff) tuple uniquely determines the result --
    invaluable when the species checkbox triggers analyze_topology for many
    sites in quick succession."""
    cache = getattr(bundle, "_neighbor_pool_cache", None)
    if cache is None:
        cache = {}
        try:
            bundle._neighbor_pool_cache = cache
        except Exception:
            return _neighbor_pool_uncached(bundle, center_fragment, cutoff)
    key = (int(center_fragment.get("index", -1)), float(cutoff))
    if key not in cache:
        cache[key] = _neighbor_pool_uncached(bundle, center_fragment, cutoff)
    return cache[key]


DEFAULT_CENTROID_OFFSET_FRAC = 0.15


def _hull_encloses_center(
    coords: np.ndarray,
    center: np.ndarray,
    *,
    centroid_offset_frac: float = DEFAULT_CENTROID_OFFSET_FRAC,
    face_tol: float = 1e-3,
) -> bool:
    """True iff ``center`` is *centred inside* the convex hull of ``coords``.

    A coordination polyhedron XYn is geometrically meaningful only when X
    sits roughly at the middle of the cage of Y. Two conditions have to hold:

    1. **Topological enclosure** - ``center`` must be on the interior side
       of every face of ``ConvexHull(coords)``. This rejects shells that
       bulge entirely onto one side of X.
    2. **Centrality** - the centroid of ``coords`` must be within
       ``centroid_offset_frac`` of the mean shell radius from ``center``.
       This rejects "tetrahedral pocket" configurations where X is
       topologically enclosed but pressed against one face - chemically
       those are not XYn polyhedra, they are partial coordination spheres.

    Together the two checks turn what was a topological enclosure test
    into a chemically defensible "X really sits inside Yn" predicate.
    """
    coords = np.asarray(coords, dtype=float)
    center = np.asarray(center, dtype=float)
    if len(coords) < 4 or ConvexHull is None:
        return False
    try:
        hull = ConvexHull(coords)
    except Exception:
        return False
    plane_vals = hull.equations[:, :3] @ center + hull.equations[:, 3]
    if np.any(plane_vals > face_tol):
        return False
    centroid = coords.mean(axis=0)
    radii = np.linalg.norm(coords - centroid, axis=1)
    mean_radius = float(np.mean(radii)) if len(radii) else 0.0
    if mean_radius < 1e-6:
        return True
    offset = float(np.linalg.norm(center - centroid))
    return offset <= centroid_offset_frac * mean_radius


def detect_coordination_number(
    distances: Sequence[float],
    fallback_max: int | None = None,
    *,
    coords: Sequence[Sequence[float]] | None = None,
    center: Sequence[float] | None = None,
    enforce_enclosure: bool = True,
    centroid_offset_frac: float = DEFAULT_CENTROID_OFFSET_FRAC,
) -> dict[str, Any]:
    """Choose a coordination number (CN) for an ordered list of neighbour distances.

    The base heuristic is the largest gap in the sorted distance list - the
    classical "first coordination shell" cut. When ``coords`` and ``center``
    are provided **and** ``enforce_enclosure`` is true, the chosen shell is
    additionally required to satisfy the XYn definition: the chosen shell
    must topologically enclose X **and** keep X within
    ``centroid_offset_frac`` of the mean shell radius from the shell
    centroid (chemical centrality). If the gap-defined CN fails either of
    those, the search walks CN monotonically upward until both conditions
    are met or the candidate pool is exhausted.

    The returned payload includes the raw gap-only CN under
    ``primary_gap_cn`` and an ``enclosed`` flag describing whether the
    *final* shell actually wraps the centre, so the UI can warn loudly
    when no enclosing shell is reachable inside the search cutoff.
    """
    sorted_distances = np.sort(np.array(distances, dtype=float))
    n = len(sorted_distances)
    if n == 0:
        return {
            "coordination_number": 0, "gap_index": None, "gap_value": None,
            "enclosed": False, "enclosure_expanded": False,
            "primary_gap_cn": 0, "sorted_distances": [], "gaps": [],
        }
    if n == 1:
        return {
            "coordination_number": 1, "gap_index": 0, "gap_value": 0.0,
            "enclosed": False, "enclosure_expanded": False,
            "primary_gap_cn": 1, "sorted_distances": sorted_distances.tolist(), "gaps": [],
        }

    gaps = np.diff(sorted_distances)
    primary_cn = int(np.argmax(gaps) + 1)
    cn = primary_cn
    enclosed = False
    expanded = False

    coords_arr = np.asarray(coords, dtype=float) if coords is not None else None
    center_arr = np.asarray(center, dtype=float) if center is not None else None
    if (
        enforce_enclosure
        and coords_arr is not None
        and center_arr is not None
        and len(coords_arr) >= 4
    ):
        # Coords arrive in distance-sorted order (caller's contract).
        if _hull_encloses_center(
            coords_arr[:primary_cn], center_arr,
            centroid_offset_frac=centroid_offset_frac,
        ):
            enclosed = True
        else:
            # Walk CN monotonically upward from primary_gap_cn. This is
            # less clever than a gap-ranked walk but it matches chemistry:
            # if the natural first-shell cut doesn't actually wrap X then
            # the next "complete" shell is at the smallest super-shell that
            # achieves both topological enclosure and centrality, regardless
            # of where the gap maxima fall.
            for candidate_cn in range(primary_cn + 1, len(coords_arr) + 1):
                if candidate_cn < 4:
                    continue
                if _hull_encloses_center(
                    coords_arr[:candidate_cn], center_arr,
                    centroid_offset_frac=centroid_offset_frac,
                ):
                    cn = candidate_cn
                    enclosed = True
                    expanded = True
                    break

    if fallback_max is not None:
        cn = min(cn, int(fallback_max))
    cn = max(1, cn)
    gap_index = min(cn - 1, len(gaps) - 1) if len(gaps) > 0 else None
    gap_value = float(gaps[gap_index]) if gap_index is not None else None
    return {
        "coordination_number": cn,
        "gap_index": gap_index,
        "gap_value": gap_value,
        "sorted_distances": sorted_distances.tolist(),
        "gaps": gaps.tolist(),
        "primary_gap_cn": primary_cn,
        "enclosed": enclosed,
        "enclosure_expanded": expanded,
    }


def _extract_coordination_shell_static(
    bundle,
    center_index: int,
    cutoff: float,
) -> dict[str, Any]:
    """Run the geometric part of ``extract_coordination_shell`` -- everything
    that depends only on (bundle, center_index, cutoff) and not on the
    per-call display-coordinate offsets. The result is cacheable; the
    public wrapper layers display fields on top of a shallow copy."""
    fragments = classify_fragments(bundle)
    center_fragment = next((frag for frag in fragments if int(frag["index"]) == int(center_index)), None)
    if center_fragment is None:
        raise IndexError(f"Unknown fragment index: {center_index}")
    source_center = np.array(center_fragment["center"], dtype=float)
    candidates = _neighbor_pool(bundle, center_fragment, cutoff=cutoff)
    candidate_coords = (
        np.array([item["center"] for item in candidates], dtype=float)
        if candidates else np.zeros((0, 3), dtype=float)
    )
    cn_info = detect_coordination_number(
        [item["distance"] for item in candidates],
        coords=candidate_coords,
        center=source_center,
        enforce_enclosure=True,
    )
    cn = int(cn_info["coordination_number"])
    shell = candidates[:cn]
    source_shell_coords = (
        np.array([item["center"] for item in shell], dtype=float)
        if shell else np.zeros((0, 3), dtype=float)
    )
    shell_distances = [float(item["distance"]) for item in shell]
    return {
        "center_index": int(center_index),
        "default_label": center_fragment.get("label", f"site-{center_index}"),
        "default_type": center_fragment.get("type", "?"),
        "center_formula": center_fragment.get("formula") or center_fragment.get("species"),
        "source_center_coords": source_center,
        "cutoff": float(cutoff),
        "neighbor_pool_size": len(candidates),
        "coordination_number": cn,
        "gap_info": cn_info,
        "shell": shell,
        "candidate_fragments": candidates,
        "source_shell_coords": source_shell_coords,
        "distances": shell_distances,
        "all_distances": [float(item["distance"]) for item in candidates],
    }


def _cached_extract_static(bundle, center_index: int, cutoff: float) -> dict[str, Any]:
    cache = getattr(bundle, "_shell_cache", None)
    if cache is None:
        cache = {}
        try:
            bundle._shell_cache = cache
        except Exception:
            return _extract_coordination_shell_static(bundle, center_index, cutoff)
    key = (int(center_index), float(cutoff))
    if key not in cache:
        cache[key] = _extract_coordination_shell_static(bundle, center_index, cutoff)
    return cache[key]


def extract_coordination_shell(
    bundle,
    center_index: int,
    cutoff: float = 10.0,
    *,
    display_center: Iterable[float] | None = None,
    display_label: str | None = None,
    display_type: str | None = None,
) -> dict[str, Any]:
    static = _cached_extract_static(bundle, int(center_index), float(cutoff))
    source_center = np.asarray(static["source_center_coords"], dtype=float)
    plot_center = source_center if display_center is None else np.array(display_center, dtype=float)
    delta = plot_center - source_center

    source_shell_coords = np.asarray(static["source_shell_coords"], dtype=float)
    shell_coords = (
        source_shell_coords + delta if len(source_shell_coords) else np.zeros((0, 3), dtype=float)
    )
    candidates = static["candidate_fragments"]
    pool_coords_arr = (
        np.array([item["center"] for item in candidates], dtype=float) + delta
        if candidates else np.zeros((0, 3), dtype=float)
    )
    return {
        "center_index": int(center_index),
        "center_label": display_label or static["default_label"],
        "center_type": display_type or static["default_type"],
        "center_formula": static["center_formula"],
        "center_coords": plot_center.tolist(),
        "source_center_coords": source_center.tolist(),
        "cutoff": float(cutoff),
        "neighbor_pool_size": static["neighbor_pool_size"],
        "coordination_number": static["coordination_number"],
        "gap_info": static["gap_info"],
        "shell": static["shell"],
        "candidate_fragments": candidates,
        "shell_coords": shell_coords.tolist(),
        "source_shell_coords": source_shell_coords.tolist(),
        "distances": static["distances"],
        "all_distances": static["all_distances"],
        "pool_coords": pool_coords_arr.tolist(),
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


def _analyze_topology_uncached(
    bundle,
    center_index: int,
    cutoff: float,
    display_center,
    display_label,
    display_type,
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


def analyze_topology(
    bundle,
    center_index: int,
    cutoff: float = 10.0,
    *,
    display_center: Iterable[float] | None = None,
    display_label: str | None = None,
    display_type: str | None = None,
) -> dict[str, Any]:
    """Cached primary-site analysis. The heavy ``planarity_analysis`` pass
    runs ``itertools.combinations`` of size 5 over the shell, which gets
    expensive for CN=12 / large neighbour pools. We key the cache on the
    static (center_index, cutoff) tuple -- the full bundle topology is
    immutable once loaded -- so flipping species checkboxes back and forth
    no longer redoes the work."""
    cache = getattr(bundle, "_analyze_topology_cache", None)
    if cache is None:
        cache = {}
        try:
            bundle._analyze_topology_cache = cache
        except Exception:
            return _analyze_topology_uncached(
                bundle, center_index, cutoff,
                display_center, display_label, display_type,
            )
    key = (int(center_index), float(cutoff))
    cached = cache.get(key)
    if cached is None:
        cached = _analyze_topology_uncached(
            bundle, center_index, cutoff,
            None, None, None,  # cache on the static result; overlay display fields below
        )
        cache[key] = cached
    # Display fields shift per call (camera / formula-unit centering); patch
    # them onto a shallow copy so the cache stays generic.
    out = dict(cached)
    if display_center is not None:
        plot_center = np.array(display_center, dtype=float)
        source_center = np.array(out.get("source_center_coords", plot_center), dtype=float)
        delta = plot_center - source_center
        out["center_coords"] = plot_center.tolist()
        if out.get("source_shell_coords"):
            shell = np.array(out["source_shell_coords"], dtype=float) + delta
            out["shell_coords"] = shell.tolist()
    if display_label is not None:
        out["center_label"] = display_label
    if display_type is not None:
        out["center_type"] = display_type
    return out
