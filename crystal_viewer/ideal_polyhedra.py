from __future__ import annotations

import math
from typing import Dict, Iterable

import numpy as np


def _normalize_points(points: Iterable[Iterable[float]]) -> np.ndarray:
    arr = np.array(list(points), dtype=float)
    arr -= arr.mean(axis=0)
    norms = np.linalg.norm(arr, axis=1)
    nonzero = norms > 1e-8
    if np.any(nonzero):
        arr[nonzero] /= norms[nonzero][:, None]
    return arr


def _ring(n: int, z: float, phase_deg: float = 0.0, radius: float | None = None) -> list[list[float]]:
    if radius is None:
        radius = math.sqrt(max(1.0 - z * z, 1e-8))
    pts = []
    for idx in range(n):
        ang = math.radians(phase_deg + idx * 360.0 / n)
        pts.append([radius * math.cos(ang), radius * math.sin(ang), z])
    return pts


def _cube():
    return _normalize_points(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ]
    )


def _square_antiprism():
    return _normalize_points(_ring(4, 0.48, 0.0) + _ring(4, -0.48, 45.0))


def _dodecahedron8():
    pts = []
    for sign_x in (-1, 1):
        for sign_y in (-1, 1):
            pts.append([sign_x, sign_y, 0.0])
            pts.append([0.0, sign_x / math.sqrt(2), sign_y * math.sqrt(0.5)])
    return _normalize_points(pts[:8])


def _tricapped_trigonal_prism():
    top = _ring(3, 0.55, 0.0)
    bottom = _ring(3, -0.55, 60.0)
    caps = [[1.0, 0.0, 0.0], [-0.5, math.sqrt(3) / 2.0, 0.0], [-0.5, -math.sqrt(3) / 2.0, 0.0]]
    return _normalize_points(top + bottom + caps)


def _capped_square_antiprism():
    return _normalize_points(_ring(4, 0.42, 0.0) + _ring(4, -0.42, 45.0) + [[0.0, 0.0, 1.0]])


def _bicapped_square_antiprism():
    return _normalize_points(_ring(4, 0.36, 0.0) + _ring(4, -0.36, 45.0) + [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])


def _bicapped_dodecahedron():
    base = _ring(5, 0.15, 0.0, radius=0.95) + _ring(5, -0.15, 36.0, radius=0.95)
    return _normalize_points(base)


def _capped_pentagonal_antiprism():
    return _normalize_points(_ring(5, 0.36, 0.0) + _ring(5, -0.36, 36.0) + [[0.0, 0.0, 1.0]])


def _capped_pentagonal_prism():
    return _normalize_points(_ring(5, 0.42, 0.0) + _ring(5, -0.42, 0.0) + [[0.0, 0.0, 1.0]])


def _edge_bicapped_square_antiprism():
    base = _ring(4, 0.34, 0.0) + _ring(4, -0.34, 45.0)
    caps = [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
    return _normalize_points(base + caps)


def _icosahedron():
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    pts = []
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            pts.extend(
                [
                    [0.0, s1, s2 * phi],
                    [s1, s2 * phi, 0.0],
                    [s2 * phi, 0.0, s1],
                ]
            )
    return _normalize_points(pts)


def _cuboctahedron():
    pts = []
    for zero in range(3):
        for s1 in (-1, 1):
            for s2 in (-1, 1):
                pt = [0.0, 0.0, 0.0]
                axes = [0, 1, 2]
                axes.remove(zero)
                pt[axes[0]] = s1
                pt[axes[1]] = s2
                pts.append(pt)
    return _normalize_points(pts)


IDEAL_POLYHEDRA: Dict[int, Dict[str, np.ndarray]] = {
    8: {
        "cube": _cube(),
        "square_antiprism": _square_antiprism(),
        "dodecahedron": _dodecahedron8(),
    },
    9: {
        "capped_square_antiprism": _capped_square_antiprism(),
        "tricapped_trigonal_prism": _tricapped_trigonal_prism(),
    },
    10: {
        "bicapped_square_antiprism": _bicapped_square_antiprism(),
        "bicapped_dodecahedron": _bicapped_dodecahedron(),
    },
    11: {
        "capped_pentagonal_antiprism": _capped_pentagonal_antiprism(),
        "capped_pentagonal_prism": _capped_pentagonal_prism(),
        "edge_bicapped_square_antiprism": _edge_bicapped_square_antiprism(),
    },
    12: {
        "icosahedron": _icosahedron(),
        "cuboctahedron": _cuboctahedron(),
    },
}


def ideal_polyhedra_for_cn(cn: int) -> Dict[str, np.ndarray]:
    return IDEAL_POLYHEDRA.get(cn, {})


def all_ideal_polyhedra() -> Dict[int, Dict[str, np.ndarray]]:
    return IDEAL_POLYHEDRA
