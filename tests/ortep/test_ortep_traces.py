from __future__ import annotations

import numpy as np

from crystal_viewer.loader import build_empty_bundle
from crystal_viewer.ortep import (
    DEFAULT_HYDROGEN_ORTEP_UISO,
    DEFAULT_MAX_ORTEP_UISO,
    DEFAULT_ORTEP_UISO,
    MAX_ORTEP_UISO_BY_ELEMENT,
    _atom_u,
    ortep_atom_mesh_traces,
    ortep_axis_dash_traces,
)


def test_ortep_traces_include_mesh_and_optional_axes():
    scene = build_empty_bundle().scene
    scene["draw_atoms"] = [
        {
            "label": "C1",
            "elem": "C",
            "cart": [0.0, 0.0, 0.0],
            "color": "#555555",
            "is_minor": False,
            "U": np.eye(3) * 0.04,
            "uiso": 0.04,
        }
    ]
    style = {"ortep_probability": 0.5, "ortep_show_principal_axes": True}
    assert ortep_atom_mesh_traces(scene, style)
    assert ortep_axis_dash_traces(scene, style)
    assert not ortep_axis_dash_traces(scene, {**style, "ortep_show_principal_axes": False})


def test_ortep_fallback_uiso_shrinks_hydrogen():
    _, h_uiso = _atom_u({"elem": "H"})
    _, c_uiso = _atom_u({"elem": "C"})
    _, explicit_h_uiso = _atom_u({"elem": "H", "uiso": DEFAULT_ORTEP_UISO})

    assert h_uiso == DEFAULT_HYDROGEN_ORTEP_UISO
    assert h_uiso < c_uiso
    assert explicit_h_uiso == DEFAULT_ORTEP_UISO


def test_ortep_caps_disorder_inflated_uiso():
    """Some CIFs encode disorder by inflating Uiso instead of writing
    proper PART/disorder records. The renderer must clamp those values
    so a single H8/H21-style atom doesn't dominate the scene as a giant
    white blob.
    """

    # NH4 H atoms in DAP-4-style CIFs ship with Uiso = 0.20-0.25.
    _, clamped_h = _atom_u({"elem": "H", "uiso": 0.25})
    _, clamped_heavy = _atom_u({"elem": "C", "uiso": 0.50})

    assert clamped_h == MAX_ORTEP_UISO_BY_ELEMENT["H"]
    assert clamped_heavy == DEFAULT_MAX_ORTEP_UISO

    # A reasonable explicit Uiso must pass through unchanged.
    _, normal_h = _atom_u({"elem": "H", "uiso": 0.025})
    assert normal_h == 0.025


def test_ortep_caps_anisotropic_u_eigenvalues():
    """Anisotropic U bloat (worst eigenvalue ≫ cap) is rescaled rather
    than truncated component-by-component, so the ellipsoid keeps its
    shape (orientation + axial ratios) while its overall size stops
    swallowing neighbouring atoms.
    """

    inflated = np.diag([0.5, 0.05, 0.02])
    U_render, _ = _atom_u({"elem": "H", "U": inflated, "uiso": 0.5})
    eigs = np.linalg.eigvalsh(U_render)
    cap = MAX_ORTEP_UISO_BY_ELEMENT["H"]
    assert eigs.max() <= cap + 1e-9
    # Ratios preserved within numerical tolerance.
    assert np.isclose(eigs.max() / eigs.min(), 25.0, rtol=1e-6)
