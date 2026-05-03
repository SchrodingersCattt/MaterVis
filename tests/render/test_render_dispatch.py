from __future__ import annotations

import itertools

from crystal_viewer.loader import build_empty_bundle
from crystal_viewer.renderer import DISORDER_DISPATCH, MATERIAL_DISPATCH, STYLE_DISPATCH, build_figure


def _scene():
    scene = build_empty_bundle().scene
    scene["draw_atoms"] = [
        {
            "label": "C1",
            "elem": "C",
            "cart": [0.0, 0.0, 0.0],
            "atom_radius": 0.18,
            "color": "#555555",
            "color_light": "#888888",
            "is_minor": False,
            "uiso": 0.04,
            "U": None,
        },
        {
            "label": "O1",
            "elem": "O",
            "cart": [1.2, 0.0, 0.0],
            "atom_radius": 0.17,
            "color": "#B85060",
            "color_light": "#D48A88",
            "is_minor": True,
            "uiso": 0.04,
            "U": None,
        },
    ]
    scene["bonds"] = [
        {
            "i": 0,
            "j": 1,
            "start": [0.0, 0.0, 0.0],
            "end": [1.2, 0.0, 0.0],
            "color_i": "#555555",
            "color_j": "#B85060",
            "is_minor": True,
        }
    ]
    return scene


def test_render_dispatch_cartesian_product_builds_figures():
    for material, render_style, disorder in itertools.product(MATERIAL_DISPATCH, STYLE_DISPATCH, DISORDER_DISPATCH):
        fig = build_figure(
            _scene(),
            {
                "material": material,
                "style": render_style,
                "disorder": disorder,
                "atom_scale": 1.0,
                "bond_radius": 0.1,
                "axis_scale": 0.1,
                "show_axes": False,
                "show_labels": False,
                "topology_enabled": False,
            },
        )
        assert len(fig.data) > 0
        if material == "flat":
            assert any(trace.type == "scatter3d" for trace in fig.data)
        else:
            assert any(trace.type == "mesh3d" for trace in fig.data)


def test_wireframe_builds_ring_and_bond_meshes_without_spheres():
    fig = build_figure(
        _scene(),
        {
            "material": "mesh",
            "style": "wireframe",
            "disorder": "none",
            "atom_scale": 1.0,
            "bond_radius": 0.1,
            "axis_scale": 0.1,
            "show_axes": False,
            "show_labels": False,
            "topology_enabled": False,
        },
    )
    names = {trace.name for trace in fig.data if getattr(trace, "name", None)}
    assert "wireframe-atoms" in names
    assert "wireframe-bonds" in names


def test_dashed_disorder_fast_path_sets_dash_style():
    fig = build_figure(
        _scene(),
        {
            "material": "flat",
            "style": "ball_stick",
            "disorder": "dashed_bonds",
            "atom_scale": 1.0,
            "bond_radius": 0.1,
            "axis_scale": 0.1,
            "show_axes": False,
            "show_labels": False,
            "topology_enabled": False,
        },
    )
    bond_lines = [trace for trace in fig.data if trace.type == "scatter3d" and getattr(trace, "mode", None) == "lines"]
    assert any(getattr(trace.line, "dash", None) == "dash" for trace in bond_lines)


def test_monochrome_style_renders_black_atoms_and_bonds():
    fig = build_figure(
        _scene(),
        {
            "material": "flat",
            "style": "ball_stick",
            "disorder": "none",
            "atom_scale": 1.0,
            "bond_radius": 0.1,
            "axis_scale": 0.1,
            "show_axes": False,
            "show_labels": False,
            "topology_enabled": False,
            "monochrome": True,
        },
    )
    marker_colors = [
        trace.marker.color
        for trace in fig.data
        if trace.type == "scatter3d" and getattr(trace, "mode", None) == "markers" and getattr(trace.marker, "color", None)
    ]
    assert "#000000" in marker_colors
