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
