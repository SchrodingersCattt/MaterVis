from __future__ import annotations

import numpy as np

from crystal_viewer.loader import build_empty_bundle
from crystal_viewer.ortep import ortep_atom_mesh_traces, ortep_axis_dash_traces


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
