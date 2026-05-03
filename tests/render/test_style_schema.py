from __future__ import annotations

import pytest

from crystal_viewer.presets import DEFAULT_STYLE, ORTEP_MODES, deep_merge, json_safe
from crystal_viewer.renderer import DISORDER_DISPATCH, MATERIAL_DISPATCH, STYLE_DISPATCH, validate_style_schema


def test_style_schema_round_trips_and_enums_are_public():
    style = deep_merge(DEFAULT_STYLE, {"material": "flat", "style": "wireframe", "disorder": "none"})
    assert json_safe(style)["material"] == "flat"
    assert set(MATERIAL_DISPATCH) == {"flat", "mesh"}
    assert "ortep" in STYLE_DISPATCH
    assert "outline_rings" in DISORDER_DISPATCH
    assert set(ORTEP_MODES) == {"ortep_solid", "ortep_axes", "ortep_octant"}


@pytest.mark.parametrize(
    ("key", "value"),
    [("material", "bogus"), ("style", "bogus"), ("disorder", "bogus"), ("ortep_mode", "bogus")],
)
def test_style_schema_rejects_unknown_values(key, value):
    with pytest.raises(ValueError):
        validate_style_schema({**DEFAULT_STYLE, key: value})


def test_ortep_mode_sets_trace_knobs():
    style = validate_style_schema({**DEFAULT_STYLE, "style": "ortep", "ortep_mode": "ortep_octant"})
    assert style["ortep_octant_shading"] is True
    assert style["ortep_show_principal_axes"] is False
