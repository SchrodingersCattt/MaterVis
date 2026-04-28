# Dash viewer HTTP/WebSocket service

The interactive Dash app exposes a REST + WebSocket API for driving the
running viewer programmatically — uploads, state changes, screenshots,
preset save/load.

- Base URL: `http://{host}:{port}/api/v1`
- WebSocket: `ws://{host}:{port}/api/v1/ws`

## REST endpoints

- `GET /state`
  Returns the full viewer state.
- `POST /state`
  Accepts any subset of:
  `structure`, `display_mode`, `atom_scale`, `bond_radius`,
  `minor_opacity`, `axis_scale`, `display_options`,
  `topology_species_keys` (list of stoichiometric formulas like
  `"C8N1"`, `"ClO4"`, `"N1"` -- one polyhedron per matching fragment
  for every key in the list, which gives a tiled view "for free"),
  `topology_site_index` (primary site for the histogram /
  results panel), `topology_enabled`, `topology_hull_color`,
  `fast_rendering`, `camera`, `cutoff`.

  Legacy aliases that still work: `topology_fragment_type` (`"A"` /
  `"B"` / `"X"`) is translated to the matching list of species keys
  in the active scene, and `topology_show_all_sites: true` selects
  every species at once.
  `display_mode` accepts `formula_unit`, `unit_cell`, `asymmetric_unit`,
  or `cluster` (free molecular cluster — every parsed atom is drawn,
  no formula-unit trim, no periodic imaging of bonds).
- `GET /camera`
  Returns the current Plotly camera.
- `POST /camera`
  Sets the full Plotly camera directly.
- `POST /camera/action`
  Convenience camera controls. Examples:
  `{"action": "zoom", "factor": 1.15}`,
  `{"action": "orbit", "yaw_deg": 12, "pitch_deg": -6}`,
  `{"action": "pan", "dx": 0.05, "dy": -0.03, "dz": 0.0}`,
  `{"action": "reset"}`.
- `POST /upload`
  Multipart form upload with field `file`.
- `GET /structures`
  Lists the loaded catalog and uploaded structures.
- `GET /scene/{name}`
  Returns the scene JSON and fragment table.
- `POST /topology`
  JSON body: `{"structure": "SY", "center_index": 0, "cutoff": 10.0}`.
- `GET /screenshot`
  Returns a PNG snapshot of the current Plotly view.
- `POST /preset/save`
  Optional JSON body: `{"path": "custom_preset.json"}`.
- `POST /preset/load`
  JSON body: `{"path": "custom_preset.json"}`.
- `POST /export`
  Triggers the vendored `crystal_viewer.legacy.plot_crystal` exporter
  with the current preset.

## Stable UI element IDs

Use these when scripting through Selenium / Playwright / Dash testing
hooks rather than the REST surface.

- `structure-selector`: structure radio list
- `cif-upload`: upload zone
- `display-options`: labels / axes / minor-only / wireframe checklist
- `display-mode-selector`: `formula_unit`, `unit_cell`,
  `asymmetric_unit`, `cluster`
- `fast-rendering-toggle`: mesh fallback toggle
- `atom-scale-slider`
- `bond-radius-slider`
- `minor-opacity-slider`
- `axis-scale-slider`
- `topology-fragment-type`
- `topology-site-index`
- `topology-toggle`
- `crystal-graph`
- `topology-histogram`
- `topology-results`
- `save-preset-btn`
- `export-btn`

## Suggested automation pattern

1. `POST /upload` with a CIF file.
2. `POST /state` to select the uploaded structure or set
   render/display controls.
3. `POST /topology` with a chosen `center_index`.
4. `POST /camera/action` to zoom/orbit/pan if needed.
5. `GET /screenshot` to capture the current viewport.
6. `POST /preset/save` if the tuned state should be persisted.

## WebSocket messages

- Server → client:
  `{"version": <int>, "state": {...}, "structures": [...]}`
  whenever state or structure inventory changes.
- Client → server:
  `{"type": "set_state", "payload": {...}}`

## Local-only data

- No CIF files are bundled in the repository.
- Default presets are written under `.local/`.
- Optional local catalog files can be provided via
  `catalog.local.json` or `.local/catalog.local.json`.
