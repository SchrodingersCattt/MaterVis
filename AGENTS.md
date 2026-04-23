# Crystal Viewer Agent Contract

Base URL:

`http://{host}:{port}/api/v1`

WebSocket:

`ws://{host}:{port}/api/v1/ws`

## REST endpoints

- `GET /state`
  Returns the full viewer state.
- `POST /state`
  Accepts any subset of:
  `structure`, `display_mode`, `atom_scale`, `bond_radius`, `minor_opacity`, `axis_scale`, `display_options`, `topology_fragment_type`, `topology_site_index`, `topology_enabled`, `fast_rendering`, `camera`, `cutoff`.
  `display_mode` accepts `formula_unit`, `unit_cell`, `asymmetric_unit`, or
  `cluster` (free molecular cluster — every parsed atom is drawn, no
  formula-unit trim, no periodic imaging of bonds).
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
  Triggers the vendored `crystal_viewer.legacy.plot_crystal` exporter with the current preset.

## Stable UI element IDs

- `structure-selector`: structure radio list
- `cif-upload`: upload zone
- `display-options`: labels / axes / minor-only / wireframe checklist
- `display-mode-selector`: `formula_unit`, `unit_cell`, `asymmetric_unit`, `cluster`
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
2. `POST /state` to select the uploaded structure or set render/display controls.
3. `POST /topology` with a chosen `center_index`.
4. `POST /camera/action` to zoom/orbit/pan if needed.
5. `GET /screenshot` to capture the current viewport.
6. `POST /preset/save` if the tuned state should be persisted.

## Local-only data

- No CIF files are bundled in the repository.
- Default presets are written under `.local/`.
- Optional local catalog files can be provided via `catalog.local.json` or `.local/catalog.local.json`.

## WebSocket messages

- Server to client:
  `{"version": <int>, "state": {...}, "structures": [...]}` whenever state or structure inventory changes.
- Client to server:
  `{"type": "set_state", "payload": {...}}`

## Programmatic scene API

Automation scripts that bypass the Dash UI and drive `build_figure` directly
have access to a small set of helpers for polishing publication figures:

- `crystal_viewer.scene.build_scene_from_cif(...)`
  Accepts `display_mode="cluster"` for free molecular clusters. In that mode
  every parsed atom is drawn unchanged, no formula-unit selection or periodic
  image reassembly is performed, and bonds are found purely from the stored
  Cartesian coordinates. The 100 Å dummy cells that CIF exporters sometimes
  write around clusters are ignored.
- `crystal_viewer.scene.apply_element_colors(scene, element_colors, element_colors_light)`
  Re-skin element palettes on a finished scene. Also invoked automatically by
  `build_scene_from_atoms` when `style["element_colors"]` is provided.
- `crystal_viewer.renderer.uniform_viewport(scenes, *, padding=0.0)`
  Stamp a shared world-cube `viewport` on a list of scenes so that every
  subsequent `build_figure` call renders at an identical physical length
  scale. The cube is the radius-aware bounding cube of the largest input
  scene. This is the hook for N-up grid figures where each panel must depict
  the same length per pixel.

Style keys honoured by `build_figure` beyond the Dash-driven defaults:

- `show_title` — set to `False` to suppress the Plotly panel title when the
  caller composes panels externally (e.g. with Matplotlib subplot titles).
- `axes_labels` — list of three strings substituted for the default
  `["a", "b", "c"]` legend on the axis triad. Clusters typically set
  `["x", "y", "z"]`.
- `element_colors`, `element_colors_light` — per-element hex overrides
  layered on top of the vendored palette.
