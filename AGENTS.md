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

## Cube / orbital rendering API (`crystal_viewer.cube`)

Static cube isosurface figures (HOMO/LUMO, spin density, charge density…)
are produced by helpers in `crystal_viewer.cube`. The library is
journal-agnostic; project-specific styling (typography, dpi, column
widths) lives in caller code, not here.

Hard contracts the library guarantees so callers do not have to
re-derive them:

- **Trace insertion order.** `build_orbital_panel_figure` defaults to
  `DEFAULT_TRACE_ORDER = ("cell", "orbital", "bonds", "atoms")` so
  half-transparent isosurfaces are always composited UNDER opaque atoms
  and bonds. This keeps the molecular skeleton legible regardless of
  orbital density and ensures panels with sparse vs dense orbitals look
  visually consistent. Override only when deliberately wanting the
  inverse stacking; pass `trace_order=(...)` with any subset of
  `{"cell", "orbital", "bonds", "atoms"}`.
- **Tiled-cube cleanliness.** When `tile_cube` has been used to extend
  the volumetric data over PBC images, callers MUST pass both
  `min_volume_voxels > 0` (drops tiny disconnected lobes from connected-
  component analysis) and `atom_mask_radius > 0` (zeroes voxels farther
  than R from any atom) to `orbital_mesh_traces` /
  `build_orbital_panel_figure`. Without either, marching-cubes will emit
  floating phantom lobes from PBC-image background noise.
- **Static export.** Use `export_static` (Kaleido) and prefer
  `use_mesh=True` (marching-cubes Mesh3d). `go.Isosurface` is an
  interactive-only fallback because Kaleido currently rasterises it
  inconsistently across versions.
- **Atom + bond geometry.** `atom_sphere_traces` and `bond_traces` emit
  opaque `Mesh3d` primitives with bright lighting defaults
  (`ambient ≥ 0.75`) so phenyl-heavy or dark-element-heavy structures
  remain legible in print. Element colors come from `ELEMENT_COLORS`;
  override per-call via positional or keyword arguments rather than
  mutating the module dict.
- **Sign legend.** `sign_legend_annotations` emits paper-coord swatches
  using unicode `\u25A0` / `\u2212`. HTML entities (`&#9632;`,
  `&minus;`) corrupt SVG export and must not be reintroduced.

## Camera-projected paper-coord indicators (`crystal_viewer.compass`)

Direction indicators (a/b/c lattice triads, k-paths, dipole/force
vectors…) should be rendered as paper-coord annotations rather than
in-scene 3D arrows whenever the 3D content does not have guaranteed
empty space. The module is layered so each tier is reusable:

1. `camera_screen_basis(camera)` — pure camera math; returns
   `(right, up)` unit vectors in data space.
2. `project_to_screen(camera, vectors)` — `(N, 3) → (N, 2)` projection
   onto the camera image plane.
3. `paper_arrow_annotations(anchor_xy, deltas_2d, *, fig_size, ...)` —
   2D arrow + label rendering at a paper anchor; agnostic of where the
   2D directions came from.
4. `lattice_compass_annotations(camera, lattice, *, panel_x_domains,
   fig_size, ...)` — convenience wrapper around 1+2+3 for the common
   crystal-axes case. Defaults to Wong (2011) colorblind-safe colours
   and `("a", "b", "c")` labels; every styling parameter (colours,
   labels, font, anchor, pixel length, label offset, arrow width) is
   overridable.

For non-crystal use cases (e.g. force vectors on a molecule), compose
layers 1–3 directly rather than abusing layer 4.
