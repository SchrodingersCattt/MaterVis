# Crystal Viewer

`crystal_viewer/` is a standalone Dash/Plotly frontend for ABX4 crystal structures. It vendors the legacy scene-building code inside the package, adds CIF upload, coordination-topology analysis, zoom-correct `Mesh3d` rendering, and exposes an automation API for other agents.

## Features

- Browser-based 3D viewer for local catalog structures and uploaded CIFs
- Drag-and-drop CIF upload
- Display scope switcher for formula unit, unit cell, and asymmetric unit
- Hydrogen and unit-cell-box display toggles
- `Mesh3d` atom/bond rendering with a fast `Scatter3d` fallback
- Stronger highlight layer and minor-disorder wireframe overlay
- Coordination shell extraction, convex-hull overlay, angular RMSD against ideal polyhedra, planarity, and prism/antiprism heuristics
- REST and WebSocket automation on the same Flask server, including camera control and screenshots

## Install

```bash
python -m pip install -r requirements.txt
```

`molcrys_kit` is optional. If it is available, fragment A/B/X classification uses it. Otherwise the app falls back to simpler element/site heuristics.

## Run

```bash
python -m crystal_viewer
python -m crystal_viewer --port 8051
python -m crystal_viewer --structure SY PEP
python -m crystal_viewer --cif my_structure.cif
```

The app serves a local browser UI and an API under `/api/v1/`.

No CIF files are bundled in this repository. Catalog entries are discovered only if their local paths exist, or from an untracked `catalog.local.json`.

Useful API calls:

```bash
curl http://127.0.0.1:8051/api/v1/state
curl http://127.0.0.1:8051/api/v1/camera
curl -X POST http://127.0.0.1:8051/api/v1/camera/action -H "Content-Type: application/json" -d "{\"action\":\"zoom\",\"factor\":1.15}"
curl http://127.0.0.1:8051/api/v1/screenshot --output view.png
```

## Package layout

- `app.py`: Dash layout, callbacks, backend state
- `api.py`: REST + WebSocket registration
- `loader.py`: CIF parsing and fragment bundle loading
- `scene.py`: wrapper around the vendored legacy scene helpers
- `renderer.py`: Plotly trace generation
- `topology.py`: coordination-shell and shape analysis
- `ideal_polyhedra.py`: reference CN=8..12 polyhedra
- `presets.py`: preset I/O and default style/catalog
- `legacy/`: vendored static-export and scene-building modules

## Notes

- Screenshot export uses Plotly image export via `kaleido`.
- Static publication export runs the vendored `crystal_viewer.legacy.plot_crystal` module and writes outputs under `.exports/`.
- Local presets default to `.local/crystal_view_preset.json`.
- Local catalog overrides can be supplied via `catalog.local.json` or `.local/catalog.local.json`.
- For automation details, see `AGENTS.md`.
