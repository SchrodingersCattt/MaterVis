# MatterVis performance notes

This file records developer benchmark results for the Phase 1 performance
cleanup. Benchmarks are run with:

```bash
python -m crystal_viewer.perf.bench --repeat 3
python -m crystal_viewer.perf.profile_app
```

## Baseline

Captured on 2026-05-01 with:

```bash
python -m crystal_viewer.perf.bench --repeat 3 --json
python -m crystal_viewer.perf.profile_app --output /tmp/mattervis-profile-baseline.txt
```

Test structure: `scripts/data/DAP-4.cif` (`atom_count_unit_cell=192`,
`fragment_count=40`).

| Benchmark | Mean (s) | Median (s) | Notes |
| --- | ---: | ---: | --- |
| `neighbor_pool` | 0.0057 | 0.0057 | 30 candidates |
| `topology_full` | 0.0082 | 0.0074 | CN 6, pool 30 |
| `atom_mesh_unit_cell` | 0.7065 | 0.6872 | 192 atoms, 168 bonds, 8 traces |
| `planarity cn_8` | 0.0054 | 0.0056 | exhaustive combinations |
| `planarity cn_10` | 0.0214 | 0.0219 | exhaustive combinations |
| `planarity cn_12` | 0.0685 | 0.0690 | exhaustive combinations |
| `planarity cn_14` | 0.2079 | 0.2131 | exhaustive combinations |

Profile scenario: 5 representative `ViewerBackend.figure_for_state` calls
took 11.559 s total. Top cumulative hot spots:

| Function | Cumulative (s) | Note |
| --- | ---: | --- |
| `renderer.build_figure` | 9.317 | main figure assembly |
| `renderer._cached_atom_bond_meshes` | 4.816 | atom/bond payload construction |
| `copy.deepcopy` | 4.068 | Plotly validation / object copying |
| `renderer._atom_mesh_traces` | 2.350 | atom sphere tessellation |
| `app.scene_for_state` / `loader.build_bundle_scene` | 1.808 | scene build/cache path |
| `crystal_scene._label_payload` | 1.186 | label collision placement |

## After Phase 1

Captured after the topology, renderer, Dash callback, scene-cache, and cleanup
changes in the same environment.

| Benchmark | Mean (s) | Median (s) | Baseline mean (s) | Change |
| --- | ---: | ---: | ---: | ---: |
| `neighbor_pool` | 0.0010 | 0.0009 | 0.0057 | 5.7x faster |
| `topology_full` | 0.0023 | 0.0020 | 0.0082 | 3.5x faster |
| `atom_mesh_unit_cell` | 0.0654 | 0.0177 | 0.7065 | 10.8x faster |
| `planarity cn_8` | 0.0008 | 0.0007 | 0.0054 | 6.5x faster |
| `planarity cn_10` | 0.0026 | 0.0025 | 0.0214 | 8.4x faster |
| `planarity cn_12` | 0.0075 | 0.0073 | 0.0685 | 9.1x faster |
| `planarity cn_14` | 0.0181 | 0.0179 | 0.2079 | 11.5x faster |

Profile scenario: 5 representative `ViewerBackend.figure_for_state` calls took
4.238 s total, down from 11.559 s (2.7x faster). Top remaining cumulative hot
spots:

| Function | Cumulative (s) | Note |
| --- | ---: | --- |
| `renderer.build_figure` | 2.161 | mostly Plotly object construction/layout |
| `app.scene_for_state` / `loader.build_bundle_scene` | 1.858 | scene build/cache path |
| `crystal_scene._label_payload` | 1.228 | label collision placement |
| `renderer.topology_foreground_traces` | 0.806 | primary/extra overlay markers |
| `loader._fragment_table_from_atoms` | 0.484 | only cold display scopes |
