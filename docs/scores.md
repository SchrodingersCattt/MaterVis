# Topology scores reference

`crystal_viewer.topology.analyze_topology` returns one JSON-serialisable dict
that bundles **five** distinct scores characterising a coordination shell.
This page explains what each score means, how it is computed, and how to
interpret the value.

All scores are derived from the **shell** (the `CN` nearest neighbouring
fragments of the chosen centre). The shell itself is extracted by
`extract_coordination_shell`, which walks every periodic image of every
fragment inside `cutoff` Å and sorts them by distance.

```python
from crystal_viewer.loader import build_loaded_crystal
from crystal_viewer.topology import analyze_topology

bundle = build_loaded_crystal(name="DAP-4", cif_path="scripts/data/DAP-4.cif")
# The first A-site fragment in DAP-4 is at index 8; tune cutoff to your lattice.
result = analyze_topology(bundle, center_index=8, cutoff=8.0)
print(result.keys())
```

The dict has the following top-level keys (schema v1):

| Key | Type | Provenance |
| --- | --- | --- |
| `center_index` / `center_label` / `center_type` | int / str / str | echoes the input |
| `center_coords` / `source_center_coords` | `list[float]` (3) | centre in plot and source space |
| `cutoff` | float | echoes the input, Å |
| `neighbor_pool_size` | int | candidates within `cutoff` before the gap cut |
| `coordination_number` | int | see **1. Coordination number** below |
| `gap_info` | dict | see **1. Coordination number** |
| `shell` / `shell_coords` / `distances` | list / list / list[float] | the selected `CN` neighbours |
| `all_distances` / `pool_coords` | list[float] / list | every candidate in the pool |
| `angular` | dict | see **2. Angular RMSD** |
| `planarity` | dict | see **3. Planarity RMS** |
| `prism_analysis` | dict | see **4. Prism vs antiprism twist** |
| `hull` | dict | see **5. Convex-hull overlay** |

---

## 1. Coordination number (`coordination_number`, `gap_info`)

**What it is:** the number of neighbouring fragments deemed "inside the first
shell".

**How it is computed** (`detect_coordination_number`):

1. Sort every candidate distance ascending → `sorted_distances`.
2. Compute successive differences → `gaps`.
3. `CN = argmax(gaps) + 1` (one past the biggest jump).

So `CN` is the cluster size that maximises the distance gap between "in the
shell" and "out of the shell" — no manual cutoff required.

**Output sub-dict `gap_info`:**

| Field | Type | Meaning |
| --- | --- | --- |
| `coordination_number` | int | the chosen CN |
| `gap_index` | int | position of the chosen gap in `gaps` |
| `gap_value` | float (Å) | size of the jump the chosen gap represents |
| `sorted_distances` | `list[float]` (Å) | every candidate in the pool, ascending |
| `gaps` | `list[float]` (Å) | first-differences of `sorted_distances` |

**Interpretation tips:**

- A large `gap_value` (> 0.3 Å) means a clean, textbook coordination shell.
- A small `gap_value` (< 0.1 Å) means the shell is fuzzy — e.g. the A-site in a
  distorted perovskite where several next-nearest anions sit just outside.
- If `len(pool_coords) < CN + 2` the algorithm has too few candidates; widen
  `cutoff` or the result is unreliable.

---

## 2. Angular RMSD vs ideal polyhedra (`angular.*`)

**What it is:** a dimensionless comparison of the shell's angular fingerprint
against a library of ideal polyhedra of matching CN.

**How it is computed** (`angular_rmsd_vs_ideals`):

1. Build the full list of pairwise centre-to-neighbour bond angles for the
   actual shell (`C(CN, 2)` angles).
2. Sort that list ascending → the **angular signature**.
3. For every ideal polyhedron of the same CN, sort its signature the same way.
4. RMSD across the sorted signatures:

  ```
  angular_rmsd = sqrt(mean((actual - ideal) ** 2))   # in degrees
  ```

5. Report results sorted ascending and single out `best_match`.

**Library** (`crystal_viewer.ideal_polyhedra`):

| CN | Ideal shapes |
| --- | --- |
| 8 | cube · square antiprism · dodecahedron |
| 9 | capped square antiprism · tricapped trigonal prism |
| 10 | bicapped square antiprism · bicapped dodecahedron |
| 11 | capped pentagonal antiprism · capped pentagonal prism · edge-bicapped square antiprism |
| 12 | icosahedron · cuboctahedron |

If the CN is outside 8..12, `results == []` and `best_match is None`.

**Output sub-dict `angular`:**

| Field | Type | Meaning |
| --- | --- | --- |
| `coordination_number` | int | echoes the CN actually compared |
| `results` | `list[{name, angular_rmsd}]` | every ideal, sorted ascending |
| `best_match` | dict \| None | `results[0]` or `None` |

**Interpretation tips:**

- `angular_rmsd` < 5° — essentially the ideal polyhedron (allow for thermal
  motion and numerical precision).
- 5–15° — clearly distorted but still in the same family.
- > 20° — the shape has drifted far from any textbook polyhedron; pick up the
  `prism_analysis.twist_deg` and `planarity.best_rms` below to triangulate.

---

## 3. Planarity RMS (`planarity.*`)

**What it is:** the lowest out-of-plane RMS displacement you can achieve by
picking any 5-neighbour subset of the shell, in Å.

**How it is computed** (`planarity_analysis`):

1. For every combination of `group_size` (default **5**) shell atoms:
   - Centre the subset at its own centroid.
   - Fit the best plane via SVD (the last right-singular vector is the normal).
   - Record the RMS distance from the subset to that plane.
2. Keep the smallest RMS and the atom indices that produced it.

**Output sub-dict `planarity`:**

| Field | Type | Meaning |
| --- | --- | --- |
| `best_rms` | float \| None (Å) | best RMS out-of-plane displacement |
| `best_indices` | `list[int]` | which shell positions form that plane |
| `group_size` | int | number of atoms per plane (5 by default) |

**Interpretation tips:**

- A five-atom plane with `best_rms` ≲ 0.05 Å is effectively flat — a signal
  that the coordination shell contains a pentagonal face (icosahedron,
  bicapped square antiprism, etc.).
- `best_indices` can be fed back into the renderer to colour-highlight that
  face.
- The routine returns `None` if the shell has fewer than `group_size` members.

---

## 4. Prism / antiprism twist (`prism_analysis.*`)

**What it is:** a quick check for whether a CN ≥ 10 polyhedron is better
described as a **prism** (stacked faces) or **antiprism** (rotated faces).

**How it is computed** (`detect_prism_vs_antiprism`):

1. Require `len(shell_coords) >= 10`; otherwise return `None` for both fields.
2. Sort shell atoms by `z`, take the bottom 5 and top 5 as two pentagonal
   rings.
3. For each paired atom compute the azimuthal rotation `Δφ` (mod ±180°).
4. Take the **average absolute twist** — this is `twist_deg`.
5. `classification = "antiprism" if twist_deg > 18° else "prism"`.

**Output sub-dict `prism_analysis`:**

| Field | Type | Meaning |
| --- | --- | --- |
| `classification` | `"prism"` \| `"antiprism"` \| None | verdict based on the 18° threshold |
| `twist_deg` | float \| None | average inter-ring twist in degrees |

**Interpretation tips:**

- The 18° threshold sits halfway between an ideal pentagonal prism (0°) and
  an ideal pentagonal antiprism (36°).
- Use this alongside `angular.best_match` — a "bicapped square antiprism"
  reported as a prism (twist < 18°) is probably a mis-assignment.
- For CN < 10 the twist is undefined and `classification` is `None`.

---

## 5. Convex-hull overlay (`hull.*`)

Not strictly a *score* but a very useful geometric summary for visualisation:

| Field | Type | Meaning |
| --- | --- | --- |
| `vertices` | `list[list[float]]` | Cartesian coordinates of every shell atom |
| `simplices` | `list[list[int]]` | triangle indices of the convex hull |
| `edges` | `list[list[int]]` | unique undirected edges of the hull |

The renderer draws `hull.simplices` as a translucent mesh and `hull.edges` as
a thick line trace to produce the purple polyhedra in the README screenshots.
If `scipy.spatial.ConvexHull` is unavailable the hull gracefully falls back
to `{vertices: [...], simplices: [], edges: []}` and only the vertices are
shown.

---

## Example output (DAP-4, first A-site cation)

`python scripts/02_coordination_analysis.py` writes
`scripts/_outputs/02_coordination_summary.json` with an abridged version of
the dict:

```json
{
  "structure": "DAP-4",
  "center_label": "A0",
  "center_type": "A",
  "cutoff_A": 8.0,
  "neighbor_pool_size": 12,
  "coordination_number": 9,
  "gap_value_A": 0.124,
  "shell_distances_A": [4.98, 4.98, 4.98, 5.10, 5.10, 5.10, 5.10, 5.10, 5.10],
  "angular": {
    "best_match": {"name": "tricapped_trigonal_prism", "angular_rmsd_deg": 15.06},
    "ranked": [
      {"name": "tricapped_trigonal_prism", "angular_rmsd_deg": 15.06},
      {"name": "capped_square_antiprism",  "angular_rmsd_deg": 16.71}
    ]
  },
  "planarity":      {"best_rms_A": 0.093, "best_indices": [1, 4, 5, 6, 7], "group_size": 5},
  "prism_analysis": {"classification": null, "twist_deg": null}
}
```

**What this says about DAP-4:**

- The diaminopropane A-cation sits in a **9-coordinate pocket** of perchlorate
  anions.
- Three anions form a tight inner shell at 4.98 Å, six more at 5.10 Å, and
  the pool drops off ~0.12 Å later (`gap_value`) — a clean shell.
- Of the two CN = 9 ideals, the geometry is closest to a
  **tricapped trigonal prism** (15° RMSD — visibly distorted but recognisable).
- A sub-set of 5 neighbours forms an almost perfect plane
  (`planarity.best_rms_A ≈ 0.09 Å`) — the capping triangle of the prism.
- No prism/antiprism verdict is emitted because the twist-test requires CN ≥ 10.

Run example 02 to regenerate the full JSON with the raw gaps, hull edges,
planarity indices and every pool distance included.
