from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import plotly.graph_objects as go


BOHR_TO_ANGSTROM = 0.529177210903

ELEMENT_SYMBOLS = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    15: "P",
    16: "S",
    17: "Cl",
    29: "Cu",
}

ELEMENT_COLORS = {
    "H": "#FFFFFF",
    "C": "#404040",
    "N": "#3050F8",
    "O": "#FF0D0D",
    "P": "#FF8000",
    "S": "#FFFF30",
    "Cl": "#1FF01F",
    "Cu": "#C88033",
}


@dataclass
class CubeAtom:
    atomic_number: int
    charge: float
    coord: np.ndarray

    @property
    def element(self) -> str:
        return ELEMENT_SYMBOLS.get(self.atomic_number, str(self.atomic_number))


@dataclass
class CubeData:
    title: str
    comment: str
    atoms: list[CubeAtom]
    origin: np.ndarray
    axes: np.ndarray
    values: np.ndarray
    path: Path

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(x) for x in self.values.shape)


def _as_angstrom(vec: Iterable[float]) -> np.ndarray:
    return np.asarray(list(vec), dtype=float) * BOHR_TO_ANGSTROM


def read_cube(path: str | Path) -> CubeData:
    """Read a Gaussian/CP2K cube file and return coordinates in Angstrom."""
    cube_path = Path(path)
    with cube_path.open("r", encoding="utf-8", errors="replace") as handle:
        title = handle.readline().rstrip()
        comment = handle.readline().rstrip()

        natom_line = handle.readline().split()
        natoms = abs(int(natom_line[0]))
        origin = _as_angstrom(float(x) for x in natom_line[1:4])

        shape: list[int] = []
        axes: list[np.ndarray] = []
        for _ in range(3):
            parts = handle.readline().split()
            shape.append(abs(int(parts[0])))
            axes.append(_as_angstrom(float(x) for x in parts[1:4]))

        atoms: list[CubeAtom] = []
        for _ in range(natoms):
            parts = handle.readline().split()
            atoms.append(
                CubeAtom(
                    atomic_number=int(parts[0]),
                    charge=float(parts[1]),
                    coord=_as_angstrom(float(x) for x in parts[2:5]),
                )
            )

        raw_values = np.fromiter((float(x) for line in handle for x in line.split()), dtype=float)

    expected = int(np.prod(shape))
    if raw_values.size != expected:
        raise ValueError(f"{cube_path} contains {raw_values.size} values, expected {expected}")

    return CubeData(
        title=title,
        comment=comment,
        atoms=atoms,
        origin=origin,
        axes=np.asarray(axes, dtype=float),
        values=raw_values.reshape(shape),
        path=cube_path,
    )


def cube_grid(cube: CubeData, stride: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return flattened x/y/z/value arrays, optionally downsampled."""
    stride = max(1, int(stride))
    values = cube.values[::stride, ::stride, ::stride]
    ii, jj, kk = np.indices(values.shape, dtype=float)
    coords = (
        cube.origin[:, None, None, None]
        + ii[None, ...] * cube.axes[0, :, None, None, None] * stride
        + jj[None, ...] * cube.axes[1, :, None, None, None] * stride
        + kk[None, ...] * cube.axes[2, :, None, None, None] * stride
    )
    return coords[0].ravel(), coords[1].ravel(), coords[2].ravel(), values.ravel()


def default_isovalue(values: np.ndarray, percentile: float = 98.5) -> float:
    """Pick a robust orbital isovalue from the absolute-value distribution."""
    nonzero = np.abs(values[np.nonzero(values)])
    if nonzero.size == 0:
        raise ValueError("Cube values are all zero")
    return float(np.percentile(nonzero, percentile))


def orbital_isosurface_traces(
    cube: CubeData,
    *,
    isovalue: float | None = None,
    percentile: float = 98.5,
    stride: int = 2,
    positive_color: str = "#D55E00",
    negative_color: str = "#0072B2",
    opacity: float = 0.55,
) -> list[go.Isosurface]:
    """Create positive and negative orbital isosurface traces."""
    x, y, z, values = cube_grid(cube, stride=stride)
    iso = float(isovalue) if isovalue is not None else default_isovalue(values, percentile=percentile)
    vmax = float(np.max(values))
    vmin = float(np.min(values))

    traces: list[go.Isosurface] = []
    if vmax >= iso:
        traces.append(
            go.Isosurface(
                x=x,
                y=y,
                z=z,
                value=values,
                isomin=iso,
                isomax=vmax,
                surface_count=1,
                opacity=opacity,
                colorscale=[[0.0, positive_color], [1.0, positive_color]],
                caps=dict(x_show=False, y_show=False, z_show=False),
                showscale=False,
                name="+ orbital",
            )
        )
    if vmin <= -iso:
        traces.append(
            go.Isosurface(
                x=x,
                y=y,
                z=z,
                value=values,
                isomin=vmin,
                isomax=-iso,
                surface_count=1,
                opacity=opacity,
                colorscale=[[0.0, negative_color], [1.0, negative_color]],
                caps=dict(x_show=False, y_show=False, z_show=False),
                showscale=False,
                name="- orbital",
            )
        )
    return traces


def cube_atom_trace(cube: CubeData, *, atom_scale: float = 5.0) -> go.Scatter3d:
    """Create a light atom overlay from the cube atom records."""
    labels = [f"{atom.element}{idx + 1}" for idx, atom in enumerate(cube.atoms)]
    colors = [ELEMENT_COLORS.get(atom.element, "#999999") for atom in cube.atoms]
    coords = np.asarray([atom.coord for atom in cube.atoms], dtype=float)
    return go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode="markers",
        text=labels,
        hovertemplate="%{text}<extra></extra>",
        marker=dict(size=atom_scale, color=colors, opacity=0.9, line=dict(color="#333333", width=0.5)),
        showlegend=False,
        name="atoms",
    )


def build_orbital_figure(
    cube: CubeData,
    *,
    isovalue: float | None = None,
    percentile: float = 98.5,
    stride: int = 2,
    show_atoms: bool = True,
    title: str | None = None,
) -> go.Figure:
    """Build a standalone Plotly figure for a cube orbital."""
    fig = go.Figure()
    for trace in orbital_isosurface_traces(cube, isovalue=isovalue, percentile=percentile, stride=stride):
        fig.add_trace(trace)
    if show_atoms and cube.atoms:
        fig.add_trace(cube_atom_trace(cube))

    coords = np.asarray([atom.coord for atom in cube.atoms], dtype=float) if cube.atoms else np.zeros((0, 3))
    if coords.size:
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
    else:
        x, y, z, _ = cube_grid(cube, stride=max(stride, 4))
        mins = np.array([x.min(), y.min(), z.min()])
        maxs = np.array([x.max(), y.max(), z.max()])
    center = 0.5 * (mins + maxs)
    half = 0.5 * max(maxs - mins) + 1.5
    ranges = [[float(c - half), float(c + half)] for c in center]

    fig.update_layout(
        title=dict(text=title or cube.path.name, x=0.5),
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="white",
        scene=dict(
            xaxis=dict(visible=False, range=ranges[0]),
            yaxis=dict(visible=False, range=ranges[1]),
            zaxis=dict(visible=False, range=ranges[2]),
            aspectmode="cube",
            bgcolor="white",
        ),
    )
    return fig
