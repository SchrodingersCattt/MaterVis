"""Render a CP2K/Gaussian cube orbital as Plotly isosurfaces.

Run from the repository root:

    python scripts/06_cp2k_cube_orbital.py --cube /path/to/orbital.cube

The HTML output is always written. PNG export is attempted when Kaleido is
available in the local environment.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from crystal_viewer.cube import build_orbital_figure, read_cube  # noqa: E402


OUTPUT_DIR = Path(__file__).resolve().parent / "_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cube", required=True, help="Input .cube file")
    parser.add_argument("--output-prefix", default=None, help="Output file stem under scripts/_outputs")
    parser.add_argument("--stride", type=int, default=2, help="Grid stride for interactive rendering")
    parser.add_argument("--percentile", type=float, default=98.5, help="Abs(value) percentile for isovalue")
    parser.add_argument("--isovalue", type=float, default=None, help="Explicit isovalue; overrides percentile")
    parser.add_argument("--no-atoms", action="store_true", help="Hide atom overlay")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cube_path = Path(args.cube)
    stem = args.output_prefix or cube_path.stem

    cube = read_cube(cube_path)
    fig = build_orbital_figure(
        cube,
        isovalue=args.isovalue,
        percentile=args.percentile,
        stride=args.stride,
        show_atoms=not args.no_atoms,
        title=cube_path.name,
    )

    html = OUTPUT_DIR / f"{stem}.html"
    png = OUTPUT_DIR / f"{stem}.png"
    fig.write_html(str(html), include_plotlyjs="cdn", full_html=True)
    print(f"Wrote HTML: {html} ({os.path.getsize(html)} bytes)")

    try:
        fig.write_image(str(png), width=900, height=720, scale=2)
    except Exception as exc:  # pragma: no cover - depends on local Kaleido/Chrome
        print(f"PNG export skipped: {exc}")
    else:
        print(f"Wrote PNG : {png} ({os.path.getsize(png)} bytes)")


if __name__ == "__main__":
    main()
