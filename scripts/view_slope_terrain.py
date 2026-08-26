"""Inspect the slope-mode ramp (roller_slope) in the MuJoCo viewer.

Builds ONLY the "flat + ramp" terrain (FlatRampTerrainCfg), across several rows
of increasing difficulty (steepness 2° -> 20°), and opens the native MuJoCo
viewer. No trained policy is needed — this exists to eyeball the geometry
(flat/ramp joint, direction of descent).

Usage:
    uv run python scripts/view_slope_terrain.py
    uv run python scripts/view_slope_terrain.py --rows 6 --ramp-max 8 --runout 4
    uv run python scripts/view_slope_terrain.py --build-only   # test without GUI

In the viewer: scroll wheel to zoom, left-click drag to orbit, right-click drag
to pan. Each row is a progressively steeper ramp (difficulty 0 -> 1), of a length
drawn at random from [ramp-min, ramp-max], ending in a flat runout. "Forward"
(+x) must go downhill.
"""

import argparse

import mujoco
import mujoco.viewer

from mjlab.terrains.terrain_generator import TerrainGenerator, TerrainGeneratorCfg
from mjlab_microduck.tasks.slope_terrain import (
    FlatRampTerrainCfg,
    RAMP_DEG_MIN,
    RAMP_DEG_MAX,
)


def build_model(rows, size, flat_length, ramp_range, runout, deg_min, deg_max):
    """Build the MuJoCo model of the terrain alone (rows ramps of increasing steepness)."""
    cfg = TerrainGeneratorCfg(
        seed=0,
        size=size,
        num_rows=rows,
        num_cols=1,
        curriculum=True,  # difficulty increases along the rows
        difficulty_range=(0.0, 1.0),
        add_lights=True,
        sub_terrains={
            "flat_ramp": FlatRampTerrainCfg(
                flat_length=flat_length,
                ramp_length_range=ramp_range,
                runout_length=runout,
                deg_min=deg_min,
                deg_max=deg_max,
            )
        },
    )
    generator = TerrainGenerator(cfg)
    spec = mujoco.MjSpec()
    generator.compile(spec)
    model = spec.compile()
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows", type=int, default=5, help="Number of rows = number of steepness levels shown (default 5)")
    p.add_argument("--size", type=float, nargs=2, default=(15.0, 4.0), help="Tile size (x y) in m")
    p.add_argument("--flat-length", type=float, default=2.0, help="Length of the starting flat (m)")
    p.add_argument("--ramp-min", type=float, default=3.0, help="Minimum horizontal ramp length (m)")
    p.add_argument("--ramp-max", type=float, default=8.0, help="Maximum horizontal ramp length (m)")
    p.add_argument("--runout", type=float, default=4.0, help="Length of the flat runout (m)")
    p.add_argument("--deg-min", type=float, default=RAMP_DEG_MIN, help=f"Minimum steepness in degrees (default {RAMP_DEG_MIN})")
    p.add_argument("--deg-max", type=float, default=RAMP_DEG_MAX, help=f"Maximum steepness in degrees (default {RAMP_DEG_MAX})")
    p.add_argument("--build-only", action="store_true", help="Build the model and exit (test without GUI)")
    args = p.parse_args()

    model = build_model(
        rows=args.rows,
        size=tuple(args.size),
        flat_length=args.flat_length,
        ramp_range=(args.ramp_min, args.ramp_max),
        runout=args.runout,
        deg_min=args.deg_min,
        deg_max=args.deg_max,
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print(
        f"Terrain built: {args.rows} ramps, steepness {args.deg_min}°->{args.deg_max}°, "
        f"ramp length {args.ramp_min}-{args.ramp_max}m + runout {args.runout}m, "
        f"{model.ngeom} geoms."
    )
    if args.build_only:
        print("--build-only: OK, no GUI.")
        return

    print("Opening the MuJoCo viewer (Ctrl+C to quit)…")
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        while viewer.is_running():
            mujoco.mj_forward(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
