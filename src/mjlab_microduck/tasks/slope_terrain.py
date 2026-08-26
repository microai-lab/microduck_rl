"""Custom "flat + downhill ramp" terrain for the roller_slope task.

The robot spawns on a flat area, gets an impulse toward +x, rolls to the ramp
and lets itself glide. The ramp angle is interpolated by the difficulty
(curriculum) over [RAMP_DEG_MIN, RAMP_DEG_MAX] degrees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import (
    SubTerrainCfg,
    TerrainGeometry,
    TerrainOutput,
)

RAMP_DEG_MIN = 2.0
RAMP_DEG_MAX = 20.0


def ramp_angle_by_difficulty(
    difficulty: float, deg_min: float = RAMP_DEG_MIN, deg_max: float = RAMP_DEG_MAX
) -> float:
    """Ramp angle (radians), linearly interpolated by the difficulty [0,1]."""
    d = float(np.clip(difficulty, 0.0, 1.0))
    return math.radians(deg_min + d * (deg_max - deg_min))


@dataclass(kw_only=True)
class FlatRampTerrainCfg(SubTerrainCfg):
    """Starting flat → downhill ramp → flat runout.

    Three boxes aligned along +x:
      1. the starting flat (surface at z=0) where the robot spawns;
      2. the downhill ramp, angle interpolated by the difficulty, HORIZONTAL
         length drawn at random from ``ramp_length_range`` (one value per tile,
         fixed at generation time);
      3. the flat runout at the level of the bottom of the ramp, so the robot
         lands on solid ground instead of the void.
    """

    flat_length: float = 2.0                       # starting flat (m)
    ramp_length_range: tuple = (3.0, 8.0)          # horizontal ramp length (m), drawn at random
    runout_length: float = 4.0                     # flat runout at the bottom (m)
    spawn_on_ramp: float = 0.3                      # spawn this many m ONTO the ramp (gravity => rolling)
    deg_min: float = RAMP_DEG_MIN
    deg_max: float = RAMP_DEG_MAX
    thickness: float = 0.5                          # box thickness (m)

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng
    ) -> TerrainOutput:
        total_max = self.flat_length + self.ramp_length_range[1] + self.runout_length
        assert total_max <= self.size[0], (
            f"flat+ramp_max+runout ({total_max}) must fit in size[0] ({self.size[0]})"
        )
        body = spec.body("terrain")
        angle = ramp_angle_by_difficulty(difficulty, self.deg_min, self.deg_max)
        width = self.size[1]
        t = self.thickness
        # Ramp length drawn at random (deterministic for a given rng).
        ramp_length = float(rng.uniform(self.ramp_length_range[0], self.ramp_length_range[1]))
        drop = ramp_length * math.tan(angle)  # elevation drop (m), positive

        # 1) Starting flat: surface at z=0, x in [0, flat_length].
        flat = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(self.flat_length / 2.0, width / 2.0, t / 2.0),
            pos=(self.flat_length / 2.0, 0.0, -t / 2.0),
        )

        # 2) Ramp: box rotated by +angle around +y (the +x edge goes down).
        # The -(t/2)·sin(angle) offset in x: without it, the TOP edge of the
        # tilted surface lands at x=flat_length+(t/2)sin(a) -> a small gap between
        # the flat platform (which ends at flat_length) and the ramp. With it, the
        # top of the ramp meets the platform edge exactly (clean joint), and the
        # bottom meets the runout exactly.
        surf_len = ramp_length / math.cos(angle)
        ramp_cx = self.flat_length + ramp_length / 2.0 - (t / 2.0) * math.sin(angle)
        ramp_cz = -(drop / 2.0) - (t / 2.0) * math.cos(angle)
        half = angle / 2.0
        ramp = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(surf_len / 2.0, width / 2.0, t / 2.0),
            pos=(ramp_cx, 0.0, ramp_cz),
            quat=(math.cos(half), 0.0, math.sin(half), 0.0),
        )

        # 3) Flat runout: surface at the level of the bottom of the ramp (z = -drop).
        runout_cx = self.flat_length + ramp_length + self.runout_length / 2.0
        runout = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(self.runout_length / 2.0, width / 2.0, t / 2.0),
            pos=(runout_cx, 0.0, -drop - t / 2.0),
        )

        # Spawn slightly ONTO the ramp: gravity spins the wheels right away
        # (momentum IN THE WHEELS, not a base push that would skid), and the robot
        # is already on the slope. z on the tilted surface at that distance.
        spawn_x = self.flat_length + self.spawn_on_ramp
        spawn_z = -self.spawn_on_ramp * math.tan(angle)
        origin = np.array([spawn_x, 0.0, spawn_z])
        return TerrainOutput(
            origin=origin,
            geometries=[
                TerrainGeometry(geom=flat, color=(0.5, 0.5, 0.5, 1.0)),
                TerrainGeometry(geom=ramp, color=(0.45, 0.55, 0.75, 1.0)),
                TerrainGeometry(geom=runout, color=(0.5, 0.5, 0.5, 1.0)),
            ],
        )
