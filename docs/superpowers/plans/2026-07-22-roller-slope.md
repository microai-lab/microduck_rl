# Slope mode `roller_slope` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a dedicated policy where the microduck (on rollers) starts on the flat with an impulse, rolls onto a downhill ramp, and lets itself glide to the bottom while staying upright — with no steering at all.

**Architecture:** A new isolated task cloned from `velocity_rollers` (same robot, same 61D obs → hot-swappable at runtime). A custom "flat + ramp" terrain whose angle is interpolated by difficulty, a home-grown steepness curriculum, a neutralized command, and balance + nominal standing posture rewards. A `Y` toggle key in `infer_policy.py`.

**Tech Stack:** Python, mjlab 1.3.x, MuJoCo (MjSpec terrains), rsl_rl (PPO), PyTorch, onnxruntime (deployment), pytest.

## Global Constraints

- **Unified 61D observation**: twist (3D) + head_command (4D) + body_command (6D) zero-padded. Never change this layout — the policy must load via `--new-cmd-obs`.
- **Joints resolved BY NAME**, never by index (the passive wheels are interleaved).
- **Entry velocity through `reset_root_state_uniform` (velocity_range)**, NEVER through `push_by_setting_velocity` in reset mode (it accumulates on the root state → the free joint diverges → NaN). A `roller_crouch` lesson.
- **Angles in radians** in the physics code; the steepness constants are expressed in degrees (`RAMP_DEG_MIN=2.0`, `RAMP_DEG_MAX=20.0`) and converted.
- **Simple commits**, in the repo's style (no `Co-authored-by`).
- Tests in `tests/`, run with `uv run pytest`.

---

## File Structure

- **Create** `src/mjlab_microduck/tasks/slope_terrain.py` — `ramp_angle_by_difficulty()` + `FlatRampTerrainCfg` (the flat+ramp terrain geometry). Single responsibility: the terrain.
- **Modify** `src/mjlab_microduck/tasks/mdp.py` — add `slope_move_masks()` (pure) + `terrain_levels_slope()` (the steepness curriculum).
- **Create** `src/mjlab_microduck/tasks/microduck_roller_slope_env_cfg.py` — `make_microduck_roller_slope_env_cfg()` + `MicroduckRollerSlopeRlCfg`.
- **Modify** `src/mjlab_microduck/tasks/__init__.py` — register the task.
- **Modify** `scripts/infer_policy.py` — the `--slope` flag + the `Y` key.
- **Create** `tests/test_slope_terrain.py`, `tests/test_slope_curriculum.py`, `tests/test_roller_slope_cfg.py`.

---

## Task 1: ramp angle by difficulty (pure function)

**Files:**
- Create: `src/mjlab_microduck/tasks/slope_terrain.py`
- Test: `tests/test_slope_terrain.py`

**Interfaces:**
- Produces: `ramp_angle_by_difficulty(difficulty: float, deg_min: float = 2.0, deg_max: float = 20.0) -> float` (returns **radians**). Module constants `RAMP_DEG_MIN = 2.0`, `RAMP_DEG_MAX = 20.0`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_slope_terrain.py
import math
from mjlab_microduck.tasks.slope_terrain import (
    ramp_angle_by_difficulty,
    RAMP_DEG_MIN,
    RAMP_DEG_MAX,
)


def test_ramp_angle_endpoints():
    assert math.isclose(ramp_angle_by_difficulty(0.0), math.radians(RAMP_DEG_MIN), abs_tol=1e-9)
    assert math.isclose(ramp_angle_by_difficulty(1.0), math.radians(RAMP_DEG_MAX), abs_tol=1e-9)


def test_ramp_angle_midpoint():
    mid_deg = (RAMP_DEG_MIN + RAMP_DEG_MAX) / 2.0
    assert math.isclose(ramp_angle_by_difficulty(0.5), math.radians(mid_deg), abs_tol=1e-9)


def test_ramp_angle_clamps_out_of_range():
    assert math.isclose(ramp_angle_by_difficulty(-1.0), math.radians(RAMP_DEG_MIN), abs_tol=1e-9)
    assert math.isclose(ramp_angle_by_difficulty(2.0), math.radians(RAMP_DEG_MAX), abs_tol=1e-9)
```

- [ ] **Step 2: Run the test — it must fail**

Run: `uv run pytest tests/test_slope_terrain.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mjlab_microduck.tasks.slope_terrain'`

- [ ] **Step 3: Minimal implementation**

```python
# src/mjlab_microduck/tasks/slope_terrain.py
"""Custom "flat + downhill ramp" terrain for the roller_slope task.

The robot spawns on a flat area, gets an impulse toward +x, rolls to the ramp
and lets itself glide. The ramp angle is interpolated by the difficulty
(curriculum) over [RAMP_DEG_MIN, RAMP_DEG_MAX] degrees.
"""

from __future__ import annotations

import math

import numpy as np

RAMP_DEG_MIN = 2.0
RAMP_DEG_MAX = 20.0


def ramp_angle_by_difficulty(
    difficulty: float, deg_min: float = RAMP_DEG_MIN, deg_max: float = RAMP_DEG_MAX
) -> float:
    """Ramp angle (radians), linearly interpolated by the difficulty [0,1]."""
    d = float(np.clip(difficulty, 0.0, 1.0))
    return math.radians(deg_min + d * (deg_max - deg_min))
```

- [ ] **Step 4: Run the test — it must pass**

Run: `uv run pytest tests/test_slope_terrain.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/mjlab_microduck/tasks/slope_terrain.py tests/test_slope_terrain.py
git commit -m "roller-slope: ramp angle by difficulty (pure function + tests)"
```

---

## Task 2: custom terrain `FlatRampTerrainCfg`

**Files:**
- Modify: `src/mjlab_microduck/tasks/slope_terrain.py`
- Test: `tests/test_slope_terrain.py`

**Interfaces:**
- Consumes: `ramp_angle_by_difficulty` (Task 1), and `SubTerrainCfg`, `TerrainGeometry`, `TerrainOutput` from `mjlab.terrains.terrain_generator`.
- Produces: `FlatRampTerrainCfg(SubTerrainCfg)` with fields `flat_length: float = 2.0`, `ramp_length: float = 5.0`, `deg_min: float = 2.0`, `deg_max: float = 20.0`, `thickness: float = 0.5`; and a `function(difficulty, spec, rng) -> TerrainOutput` method. The spawn origin is on the flat.

**Geometry notes (worth remembering):** the flat's surface is at local `z=0`. The ramp is a box rotated about `+y` by a quaternion `[cos(a/2), 0, sin(a/2), 0]` — a `+a` rotation about `+y` lowers the `+x` edge (the ramp descends as `x` increases). The exact flat/ramp joint (no step, no gap) **must be verified in the viewer** (Step 6) because the ramp center's `z` is sensitive.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_slope_terrain.py  (add)
import mujoco
import numpy as np
from mjlab_microduck.tasks.slope_terrain import FlatRampTerrainCfg


def _empty_terrain_spec():
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    return spec


def test_flat_ramp_builds_geoms_and_origin_on_flat():
    cfg = FlatRampTerrainCfg(flat_length=2.0, ramp_length=5.0)
    cfg.size = (8.0, 4.0)  # normally set by the generator
    spec = _empty_terrain_spec()
    out = cfg.function(difficulty=0.5, spec=spec, rng=np.random.default_rng(0))
    # two geoms: flat + ramp
    assert len(out.geometries) == 2
    # origin on the flat (x in [0, flat_length], z ~ 0)
    assert 0.0 <= out.origin[0] <= 2.0
    assert abs(out.origin[2]) < 1e-6


def test_flat_ramp_steeper_at_higher_difficulty():
    # at higher difficulty the end of the ramp goes lower
    cfg = FlatRampTerrainCfg()
    cfg.size = (8.0, 4.0)
    easy = cfg.function(0.0, _empty_terrain_spec(), np.random.default_rng(0))
    hard = cfg.function(1.0, _empty_terrain_spec(), np.random.default_rng(0))
    # the ramp (2nd geom) sits lower (more negative center z) at high difficulty
    assert hard.geometries[1].geom.pos[2] < easy.geometries[1].geom.pos[2]
```

- [ ] **Step 2: Run the test — it must fail**

Run: `uv run pytest tests/test_slope_terrain.py -k flat_ramp -v`
Expected: FAIL — `ImportError: cannot import name 'FlatRampTerrainCfg'`

- [ ] **Step 3: Minimal implementation**

```python
# src/mjlab_microduck/tasks/slope_terrain.py  (add at the top)
from dataclasses import dataclass

import mujoco

from mjlab.terrains.terrain_generator import (
    SubTerrainCfg,
    TerrainGeometry,
    TerrainOutput,
)


@dataclass(kw_only=True)
class FlatRampTerrainCfg(SubTerrainCfg):
    """A flat starting area followed by a downhill ramp (angle by difficulty)."""

    flat_length: float = 2.0   # length of the starting flat along +x (m)
    ramp_length: float = 5.0   # horizontal length of the ramp along +x (m)
    deg_min: float = RAMP_DEG_MIN
    deg_max: float = RAMP_DEG_MAX
    thickness: float = 0.5     # box thickness (m)

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng
    ) -> TerrainOutput:
        del rng  # unused
        body = spec.body("terrain")
        angle = ramp_angle_by_difficulty(difficulty, self.deg_min, self.deg_max)
        width = self.size[1]
        t = self.thickness

        # Flat: a box whose top surface is at z=0, x in [0, flat_length].
        flat = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(self.flat_length / 2.0, width / 2.0, t / 2.0),
            pos=(self.flat_length / 2.0, 0.0, -t / 2.0),
        )

        # Ramp: a box rotated by +angle about +y (the +x edge goes down).
        # Surface length = ramp_length / cos(angle).
        surf_len = self.ramp_length / math.cos(angle)
        ramp_cx = self.flat_length + self.ramp_length / 2.0
        # Center z: halfway down the surface, minus the projected half-thickness.
        ramp_cz = -(self.ramp_length * math.tan(angle) / 2.0) - (t / 2.0) * math.cos(angle)
        half = angle / 2.0
        ramp = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(surf_len / 2.0, width / 2.0, t / 2.0),
            pos=(ramp_cx, 0.0, ramp_cz),
            quat=(math.cos(half), 0.0, math.sin(half), 0.0),
        )

        origin = np.array([self.flat_length * 0.4, 0.0, 0.0])
        return TerrainOutput(
            origin=origin,
            geometries=[
                TerrainGeometry(geom=flat, color=(0.5, 0.5, 0.5, 1.0)),
                TerrainGeometry(geom=ramp, color=(0.45, 0.55, 0.75, 1.0)),
            ],
        )
```

- [ ] **Step 4: Run the tests — they must pass**

Run: `uv run pytest tests/test_slope_terrain.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/mjlab_microduck/tasks/slope_terrain.py tests/test_slope_terrain.py
git commit -m "roller-slope: custom flat+ramp terrain (FlatRampTerrainCfg + tests)"
```

- [ ] **Step 6: Visual verification (human checkpoint)**

The geometry (especially `ramp_cz` and the quaternion's sign) must be confirmed by eye.
After Task 4 (env assembled), launch the play viewer (see Task 4 Step 6) and check:
the flat area meets the ramp **with no step and no gap**, and the ramp **descends** in
the `+x` direction (in front of the robot). If a vertical offset appears, adjust `ramp_cz`;
if the ramp goes up instead of down, flip the sign (`-half`) of the quaternion.

---

## Task 3: steepness curriculum `terrain_levels_slope`

**Files:**
- Modify: `src/mjlab_microduck/tasks/mdp.py`
- Test: `tests/test_slope_curriculum.py`

**Interfaces:**
- Produces:
  - `slope_move_masks(distance: torch.Tensor, size_x: float) -> tuple[torch.Tensor, torch.Tensor]` — a pure helper. `move_up = distance > size_x * 0.5` (reached the bottom → steeper ramp); `move_down = (distance < size_x * 0.2) & ~move_up` (early fall/stall → gentler ramp). Returns `(move_up, move_down)` as `bool`.
  - `terrain_levels_slope(env, env_ids) -> torch.Tensor` — the mjlab curriculum signature; computes the distance travelled in `x` from the origin, applies `slope_move_masks`, calls `terrain.update_env_origins`, and returns the mean level.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_slope_curriculum.py
import torch
from mjlab_microduck.tasks.mdp import slope_move_masks


def test_move_up_when_reached_bottom():
    # distance > size_x/2 → promote to a harder slope
    dist = torch.tensor([5.0, 4.1])
    up, down = slope_move_masks(dist, size_x=8.0)
    assert bool(up[0]) and bool(up[1])
    assert not bool(down[0]) and not bool(down[1])


def test_move_down_when_stuck_early():
    # distance < size_x*0.2 (=1.6) → demote to an easier slope
    dist = torch.tensor([0.5, 1.0])
    up, down = slope_move_masks(dist, size_x=8.0)
    assert not bool(up[0]) and not bool(up[1])
    assert bool(down[0]) and bool(down[1])


def test_stay_in_middle_band():
    # between 1.6 and 4.0 → neither up nor down
    dist = torch.tensor([2.5])
    up, down = slope_move_masks(dist, size_x=8.0)
    assert not bool(up[0]) and not bool(down[0])
```

- [ ] **Step 2: Run the test — it must fail**

Run: `uv run pytest tests/test_slope_curriculum.py -v`
Expected: FAIL — `ImportError: cannot import name 'slope_move_masks'`

- [ ] **Step 3: Minimal implementation**

Add to `src/mjlab_microduck/tasks/mdp.py` (near the other curricula, e.g. after `com_range_curriculum`). Check at the top of the file that `torch` is imported (it is).

```python
def slope_move_masks(distance: "torch.Tensor", size_x: float):
    """Promotion/demotion masks for the slope curriculum.

    move_up   : travelled more than half the tile → it rode the ramp down, so we
                make it steeper.
    move_down : barely moved (< 20% of the tile) → early fall/stall, so we
                flatten the ramp.
    """
    move_up = distance > size_x * 0.5
    move_down = (distance < size_x * 0.2) & (~move_up)
    return move_up, move_down


def terrain_levels_slope(env, env_ids):
    """Steepness curriculum for roller_slope (no commanded velocity).

    Progression based on the x distance travelled from the spawn origin.
    """
    asset = env.scene["robot"]
    terrain = env.scene.terrain
    assert terrain is not None
    terrain_generator = terrain.cfg.terrain_generator
    assert terrain_generator is not None

    distance = (
        asset.data.root_link_pos_w[env_ids, 0] - env.scene.env_origins[env_ids, 0]
    )
    move_up, move_down = slope_move_masks(distance, terrain_generator.size[0])
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())
```

- [ ] **Step 4: Run the test — it must pass**

Run: `uv run pytest tests/test_slope_curriculum.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/mjlab_microduck/tasks/mdp.py tests/test_slope_curriculum.py
git commit -m "roller-slope: terrain_levels_slope steepness curriculum (+ tested pure helper)"
```

---

## Task 4: `roller_slope` env cfg + registration

**Files:**
- Create: `src/mjlab_microduck/tasks/microduck_roller_slope_env_cfg.py`
- Modify: `src/mjlab_microduck/tasks/__init__.py`
- Test: `tests/test_roller_slope_cfg.py`

**Interfaces:**
- Consumes: `make_microduck_velocity_rollers_env_cfg` (the physics/DR/obs base), `FlatRampTerrainCfg` (Task 2), `terrain_levels_slope` (Task 3), and the existing mdp functions: `body_upright_gaussian`, `is_alive`, `pose_target_match`, `pose_l1_penalty`, `feet_flat_penalty`, `neck_action_rate_l2`, `joint_torques_l2`, `robot_state_is_nan`, `reset_action_history`, `zero_command_padding`.
- Produces: `make_microduck_roller_slope_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg` and `MicroduckRollerSlopeRlCfg` (`RslRlOnPolicyRunnerCfg`, `experiment_name="roller_slope"`).

> Reuse the roller env's DR/obs/reset blocks: we **start from** `make_microduck_velocity_rollers_env_cfg()` and modify ONLY the terrain, command, rewards, terminations and curriculum. Do not rewrite the DR.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_roller_slope_cfg.py
from mjlab_microduck.tasks.microduck_roller_slope_env_cfg import (
    make_microduck_roller_slope_env_cfg,
)
from mjlab_microduck.tasks.slope_terrain import FlatRampTerrainCfg


def test_terrain_is_flat_ramp_generator():
    cfg = make_microduck_roller_slope_env_cfg()
    assert cfg.scene.terrain.terrain_type == "generator"
    gen = cfg.scene.terrain.terrain_generator
    assert gen is not None and gen.curriculum is True
    assert any(isinstance(st, FlatRampTerrainCfg) for st in gen.sub_terrains.values())


def test_command_is_neutralised():
    cfg = make_microduck_roller_slope_env_cfg()
    cmd = cfg.commands["twist"]
    assert cmd.rel_standing_envs == 1.0
    assert cmd.rel_heading_envs == 0.0


def test_entry_velocity_set_on_reset_base():
    cfg = make_microduck_roller_slope_env_cfg()
    vr = cfg.events["reset_base"].params["velocity_range"]
    assert vr["x"][0] > 0.0  # forward impulse


def test_has_upright_and_pose_rewards():
    cfg = make_microduck_roller_slope_env_cfg()
    for name in ("upright", "alive", "standing_pose", "feet_flat"):
        assert name in cfg.rewards
```

- [ ] **Step 2: Run the test — it must fail**

Run: `uv run pytest tests/test_roller_slope_cfg.py -v`
Expected: FAIL — `ModuleNotFoundError` (the env cfg module is missing)

- [ ] **Step 3: Implementation**

```python
# src/mjlab_microduck/tasks/microduck_roller_slope_env_cfg.py
"""Microduck roller slope — balanced passive descent.

The robot spawns on the flat (with a forward impulse), rolls onto a downhill
ramp and lets itself glide while staying upright. No steering: the twist
command is neutralized (rel_standing_envs=1.0). Custom flat+ramp terrain
(FlatRampTerrainCfg), steepness curriculum (terrain_levels_slope).
Unified 61D obs → hot-swappable at runtime (--new-cmd-obs).
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg, EventTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlModelCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.tasks.velocity import mdp
from mjlab.envs import mdp as base_mdp

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.slope_terrain import FlatRampTerrainCfg
from mjlab_microduck.tasks.microduck_velocity_rollers_env_cfg import (
    make_microduck_velocity_rollers_env_cfg,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg

ENTRY_VELOCITY_X = (0.2, 0.5)  # forward impulse at reset (m/s)


def make_microduck_roller_slope_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_velocity_rollers_env_cfg(play=play)

    # === TERRAIN: flat + ramp, steepness curriculum ===
    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=(8.0, 4.0),
            curriculum=True,
            num_rows=10,          # 10 steepness levels
            num_cols=1,
            difficulty_range=(0.0, 1.0),
            sub_terrains={"flat_ramp": FlatRampTerrainCfg(flat_length=2.0, ramp_length=5.0)},
        ),
        max_init_terrain_level=0,  # start on the gentlest ramp
    )

    # === Neutralized COMMAND (pure balance) ===
    command = cfg.commands["twist"]
    command.rel_standing_envs = 1.0
    command.rel_heading_envs = 0.0
    command.ranges.lin_vel_x = (0.0, 0.0)
    command.ranges.lin_vel_y = (0.0, 0.0)
    if getattr(command.ranges, "ang_vel_z", None) is not None:
        command.ranges.ang_vel_z = (0.0, 0.0)

    # === RESET: forward impulse on the flat ===
    cfg.events["reset_base"].params["velocity_range"] = {"x": ENTRY_VELOCITY_X}

    # === REWARDS: balance + nominal standing posture ===
    keep = {"action_rate_l2"}
    for name in list(cfg.rewards.keys()):
        if name not in keep:
            del cfg.rewards[name]

    cfg.rewards["upright"] = RewardTermCfg(
        func=microduck_mdp.body_upright_gaussian,
        weight=3.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)), "std": 0.2},
    )
    cfg.rewards["alive"] = RewardTermCfg(func=microduck_mdp.is_alive, weight=1.0)
    # nominal standing posture (fixed target = default_joint_pos, no override)
    cfg.rewards["standing_pose"] = RewardTermCfg(
        func=microduck_mdp.pose_target_match, weight=3.0, params={"std": 0.4},
    )
    cfg.rewards["standing_pose_l1"] = RewardTermCfg(
        func=microduck_mdp.pose_l1_penalty, weight=1.0,
    )
    cfg.rewards["feet_flat"] = RewardTermCfg(
        func=microduck_mdp.feet_flat_penalty,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
            "sensor_name": "feet_ground_contact",
        },
    )
    cfg.rewards["neck_action_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.neck_action_rate_l2, weight=-0.5,
    )
    cfg.rewards["joint_torques_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torques_l2, weight=-1e-3,
    )
    cfg.rewards["action_rate_l2"].weight = -1.0

    # === TERMINATIONS: fall + bottom reached ===
    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=base_mdp.bad_orientation,
        params={"limit_angle": 1.0, "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    cfg.terminations["out_of_bounds"] = TerminationTermCfg(func=mdp.out_of_terrain_bounds)
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan, time_out=False,
    )

    # === EVENTS ===
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history, mode="reset",
    )

    # === CURRICULUM: ramp steepness ===
    for name in list(cfg.curriculum.keys()):
        del cfg.curriculum[name]
    cfg.curriculum["terrain_levels"] = CurriculumTermCfg(func=microduck_mdp.terrain_levels_slope)

    return cfg


MicroduckRollerSlopeRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
        entropy_coef=0.01, num_learning_epochs=5, num_mini_batches=4,
        learning_rate=1.0e-3, schedule="adaptive", gamma=0.99, lam=0.95,
        desired_kl=0.01, max_grad_norm=1.0, symmetry_cfg=None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="roller_slope",
    run_name="roller_slope",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=8_000,
)
```

Then register it in `src/mjlab_microduck/tasks/__init__.py`, following EXACTLY the registration pattern of the `roller_crouch` block already present (import `make_...` + `Microduck...RlCfg`, then `register_mjlab_task(...)` with an id in the style of `"Microduck-Roller-Slope"`). Copy the `roller_crouch` block and replace `crouch`→`slope`.

- [ ] **Step 4: Run the tests — they must pass**

Run: `uv run pytest tests/test_roller_slope_cfg.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Verify the task registration + a full build**

Run:
```bash
uv run python -c "import gymnasium as gym; import mjlab_microduck.tasks; print([e for e in gym.registry if 'Slope' in e])"
```
Expected: the list contains the id `Microduck-Roller-Slope` (or whichever variant was registered).

- [ ] **Step 6: Visual verification of the terrain + descent (human checkpoint — closes Task 2 Step 6)**

Run a short training, then play (or `scripts/play_latest.py`, depending on the repo's usage) and observe:
1. Flat + ramp assembled with no step/gap; the ramp **descends** in front of the robot.
2. The robot spawns on the flat, moves forward, and reaches the ramp.
If the geometry is wrong, fix `slope_terrain.py` (see Task 2 Step 6) and re-commit.

- [ ] **Step 7: Commit**

```bash
git add src/mjlab_microduck/tasks/microduck_roller_slope_env_cfg.py src/mjlab_microduck/tasks/__init__.py tests/test_roller_slope_cfg.py
git commit -m "roller-slope: passive descent env (flat+ramp terrain, zero cmd, balance rewards) + registration"
```

---

## Task 5: deployment — the `--slope` flag + the `Y` key

**Files:**
- Modify: `scripts/infer_policy.py`

**Interfaces:**
- Consumes: the exported `.onnx` of the `roller_slope` policy.
- Produces: the `--slope <path>` CLI argument; a `self.slope_session` attribute + a `self.slope_mode` flag; a `toggle_slope_mode()` method; the `GLFW_KEY_Y = 89` key wired up.

> The slope policy runs with a zero twist command (like standing mode). In slope mode, the automatic walking/standing switch must be disabled.

- [ ] **Step 1: Add the CLI argument and load the session**

In `main()` (near the other `add_argument` calls, ~line 471):
```python
    parser.add_argument("--slope", type=str, default=None, help="Path to slope policy ONNX file (press Y to toggle)")
```
Pass `slope_onnx_path=args.slope` to the controller's constructor (add the `slope_onnx_path=None` parameter to `__init__`, ~lines 51-57, and load it like the others):
```python
        self.slope_session = None
        self.slope_mode = False
        if slope_onnx_path:
            print(f"\nLoading slope policy from: {slope_onnx_path}")
            self.slope_session = ort.InferenceSession(slope_onnx_path)
```

- [ ] **Step 2: Add `toggle_slope_mode` and disable the automatic switch**

After `toggle_body_pose_mode` (~line 285):
```python
    def toggle_slope_mode(self):
        """Toggle to/from the slope policy (passive descent)."""
        if self.slope_session is None:
            print("Slope unavailable: no --slope policy loaded")
            return
        self.slope_mode = not self.slope_mode
        if self.slope_mode:
            self.ort_session = self.slope_session
            self.current_policy = "slope"
            self.set_vel_cmd(0.0, 0.0, 0.0)  # passive descent: zero command
            print("Slope mode: ON (passive descent)")
        else:
            self.ort_session = self.walking_session or self.standing_session
            self.current_policy = "walking" if self.walking_session else "standing"
            print("Slope mode: OFF")
```
In `_update_policy_session` (~line 250), add the guard at the top (after the `ground_pick_mode` guard):
```python
        if self.slope_mode:
            return  # Do not switch while in slope mode
```

- [ ] **Step 3: Wire up the `Y` key**

Add the key code near the others (~line 680):
```python
    GLFW_KEY_Y = 89
```
In `key_callback`, add a branch (e.g. after the `GLFW_KEY_B` branch):
```python
            elif key == GLFW_KEY_Y:
                policy.toggle_slope_mode()
```
Add the keyboard help line (near the `print` calls, ~line 821):
```python
    print("  Y:                toggle slope mode (requires --slope, passive descent)")
```

- [ ] **Step 4: Check that the script loads without error**

Run: `uv run python scripts/infer_policy.py --help`
Expected: the help is displayed and lists `--slope`.

- [ ] **Step 5: Commit**

```bash
git add scripts/infer_policy.py
git commit -m "roller-slope: --slope deployment + Y key (slope mode toggle)"
```

---

## Self-Review (by the plan's author)

- **Spec coverage**: dedicated task (Task 4) ✓; custom flat+ramp terrain (Task 2) ✓; flat start + impulse (Task 4 reset velocity_range) ✓; zero command (Task 4) ✓; balance + standing pose + anti-flattening rewards (Task 4) ✓; fall/bottom/nan terminations (Task 4) ✓; 0→20° curriculum (Task 1 angle + Task 3 promotion) ✓; hot-swappable 61D obs (inherited from the roller env, unmodified) ✓; the Y button (Task 5) ✓.
- **Placeholders**: no "TBD/TODO"; the two human checkpoints (viewer geometry) are explicit verifications, not implementation gaps.
- **Type consistency**: `ramp_angle_by_difficulty` (Task 1) reused by `FlatRampTerrainCfg` (Task 2); `slope_move_masks` (Task 3) consumed by `terrain_levels_slope` (Task 3); the reward names tested in Task 4 (`upright`, `alive`, `standing_pose`, `feet_flat`) aligned with the implementation.
- **Flagged risks**: the ramp geometry (`ramp_cz`, the quaternion's sign) to be confirmed in the viewer; the exact mjlab API names (`terrain.terrain_levels`, `TerrainEntityCfg`, the registration id) to be validated against the existing `roller_crouch` pattern during implementation.
