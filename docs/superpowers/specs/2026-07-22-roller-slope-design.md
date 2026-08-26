# Slope mode — `roller_slope` (balanced passive descent)

Date: 2026-07-22
Status: design approved, ready for the implementation plan.

## Objective

Train a dedicated policy where **the microduck (on rollers) starts on the flat
with a small forward impulse, rolls to a downhill ramp, and lets itself glide all
the way down while staying upright and balanced**. No steering during the
descent: the policy's only objective is to **not fall**.

The policy must handle ramps of increasing steepness (**~2° → ~20°**) via a
difficulty curriculum.

## Settled decisions (brainstorming)

| Topic | Decision |
|---|---|
| Behavior | Balanced passive descent (gravity provides the motion, no imposed pedalling) |
| Steering | None — pure balance, `twist` command forced to zero |
| Approach | **A** — dedicated, isolated task (like `roller_crouch`) |
| Terrain shape | **Simple ramp**: starting flat + downhill ramp (not a pyramid) |
| Episode scenario | Spawn on the flat → forward impulse velocity → glide down the ramp |
| Steepness | Curriculum **0/2° → 20°** |
| Deployment | `--slope <onnx>` flag + the **`Y`** key in `infer_policy.py` (Y is free) |

## Architecture

### 1. New task

File: `src/mjlab_microduck/tasks/microduck_roller_slope_env_cfg.py`, cloned from
`microduck_velocity_rollers_env_cfg.py`.

- Same roller robot (`MICRODUCK_WALK_ROLLERS_ROBOT_CFG`), same physics, same
  domain randomization / noise / delays.
- **Same 61D observation** (twist + zero-padded head/body) → the policy loads
  through the runtime's `--new-cmd-obs` path and stays hot-swappable with
  the other roller policies.
- Registered in `src/mjlab_microduck/tasks/__init__.py` via
  `register_mjlab_task`, with a PPO config `MicroduckRollerSlopeRlCfg`
  (`experiment_name`/`run_name` = `roller_slope`).

### 2. "Flat + ramp" terrain (custom)

The sloped terrains shipped with mjlab are pyramids; we therefore write a
dedicated `SubTerrainCfg` (e.g. `FlatRampTerrainCfg`) whose
`function(difficulty, spec, rng)` method builds:

- a **starting flat area** (~1–2 m long) where the robot spawns;
- a **downhill ramp** after it, whose angle is
  **interpolated by `difficulty`** over `[~2°, ~20°]`.

The terrain is mounted through `TerrainEntityCfg(terrain_type="generator", ...)` with a
`TerrainGeneratorCfg` that generates several difficulty levels (hence several
ramp angles). Each environment's origin must land **on the flat area**, with the
ramp in front of it.

> Implementation risk to address in the plan: placing the spawn origin on the
> flat (not at the tile center), and orienting the ramp so that "forward" =
> "downhill".

### 3. Command = none

The `twist` slot is neutralized: `rel_standing_envs = 1.0`, velocity ranges at 0,
`rel_heading_envs = 0.0`. Head/body stay zero-padded. The policy receives no
movement instruction at all.

### 4. Reset & impulse velocity

- `reset_base`: spawn at rest on the flat, at the nominal roller `z` height
  (~`0.1335–0.1435`, like the roller env).
- **Entry velocity** injected through the `velocity_range` of
  `reset_root_state_uniform` (clean state + range), **not** through
  `push_by_setting_velocity` (which adds to the current state and can make the
  free joint diverge → NaN — a lesson already learned on `roller_crouch`):
  `x ≈ (0.2, 0.5) m/s` forward.
- Light random pushes kept during the episode (robustness), like the
  roller env.

### 5. Rewards

Core "stay upright + natural posture", anti-lazy-optimum (preventing it from
flattening itself onto the ground to maximize stability):

- `upright` (vertical trunk) — **primary**
- `alive` (per-step survival bonus)
- **nominal standing pose**: reward toward the HOME pose (pose-interpolation
  machinery taken from `roller_crouch`, but with a fixed target = standing),
  to keep a normal roller stance rather than a defensive crouch
- `feet_flat` (rollers flat on the ground)
- `body_ang_vel`, `angular_momentum` (no shaking / twisting)
- `action_rate_l2`, `neck_action_rate_l2`, `joint_torques_l2`,
  `self_collisions` (smoothness + sim2real)

> No speed/braking reward: the descent is passive. We do not reward "going
> fast", only "staying upright while going down".

### 6. Terminations

- **Fall**: `bad_orientation` (trunk tilted too far).
- **Bottom reached**: `out_of_terrain_bounds` (the robot has reached the bottom
  of the ramp → reset).
- `nan_state`, time-out.

### 7. Difficulty (steepness) curriculum

Progression **gentle → steep**: start on nearly flat ramps and increase the
angle toward 20° as successes accumulate.

> Implementation risk: the standard `terrain_levels_vel` curriculum promotes
> based on distance travelled relative to the commanded velocity. Here the
> command is zero, so **a custom promotion criterion is needed**: promote if the
> robot survived / reached the bottom without falling, demote if it falls early.

### 8. Deployment — the `Y` button

In `scripts/infer_policy.py`:

- a new `--slope <onnx>` flag loading the slope policy as an additional session
  (same scheme as `--walking` / `--standing` / `--ground-pick`);
- `GLFW_KEY_Y = 89` (currently **free** — the head is on `H`) which **toggles**
  the active session to/from the slope policy;
- a keyboard help line added.

No existing control is broken (unlike sharing the head control's `H` key).

## What is NOT in scope (YAGNI)

- No left/right steering and no braking on the descent.
- No uphill climbing and no traversal.
- No pyramid and no multi-direction terrains.
- No fine-tuning from the existing roller weights (training from scratch).

## Deliverables

1. `microduck_roller_slope_env_cfg.py` (env + `FlatRampTerrainCfg` + PPO cfg).
2. Task registration in `tasks/__init__.py`.
3. Any custom rewards/curriculum needed in `tasks/mdp.py` (standing pose, level
   promotion).
4. `--slope` wiring + the `Y` key in `scripts/infer_policy.py`.
5. Unit tests for the pure functions (ramp angle by difficulty, and the
   promotion criterion if any).
