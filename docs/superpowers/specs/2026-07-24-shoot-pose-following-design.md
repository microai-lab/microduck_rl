# Spec — "Kick a ball" RL task by pose following

**Date**: 2026-07-24
**Branch**: `new_pre_alpha_ground_pick`
**Task id**: `Mjlab-Shoot-Flat-MicroDuck`

## Objective

Learn a **one-shot kick gesture** (striking a ball) by **following a 4-keyframe
joint-pose trajectory** interpolated along the phase:

```
STAND → FOOT_BACK (wind-up) → FOOT_FORWARD (strike) → STAND (rest)
```

- The **right leg** kicks, the **left leg** provides support.
- **No simulated ball**: we learn the *gesture* through pose following (like
  `ground_pick` / crouch). If a real ball is in front of the robot at deployment,
  it gets kicked.
- **Unified 61D obs**, identical to the other microduck policies → the exported
  ONNX deploys as-is into a runtime **button slot** (one-shot: it plays the
  gesture then hands control back to the main policy).

Same mold as this branch's `ground_pick` task (phase encoded as `[cos, sin, 0]`
in the twist slot, per-phase pose following, 61D obs, sim2real DR inherited from velocity).

## Non-objectives (YAGNI)

- No physical ball, no contact/ball-velocity reward.
- No configurable side (right only; left = mirrorable later if needed).
- No walking / fall recovery: all locomotion terms are removed.

## Architecture

### File & registration
- `src/mjlab_microduck/tasks/microduck_shoot_env_cfg.py`
  - `make_microduck_shoot_env_cfg(play: bool = False, rough: bool = False) -> ManagerBasedRlEnvCfg`
  - `MicroduckShootRlCfg` (RslRlOnPolicyRunnerCfg, `experiment_name="shoot"`)
- Registration in `src/mjlab_microduck/tasks/__init__.py`:
  `Mjlab-Shoot-Flat-MicroDuck` (optional `-Rough-` variant).
- Base: inherits from the velocity env (via `make_velocity_env_cfg` like ground_pick),
  then aggressively strips everything locomotion-related.
- Robot: `MICRODUCK_WALK_ROBOT_CFG` (standard walker, 14 joints, no rollers).
- `action.scale = 1.0`.

### Poses (placeholders → read off the real robot via `read_pose.py`)
`{joint_name: rad}` dicts, **14 joints** (mouth excluded). At the top of the env file.
- `STAND_POSE`: neutral stance (~the sim's HOME).
- `KICK_BACK_POSE`: right hip in **backward extension** + right knee flexed
  (wind-up); left leg + neck ≈ HOME.
- `KICK_FWD_POSE`: right hip **flexed forward** + right knee extended (strike);
  left leg + neck ≈ HOME.

Plausible placeholders to begin with (adjustable), to be replaced by real readings.

### Command & phase
- Reuses `GroundPickPhaseCommand`: `command = [cos(2π·φ), sin(2π·φ), 0]` in the
  twist slot.
- **Period**: `SHOOT_PERIOD ≈ 2.5 s` (configurable via `cfg.period`).
- **New `randomize_phase` flag** on `GroundPickPhaseCommandCfg` /
  `GroundPickPhaseCommand`:
  - Default `True` (non-breaking: `ground_pick` keeps its current behavior).
  - Shoot sets it to `False` → `reset()` sets φ=0 instead of `rand()`.
  - Reason: every episode starts at STAND (robot state = `default_joint_pos`)
    with φ=0 = the STAND target → state/target consistency at reset (otherwise the
    policy is required to be instantly in the "strike" pose from a standstill).
  - **Consistency invariant**: `STAND_POSE` MUST equal the sim's reset joint pose
    (`HOME_FRAME` / `default_joint_pos`, which is non-zero: hip_pitch ±0.4579,
    ankle ±0.4530, hip_roll ±0.0873, neck/head_pitch 0.3491). Verified by
    `test_stand_pose_matches_home_standing_pose`. The placeholders initially set to
    zero broke this invariant (fixed after the final review).

### Reset (standing height, no run-up)
- `reset_base.pose_range.z = (0.12, 0.13)` — an **absolute standing height** (the
  default root `pos` of `InitialStateCfg` is (0,0,0), so the reset z is 0.12–0.13 m,
  not an additive offset; the same value as the velocity env, which works). No fall.
- **No entry-velocity injection** (kicking from standing, unlike crouch-glide).

### Inherited rewards not listed
The table above is not exhaustive: the env inherits from velocity a few generic
low-weight regularizers not specific to the kick —
`angular_momentum` (-0.02), `dof_pos_limits` — kept (stability, negligible).
⚠️ `soft_landing` (a walking reward) is **removed**: it reads the two-foot
`feet_ground_contact` sensor, dropped in favor of the left-foot sensor → KeyError on the
first step otherwise, and it is inert for a standing kick.

### Sensor-renaming gotcha (⚠️)
Renaming the foot sensor (`feet_ground_contact` → `left_foot_ground_contact`) breaks
everything the velocity/ground_pick inheritance references by that name. To handle:
- **critic obs** `foot_air_time`/`foot_contact`/`foot_contact_forces` → repointed at
  the left-foot sensor (the critic keeps the support information; otherwise KeyError at
  env construction).
- **reward** `soft_landing` → removed (see above; otherwise KeyError on the first step).
Always validate with a live construction + **at least one `step()`** (the reward
manager only runs at step time), not just the cfg build or the unit tests.

### ⚠️ Learned weight transfer (revision after the first training run)
Finding: the BACK/FWD poses recorded with the **robot held by hand (double support)** keep the
CoM **centered between both feet** (~4-5 cm inside the left foot) at every phase.
With `upright` imposed, as soon as the right foot lifts the robot tips over → no
policy can hold it (geometric, not a tuning issue). Verified in sim (CoM vs foot sites).

Chosen fix (let RL learn the balance):
- `mdp.com_over_support_foot`: a gaussian reward (std 4 cm) pulling the CoM projection
  (`root_com_pos_w`) toward the support foot, **gated** by `mdp.kick_engagement` (0 at STAND
  rest, 1 during the strike). Weight 3.0.
- **split pose following** (a `joint_names` param on `kick_pose_track`/`_l1`):
  GESTURE = right leg + neck/head (std 0.35, tight); SUPPORT = left leg (std 0.9,
  weight 1.0, **loose**) → the policy can adduct/shift the pelvis to transfer the
  weight without the pose tracking freezing the pelvis centered.
The "Balance / support" table above is therefore extended: `support_leg_pose` (1.0)
and `com_over_support` (3.0) are added, and `kick_pose_track`/`kick_pose_l1` now cover
only the 9 gesture joints (right + neck).

### Objective: following the phase-interpolated pose
A new **pure** function in `mdp.py`:
```python
kick_pose_target(phase, stand, back, forward, windup_end, kick_end, return_end) -> Tensor
```
Interpolates between the pose vectors across 4 segments (normalized period [0,1)):
```
[0, windup_end)        STAND   → BACK      (wind-up,      default 0.35)
[windup_end, kick_end) BACK    → FORWARD   (sharp strike, default 0.10 = "snap")
[kick_end, return_end) FORWARD → STAND     (return,       default 0.30)
[return_end, 1.0)      STAND              (rest)
```
The "snap" comes from the short strike segment: the joint target moves fast → a fast
foot swing. All 3 timing bounds are parameterizable.

Joints resolved **by name** (`asset.find_joints([name])`) — robust to ordering.

Tracking rewards (always active, symmetric like crouch):
| Reward | Weight | Role |
|---|---|---|
| `kick_pose_tracking` | 6.0 | gaussian tracking `exp(-((q-target)/std)²).mean`, std=0.4 |
| `kick_pose_l1` | 2.0 | L1 bootstrap (constant gradient early on) |

### Balance / support (single leg = tipping risk)
| Reward | Weight | Role |
|---|---|---|
| `upright` | 2.0 | vertical trunk |
| `support_foot_grounded` (left foot) | 6.0 | keep the support foot planted (single-foot sensor → `found∈{0,1}` → reward∈{0,0.5} after `/2`, hence weight 6.0 ≈ max contribution 3.0) |
| `feet_flat` (left) | -1.0 | left blade flat |
| `self_collisions` | -1.0 | |
| `body_ang_vel` | -0.05 | |

`support_foot_grounded`: reuse the ground_pick `feet_grounded_reward` mechanism but
restricted to the **left foot** (contact sensor on
`left_foot_collision`).

### Regularization (lighter than ground_pick — let the snap through)
| Reward | Weight | Role |
|---|---|---|
| `action_rate_l2` | -0.5 | light: a heavy weight would kill the fast strike |
| `neck_action_rate_l2` | -0.5 | stable head |
| `joint_torques_l2` | -1e-3 | |

**Removed** (walking terms): `track_linear_velocity`, `track_angular_velocity`,
`air_time`, `foot_clearance`, `foot_swing_height`, `foot_slip`, `pose`.

### Observations / deployment (parity)
- **61D obs identical** to ground_pick/roller: `[gyro(3), projected_gravity(3),
  joint_pos(14), joint_vel(14), last_action(14), command(13)]` with the
  head(4)+body(6) slots **zero-padded** (`zero_command_padding`).
- The same sim2real DR inherited from velocity (CoM, mass/inertia, BAM friction, armature,
  obs-level IMU misalignment, encoder bias, ±0.3 pushes), ending with the NaN guard.
- ONNX export (normalizer baked in) via the existing export script.
- Deployed into a runtime phase slot, e.g.:
  ```
  --ground-pick shoot.onnx --ground-pick-period 2.5 \
  --ground-pick-kp-ratio 1.0 --ground-pick-action-scale <match>
  ```
  Button → kick → automatic return to the main policy.

## Tests

- `tests/test_shoot.py` — pure functions:
  - `kick_pose_target` at the keypoints: STAND at φ=0, BACK at `windup_end`,
    FORWARD at `kick_end`, STAND in the rest segment; mid-segment interpolation;
    bounds (each component between the min/max of the poses).
  - Values of the `kick_pose_tracking` / `kick_pose_l1` rewards on simple cases.
- `tests/test_shoot_cfg.py` — the env builds with the right command
  (`GroundPickPhaseCommand`, `randomize_phase=False`, period) and the expected
  rewards present / walking terms absent.
- Run: `uv run --with pytest pytest tests/ -q`.

## Training

```bash
uv run train Mjlab-Shoot-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max_iterations <N>
```
Watch `Episode_Reward/kick_pose_tracking` (it must rise). Play: the play_latest script.

## Open points / to settle during training
- **Timings** (windup/kick/return) and **period**: reasonable snap defaults,
  to be adjusted according to the foot speed achieved and the stability.
- **`action_rate` weight**: snap vs sim2real smoothing tension; start light (-0.5).
- **Optional enrichment (not retained for v1)**: a small "right foot forward velocity"
  reward gated on the strike segment, to push for power without a simulated ball.
  To be added only if pose following alone lacks punch.
- **Transitions at deployment**: if `STAND_POSE` ≠ the main policy's neutral,
  a slight jolt on trigger/return (as noted for crouch).
