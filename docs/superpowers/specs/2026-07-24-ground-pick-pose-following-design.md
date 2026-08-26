# Ground-pick by phase-interpolated pose following

**Date**: 2026-07-24
**Branch**: `new_pre_alpha_ground_pick`
**Target file**: `src/mjlab_microduck/tasks/microduck_ground_pick_env_cfg.py` (rewritten in place)
**Task id**: `Mjlab-GroundPick-Flat-MicroDuck` (unchanged)

## 1. Objective

Replace the current ground_pick's *task-space* objective (reward the mouth going
down to the ground, then separately reward the return to standing) with a
**directive pose-following objective**: we define two target joint poses — STAND
and DOWN — and reward tracking of the **phase-interpolated pose**
(STAND→DOWN→STAND).

Motivation (taken from the roller_crouch approach, which is validated): the
interpolated-pose objective is **symmetric by construction** — "standing back up"
(target → STAND) is rewarded exactly like "going down" (target → DOWN), which
fixes the lazy-optimum problem where the policy goes down but comes back up
poorly. The signal is **dense at every phase** (a continuously moving target),
unlike a fixed target weighted by `sin`, which gives no signal at the transitions.

The gesture stays triggered by **button A** through the runtime's `--ground-pick`
slot (one-shot, automatic return to the main policy). The unified 61D obs is
unchanged → the policy stays hot-swappable in the slot.

## 2. Target poses

Joints resolved **BY NAME** (`asset.find_joints([name])`) — robust, and consistent
with the roller approach. 14 joints (mouth excluded).

- **STAND_POSE** = HOME (the model's `default_joint_pos`). The blend source; do not
  redefine it by hand — use the model default as the source (blend=0).
  At deployment, the main policy resumes from HOME → clean return.

- **DOWN_POSE** = initial values taken from the **FOLD keyframe** of `scene_walk.xml`
  (deep forward fold, head down → mouth toward the ground). A by-name dict at the top of
  the file, **commented as replaceable with a `read_pose.py` reading** of the real
  robot placed mouth-to-ground. Starting values:

  ```python
  DOWN_POSE = {
      "left_hip_yaw": 0.0, "left_hip_roll": 0.0, "left_hip_pitch": 1.57,
      "left_knee": 1.57, "left_ankle": 0.0,
      "neck_pitch": 1.0, "head_pitch": 1.0, "head_yaw": 0.0, "head_roll": 0.0,
      "right_hip_yaw": 0.0, "right_hip_roll": 0.0, "right_hip_pitch": -1.57,
      "right_knee": -1.57, "right_ankle": 0.0,
  }
  ```

## 3. Phase profile (4 segments)

`GroundPickPhaseCommand` command: `[cos(2πφ), sin(2πφ), 0]`, period **4.0 s**
(the runtime slot's default → no period flag to change at deployment).

```
DESCENT_END=0.15  HOLD_END=0.50  RISE_END=0.65   (4 s period)
[0, 0.15)     descent  STAND->DOWN   ~0.6 s   blend 0->1
[0.15, 0.50)  low      DOWN          ~1.4 s   blend 1
[0.50, 0.65)  rise     DOWN->STAND   ~0.6 s   blend 1->0
[0.65, 1.0)   high     STAND (rest)  ~1.4 s   blend 0
```

`blend ∈ [0,1]`: 0 = STAND (HOME), 1 = DOWN. Target = `stand + blend·(down - stand)`.
Tunable bounds (constants at the top of the file).

**`randomize_phase=False`**: every episode starts at φ=0 (= standing), like the
button-A trigger at deployment. Since episodes reset at staggered times, the envs
naturally decorrelate in phase (no need to randomize). This requires adding a
`randomize_phase` flag to `GroundPickPhaseCommandCfg` (default `True` → the other
sit/stand tasks are unchanged), honored in `reset()`.

## 4. New mdp functions (ported from roller, adapted, by name)

In `src/mjlab_microduck/tasks/mdp.py`. Names deliberately distinct from the existing
`phase_pose_match` (which is the fixed-target, sin-weighted variant) to avoid confusion.

- **`phase_pose_blend(phase, descent_end, hold_end, rise_end) -> Tensor`** — pure,
  4-segment blend 0..1 (testable in isolation).
- **`_phase_pose_error(env, asset_cfg, command_name, target_pose, descent_end,
  hold_end, rise_end, source_pose=None) -> (cur, target)`** — resolves joints by
  name; `source_pose` = HOME (`default_joint_pos`) if `None`; computes
  `phase = atan2(sin,cos)/2π % 1`, `blend`, then `target = source + blend·(target_pose - source)`.
- **`phase_pose_track(env, command_name, target_pose, source_pose=None, std=0.3,
  descent_end, hold_end, rise_end, asset_cfg) -> Tensor`** — gaussian
  `exp(-((cur-target)/std)²).mean(-1)`.
- **`phase_pose_track_l1(env, ...same args without std...) -> Tensor`** — bootstrap
  `-(cur-target).abs().mean(-1)` (constant gradient when the gaussian saturates).

`target_pose` = `DOWN_POSE` (by-name dict). `source_pose=None` → HOME.

## 5. Rewards

A minimal rewrite relative to the current one — we replace the pose-return
machinery and keep the stability/regularization/sim2real stack.

| Reward | Weight | Status | Role |
|---|---|---|---|
| `phase_pose_track` (std 0.3) | **6.0** | **NEW** | tracks the interpolated STAND↔DOWN pose |
| `phase_pose_track_l1` | **2.0** | **NEW** | L1 bootstrap |
| `mouth_ground_proximity` (std 0.10) | **1.0** | retuned (was 2.0) | safety net: guarantees the mouth reaches the ground if DOWN is imperfect; gated on the approach (+sin) |
| `upright` | 0.2 | kept | trunk ~vertical (low weight, the robot leans) |
| `feet_grounded` | 3.0 | kept | both feet on the ground throughout the gesture |
| `self_collisions` | -1.0 | kept | |
| `head_impact_penalty` (threshold 2 N) | -0.5 | kept | no head slam (DOWN brings the head low) |
| `action_rate_l2` | -0.8→-2.0 (curric) | kept | smoothing |
| `neck_action_rate_l2` | -1.0 | kept | |
| `joint_torques_l2` | -5e-3 | kept | |
| `body_ang_vel` | -0.05 | kept | |
| `angular_momentum` | -0.02 | kept | |
| `soft_landing` | -1e-5 | kept | |

**Removed**: `mouth_perpendicular_to_ground`, `ground_pick_return_pose_legs`,
`ground_pick_return_pose_neck` (replaced by pose following).

Everything else is **unchanged**: the DR block (CoM/head-CoM/mass-inertia/friction/armature/
IMU-misalign/encoder-bias/pushes), the 61D obs + zero head/body padding, the terminations
(`nan_state`), the curricula (`action_rate_weight`, `com_range`, `head_com_range`),
and the RlCfg (`experiment_name="ground_pick"`).

## 6. Deployment (sim2real parity)

```bash
microduck_runtime ... \
  --ground-pick ground_pick.onnx \
  --ground-pick-period 4.0 \       # = env period (the default, nothing to change)
  --ground-pick-kp-ratio 1.0 \     # trained at kp 200 → force 1.0 (the 0.6 default lowers it to 120)
  --ground-pick-action-scale 1.0   # = env action.scale
```

## 7. Tests

In `tests/` (run `uv run --with pytest pytest tests/ -q`):

- **Pure functions**: `phase_pose_blend` at the key points
  (φ=0→0, φ=0.075→0.5, φ=0.3→1, φ=0.575→0.5, φ=0.8→0, monotone per segment);
  `phase_pose_track`/`_l1`: maximum value (cur==target) and sign.
- **Env construction**: `make_microduck_ground_pick_env_cfg()` builds;
  the command is a `GroundPickPhaseCommand` with `randomize_phase=False`, `period=4.0`;
  the `phase_pose_track`/`phase_pose_track_l1` rewards are present;
  `mouth_perpendicular_to_ground`/`ground_pick_return_pose_*` are absent;
  `mouth_ground_proximity` is present with weight 1.0.

## 8. Training / play / export

```bash
uv run train Mjlab-GroundPick-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max_iterations 20000
uv run scripts/play_latest.py     # md-play
uv run scripts/export_latest.py   # normalizer baked into the ONNX
```
Watch `Episode_Reward/phase_pose_track` (it must rise).

## 9. Out of scope / notes

- **Duplicate `pose_target_match`** (mdp.py 1577 and 1914): latent, not addressed here.
- **Tuning DOWN_POSE**: if the mouth does not reach the ground closely enough with the
  FOLD values, adjust the dict (ideally a `read_pose.py` reading of the real robot
  placed mouth-to-ground) rather than inflating `mouth_ground_proximity`.
- **Transition at deployment**: STAND=HOME = the main policy's neutral →
  no jolt on return (unlike the issue noted on roller, where STAND≠HOME).
