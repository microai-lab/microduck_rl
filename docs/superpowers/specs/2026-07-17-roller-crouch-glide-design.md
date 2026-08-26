# Design — Roller Crouch-Glide ("crouch while gliding", on a button press)

**Date:** 2026-07-17
**Status:** design approved, ready for the implementation plan

## Context

The microduck robot can skate (the roller policy, task `Mjlab-Velocity-Flat-MicroDuck-Rollers`).
We want a new gesture: on a button press, it **crouches and keeps gliding on its
momentum** (like a skater in a low stance), holds for ~1 s, then **stands back up**
by itself and resumes skating.

Hard constraint from the user: **do not modify the Rust runtime**
(`apirrone/microduck_runtime`, installed as a binary). The gesture must therefore reuse a
mechanism already present in the runtime.

**Key discovery:** the runtime already has a "one-shot button-triggered behavior"
slot: `--ground-pick`. It is triggered by **button A** (rising edge), plays a
phase-driven ONNX policy for a fixed duration, then automatically returns to the
main policy. Crucially, it uses **exactly the same 61D observation layout** as the
roller policy — the two are hot-swappable at runtime. It is the ideal vehicle,
without a line of Rust.

Accepted trade-off: the gesture is **one-shot** (fixed duration, no "hold to keep it").
The crouch duration is set by the slot's period.

## Chosen approach (approach B)

Create a **new mjlab task** trained on the roller robot, which plays
descent → crouched glide → rise, driven by the ground-pick slot's phase.
Export it to ONNX and load it via `--ground-pick`. No Rust changes.

### Files involved

| File | Action |
|---|---|
| `src/mjlab_microduck/tasks/microduck_roller_crouch_env_cfg.py` | **New.** The env, a roller + ground-pick hybrid. |
| `src/mjlab_microduck/tasks/mdp.py` | **Add** the `crouch_glide_height_by_phase` reward. |
| `src/mjlab_microduck/tasks/__init__.py` | **Add**: register `Mjlab-RollerCrouch-Flat-MicroDuck`. |

### Reuse (do not reinvent anything)

- **Roller physics / robot** ← `microduck_velocity_rollers_env_cfg.py`:
  `MICRODUCK_WALK_ROLLERS_ROBOT_CFG` (14 active joints + 4 passive wheels),
  contact sensor on the `roller_blade`s, bearing-friction DR
  (`randomize_wheel_friction` + curriculum), 14-dim obs (wheels excluded via
  `SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))`), `action.scale=1.0`,
  `kp_fw=200`.
- **Phase / one-shot machinery** ← `microduck_ground_pick_env_cfg.py`:
  the `microduck_mdp.GroundPickPhaseCommand` command **reused as-is**
  (it produces the `[cos(2πφ), sin(2πφ), 0]` the runtime will send into the twist slot),
  zero head/body padding (`zero_command_padding`), the `robot_state_is_nan` termination,
  `reset_action_history`.
- **sim2real DR** ← taken from the roller env unchanged (obs-level IMU misalignment,
  encoder bias, mass/inertia, BAM friction, armature, gentle ±0.2 pushes).

## The core: a trapezoidal, phase-driven height target

The only genuine novelty. Instead of lowering the mouth (ground-pick), we drive the
**trunk height** (`com_height` of `trunk_base`) along the phase, with a low plateau:

```
height
  high ┐                    ┌──   standing (hands control back to the roller policy)
       │ \                 /
   low │  \_______________/       crouched + gliding (1 s plateau)
       └───────────────────────► phase
       0   0.375      0.625   1
```

- φ ∈ [0, 0.375]: descent toward the crouched height
- φ ∈ [0.375, 0.625]: **crouch hold** (= 1 s over a 4 s period) → glide
- φ ∈ [0.625, 1.0]: rise back to the standing roller pose

**New reward `crouch_glide_height_by_phase(env, command_name, height_low,
height_high, hold_lo=0.375, hold_hi=0.625, std=...)`** in `mdp.py`:
it reads the phase from the command, computes the target height (interpolated
high→low→high, flat over the plateau), and rewards `exp(-((h_measured - h_target)/std)²)`.
Take inspiration from `com_height_target` (mdp.py:694) and the
`interpolated/multistage height target` functions already present.

Starting values: `height_high ≈ 0.11` m (standing roller height, cf. the roller
`com_height_target` band 0.0935–0.1235), `height_low ≈ 0.075` m (crouched;
to be refined at play time). The phase is reconstructed from `atan2(sin, cos)` of the command.

## Rewards

| Reward | Role | Origin |
|---|---|---|
| `crouch_glide_height_by_phase` | Main target (high→low→high) | **new** |
| `wheel_speed` (reduced weight ~2–3) | Keep the momentum, do not brake during the crouch | roller env (`wheel_speed_reward`) |
| `upright` (≈2), `body_ang_vel` (−0.05), `angular_momentum` (−0.02) | Balance / stability | roller env |
| `return_pose` (end of phase) | Converge to the standing roller pose for a clean handover | adapted from `ground_pick_return_pose` |
| `feet_flat` (−2) | Blades flat → stable glide | roller env |
| `action_rate_l2`, `neck_action_rate_l2`, `joint_torques_l2`, `self_collisions` | Smoothing / sim2real transfer | both envs |

**Explicitly NOT included:** `braking` (we do not want to stop), `mouth_ground_proximity`
/ `mouth_perpendicular_to_ground` (we do not touch the ground), `skating_air_time` /
`single_support` / `glide` (no stride during the trick — we glide passively).

## Training

- `MicroduckRollerCrouchRlCfg` = a copy of `MicroduckRollersRlCfg`
  (MLP 512/256/128, ELU, obs_normalization, PPO, `experiment_name="roller_crouch"`).
- Register in `tasks/__init__.py`:
  `register_mjlab_task(task_id="Mjlab-RollerCrouch-Flat-MicroDuck", ...)`.
- Launch:
  ```bash
  uv run train Mjlab-RollerCrouch-Flat-MicroDuck \
    --env.scene.num-envs 4096 --agent.max_iterations 8000
  ```
- Episodes started with a **realistic entry velocity** (the robot arrives already rolling),
  otherwise it will have no momentum to preserve during the crouch. To be wired through a
  reset event (non-zero initial velocity) or a push at the start of the episode.

## Export + deployment (exact runtime flags)

ONNX export (the normalizer is baked in by `export.py`), then:

```bash
microduck_runtime --variant pre-alpha --new-cmd-obs --roller \
  --model output.onnx \
  --new-dxl-imu --kp 200 --action-scale 0.8 \
  --max-linear-vel 0.6 --max-linear-vel-backward 0.5 --max-angular-vel 0.0 \
  --ground-pick roller_crouch.onnx \
  --ground-pick-period 5.0 \
  --ground-pick-kp-ratio 1.0 \
  --ground-pick-action-scale 0.8
```

Button **A** → crouch-glide, then automatic return to the roller policy.

**Training/deployment parity pitfalls (important for sim2real):**
- `--ground-pick-kp-ratio 1.0`: the default is **0.6** (it lowers kp to 120 during the trick).
  We train at kp=200 → we must force **1.0** to match.
- `--ground-pick-action-scale` must match the training `action_scale` (0.8 above).
- `--ground-pick-period 5.0` must match the trained period/motion length
  (the default is 4.0; we keep ours).

## Risks and verification

- **One-shot, fixed duration:** the crouch lasts `ground-pick-period` then rises by itself.
  No free hold — an accepted limitation of approach B.
- **Momentum during the trick:** the phase replaces the velocity command → **no active push**
  during the crouch. If the entry momentum is too low, it slows down. Hence
  training with a realistic entry velocity.
- **Verification:**
  1. In sim (`play`): it goes down, keeps the wheels turning during the plateau,
     stands back up without falling, and the final pose cleanly rejoins the standing roller pose.
  2. On the real robot: launch at low speed, press A, observe.
  3. Confirm that the roller policy cleanly takes back control after the return.

## Open questions / to confirm during implementation

- The exact value of `height_low` (crouched) — to be tuned at play time.
- The best way to inject the entry velocity at episode start (reset event vs initial push).
- The relative weight of `wheel_speed` vs `crouch_glide_height_by_phase` (keep the momentum
  without preventing the crouch).
