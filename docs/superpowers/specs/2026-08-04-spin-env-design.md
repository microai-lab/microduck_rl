# Spec — "Spin" env (fast spin in place, on rollers)

Date: 2026-08-04. Branch: `new_pre_alpha_rollers`.

> **Amendment (after the first run)**: the first calibration run (500 it.)
> showed that the robot falls systematically around 1.16 s, well before the
> braking segment. In response, the target was halved — `SPIN_RATE_MAX`
> 6.0 → **3.0 rad/s**, i.e. **1 turn per cycle instead of 2** — and
> `spin_stay_in_place` strengthened to **−3.0**, **without a curriculum** on speed.
> See "Initial verification results" for the evidence and the configuration
> currently in force.

## Goal

A new RL task that teaches the microduck on rollers to perform a **spin**:
~2 counter-clockwise turns in place at ~6 rad/s (360°/s) *(the initial target; reduced
to 3 rad/s, see the amendment)*, then a clean stop standing up.
A **cyclic, phase-driven gesture**, deployed in a **one-shot button slot**
of the runtime, like the existing `roller_crouch` task.

## Settled decisions

| Question | Decision |
|---|---|
| Support | On rollers (`MICRODUCK_WALK_ROLLERS_ROBOT_CFG`, 4 passive wheels) |
| Steering | One-shot button slot, command = phase `[cos(2πφ), sin(2πφ), 0]` |
| Target | ~6 rad/s, 2 turns, then braking to a stop (the initial target; reduced to 3 rad/s, see the amendment) |
| Entry state | At a standstill **or** rolling slowly (0 → 0.3 m/s) |
| Direction | Left only (positive yaw, counter-clockwise) |
| Approach | "Outcome" objective (tracking ω_z) + a decaying antisymmetric priming term |

**Runtime constraint**: the slot only sends `[cos, sin, 0]` — no free channel
for the rotation direction. The policy therefore **always spins left**. A mirrored
policy could later go into another slot (button B, `--fold-policy`).

## Target physical mechanism

On 4 passive wheels, a "clean" spin in place is achieved through **differential
rolling**: the left skate goes backward, the right one forward (the wheels **roll**,
they do not skid). This is an *antisymmetric swizzle*: the legs do the opposite of
each other, instead of the classic swizzle's mirror.

Sign check for a counter-clockwise rotation (frame: x forward, y left,
z up; ω_z > 0): a point on the left (+y) has velocity `ω ẑ × y ŷ = −ω y x̂`,
i.e. **backward**. All 4 wheels spin positive when moving forward (verified by
`test_wheel_direction.py`), so for a counter-clockwise spin:
`ω_left_wheels < 0`, `ω_right_wheels > 0`, i.e. **`ω_R − ω_L > 0`**.

## Chosen approach (C) and why

Three approaches were considered:

- **A — pure "outcome" objective**: reward the yaw rate and let PPO find the gesture.
  Risk documented in this repo: a lazy optimum /
  skid-hopping instead of clean rolling.
- **B — "directive" objective through poses**: two scissor poses interpolated along the
  phase, like `roller_crouch`. Works quickly *if* the poses are good; but for the
  crouch they were **read off the real robot**, whereas here the gesture is unknown.
  It would have to be composed by hand: expensive and risky (poses with no useful
  torque produce nothing).
- **C — A + a decaying antisymmetric priming term** ← **chosen**. A's structure, plus
  two weak *shaping* terms that inject the only certain physical knowledge (differential
  rolling), and whose weight decays by curriculum so the policy can refine its own
  gesture. The **pumping frequency stays free**.

## Architecture

**File**: `src/mjlab_microduck/tasks/microduck_spin_env_cfg.py`
- factory `make_microduck_spin_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg`
- PPO config `MicroduckSpinRlCfg`
- task id `Mjlab-Spin-Flat-MicroDuck`, registered in `tasks/__init__.py`

Clones the structure of `microduck_roller_crouch_env_cfg.py`: roller robot, unified 61D
obs, full DR, `action.scale = 1.0`, flat terrain.

**`ENABLE_SYMMETRY = False`** — mandatory: left/right symmetry augmentation
would turn a left spin into a right spin and destroy learning.

**Command**: `GroundPickPhaseCommandCfg(period=4.0, randomize_phase=False)`.
`period=4.0` is the default of `--ground-pick-period` → nothing to pass at runtime.
`randomize_phase=False` → every episode starts at φ=0 (standing), like at deployment.

## Phase envelope

The phase drives a **target yaw rate** ω\*(φ), trapezoidal over 4 segments
(4 s period, `SPIN_RATE_MAX = 6.0` rad/s — the initial target; reduced to 3 rad/s,
see the amendment; the segments and the period are unchanged):

```
ACCEL_END = 0.125   [0,     0.125)  0.5 s  ω*: 0 → 6 rad/s   (launch, linear ramp)
HOLD_END  = 0.525   [0.125, 0.525)  1.6 s  ω* = 6 rad/s       (steady state)
BRAKE_END = 0.650   [0.525, 0.650)  0.5 s  ω*: 6 → 0          (braking, linear ramp)
            1.0     [0.650, 1.0)    1.4 s  ω* = 0             (standing rest)
```

*(The ω\* = 6 rad/s values above correspond to `SPIN_RATE_MAX` = 6.0, the
initial target; see the amendment for the value in force.)*

Integral over one cycle: `0.5·3 + 1.6·6 + 0.5·3 = 12.6 rad ≈ 2.0 turns`. ✅
*(at `SPIN_RATE_MAX = 6.0`, the initial target.)* General form: the integral equals
`2.1 × SPIN_RATE_MAX` whatever `rate_max` is (0.25 + 1.6 + 0.25 = 2.1). With the
target in force (3.0 rad/s): `2.1 × 3.0 = 6.3 rad ≈ 1 turn` per cycle — see
the amendment.

Episode = 20 s = **5 cycles**: the robot repeats launch → steady state → braking → rest
five times per episode. More data per episode, and the "rest" segment also trains
the clean exit from the trick. **Note (post-run)**: this remains true
geometrically (20 s / 4 s), but no episode of the calibration run survived
beyond ~1.16 s, i.e. only a fraction of the first cycle — see
"Initial verification results".

**Pure function** `spin_rate_by_phase(phase, rate_max, accel_end, hold_end, brake_end)`
in `mdp.py`, next to `crouch_pose_blend`. Testable without a simulator.

**Shaping gate**: `gate(φ) = spin_rate_by_phase(φ) / rate_max ∈ [0, 1]`. It is 0
over the rest segment → no priming term pushes toward the scissor at that moment, so the robot
returns to a neutral stance. That is what gives a clean trick exit back to the roller
policy.

## Rewards

### Pitfalls verified in mjlab (to be handled explicitly)

- `body_ang_vel` (`body_angular_velocity_penalty`) only penalizes **x/y**
  (`ang_vel_xy`, comment "Don't penalize z-angular velocity") → **kept**
  (weight −0.05): it suppresses roll/pitch wobble without hindering the spin.
- `angular_momentum` (`angular_momentum_penalty`) penalizes the **3D norm** of angular
  momentum → it would fight the spin head-on. **Removed.**

### New rewards (to be written in `mdp.py`)

| Reward | Weight | Definition |
|---|---|---|
| `spin_rate_track` | 6.0 | `exp(−((ω_z − ω*(φ))/std)²)`, `std = 1.5` rad/s. ω_z = trunk yaw in the body frame (what the IMU sees). Main objective. |
| `spin_rate_l1` | 0.5 | `−|ω_z − ω*(φ)|`: constant-gradient bootstrap when the gaussian saturates far from the target (the same trick as `crouch_glide_pose_l1`) |
| `spin_stay_in_place` | −3.0 (initially −1.0, see the amendment) | trunk `‖v_xy‖²` → "in place", and kills the entry momentum. No reference state, hence robust across the 5 cycles per episode |
| `spin_wheel_differential` | 1.0 | `gate(φ) · tanh(clamp(ω_R − ω_L, min=0) / omega_scale)` with `ω_L = (LF+LR)/2`, `ω_R = (RF+RR)/2`: rewards skates rolling in opposite directions consistent with counter-clockwise → spinning **by rolling**, not by skidding. Wheels resolved by name (`passive_LF_?wheel`, …). `omega_scale = 17.0` rad/s in force (see the calibration paragraph below) |
| `leg_antisymmetry` | 1.0 → 0.25 | `gate(φ) · (−mean|q_L − q_R|)` on `hip_pitch` and `knee`. ⚠️ mirror convention: a *symmetric* pose gives `q_L + q_R ≈ 0`, so the **scissor** is `q_L ≈ q_R`. Decays by curriculum |
| `spin_grounded` | 0.5 | `gate(φ) · 1[n_contact ≥ 2]`: both blades on the ground, prevents "jump and twist in mid-air". The swizzle's `grounded_reward` is not reusable as-is (it weights itself by `cmd_x`, which here equals `cos(2πφ)`) |

**Calibrating `omega_scale`** (the tanh saturation scale): at the target steady state,
each skate advances at `v = ω_z · half_track`, so each wheel spins at
`v / r` with `r = 0.0175` m, and the differential equals `2 · ω_z · half_track / r`.
The leg roots are at `y = ±0.0175` m in the roller model, but the skates
are further apart (ankle offset): the real half-track has to be **measured on the
`left_foot` / `right_foot` sites in sim** on the first run. With a half-track
estimated at ~0.03 m and `ω_z = 6` rad/s, the expected differential was ~20 rad/s — hence
the initial default `omega_scale = 20.0`. **Measurement done (Task 3): real half-track
= 0.0499 m, expected differential = 34.2 rad/s, i.e. 71% above the estimate
— beyond the 30% threshold set by the plan.** `SPIN_WHEEL_OMEGA_SCALE` was therefore
corrected to **34.0** (an intermediate value, in force while the target was at
6 rad/s; since recalibrated to **17.0**, see the "Update" paragraph
just below). See the "Initial verification results" section
below for the details of the half-track measurement.

**Update (post-review fix wave)**: `SPIN_RATE_MAX` was reduced from 6.0 to
**3.0 rad/s** (a human decision, without a curriculum — see below). A direct mechanical
consequence for `omega_scale`, not an independent choice: the expected differential
at steady state becomes `2 · 3.0 · 0.0499 / 0.0175` = **17.1 rad/s**. Leaving
`omega_scale = 34.0` would cap the term at `tanh(17.1/34) = 0.47` of its own
maximum, which would weaken exactly the shaping we are trying to strengthen.
`SPIN_WHEEL_OMEGA_SCALE` is therefore re-corrected to **17.0**, with the same measured
half-track (0.0499 m) kept as the reference.

### Rewards taken from `roller_crouch` (stability / sim2real)

| Reward | Weight |
|---|---|
| `upright` (vertical trunk) | 2.0 |
| `feet_flat` (blades flat) | −2.0 |
| `self_collisions` | −1.0 |
| `body_ang_vel` (xy only) | −0.05 |
| `action_rate_l2` | −1.0 (curriculum −0.5 → −1.0) |
| `neck_action_rate_l2` | −0.5 |
| `joint_torques_l2` | −1e-3 |
| `neck_joint_pos_l2` **excluding `head_yaw`** | −0.2 |

**The head**: neck pitch/roll held near neutral (sim2real), but
`head_yaw` is **excluded** from the term → free to act as a flywheel to launch the
rotation. Implementation: `neck_joint_pos_l2` resolves its joints by a hardcoded
`.*(neck|head).*` regex; we must therefore either add a regex parameter to that
function or write a `neck_joint_pos_l2_no_yaw` variant. Choice: **add a
`pattern` parameter** to `neck_joint_pos_l2` (default unchanged) so as not to duplicate.

## Reset / entry state

```python
cfg.events["reset_base"].params["pose_range"]["z"] = (0.1335, 0.1435)
cfg.events["reset_base"].params["velocity_range"] = {"x": (0.0, 0.3)}
```

Injected through `reset_root_state_uniform`. **Never** `push_by_setting_velocity` in
`mode="reset"`: that is what produced the NaNs on the crouch (`root_vel +=` on
a potentially divergent root velocity → the base free joint blows up).

## Domain randomization

Identical to `roller_crouch`, with no deviation (the repo's validated sim2real recipe): trunk +
head CoM, mass/inertia, BAM joint friction, armature, wheel friction,
0.2 m/s pushes every 3–6 s, 6° IMU misalignment, ±0.015 rad encoder bias.

## Observations

The **61D layout, identical** to roller / ground_pick / crouch — the condition for the
ONNX to load in the slot:
`[gyro(3), projected_gravity(3), joint_pos(14), joint_vel(14), last_action(14), command(13)]`
with `command = [twist(3), head_pose(4), body_pose(6)]`, head/body zero-padded.

Hence: `base_lin_vel` removed from the actor (kept on the critic side), `height_scan` and
`foot_height` removed, `wheel_vel` on the critic side, passive joints excluded from the
`joint_pos`/`joint_vel` terms, delays and noise identical to the crouch.

The gyro is in the obs → the policy **observes** its own ω_z: the task is observable.

## Terminations

`time_out`, `fell_over`, `out_of_terrain_bounds` (inherited) + `nan_state`
(`microduck_mdp.robot_state_is_nan`), like the crouch.

## Curriculum

| Term | Stages |
|---|---|
| `action_rate_weight` | −0.5 (0) → −0.8 (250 it.) → −1.0 (500 it.) |
| `leg_antisym_weight` | 1.0 (0) → 0.5 (1500 it.) → 0.25 (3000 it.) |
| `com_range` | 0.003 → 0.005 (500 it.) → 0.01 (1000 it.) |
| `head_com_range` | 0.003 → 0.005 (500 it.) → 0.01 (1000 it.) |

(iterations × 24 steps/env, like the other envs)

**No curriculum on the target speed**: 6 rad/s from the start *(the initial target;
reduced to 3 rad/s, still without a curriculum, see the amendment)*. See "Plan B".

## PPO

`MicroduckSpinRlCfg` = a copy of `MicroduckRollerCrouchRlCfg`: actor/critic
(512, 256, 128) elu, obs normalization, adaptive PPO lr 1e-3, `desired_kl=0.01`,
`num_steps_per_env=24`, `symmetry_cfg=None`, `experiment_name="spin"`,
`run_name="spin"`, `max_iterations=8000`.

## Tests

`tests/test_spin.py` — pure functions, no simulator:
- `spin_rate_by_phase`: values at the boundaries of the 4 segments (0, rate_max, rate_max, 0, 0)
- increasing monotonicity on the launch ramp, decreasing on the braking ramp
- **integral over one cycle ≈ 4π** at `rate_max = 6.0` (guarantees the **shape** of the
  trapezoid, `2.1 × rate_max` rad per cycle) — no longer protects the target in force
  since the amendment, cf. the next bullet. Exact envelope value: 12.6 rad
  against 4π = 12.566 → 1% tolerance
- **the target actually shipped** (`mdp.SPIN_RATE_MAX`) does integrate to
  `2.1 × SPIN_RATE_MAX` rad per cycle, whatever `rate_max` is — added in
  7d916aa, this is the test that fails if the target changes without anyone thinking about the
  number of turns. With the value in force (3.0 rad/s): 6.3 rad ≈ 1 turn
- `gate(φ) = 0` over the whole rest segment, `∈ [0,1]` everywhere

`tests/test_spin_cfg.py` — the env builds:
- command = `GroundPickPhaseCommand`, `period == 4.0`, `randomize_phase is False`
- `"angular_momentum" not in cfg.rewards` (the pitfall from the rewards section)
- `symmetry_cfg is None`
- actor obs dimension == 61
- **exact parity of the observation term ordering** (actor + critic) with
  `roller_crouch`, group by group — added in 7d916aa, a strict condition for
  the exported ONNX to load in the runtime slot

Run: `uv run --with pytest pytest tests/ -q`

## Training / deployment

```bash
uv run train Mjlab-Spin-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max_iterations 8000
# watch Episode_Reward/spin_rate_track (it must rise)
uv run scripts/play_latest.py     # alias md-play
uv run scripts/export_latest.py   # ONNX, obs normalizer baked in
```

```bash
microduck_runtime --variant pre-alpha --new-cmd-obs --roller \
  --model output.onnx --new-dxl-imu --kp 200 --action-scale 0.8 \
  --ground-pick spin.onnx \
  --ground-pick-period 4.0 \      # = SPIN_PERIOD
  --ground-pick-kp-ratio 1.0 \    # default 0.6 -> force 1.0 (trained at kp 200)
  --ground-pick-action-scale 0.8  # match the runtime action_scale
```

Button **A** → spin, then automatic return to the roller policy.

## Success criterion

At play time: ~2 counter-clockwise turns in ~2.6 s, trunk drift < ~10 cm, robot upright
throughout, a stable neutral stance during the rest segment before the next cycle.
*(Criterion formulated for the initial target of 6 rad/s / 2 turns; at 3 rad/s, see
the amendment, it would be ~1 turn over the duration of the steady state — criterion not revised, as the
robot does not yet stay up that long.)*

## Plan B if training plateaus

In order:
1. **Speed curriculum**: `SPIN_RATE_MAX` 3 → 6 rad/s (requires making
   `rate_max` drivable by a `CurriculumTermCfg` on the reward params).
   **Partially followed**: after the calibration run, the target was indeed
   lowered to 3 rad/s (see the amendment), but **without a curriculum** — 3 rad/s
   is for now a fixed target, not a starting point ramping toward 6.
   The human chose to first see what the robot manages to do at that
   speed before considering a gradual increase.
2. Raise `spin_wheel_differential` and delay the decay of `leg_antisymmetry`.
3. Widen the `std` of `spin_rate_track` (1.5 → 2.5) for a useful gradient further out.
4. As a last resort, switch to approach B (scissor poses composed by hand
   in a pose editor) to prime the gesture, then release it.

## Out of scope

- Spinning right (a mirrored policy in another slot) — later.
- A footed variant (without rollers).
- A continuously speed-commanded spin (would require a runtime command channel).

## Initial verification results

### Measured half-track and `omega_scale`

The half-track was measured on the `left_foot` / `right_foot` sites of the roller
model: **0.0499 m**, against the spec's estimate of 0.03 m. Expected wheel
differential at steady state (6 rad/s): `2 · 6.0 · 0.0499 / 0.0175` = **34.2 rad/s**,
i.e. 71% above the 20.0 default — beyond the 30% threshold set by the plan.
`SPIN_WHEEL_OMEGA_SCALE` was therefore changed from 20.0 to **34.0**. The tests keep
passing `omega_scale=20.0` explicitly, to stay independent of the
constant.

### Smoke run (Step 2: 5 iterations, 64 envs, NaN guard)

Completed without an exception. `Episode_Termination/nan_state` stayed at 0.0000 for
the whole duration, and `/tmp/mjlab/nan_dumps/` was never created. The six spin
rewards do appear in the logged `Episode_Reward/` keys: `spin_rate_track`,
`spin_rate_l1`, `spin_stay_in_place`, `spin_wheel_differential`, `spin_grounded`,
`leg_antisymmetry`.

Observation parity (Step 1): the list of terms in the spin env's actor obs
is **identical** to `roller_crouch`'s — 8 terms, same order:
`base_ang_vel, projected_gravity, joint_pos, joint_vel, actions, command,
head_command, body_command`. That is the condition for the exported ONNX to load
in the runtime slot.

**Usage note worth remembering**: the plan's example command with `--enable-nan-guard`
as a bare flag is rejected by this repo's CLI — you must pass
`--enable-nan-guard True`.

### 500-iteration calibration run (Step 3)

4096 envs, 500 iterations, ~2.32 s/iteration, exit code 0, wandb logger (so
`scripts/play_latest.py` / `md-play` finds the run).

**What was actually established**: `Mean episode length` = **57.83 steps** out of a
1000-step episode (20 s at 50 Hz), i.e. **~1.16 s**. `Episode_Termination/fell_over`
≈ **70**, `time_out = 0.0000`, `nan_state = 0`. The robot **falls every episode**,
at a phase φ ≈ 0.29 — right in the middle of the steady-state segment. It never reaches the
braking (φ ≥ 0.525) nor the rest (φ ≥ 0.650) segment: **71% of the cycle is never
trained**.

Episode length went from 23.98 to 57.83 steps over the run: the
rise in `Episode_Reward/spin_rate_track` (0.0291 → 0.3168) therefore mainly reflects
**lengthening survival**, not improving tracking. The
success criterion for this step as stated in the plan ("the curve must
rise") **is not a valid signal** for that term: a completely
motionless robot already scores `6.0 × 0.405 = 2.43` on it — the rest segment pays full
price for staying upright without moving, so any policy that survives longer
mechanically captures more of that segment, independently of tracking quality.

### Derived diagnosis — estimates, not direct measurements

The values below come from the ratio between reward terms in the last
log block, which cancels the unknown normalization factor applied by the
logger. To be taken as estimates, reproducible from the same
method:

**What holds up**: during the ~1.2 s it stays upright, the robot tracks the target
fairly closely. The `spin_rate_l1 / spin_rate_track` ratio (−0.0097 / 0.3168, weights 0.5
and 6.0, `std = 1.5`), solving `e = 0.3674 · exp(−(e/1.5)²)`: a mean absolute
yaw-rate tracking error of ≈ **0.35 rad/s**, confirmed by two independent
routes — that `spin_rate_l1 / spin_rate_track` ratio, and a back-calculation from
the reward manager's normalization. It **can launch** the spin; it **cannot
stay upright** while doing so.

**What does not hold up**: the shaping block (`spin_wheel_differential` 1.0,
`spin_grounded` 0.5, `spin_stay_in_place` −1.0) totals ~1.0 of weight against 6.0
for the main objective — about **13%** of what a skidding policy
would give up by ignoring that block. And `spin_wheel_differential` is
**invariant to the instantaneous center of rotation**: a centered spin at 6 rad/s and a
pivot on the left skate at 6 rad/s both produce a differential of
34.2 — so that term does **not** encode centered rolling, only
`spin_stay_in_place` does. `spin_stay_in_place` ≈ −0.0069 implies
`‖v_xy‖ ≈ 0.35 m/s`: the robot is still translating, consistent with an off-center
pivot (a skate as the pivot) rather than a rotation about the body center.

### Configuration change decided from this diagnosis

Target halved — `SPIN_RATE_MAX` 6.0 → **3.0 rad/s** — and
`spin_stay_in_place` strengthened −1.0 → **−3.0** (see the rewards table and
`SPIN_WHEEL_OMEGA_SCALE` recalibrated to 17.0 above). **Deliberately without a
curriculum** on the target speed: this is a first attempt to see what the
robot manages at half speed, before considering a gradual increase
if needed.

**Attenuating the drift cost during the launch.** Strengthening
`spin_stay_in_place` to −3.0 sharpened a flaw raised in review: that
term was the only one in the spin not modulated by the phase, so it charged full
price for the transient translation during the launch ramp — precisely
the moment when the robot must push against the ground to inject angular momentum, and
when the entry momentum (up to 0.3 m/s) must be **converted** into rotation. The cost
is now multiplied by `SPIN_LAUNCH_DRIFT_SCALE = 0.2` over `[0, ACCEL_END)`
and is full price afterwards. It is deliberately **not** switched off during the
rest segment, unlike the priming terms: that is where stillness is the real criterion.

Step 4 (watching the gesture) remains to be done, reserved for the human.

⚠️ These four tests (three new ones on the attenuation, one modified) were **not**
run — the machine was reserved for something else at commit time. To be
run before any long run: `uv run --with pytest pytest tests/test_spin.py
tests/test_spin_cfg.py -q`.
