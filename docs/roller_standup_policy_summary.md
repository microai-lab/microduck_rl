# Policy `roller_standup` — standing up on rollers

**Goal**: the microduck (on rollers) starts on the ground — face down or face up — and gets back **up on its wheels**, then **holds** the stance.

- **Task**: `Mjlab-RollerStandUp-Flat-MicroDuck`
- **File**: `src/mjlab_microduck/tasks/microduck_roller_standup_env_cfg.py`
- **Base**: derived from the roller env (`velocity_rollers`) → same robot, same physics/DR, **same 61D observation** (hot-swappable at runtime, loadable via `--new-cmd-obs`).
- **Spec**: `docs/superpowers/specs/2026-08-04-roller-standup-design.md`
- **Blind policy**: no terrain scan; proprioception + `projected_gravity`.

## Heights (measured, not guessed)

| pose | feet model | roller model |
|---|---|---|
| standing | 0.1172 → `STAND_Z=0.115` under load | 0.1407 → **`ROLLER_STAND_Z=0.138`** |
| face down (rest) | 0.075 | 0.075 |
| face up (rest) | 0.048 | 0.048 |

The ground rest heights are identical on both models: it is the trunk shell that touches, not the feet.

## ⚠️ Joint indices — the wheels are INTERLEAVED

```
0-4   left leg          5-6   left wheels
7-10  neck / head      11-15  right leg          16-17  right wheels
```
`_LEG_JOINTS = [0-4, 11-15]`. The `standup` indices (`[0-4, 9-13]`) hold for the model **without** wheels and would point at wheels here. Locked in by `tests/test_roller_standup_cfg.py::test_joint_indices_match_actual_roller_model`.

## Reset — starting on the ground

`set_random_ground_state`: face down (`prone_z` 0.076–0.09, floor raised because face down only clears the ground at 0.0752) / face up / **already standing** (`standing_z` 0.134–0.144), ± 10° of pitch/roll noise. No "sitting" bucket. The "standing" bucket is necessary: without it the policy rises but does not hold.

**Curriculum `ground_state_mix`** (easy → hard, face up last):

| iter | standing | face down | face up |
|---|---|---|---|
| 0 | 0.50 | 0.50 | 0.00 |
| 600 | 0.35 | 0.45 | 0.20 |
| 1500 | 0.25 | 0.40 | 0.35 |
| 2500 | 0.20 | 0.40 | 0.40 |

## Rewards

Ten terms taken from `standup` with their already-tuned weights: `pose_stand_legs` (+8), `pose_stand_l1` (+5), `height_stand` (+4, std 0.04), `height_stand_sharp` (+4, std 0.015), `height_stand_l1` (+30), `com_upward_velocity` (+3), `gentle_rise` (−0.02), `upright_linear` (+6), `upright_sharp` (+6), `standing_composite` (+15). Plus `joint_torque_rate_l2` (−2e-3), the anti-jitter term that does not block the roll-over.

Inherited regularizers: `body_ang_vel` **−0.05** (motion blocker, keep it LIGHT), `angular_momentum` −0.02, `action_rate_l2` (ramp −0.4 → −1.0, **not** the roller's −2.0), `neck_action_rate_l2` −0.5, `neck_joint_pos_l2` −0.5 (head upright), `joint_torques_l2` −1e-3, `action_over_limit` −0.5, `self_collisions` −1.0.

Removed: all the skating rewards, plus `feet_flat` (the blades are not flat during the rise) and `hip_roll_neutral` (standing up requires spreading the legs).

## ⚠️ The hard part: the wheels roll

There is no longitudinal traction to push against the ground. The **rolling-friction curriculum is REVERSED** (the roller env ramps it up, here it goes down):

| iter | frictionloss | |
|---|---|---|
| 0 | 0.05 | near-locked wheels → rises as if on feet |
| 1000 | 0.02 | |
| 2000 | 0.008 | |
| 3000 | 0.003 | |
| 4000 | 0.0015 | the real rolling value |

**Watch `Episode_Reward/standing_composite` at each stage.** If it collapses, the "sticky feet" gesture does not transfer to free wheels → we will have to guide a skater technique (intermediate knee support, one skate at a time). That is a result, not a failure.

**ALSO watch the robot's horizontal drift at play time**, at every friction stage. `standing_composite` sees neither `root_link_pos_w[:2]` nor horizontal velocity: a policy that stands up while sliding far from its starting point collects exactly the same score as one that stands up and stops. Until that drift has been measured visually, the outcome of the friction curriculum (the very question this env exists to settle) is not trustworthy.

**Sim2real**: only checkpoints from after iter 4000 are deployment candidates. Before that, the policy relies on a friction that does not exist on the real robot.

## Command

The `twist` slot is neutralized: `lin_vel_x`/`lin_vel_y` ± 0.01, `ang_vel_z` **± 0.05** (5× wider — the same
choice as `standup`). The `head_pose` / `body_pose` slots are **zero-padded** (roller convention). Intended deployment: in `--standing` alongside the roller policy in `--walking`, with the automatic switch on command magnitude (`infer_policy.py:262`, threshold 0.05); the twist slot is left at zero there (`infer_policy.py:239`).

**Caveat**: `infer_policy.py` is the local sim/keyboard script. The robot runtime is the Rust binary `microduck_runtime`, absent from this repo — it has not been verified that it exposes a `--standing` equivalent. The crouch handover doc only lists `--model`, `--ground-pick`, `--fold-policy`. To be confirmed.

## Terminations

`fell_over` **removed** (the robot starts fallen). `nan_state` inherited. `nan_policy="sanitize"` on the actor/critic obs.

## Network / PPO

Actor and critic `(512, 256, 128)` elu, `obs_normalization=True`. PPO `lr=1e-3` adaptive, `desired_kl=0.01`, `gamma=0.99`, `lam=0.95`, `num_steps_per_env=24`, 6 s episode, `max_iterations=15000`. **Symmetry OFF** (`SYMMETRY_CFG` is wired for the 51D layout).

## Commands

```bash
uv run train Mjlab-RollerStandUp-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max_iterations 15000
uv run scripts/play_latest.py        # alias md-play
uv run scripts/export_latest.py      # alias md-export
uv run --with pytest pytest tests/test_roller_standup_cfg.py -q
```

### ⚠️ Seeing face-up starts at play time

A play run **never** shows a face-up start by default: the play env is
rebuilt from scratch, so `common_step_counter` restarts at 0 and the curriculum applies its
stage 0, where `face_up_prob = 0`. You only see 50% face down / 50% standing, whatever
the maturity of the loaded checkpoint. Yet face up is the hardest case, the very one we
want to inspect.

`STANDUP_PLAY_FACE_UP` forces the mix (same pattern as `SLOPE_PLAY_DIFFICULTY` in
`roller_slope`), **on the `play=True` path only** — training and its easy → hard
curriculum are untouched:

```bash
STANDUP_PLAY_FACE_UP=1.0 md-play    # 100% face-up starts
STANDUP_PLAY_FACE_UP=0.4 md-play    # the mix of the curriculum's last stage
STANDUP_PLAY_FACE_UP=none md-play   # default (stage 0, no face up)
```

The remainder (`1 - face_up`) is split face-down:standing in that last stage's 2:1 ratio,
so `0.4` reproduces the end-of-training mix exactly (0.40 / 0.20 / 0.40).

## 🔧 Anti-violence fix (after the first robot test)

**Symptoms** on a 4000+ checkpoint: very abrupt motions, the head banging the ground,
failure to rise from the back on the robot. **Present in sim too** → so this was
neither a sim2real issue nor a too-young checkpoint, but the reward design.

**Root cause: `gentle_rise` was rewarding violence.** `trunk_vertical_accel_penalty`
already returns `-|a_z|` (`mdp.py:2171`); multiplied by the **−0.02** weight inherited from
`standup`, that made a double negative, hence `+0.02·|a_z|` — **the more brutally the trunk
accelerated, the more the policy was paid**. Confirmed by the log: `Episode_Reward/gentle_rise
= +0.0118` on run `vweolw91`, the only penalty term logged positive.

`mdp.py` mixes two sign conventions, and that is the trap:

| term | the function returns | correct weight |
|---|---|---|
| `height_stand_l1`, `pose_stand_l1`, `gentle_rise` | `-abs(...)`, already negative | **positive** |
| `joint_torques_l2`, `joint_torque_rate_l2`, `action_rate_l2`, `body_impact_cost` | positive magnitude | **negative** |

Locked in by `test_already_negative_penalties_use_positive_weights`.

⚠️ **The walker's `standup` has exactly the same bug** (same function, same −0.02 weight).
That explains the series of unsuccessful damping attempts documented in its
comments ("*violent / shaky / overshoot-tip-repeat on the real robot*"): they were
fighting a term that actively pushed the other way. **Not fixed here** — that is
a different env, to be settled separately.

**Associated structural problem.** At convergence the task rewards totalled **≈ +41.6**
saturated at 95–99%, against **≈ −1.2** for all the dampers combined — of which
`joint_torque_rate_l2` at **−0.0002/step** and `joint_torques_l2` at **−0.0001/step**, i.e. nothing.
A ~35:1 ratio: no reason to be gentle.

**Current state of the fixes:**

| | before | now | why |
|---|---|---|---|
| `gentle_rise` | −0.02 (reward) | **+0.02** (penalty) | sign fixed; magnitude deliberately kept SMALL — `\|a_z\|` is necessarily high during a roll-over, and a large weight would be a motion blocker |
| `joint_torque_rate_l2` | −2e-3 | **−0.2** | the SAFE lever: it penalizes torque rate, not motion |
| `head_impact_penalty` | absent | **still absent** | tried at −1.0, froze the policy — see below |

### ⚠️ The head-impact penalty froze the policy — do not put it back as-is

Attempt with `velstand`'s values (`body_impact_cost`, `neck` subtree, −1.0,
threshold 2.0): **the policy converged to lying down, inert.** Measured (run `d8rnko6p`):

| term | before (violent) | with head_impact (frozen) |
|---|---|---|
| `standing_composite` | +14.32 | **+3.26** |
| `upright_sharp` | +5.76 | +1.06 |
| `head_impact_penalty` | — | **−1.01** ← largest negative term |
| `joint_torque_rate_l2` | −0.0002 | −0.255 (so **not** the culprit) |

The reasoning error: believing that a "targeted" penalty does not restrict motion.
**False here — to get up from its back, this robot pivots on its head and shoulders.** The head
is the fulcrum of the roll-over, not collateral damage; penalizing it blocks the only
available mechanism, and face up was already the failing case.

**The lazy optimum that makes this freeze possible**: `pose_stand_legs` stayed at **+7.72 out of 8**
while the robot was lying flat — the legs are at HOME in a prone position, so that
reward is collected almost for free. It is `height_stand_l1` (weight +30) that must
make "stay on the ground" net negative; it must not be weakened.

**Hypothesis under test**: head banging was a *symptom* of the violence (the sign bug
paid for brutality, and a brutal rise ends on the head), not a separate defect.
If the slam comes back now that the sign is fixed, the reintroduction must be a
**height-gated** penalty (as `upright_sharp` is), which spares the ground roll-over phase.

**Methodological lesson**: the three fixes were applied all at once, so the freeze could not
be attributed with certainty — only the most likely suspect could be named. One
fix at a time in future.

**Recalibration if it is still violent**: `|Δτ|²` is ~0.1 at convergence, so
`joint_torque_rate_l2`'s contribution ≈ `0.1 × |weight|`. Raise **that** term, **not**
`body_ang_vel` (−0.05) nor `action_rate_l2` (ramp → −1.0): those are motion
blockers, and `standup` documents that at −0.15 and −1.2 respectively they **froze** the
rise from the back. If instead face up stops working, **lower**
`joint_torque_rate_l2` first.

## Out of scope

Integrating the standup into the rolling policy (the `velstand` recipe); side-lying start buckets; a rough variant; trunk/head impact penalties.

No reward penalizes the trunk's horizontal velocity (`root_link_lin_vel_w[:, :2]`): "standing up while rolling far away" is an unpenalized outcome that scores full marks. This is a deliberate decision (not an oversight): a stillness reward that was not height-gated would also penalize the translation that rising from the ground physically requires — the "motion blocker" failure mode that `standup` documents. Candidate if the problem is confirmed: a height-gated stillness term (close to `ROLLER_STAND_Z` only).
