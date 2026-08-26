# Design — `roller_standup`: standing up on rollers

**Goal**: a dedicated policy that puts the microduck back **upright on its rollers** after a fall
(face down or face up), and that can then **hold** the stance on wheels.

A port of the `standup` recipe (walking duck) to the roller model. No changes to the
existing envs.

---

## Settled decisions

| Decision | Choice | Rejected alternatives |
|---|---|---|
| Form | **Dedicated** episodic policy | Grafting the standup onto the roller env (the `velstand` recipe) → a real risk of breaking the learned gait |
| Start poses | **face down + face up + standing** | `sitting` (only exists for the hand-off from the `sit` policy, no roller equivalent); side-lying (maximum coverage but much harder convergence); without `standing` (the policy would stand up then fall back) |
| Free wheels | **reversed rolling-friction curriculum** | Real entry friction (bootstrap too hard); imposing a skater technique through rewards (repo history: overly directive style rewards create parasitic optima — the swizzle, the crouch's lazy optimum) |
| Target pose | **HOME + measured height** | The roller-crouch's `STAND_POSE` (flagged as an open issue: ≠ the roller neutral → jolt on return); a pose read off the real robot (blocks development) |
| Command | **neutralized twist** (≈ 0) | A phase command / button slot (see "Deployment"); steerable head |

---

## Architecture

**New file**: `src/mjlab_microduck/tasks/microduck_roller_standup_env_cfg.py`
- `make_microduck_roller_standup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg`
- `MicroduckRollerStandUpRlCfg` (`experiment_name="roller_standup"`)
- Task id: `Mjlab-RollerStandUp-Flat-MicroDuck` (flat only, no rough variant)

**Derivation**: `cfg = make_microduck_velocity_rollers_env_cfg()`.

This is `roller_slope`'s pattern (246 lines), not `roller_crouch`'s (479 lines, which starts
over from `make_velocity_env_cfg()` and copies every DR block). We thus inherit without any risk
of drift:

- the `MICRODUCK_WALK_ROLLERS_ROBOT_CFG` robot (14 active joints + 4 passive wheels, BAM m6, kp_fw 200);
- the `feet_ground_contact` (subtree mode on `ankle_{l,r}_v1`) and `self_collision` sensors;
- the whole DR stack: trunk + head CoM, mass/inertia (pseudo_inertia), BAM friction, armature,
  encoder bias, obs-level IMU misalignment, bearing friction;
- **the unified 61D observation** `[gyro(3), projected_gravity(3), joint_pos(14), joint_vel(14),
  last_action(14), command(13)]` — a hard requirement for runtime hot-swappability;
- the `nan_state` termination (widened guard: joints + free joint + wheels).

The roller model **physically allows** lying down: `robot_allcollisions_rollers.xml` carries
collision geoms on the trunk (`np_f970`), the hips, the legs, the head shells and the jaw,
in addition to the 4 tires. Verified.

---

## Measured constants

Measured by exact kinematics (minimum over the mesh vertices of the colliding geoms, the keyframe's
`STAND` pose, trunk lowered to contact) on `scene_rollers.xml` vs `scene.xml`:

| pose | feet model | roller model |
|---|---|---|
| standing (`STAND` = HOME) | 0.1172 | **0.1407** |
| face down (rest) | 0.0752 | 0.0752 |
| face up (rest) | 0.0476 | 0.0475 |

Consistency cross-check: `standup` uses `STAND_Z = 0.115` measured **under load** against 0.1172
kinematically → ~2 mm of sag. We apply the same correction, and the result falls exactly inside
the `reset_base z = 0.1335–0.1435` range already used by the roller env.

```python
ROLLER_STAND_Z   = 0.138   # trunk standing on wheels, under load (+23 mm vs feet)
ROLLER_PRONE_Z   = 0.075   # face-down rest height
EPISODE_LENGTH_S = 6.0
```

The ground rest heights are **identical** on both models (it is the trunk shell that touches,
not the feet). That does not mean `standup`'s `prone_z` range can be reused as-is:
see the note under "Reset" — `prone_z_min` diverges (0.076 here, not 0.05) because a single range serves
two poses (face down, face up) whose contact heights at reset are not the same.

The measured quantity is indeed the one the rewards read: `height_target_gaussian` and
`height_l1_penalty` use `root_link_pos_w[:, 2]`, which equals exactly `xpos[trunk_base].z`
(the free joint sits on `trunk_base`) — verified numerically.

## Joint indices

The passive wheels are **interleaved** in the joint ordering. Actual ordering verified in MuJoCo
(`m.jnt_qposadr`, roller model, 18 joints after the free joint):

```
0-4   left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee, left_ankle
5-6   passive_LF_wheel, passive_LR_wheel
7-10  neck_pitch, head_pitch, head_yaw, head_roll
11-15 right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee, right_ankle
16-17 passive_RF_wheel, passive_RR_wheel
```

```python
_LEG_JOINTS   = [0, 1, 2, 3, 4, 11, 12, 13, 14, 15]   # standup: [0-4, 9-13]
_NECK_JOINTS  = [7, 8, 9, 10]                          # standup: [5-8]
_WHEEL_JOINTS = [5, 6, 16, 17]
```

Only `_LEG_JOINTS` is actually consumed (by the pose rewards). `_NECK_JOINTS` and
`_WHEEL_JOINTS` are declared for documentation and for the index test: the neck is resolved
**by name** (`neck_joint_pos_l2` calls `find_joints(r".*(neck|head).*")` every step, precisely
to be robust to the offset caused by the wheels) and the wheels by the `^passive_.*` regex.

The handover doc flags this fragility explicitly. It is locked in by a test that
builds the env and checks the joint names at those indices (see "Tests").

---

## Rewards

### Removed from the roller inheritance

| Removed | Why |
|---|---|
| `wheel_speed`, `braking`, `skating_air_time`, `glide`, `single_support`, `gait_symmetry`, `forward_lean`, `heading_hold` | stride rewards: meaningless while lying on the ground |
| `feet_flat` | during the rise the blades are not flat → this penalty would fight the gesture |
| `hip_roll_neutral` | standing up requires spreading the legs |
| `pose`, `com_height_target` | replaced by the pose/height targets below |
| `upright` (base gaussian) | replaced by `upright_linear` + `upright_sharp` |

### Kept from the roller inheritance

| Reward | Weight | Role |
|---|---|---|
| `action_over_limit` | −0.5 | sim2real protection (over-commanding past the limits), task-independent |
| `self_collisions` | −1.0 | |
| `body_ang_vel` | **−0.05** | deliberately **light**: `standup` documents that at −0.15 it froze the rise (motion blocker) |
| `angular_momentum` | −0.02 | |
| `action_rate_l2` | curriculum −0.4 → −0.8 → −1.0 | the roller env pins it flat at −1.0; we reuse `standup`'s ramp (gentle at first → helps bootstrap the large roll-over motion) |
| `neck_action_rate_l2` | −0.5 | stable head |
| `neck_joint_pos_l2` | −0.5 | keep the head upright (`roller_slope`'s choice) — **replaces** `standup`'s `head_pose` command |
| `joint_torques_l2` | −1e-3 | |

### Added

| Reward | Weight | Role |
|---|---|---|
| `joint_torque_rate_l2` | −2e-3 | anti-jitter: `standup` identified it as the only damper that does not block the roll-over (it penalizes the torque *rate*, not its magnitude nor trunk rotation) |

### Standup rewards (transplanted from `standup`, remapped)

The ten terms are copied **with their already-tuned weights** from the iterations documented in
`microduck_standup_env_cfg.py`. Only the joint indices and the two heights change.
All the mdp functions already exist — **nothing to write in `mdp.py`**.

| Reward | mdp function | Weight | Roller parameters | Role |
|---|---|---|---|---|
| `pose_stand_legs` | `pose_target_match` | +8.0 | `std=0.5`, `joint_indices=_LEG_JOINTS`, `target_overrides=None` (HOME) | target joint pose |
| `pose_stand_l1` | `pose_l1_penalty` | +5.0 | `joint_indices=_LEG_JOINTS`, `target_overrides=None` | L1 bootstrap: constant gradient even far from HOME |
| `height_stand` | `height_target_gaussian` | +4.0 | `std=0.04`, `target_height=0.138` | wide gaussian → pulls up from the ground |
| `height_stand_sharp` | `height_target_gaussian` | +4.0 | `std=0.015`, `target_height=0.138` | narrow gaussian → forces the last few cm |
| `height_stand_l1` | `height_l1_penalty` | +30.0 | `target_height=0.138` | makes "stay on the ground" net negative (otherwise a lazy optimum) |
| `com_upward_velocity` | `com_upward_velocity` | +3.0 | `max_height=0.148` | pays for the *motion* of rising (+10 mm of margin above the target, like 0.125 vs 0.115 in `standup`) |
| `gentle_rise` | `trunk_vertical_accel_penalty` | −0.02 | | penalizes `\|a_z\|` → smooth, constant-speed rise |
| `upright_linear` | `body_upright_linear` | +6.0 | | `cos(tilt)`: strong gradient while lying down |
| `upright_sharp` | `upright_gaussian_at_height` | +6.0 | `std=0.3`, `height_low=0.075`, `height_high=0.138` | tight height-gated gaussian → kills the backward lean |
| `standing_composite` | `standing_composite_score` | +15.0 | `height_std=0.04`, `upright_std=0.40`, `pose_std=0.40`, `target_height=0.138`, `joint_indices=_LEG_JOINTS` | multiplicative score height × uprightness × pose |

Every term takes `asset_cfg=SceneEntityCfg("robot", body_names=("trunk_base",))` wherever
`standup` does.

**No impact penalties** (trunk/head) for this v1: `standup` has none, only `velstand`
does. We keep the minimal set.

---

## Observation and command

**Observation**: inherited untouched from the roller env (61D). No modification — that is the reason
for deriving from that env.

We add `nan_policy = "sanitize"` on the actor and critic groups, like `roller_slope`: a rare
contact makes the free joint diverge to NaN, the obs is sanitized (→ 0) so training is not killed,
and the offending env resets on the next step.

**Command**: the `twist` slot is neutralized, exactly like `standup`:

```python
command = cfg.commands["twist"]
command.rel_standing_envs = 0.0
command.rel_heading_envs  = 0.0
command.heading_command   = False
command.ranges.heading    = None
command.resampling_time_range = (EPISODE_LENGTH_S, EPISODE_LENGTH_S * 2)
command.debug_vis = False
command.ranges.lin_vel_x = (-0.01, 0.01)
command.ranges.lin_vel_y = (-0.01, 0.01)
command.ranges.ang_vel_z = (-0.05, 0.05)
cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(command))
```

The `head_pose` (4) and `body_pose` (6) slots stay **zero-padded** — the roller family's
convention (`roller`, `roller_crouch`, `roller_slope`). This is a deliberate departure from the
walker's `standup`, which steers the head through a real 4D `head_pose` command (see "Risks").

Rationale for the neutralized twist: in `scripts/infer_policy.py`, the walker's `standup` policy
is loaded as `--standing` alongside `--walking`, and the switch is **automatic on the velocity
command's magnitude** (`infer_policy.py:262`, threshold 0.05); when `standing` is active, the
twist slot is left at zero (`infer_policy.py:239`). The phase slots (`ground_pick`, `fold`)
serve one-shot button-triggered tricks, not a standup.

---

## Reset

We add the `set_ground_state` event (mode `reset`), inserted **after** the inherited `reset_base`
and `reset_robot_joints` (event order follows insertion order in the dict):

```python
cfg.events["set_ground_state"] = EventTermCfg(
    func=microduck_mdp.set_random_ground_state,
    mode="reset",
    params={
        "face_down_prob":  0.50,   # face down — driven by the curriculum below
        "face_up_prob":    0.00,   # face up — introduced late (the hardest)
        "sitting_prob":    0.00,   # no sitting bucket → no joint overrides to remap
        "standing_prob":   0.50,
        "prone_z_min":     0.076,  # cf. the note below — not a simple inheritance from standup
        "prone_z_max":     0.09,
        "standing_z_min":  0.134,  # roller (vs 0.11–0.12 for the feet model)
        "standing_z_max":  0.144,
        "sitting_tilt_max": math.radians(10),  # ± pitch/roll noise; ALSO applies to the standing bucket
    },
)
```

Note: in `set_random_ground_state`, the `standing` bucket reuses the `sitting` bucket's quaternion —
so `sitting_tilt_max` also adds noise to standing starts, which is intended.

**On `prone_z_min` = 0.076 (and not 0.05, a value wrongly carried over from `standup`)**: the face-down
and face-up poses share a single z range, but their measured contact heights differ — face down
0.0752, face up 0.0475 — so a single range cannot be ideal for both. `standup`'s comment justifies
its `0.05` floor with a rest height measured at ~0.044 **after settling under gravity**; but what
matters at reset time is the contact height in the HOME pose, not the rest height after settling.
At 0.05, a face-down start spawns with the trunk shell **sunk 25 mm into the ground**, a pushout the
policy then pays for through `gentle_rise` / `joint_torque_rate_l2`. `prone_z_min = 0.076` eliminates
that interpenetration, at the cost of a face-up start 28–42 mm above its rest height — a far gentler
artifact than a contact pushout.

**No modification to `mdp.py`**: the base's `reset_robot_joints` uses
`joint_names=(".*",)` with `velocity_range=(0.0, 0.0)` and `default_joint_vel` (HOME_FRAME
`joint_vel={".*": 0.0}`) → the 4 passive wheels are already zeroed on every reset. Verified.

**Curriculum `ground_state_mix`** (`event_param_curriculum`), the same easy → hard logic as
`standup`: face up is introduced late and gets the most training at the end.

| iter | standing | face down | face up |
|---|---|---|---|
| 0 | 0.50 | 0.50 | 0.00 |
| 600 | 0.35 | 0.45 | 0.20 |
| 1500 | 0.25 | 0.40 | 0.35 |
| 2500 | 0.20 | 0.40 | 0.40 |

(Steps in units of `common_step_counter` = `iter × 24`.)

**Pushes**: `push_robot` is inherited from the roller env (±0.2 m/s, 3–6 s interval). We add
`standup`'s rising curriculum so as not to disturb the bootstrap: 0 → ±0.08 (iter 500) → ±0.2
(iter 1000).

**Terminations**: `fell_over` removed (the robot **starts** fallen — a tilt termination
makes no sense here). `nan_state` is inherited and kept.

**Terrain**: `plane`. No rough variant for this v1 — consistent with the roller env, which has no
`rough` parameter.

---

## Reversed rolling-friction curriculum

This is the only genuinely new piece of the design, and the heart of the question the task poses:
**the wheels roll, there is no longitudinal traction to push against the ground.**

The mechanism already exists and is inherited (`randomize_wheel_friction` via `dr.dof_frictionloss` on
`^passive_.*` + `wheel_friction_curriculum`). In the roller env it ramps **up** 0 → 0.0015. Here we ramp it
**down**:

| iter | frictionloss | effect |
|---|---|---|
| 0 | 0.05 | near-locked wheels → it rises as if it had feet |
| 1000 | 0.02 | |
| 2000 | 0.008 | |
| 3000 | 0.003 | |
| 4000 | 0.0015 | the real rolling value (the roller env's) |

`wheel_friction_curriculum` simply applies the last stage crossed
(`if env.common_step_counter > stage["step"]`) — it works just as well going down as going up.
**Zero code to write.**

**What this curriculum tells us**: if `Episode_Reward/standing_composite` collapses when the
friction drops, we have a clear answer that the "sticky feet" gesture does not transfer to free
wheels, and we will have to guide a skater technique (intermediate knee support, one skate at a
time). That is an actionable result, not a failure.

---

## Network and PPO

Identical to `standup`: actor and critic `(512, 256, 128)` elu, `obs_normalization=True`
(the normalizer is baked into the ONNX by `export.py`), PPO `lr=1e-3` adaptive schedule, `desired_kl=0.01`,
`entropy_coef=0.01`, `gamma=0.99`, `lam=0.95`, `num_steps_per_env=24`, `save_interval=250`,
`max_iterations=15_000`. **Symmetry OFF** (`SYMMETRY_CFG` is wired for the old 51D layout and breaks
on the 61D one — the same situation as every v1.5+ env).

---

## Tests

`tests/test_roller_standup_cfg.py`:

1. the env builds (`play=False` and `play=True`);
2. **the joint names at the `_LEG_JOINTS` / `_NECK_JOINTS` / `_WHEEL_JOINTS` indices are correct**
   (the lock-in against the interleaved-wheel fragility);
3. the expected standup rewards are present, the skating rewards absent
   (`wheel_speed`, `glide`, `single_support`, `feet_flat`, …);
4. `fell_over` absent, `nan_state` present;
5. the `wheel_friction` curriculum is indeed **decreasing** and ends at 0.0015;
6. the `ground_state_mix` curriculum: the last stage's probabilities sum to 1 and
   `face_up_prob` grows monotonically;
7. **obs parity**: the names and dimensions of the actor/critic terms are identical to those of
   `make_microduck_velocity_rollers_env_cfg()` (otherwise the ONNX will not load in a slot).

Run: `uv run --with pytest pytest tests/ -q`.

---

## Training and deployment

```bash
uv run train Mjlab-RollerStandUp-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max_iterations 15000
```

Watch `Episode_Reward/standing_composite` (it must rise), and above all its behavior **at the
rolling-friction stages** (iters 1000/2000/3000/4000).

Play: `uv run scripts/play_latest.py`. Export: `uv run scripts/export_latest.py`.

Intended deployment: the policy in `--standing` alongside the roller policy in `--walking`, with the
automatic switch on command magnitude. **Caveat**: `infer_policy.py` is the local
sim/keyboard script; the robot runtime is the Rust binary `microduck_runtime`, absent from this repo — it
has not been verified here that it exposes a `--standing` equivalent with the same switching. The
handover doc only lists `--model`, `--ground-pick`, `--fold-policy`. To be confirmed. This changes nothing
about training: if the runtime lacks that slot, the policy remains usable in a button slot (the
command there would be a phase instead of zero — that would then be the only point to revisit).

---

## Risks and points to watch

1. **Standing up on free wheels may be infeasible without a dedicated technique.** That is the main
   risk. The friction curriculum is designed to settle this question legibly rather than to work
   around it.
2. **The "face up" bucket is the hardest.** `standup` documents that it froze into "do nothing"
   on that pose, and that the cause was the *motion blockers* (high `body_ang_vel`,
   too strong an `action_rate`). The values reused here are those of the "gets up from
   anywhere" version — do not harden them without a reason.
3. **Zero-padded head vs a `head_pose` command.** If the policy is deployed as `--standing` and
   someone presses the head keys, `infer_policy` writes `cmd[3:7] = head_offset` and the policy
   sees out-of-distribution input. A deliberate choice to stay within the roller convention; to be revisited if
   steering the head during the standup turns out to be necessary.
4. **Frictionloss 0.05 is far from reality.** The stages from iter 0 → 2000 produce a policy that does
   not transfer; only checkpoints from after the last stage (iter 4000+) are deployment candidates.

## Out of scope

- Integrating the standup into the rolling policy (the `velstand` recipe) — decision deferred until
  feasibility is validated.
- Side-lying start buckets.
- A rough / uneven-terrain variant.
- Trunk/head impact penalties.
- Any modification to the `roller`, `roller_crouch`, `roller_slope`, `standup`, `velstand` envs, or
  to `mdp.py`.
