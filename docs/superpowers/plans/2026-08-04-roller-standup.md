# Roller StandUp — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A dedicated policy `Mjlab-RollerStandUp-Flat-MicroDuck` that puts the microduck back upright on its rollers after a fall (face down or face up) and can then hold the stance on wheels.

**Architecture:** A single new env file, derived from `make_microduck_velocity_rollers_env_cfg()` — it thereby inherits the roller robot, the sensors, the whole domain randomization stack and the 61D observation (a hard requirement for runtime hot-swappability). We remove the skating rewards, graft on `standup`'s ten standup rewards (remapped onto the roller model's joint indices, where the passive wheels are interleaved), replace the reset with a start on the ground, and reverse the rolling-friction curriculum (braked wheels → free) to bootstrap the gesture before imposing the real wheel physics.

**Tech Stack:** Python 3.12, mjlab 1.3.0, MuJoCo / mujoco-warp, rsl_rl (PPO), uv, pytest.

Reference spec: `docs/superpowers/specs/2026-08-04-roller-standup-design.md`

## Global Constraints

- **No modification** to `src/mjlab_microduck/tasks/mdp.py`, nor to the `roller`, `roller_crouch`, `roller_slope`, `standup`, `velstand` envs. All the required mdp functions already exist.
- **61D observation parity is mandatory** with `make_microduck_velocity_rollers_env_cfg()`: `[gyro(3), projected_gravity(3), joint_pos(14), joint_vel(14), last_action(14), command(13)]`. The `head_pose` (4) and `body_pose` (6) slots stay **zero-padded**. Without this parity the ONNX will not load in a runtime slot.
- **Roller-model joint indices** (passive wheels interleaved; verified in MuJoCo):
  `_LEG_JOINTS = [0, 1, 2, 3, 4, 11, 12, 13, 14, 15]`, `_NECK_JOINTS = [7, 8, 9, 10]`, `_WHEEL_JOINTS = [5, 6, 16, 17]`.
  **Never** reuse `standup`'s indices (`[0-4, 9-13]` / `[5-8]`), which hold for the model without wheels.
- **Measured heights**: `ROLLER_STAND_Z = 0.138`, `ROLLER_PRONE_Z = 0.075`. Do not replace them with `standup`'s values (0.115 / 0.07).
- `EPISODE_LENGTH_S = 6.0`, `NUM_STEPS_PER_ENV = 24`. Curriculum `step` values are expressed as `iters × NUM_STEPS_PER_ENV`.
- **Symmetry OFF**: `symmetry_cfg=None`. `SYMMETRY_CFG` is wired for the old 51D layout and breaks on the 61D one.
- Repo style: 4-space indentation, and `SceneEntityCfg` **rebuilt for each term** (never a shared object — mjlab resolves and mutates these objects in place).
- Simple commits, no `Co-Authored-By`.
- **Pre-existing, out of scope**: `tests/test_wheel_glide.py` has 4 failing tests before this work (a fake asset with an obsolete `passive_LF_?wheel` regex). Do not fix them, and do not be alarmed by them. The rest of the suite passes (46 tests).

---

## File structure

| File | Responsibility |
|---|---|
| `src/mjlab_microduck/tasks/microduck_roller_standup_env_cfg.py` (create) | The whole env config + `MicroduckRollerStandUpRlCfg`. A single file, like every other env in the repo. |
| `src/mjlab_microduck/tasks/__init__.py` (modify) | Import + `register_mjlab_task` for the new task. |
| `tests/test_roller_standup_cfg.py` (create) | Config-construction tests + the joint-index lock-in. No sim, no GPU (like `test_roller_slope_cfg.py`). |
| `docs/roller_standup_policy_summary.md` (create, Task 5) | Handover summary, modeled on `docs/roller_slope_policy_summary.md`. |

---

## Task 1: Env skeleton — derivation, neutralized command, skating removed, registration

**Files:**
- Create: `src/mjlab_microduck/tasks/microduck_roller_standup_env_cfg.py`
- Modify: `src/mjlab_microduck/tasks/__init__.py`
- Test: `tests/test_roller_standup_cfg.py`

**Interfaces:**
- Consumes: `make_microduck_velocity_rollers_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg` (existing), `microduck_mdp.VelocityCommandCommandOnlyCfg` (existing).
- Produces: `make_microduck_roller_standup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg`; `MicroduckRollerStandUpRlCfg: RslRlOnPolicyRunnerCfg`; the module constants `ROLLER_STAND_Z: float`, `ROLLER_PRONE_Z: float`, `EPISODE_LENGTH_S: float`, `NUM_STEPS_PER_ENV: int`, `_LEG_JOINTS: list[int]`, `_NECK_JOINTS: list[int]`, `_WHEEL_JOINTS: list[int]`; and the registered task `"Mjlab-RollerStandUp-Flat-MicroDuck"`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_roller_standup_cfg.py`:

```python
from mjlab_microduck.tasks.microduck_roller_standup_env_cfg import (
    EPISODE_LENGTH_S,
    make_microduck_roller_standup_env_cfg,
)
from mjlab_microduck.tasks.microduck_velocity_rollers_env_cfg import (
    make_microduck_velocity_rollers_env_cfg,
)

# SKATING rewards: they must not survive in a standup env.
SKATING_REWARDS = (
    "wheel_speed",
    "braking",
    "skating_air_time",
    "glide",
    "single_support",
    "gait_symmetry",
    "forward_lean",
    "heading_hold",
    "feet_flat",
    "hip_roll_neutral",
    "pose",
    "com_height_target",
    "upright",
)


def test_env_builds_train_and_play():
    assert make_microduck_roller_standup_env_cfg() is not None
    assert make_microduck_roller_standup_env_cfg(play=True) is not None


def test_episode_is_short():
    # Short episode: rise then stabilize, like standup (6 s).
    cfg = make_microduck_roller_standup_env_cfg()
    assert cfg.episode_length_s == EPISODE_LENGTH_S == 6.0


def test_no_skating_rewards_survive():
    cfg = make_microduck_roller_standup_env_cfg()
    for name in SKATING_REWARDS:
        assert name not in cfg.rewards, f"skating reward survived: {name}"


def test_smoothness_regularisers_kept():
    # Kept from the roller inheritance: the standup needs sim2real smoothness, but
    # body_ang_vel must stay LIGHT (standup documents that it froze at -0.15).
    cfg = make_microduck_roller_standup_env_cfg()
    for name in (
        "action_over_limit",
        "self_collisions",
        "body_ang_vel",
        "angular_momentum",
        "action_rate_l2",
        "neck_action_rate_l2",
        "neck_joint_pos_l2",
        "joint_torques_l2",
    ):
        assert name in cfg.rewards, f"regularizer lost: {name}"
    assert cfg.rewards["body_ang_vel"].weight == -0.05


def test_twist_command_is_neutralised():
    # No steering: the policy deploys in --standing, where the runtime leaves the
    # twist slot at zero (cf. infer_policy.py:239).
    cfg = make_microduck_roller_standup_env_cfg()
    cmd = cfg.commands["twist"]
    assert cmd.ranges.lin_vel_x == (-0.01, 0.01)
    assert cmd.ranges.lin_vel_y == (-0.01, 0.01)
    assert cmd.ranges.ang_vel_z == (-0.05, 0.05)
    assert cmd.heading_command is False
    assert cmd.ranges.heading is None
    assert cmd.rel_standing_envs == 0.0


def test_twist_command_is_not_heading_relative():
    # The roller env installs a RelativeHeadingVelocityCommandCfg (cmd[2] =
    # heading error, computed internally). Here cmd[2] must be a real noisy zero.
    from mjlab_microduck.tasks import mdp as microduck_mdp

    cfg = make_microduck_roller_standup_env_cfg()
    cmd = cfg.commands["twist"]
    assert isinstance(cmd, microduck_mdp.VelocityCommandCommandOnlyCfg)
    assert not isinstance(cmd, microduck_mdp.RelativeHeadingVelocityCommandCfg)


def test_obs_nan_policy_sanitize():
    # A rare contact makes the free joint diverge to NaN: we sanitize the obs
    # rather than kill training (same choice as roller_slope).
    cfg = make_microduck_roller_standup_env_cfg()
    assert cfg.observations["actor"].nan_policy == "sanitize"
    assert cfg.observations["critic"].nan_policy == "sanitize"


def test_obs_parity_with_roller_env():
    # 61D parity is mandatory: otherwise the ONNX will not load in a runtime slot.
    standup = make_microduck_roller_standup_env_cfg()
    roller = make_microduck_velocity_rollers_env_cfg()
    for grp in ("actor", "critic"):
        assert list(standup.observations[grp].terms.keys()) == list(
            roller.observations[grp].terms.keys()
        ), f"observation layout diverged on group {grp}"


def test_terrain_is_plain_plane():
    # Inherited from the roller env: flat ground, no generator. No rough variant
    # for this v1.
    cfg = make_microduck_roller_standup_env_cfg()
    assert cfg.scene.terrain.terrain_type == "plane"
    assert cfg.scene.terrain.terrain_generator is None


def test_task_is_registered():
    from mjlab.tasks.registry import list_tasks

    import mjlab_microduck.tasks  # noqa: F401  (the import triggers registration)

    assert "Mjlab-RollerStandUp-Flat-MicroDuck" in list_tasks()
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run --with pytest pytest tests/test_roller_standup_cfg.py -q
```
Expected: erreur de collecte, `ModuleNotFoundError: No module named 'mjlab_microduck.tasks.microduck_roller_standup_env_cfg'`.

- [ ] **Step 3: Create the env file**

Create `src/mjlab_microduck/tasks/microduck_roller_standup_env_cfg.py`:

```python
"""Microduck roller standup — standing up on rollers.

DEDICATED episodic policy: the robot starts on the ground (face down, face up)
or already standing, and must get back up on its rollers and then HOLD the
stance. Port of the `standup` recipe (walking duck) to the roller model.

Derives from the roller env (`make_microduck_velocity_rollers_env_cfg`) → inherits
the roller robot, the sensors, the whole DR stack and the 61D observation as-is,
so it is hot-swappable at runtime (--new-cmd-obs). Same pattern as roller_slope.

Two structural differences from `standup`:
  - the passive wheels are INTERLEAVED in the joint ordering → remapped indices
    (_LEG_JOINTS below), locked in by tests/test_roller_standup_cfg.py;
  - no head_pose command: the head/body slots stay zero-padded (roller family
    convention) and the head is held upright by neck_joint_pos_l2, which
    resolves by NAME.

The genuinely new piece is the rolling-friction curriculum, REVERSED (braked
wheels → free wheels): the wheels roll, so there is no traction at all to push
against the ground. We bootstrap with near-locked wheels and then ramp toward
the real value. If `standing_composite` collapses at a stage, the "sticky feet"
gesture does not transfer and we will have to guide a skater technique (knee
support, one skate at a time).

Intended deployment: in `--standing` alongside the roller policy in `--walking`,
with the automatic switch on velocity command magnitude (infer_policy.py:262,
threshold 0.05); the twist slot is left at zero there (infer_policy.py:239).
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    RewardTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_rollers_env_cfg import (
    make_microduck_velocity_rollers_env_cfg,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg

# ── Trunk heights (m) ─────────────────────────────────────────────────────────
# Measured by exact kinematics (minimum over the mesh vertices of the colliding
# geoms, STAND pose, trunk lowered to contact) on scene_rollers.xml:
# standing 0.1407, face-down rest 0.0752, face-up rest 0.0475.
# Cross-check: the model WITHOUT wheels gives 0.1172 kinematically against
# STAND_Z=0.115 measured under load by standup → ~2 mm of sag, applied here too.
# 0.138 falls inside the reset_base z range (0.1335–0.1435) already used by the
# roller env.
ROLLER_STAND_Z = 0.138
ROLLER_PRONE_Z = 0.075

EPISODE_LENGTH_S  = 6.0   # rise + stabilize, same as standup
NUM_STEPS_PER_ENV = 24

# ── Joint indices — the passive wheels are INTERLEAVED ────────────────────────
# Actual ordering of the roller model (18 joints after the free joint), verified
# in MuJoCo via get_walk_rollers_spec().compile():
#   0-4   left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee, left_ankle
#   5-6   passive_LF_wheel, passive_LR_wheel
#   7-10  neck_pitch, head_pitch, head_yaw, head_roll
#   11-15 right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee, right_ankle
#   16-17 passive_RF_wheel, passive_RR_wheel
# standup uses [0-4, 9-13] / [5-8]: those are the indices of the model WITHOUT
# wheels, they do NOT hold here. Locked in by tests/test_roller_standup_cfg.py.
#
# Only _LEG_JOINTS is consumed (by the pose rewards). _NECK_JOINTS and
# _WHEEL_JOINTS exist for documentation and for the index test: the neck is
# resolved by NAME (neck_joint_pos_l2 calls find_joints(r".*(neck|head).*") every
# step) and the wheels by the ^passive_.* regex.
_LEG_JOINTS   = [0, 1, 2, 3, 4, 11, 12, 13, 14, 15]
_NECK_JOINTS  = [7, 8, 9, 10]
_WHEEL_JOINTS = [5, 6, 16, 17]

# SKATING rewards from the roller env: meaningless while lying on the ground.
# feet_flat: the blades are NOT flat during the rise → would fight the gesture.
# hip_roll_neutral: standing up requires spreading the legs.
# pose / com_height_target: replaced by the standup pose/height targets.
# upright (base gaussian): replaced by upright_linear + upright_sharp.
_SKATING_REWARDS = (
    "wheel_speed",
    "braking",
    "skating_air_time",
    "glide",
    "single_support",
    "gait_symmetry",
    "forward_lean",
    "heading_hold",
    "feet_flat",
    "hip_roll_neutral",
    "pose",
    "com_height_target",
    "upright",
)


def make_microduck_roller_standup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """"Stand up on rollers" env: start on the ground, target = standing on wheels."""
    cfg = make_microduck_velocity_rollers_env_cfg(play=play)

    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── Skating rewards removed ──────────────────────────────────────────────
    for name in _SKATING_REWARDS:
        cfg.rewards.pop(name, None)

    # ── Command: twist slot neutralized (≈ 0) ────────────────────────────────
    # The roller env installs a RelativeHeadingVelocityCommandCfg (cmd[2] =
    # heading error computed internally). Here we steer nothing: we go back to
    # the neutralized command-only variant, like standup. The head_pose (4) and
    # body_pose (6) slots stay zero-padded → 61D obs parity preserved.
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

    # ── Numerical robustness (same choice as roller_slope) ───────────────────
    # A rare contact (~1 in 25M steps) makes the free joint diverge to NaN: we
    # sanitize the obs (→ 0) so training is not killed, and the offending env
    # resets on the next step.
    for grp in ("actor", "critic"):
        cfg.observations[grp].nan_policy = "sanitize"

    return cfg


# ── RL runner config — identical to standup ───────────────────────────────────
MicroduckRollerStandUpRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,  # the normalizer MUST be baked into the ONNX by export.py
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # Symmetry OFF: SYMMETRY_CFG is wired for the old 51D layout and breaks
        # on the 61D one (same situation as every v1.5+ env).
        symmetry_cfg=None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="roller_standup",
    run_name="roller_standup",
    save_interval=250,
    num_steps_per_env=NUM_STEPS_PER_ENV,
    max_iterations=15_000,
)
```

- [ ] **Step 4: Register the task**

In `src/mjlab_microduck/tasks/__init__.py`, add the import **after** the `microduck_roller_slope_env_cfg` import block:

```python
from .microduck_roller_standup_env_cfg import (
    make_microduck_roller_standup_env_cfg,
    MicroduckRollerStandUpRlCfg,
)
```

Then, at the very end of the file (after the `Mjlab-RollerSlope-Flat-MicroDuck` registration):

```python
# Roller STANDUP — standing up on rollers (dedicated policy, starts on the ground).
register_mjlab_task(
    task_id="Mjlab-RollerStandUp-Flat-MicroDuck",
    env_cfg=make_microduck_roller_standup_env_cfg(),
    play_env_cfg=make_microduck_roller_standup_env_cfg(play=True),
    rl_cfg=MicroduckRollerStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ RollerStandUp task registered: Mjlab-RollerStandUp-Flat-MicroDuck")
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
uv run --with pytest pytest tests/test_roller_standup_cfg.py -q
```
Expected: 10 passed.

If `test_obs_parity_with_roller_env` fails, something has touched the observations — fix it before continuing; that is the project's hard constraint.

- [ ] **Step 6: Check that no other test regresses**

```bash
uv run --with pytest pytest tests/ -q
```
Expected: `4 failed, 56 passed` — the 4 failures are the pre-existing ones in `tests/test_wheel_glide.py` (the suite was at `4 failed, 46 passed` before this work). No other failure.

- [ ] **Step 7 : Commit**

```bash
git add src/mjlab_microduck/tasks/microduck_roller_standup_env_cfg.py \
        src/mjlab_microduck/tasks/__init__.py \
        tests/test_roller_standup_cfg.py
git commit -m "roller-standup: env skeleton (derived from roller, neutralized twist)"
```

---

## Task 2: Standup rewards + joint-index lock-in

**Files:**
- Modify: `src/mjlab_microduck/tasks/microduck_roller_standup_env_cfg.py`
- Test: `tests/test_roller_standup_cfg.py`

**Interfaces:**
- Consumes: from Task 1 — `make_microduck_roller_standup_env_cfg`, `ROLLER_STAND_Z`, `ROLLER_PRONE_Z`, `_LEG_JOINTS`, `_NECK_JOINTS`, `_WHEEL_JOINTS`. From `mdp.py` (existing, unmodified): `pose_target_match(target_overrides, asset_cfg, std, joint_indices)`, `pose_l1_penalty(target_overrides, asset_cfg, joint_indices)`, `height_target_gaussian(target_height, asset_cfg, std)`, `height_l1_penalty(target_height, asset_cfg)`, `com_upward_velocity(asset_cfg, max_height)`, `trunk_vertical_accel_penalty(asset_cfg)`, `body_upright_linear(asset_cfg)`, `upright_gaussian_at_height(std, height_low, height_high, asset_cfg)`, `standing_composite_score(target_height, height_std, upright_std, pose_std, joint_indices, target_overrides, asset_cfg)`, `joint_torque_rate_l2()`.
- Produces: the reward terms `pose_stand_legs`, `pose_stand_l1`, `height_stand`, `height_stand_sharp`, `height_stand_l1`, `com_upward_velocity`, `gentle_rise`, `upright_linear`, `upright_sharp`, `standing_composite`, `joint_torque_rate_l2` in `cfg.rewards`.

- [ ] **Step 1: Write the failing tests**

Add to the end of `tests/test_roller_standup_cfg.py`:

```python
def test_joint_indices_match_actual_roller_model():
    """Lock-in: the passive wheels are interleaved in the joint ordering.

    Reusing standup's indices ([0-4, 9-13]) would give rewards that point at
    wheels. This test compiles the real MjSpec of the roller robot and checks the
    names at the indices used. Pure CPU, no sim.
    """
    import mujoco

    from mjlab_microduck.robot.microduck_constants import get_walk_rollers_spec
    from mjlab_microduck.tasks.microduck_roller_standup_env_cfg import (
        _LEG_JOINTS,
        _NECK_JOINTS,
        _WHEEL_JOINTS,
    )

    model = get_walk_rollers_spec().compile()
    articulated = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        for j in range(model.njnt)
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
    ]

    assert [articulated[i] for i in _LEG_JOINTS] == [
        "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
        "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
    ]
    assert [articulated[i] for i in _NECK_JOINTS] == [
        "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    ]
    assert [articulated[i] for i in _WHEEL_JOINTS] == [
        "passive_LF_wheel", "passive_LR_wheel", "passive_RF_wheel", "passive_RR_wheel",
    ]
    # No overlap, and the three lists cover every joint.
    assert len(set(_LEG_JOINTS) | set(_NECK_JOINTS) | set(_WHEEL_JOINTS)) == len(articulated)


def test_recovery_rewards_present_with_expected_weights():
    cfg = make_microduck_roller_standup_env_cfg()
    expected = {
        "pose_stand_legs":      8.0,
        "pose_stand_l1":        5.0,
        "height_stand":         4.0,
        "height_stand_sharp":   4.0,
        "height_stand_l1":     30.0,
        "com_upward_velocity":  3.0,
        "gentle_rise":         -0.02,
        "upright_linear":       6.0,
        "upright_sharp":        6.0,
        "standing_composite":  15.0,
        "joint_torque_rate_l2": -2e-3,
    }
    for name, weight in expected.items():
        assert name in cfg.rewards, f"missing standup reward: {name}"
        assert cfg.rewards[name].weight == weight, f"unexpected weight on {name}"


def test_recovery_rewards_use_roller_heights_not_walker_heights():
    from mjlab_microduck.tasks.microduck_roller_standup_env_cfg import (
        ROLLER_PRONE_Z,
        ROLLER_STAND_Z,
    )

    cfg = make_microduck_roller_standup_env_cfg()
    assert ROLLER_STAND_Z == 0.138  # NOT the 0.115 of the model without wheels
    for name in ("height_stand", "height_stand_sharp", "height_stand_l1"):
        assert cfg.rewards[name].params["target_height"] == ROLLER_STAND_Z
    assert cfg.rewards["standing_composite"].params["target_height"] == ROLLER_STAND_Z
    # com_upward_velocity cuts off just ABOVE the target (10 mm of margin),
    # otherwise the policy parks at the cutoff altitude without finishing the rise.
    assert cfg.rewards["com_upward_velocity"].params["max_height"] == ROLLER_STAND_Z + 0.010
    # upright_sharp is gated between the ground rest height and the standing height.
    assert cfg.rewards["upright_sharp"].params["height_low"] == ROLLER_PRONE_Z
    assert cfg.rewards["upright_sharp"].params["height_high"] == ROLLER_STAND_Z


def test_pose_rewards_target_legs_only_at_roller_indices():
    from mjlab_microduck.tasks.microduck_roller_standup_env_cfg import _LEG_JOINTS

    cfg = make_microduck_roller_standup_env_cfg()
    for name in ("pose_stand_legs", "pose_stand_l1", "standing_composite"):
        assert cfg.rewards[name].params["joint_indices"] == _LEG_JOINTS
        # target_overrides=None → the target is HOME (default_joint_pos).
        assert cfg.rewards[name].params["target_overrides"] is None


def test_trunk_asset_cfgs_are_distinct_objects():
    """mjlab resolves and MUTATES SceneEntityCfg in place: an object shared across
    several terms causes stale indices. Each term must have its own.
    """
    cfg = make_microduck_roller_standup_env_cfg()
    names = (
        "height_stand", "height_stand_sharp", "height_stand_l1",
        "com_upward_velocity", "gentle_rise", "upright_linear",
        "upright_sharp", "standing_composite",
    )
    seen = [id(cfg.rewards[n].params["asset_cfg"]) for n in names]
    assert len(set(seen)) == len(seen), "asset_cfg shared across several terms"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run --with pytest pytest tests/test_roller_standup_cfg.py -q
```
Expected: `test_joint_indices_match_actual_roller_model` **passes** (Task 1's constants are already correct — it is a regression lock-in, not a red test); the other 4 fail with `KeyError: 'pose_stand_legs'` or `assert 'pose_stand_legs' in cfg.rewards`.

- [ ] **Step 3: Add the standup rewards**

In `microduck_roller_standup_env_cfg.py`, insert this block **after** the "Numerical robustness" block and **before** the `return cfg`:

```python
    # ── Standup rewards — transplanted from standup, remapped ────────────────
    # The weights come from the iterations documented in
    # microduck_standup_env_cfg.py: only touch them with a reason. Only the
    # joint indices and the two heights change here.
    # NB: a FRESH SceneEntityCfg per term — mjlab resolves and mutates them in
    # place, so a shared object yields stale indices.

    # Target pose = HOME (target_overrides=None), LEGS only: the neck and head
    # are held by neck_joint_pos_l2 (inherited), which resolves by NAME.
    cfg.rewards["pose_stand_legs"] = RewardTermCfg(
        func=microduck_mdp.pose_target_match,
        weight=8.0,
        params={
            "std": 0.5,
            "joint_indices": _LEG_JOINTS,
            "target_overrides": None,
        },
    )
    # L1 bootstrap: constant gradient even far from HOME (the gaussian saturates).
    cfg.rewards["pose_stand_l1"] = RewardTermCfg(
        func=microduck_mdp.pose_l1_penalty,
        weight=5.0,
        params={
            "joint_indices": _LEG_JOINTS,
            "target_overrides": None,
        },
    )

    # Height in three layers: wide gaussian (pulls up from the ground), narrow
    # gaussian (forces the last few cm, where the wide one is saturated), and a
    # strong L1 that makes "stay on the ground" net NEGATIVE — without it the
    # policy settles for the lazy optimum "motionless on the floor".
    cfg.rewards["height_stand"] = RewardTermCfg(
        func=microduck_mdp.height_target_gaussian,
        weight=4.0,
        params={
            "std": 0.04,
            "target_height": ROLLER_STAND_Z,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.rewards["height_stand_sharp"] = RewardTermCfg(
        func=microduck_mdp.height_target_gaussian,
        weight=4.0,
        params={
            "std": 0.015,
            "target_height": ROLLER_STAND_Z,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.rewards["height_stand_l1"] = RewardTermCfg(
        func=microduck_mdp.height_l1_penalty,
        weight=30.0,
        params={
            "target_height": ROLLER_STAND_Z,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # Pays for the UPWARD MOTION, not just the destination: without it,
    # "sit still and farm the partial pose reward" dominates. The cutoff is
    # 10 mm ABOVE the target, otherwise the policy parks at the cutoff altitude
    # and never finishes the rise.
    cfg.rewards["com_upward_velocity"] = RewardTermCfg(
        func=microduck_mdp.com_upward_velocity,
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            "max_height": ROLLER_STAND_Z + 0.010,
        },
    )
    # Gentle rise: penalizes |a_z|. Compatible with com_upward_velocity — a
    # constant vertical velocity collects the latter AND has a_z = 0 → the two
    # pressures jointly select a smooth constant-speed rise.
    cfg.rewards["gentle_rise"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )

    # Trunk uprightness in two layers: cos(tilt) has a strong gradient while
    # lying down but runs out of steam near vertical; the tight height-gated
    # gaussian takes over and kills the backward lean (standup's failure mode:
    # tipping backward while extending the legs).
    cfg.rewards["upright_linear"] = RewardTermCfg(
        func=microduck_mdp.body_upright_linear,
        weight=6.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    cfg.rewards["upright_sharp"] = RewardTermCfg(
        func=microduck_mdp.upright_gaussian_at_height,
        weight=6.0,
        params={
            "std": 0.3,
            "height_low": ROLLER_PRONE_Z,
            "height_high": ROLLER_STAND_Z,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # MULTIPLICATIVE score height × uprightness × pose: since the factors
    # multiply, being good on 2 criteria out of 3 pays nothing → it breaks the
    # "leaning at the right height" compromises that additive rewards let
    # through. Stds deliberately WIDE so the score stays visible during the rise
    # (tight stds gave a score of ~5e-5, i.e. zero gradient).
    cfg.rewards["standing_composite"] = RewardTermCfg(
        func=microduck_mdp.standing_composite_score,
        weight=15.0,
        params={
            "target_height": ROLLER_STAND_Z,
            "height_std": 0.04,
            "upright_std": 0.40,
            "pose_std": 0.40,
            "joint_indices": _LEG_JOINTS,
            "target_overrides": None,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # Anti-jitter: penalizes torque RATE, not its magnitude nor trunk rotation
    # → damps the shakes without blocking the roll-over.
    # standup identified this as the only damper that does not kill the rise
    # from the back.
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torque_rate_l2,
        weight=-2e-3,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run --with pytest pytest tests/test_roller_standup_cfg.py -q
```
Expected: 15 passed.

- [ ] **Step 5 : Commit**

```bash
git add src/mjlab_microduck/tasks/microduck_roller_standup_env_cfg.py \
        tests/test_roller_standup_cfg.py
git commit -m "roller-standup: standup rewards + joint-index lock-in"
```

---

## Task 3: Starting on the ground — reset, removing `fell_over`, the pose curriculum

**Files:**
- Modify: `src/mjlab_microduck/tasks/microduck_roller_standup_env_cfg.py`
- Test: `tests/test_roller_standup_cfg.py`

**Interfaces:**
- Consumes: `microduck_mdp.set_random_ground_state(env, env_ids, asset_cfg, face_down_prob, face_up_prob, sitting_prob, standing_prob, prone_z_min, prone_z_max, sitting_z_min, sitting_z_max, standing_z_min, standing_z_max, sitting_joint_overrides, sitting_joint_noise_std, sitting_tilt_max)` and `microduck_mdp.event_param_curriculum(env, env_ids, event_name, param_stages)` — existing, unmodified.
- Produces: the `cfg.events["set_ground_state"]` event and the `cfg.curriculum["ground_state_mix"]` curriculum; `cfg.terminations` without `fell_over`.

- [ ] **Step 1: Write the failing tests**

Add to the end of `tests/test_roller_standup_cfg.py`:

```python
def test_starts_from_ground_states():
    # Face down + face up + standing. No "sitting" bucket: in standup it existed
    # only for the hand-off from the sit policy, which has no roller equivalent —
    # and its sitting_joint_overrides are indices of the model WITHOUT wheels.
    cfg = make_microduck_roller_standup_env_cfg()
    assert "set_ground_state" in cfg.events
    params = cfg.events["set_ground_state"].params
    assert params["sitting_prob"] == 0.0
    assert params["sitting_joint_overrides"] is None
    assert params["face_down_prob"] > 0.0
    assert params["standing_prob"] > 0.0
    # face_up starts at 0: introduced late by the curriculum.
    assert params["face_up_prob"] == 0.0


def test_ground_state_heights_are_roller_specific():
    cfg = make_microduck_roller_standup_env_cfg()
    params = cfg.events["set_ground_state"].params
    # Ground rest: identical geometry on both models (it is the trunk shell
    # that touches, not the feet) → standup's ranges reused.
    assert (params["prone_z_min"], params["prone_z_max"]) == (0.05, 0.09)
    # [Corrected to 0.076 after the final review — see docs/superpowers/specs/2026-08-04-roller-standup-design.md]
    # Standing: ROLLER height (+23 mm vs the model without wheels, at 0.11–0.12).
    assert params["standing_z_min"] == 0.134
    assert params["standing_z_max"] == 0.144
    assert params["standing_z_min"] < 0.138 < params["standing_z_max"]


def test_ground_state_event_runs_after_base_reset():
    # set_ground_state overwrites the pose set by reset_base / reset_robot_joints:
    # event order follows insertion order, so it must come AFTER them.
    cfg = make_microduck_roller_standup_env_cfg()
    order = list(cfg.events.keys())
    assert order.index("set_ground_state") > order.index("reset_base")
    assert order.index("set_ground_state") > order.index("reset_robot_joints")


def test_no_fall_termination():
    # The robot STARTS fallen: a tilt termination would kill the episode on the
    # first step. nan_state (inherited) does stay.
    cfg = make_microduck_roller_standup_env_cfg()
    assert "fell_over" not in cfg.terminations
    assert "nan_state" in cfg.terminations


def test_ground_state_curriculum_ramps_easy_to_hard():
    cfg = make_microduck_roller_standup_env_cfg()
    assert "ground_state_mix" in cfg.curriculum
    stages = cfg.curriculum["ground_state_mix"].params["param_stages"]
    assert cfg.curriculum["ground_state_mix"].params["event_name"] == "set_ground_state"
    # The steps are increasing and start at 0.
    steps = [s["step"] for s in stages]
    assert steps[0] == 0 and steps == sorted(steps) and len(set(steps)) == len(steps)
    # face_up is introduced late and then grows monotonically.
    face_up = [s["params"]["face_up_prob"] for s in stages]
    assert face_up[0] == 0.0
    assert face_up == sorted(face_up)
    assert face_up[-1] >= 0.35
    # Every stage is a valid distribution, and "already standing" never
    # disappears (otherwise the policy stands up then falls back, never learning
    # to hold).
    for stage in stages:
        p = stage["params"]
        total = (
            p["standing_prob"] + p["sitting_prob"]
            + p["face_down_prob"] + p["face_up_prob"]
        )
        assert abs(total - 1.0) < 1e-9
        assert p["sitting_prob"] == 0.0
        assert p["standing_prob"] > 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run --with pytest pytest tests/test_roller_standup_cfg.py -q
```
Expected: the 5 new ones fail — `assert 'set_ground_state' in cfg.events` (KeyError / AssertionError), `assert 'fell_over' not in cfg.terminations`, `assert 'ground_state_mix' in cfg.curriculum`.

- [ ] **Step 3: Add the ground reset, the removal of `fell_over` and the curriculum**

In `microduck_roller_standup_env_cfg.py`, insert this block **after** the standup rewards and **before** the `return cfg`:

```python
    # ── Start ON THE GROUND: face down / face up / already standing ─────────
    # Added LAST in cfg.events: execution order follows insertion order, and this
    # term must overwrite the pose set by reset_base / reset_robot_joints.
    # The "already standing" bucket is not decorative: without it the policy
    # learns to rise but not to HOLD, and it falls right back down after standing
    # up.
    # No "sitting" bucket → no sitting_joint_overrides to remap (standup's are
    # indices of the model WITHOUT wheels).
    # The probabilities below = stage 0 of the ground_state_mix curriculum.
    cfg.events["set_ground_state"] = EventTermCfg(
        func=microduck_mdp.set_random_ground_state,
        mode="reset",
        params={
            "face_down_prob": 0.50,   # face down (+90° of pitch)
            "face_up_prob":   0.00,   # face up — hardest, introduced late
            "sitting_prob":   0.00,
            "standing_prob":  0.50,
            "sitting_joint_overrides": None,
            # Ground rest: measured at 0.075 (face down) / 0.048 (face up), identical
            # on both models — it is the trunk shell that touches, not the feet.
            "prone_z_min":    0.05,
            # [Corrected to 0.076 after the final review — see docs/superpowers/specs/2026-08-04-roller-standup-design.md]
            "prone_z_max":    0.09,
            # Standing on wheels: ROLLER_STAND_Z = 0.138 (vs 0.11–0.12 without wheels).
            "standing_z_min": 0.134,
            "standing_z_max": 0.144,
            # Pitch/roll noise at start. Careful: in set_random_ground_state the
            # "standing" bucket reuses the "sitting" bucket's quaternion, so this
            # noise applies to standing starts TOO — that is intended (no
            # overfitting to perfectly upright).
            "sitting_tilt_max": math.radians(10),
        },
    )

    # The robot STARTS fallen → the tilt termination makes no sense here (it
    # would kill the episode on the first step). nan_state, inherited, stays.
    cfg.terminations.pop("fell_over", None)

    # Start-pose curriculum, easy → hard. With a flat mix from the start, the
    # policy optimizes the easy majority and leaves the face-up case undertrained
    # (standup's lesson: it froze into "do nothing" on that pose). So we
    # introduce standing+face-down first, face-up late, and bias toward the hard
    # poses at the end so they get the most training.
    cfg.curriculum["ground_state_mix"] = CurriculumTermCfg(
        func=microduck_mdp.event_param_curriculum,
        params={
            "event_name": "set_ground_state",
            "param_stages": [
                {"step": 0, "params": {
                    "standing_prob": 0.50, "sitting_prob": 0.00,
                    "face_down_prob": 0.50, "face_up_prob": 0.00}},
                {"step": 600 * NUM_STEPS_PER_ENV, "params": {
                    "standing_prob": 0.35, "sitting_prob": 0.00,
                    "face_down_prob": 0.45, "face_up_prob": 0.20}},
                {"step": 1500 * NUM_STEPS_PER_ENV, "params": {
                    "standing_prob": 0.25, "sitting_prob": 0.00,
                    "face_down_prob": 0.40, "face_up_prob": 0.35}},
                {"step": 2500 * NUM_STEPS_PER_ENV, "params": {
                    "standing_prob": 0.20, "sitting_prob": 0.00,
                    "face_down_prob": 0.40, "face_up_prob": 0.40}},
            ],
        },
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run --with pytest pytest tests/test_roller_standup_cfg.py -q
```
Expected: 20 passed.

- [ ] **Step 5 : Commit**

```bash
git add src/mjlab_microduck/tasks/microduck_roller_standup_env_cfg.py \
        tests/test_roller_standup_cfg.py
git commit -m "roller-standup: depart au sol (ventre/dos/debout) + curriculum des poses"
```

---

## Task 4: Curricula — reversed rolling friction, pushes, action_rate

**Files:**
- Modify: `src/mjlab_microduck/tasks/microduck_roller_standup_env_cfg.py`
- Test: `tests/test_roller_standup_cfg.py`

**Interfaces:**
- Consumes: `microduck_mdp.wheel_friction_curriculum(env, env_ids, event_name, ranges_stages)`, `microduck_mdp.push_curriculum(env, env_ids, event_name, push_stages)`, `microduck_mdp.reward_weight(env, env_ids, reward_name, weight_stages)` — existing, unmodified. Events inherited from the roller env: `randomize_wheel_friction`, `push_robot`.
- Produces: `cfg.curriculum["wheel_friction"]` (decreasing), `cfg.curriculum["push_magnitude"]`, `cfg.curriculum["action_rate_weight"]` (replaced).

- [ ] **Step 1: Write the failing tests**

Add to the end of `tests/test_roller_standup_cfg.py`:

```python
def test_wheel_friction_curriculum_is_decreasing():
    """The new piece: BRAKED → FREE wheels.

    The wheels roll, so there is no longitudinal traction to push against the
    ground. We bootstrap with near-locked bearings (the rise happens as if on
    feet), then ramp toward the real value. The roller env, by contrast, ramps
    this friction UP (0 → 0.0015): the direction really is reversed here.
    """
    cfg = make_microduck_roller_standup_env_cfg()
    stages = cfg.curriculum["wheel_friction"].params["ranges_stages"]
    assert cfg.curriculum["wheel_friction"].params["event_name"] == "randomize_wheel_friction"

    steps = [s["step"] for s in stages]
    assert steps[0] == 0 and steps == sorted(steps) and len(set(steps)) == len(steps)

    lows = [s["ranges"][0] for s in stages]
    assert lows == sorted(lows, reverse=True), "friction must DECREASE"
    assert lows[0] >= 0.02, "start clearly braked to bootstrap the gesture"
    # Ends at the real rolling value (the roller env's).
    assert stages[-1]["ranges"] == (0.0015, 0.0015)
    for stage in stages:
        assert stage["ranges"][0] == stage["ranges"][1]


def test_wheel_friction_event_starts_at_stage_zero():
    # The curriculum is only evaluated from the first step onward: without this,
    # the very first resets would use the (0, 0) value inherited from the roller
    # env, i.e. FREE wheels during the bootstrap — exactly the opposite of the goal.
    cfg = make_microduck_roller_standup_env_cfg()
    stage0 = cfg.curriculum["wheel_friction"].params["ranges_stages"][0]["ranges"]
    assert cfg.events["randomize_wheel_friction"].params["ranges"] == stage0


def test_action_rate_ramp_is_the_standup_one_not_the_roller_one():
    # The roller env ramps to -2.0 (calm gait): that is a motion blocker, it slows
    # the fast action the rise from the back needs. We reuse standup's ramp, which
    # tops out at -1.0.
    cfg = make_microduck_roller_standup_env_cfg()
    weights = [
        s["weight"] for s in cfg.curriculum["action_rate_weight"].params["weight_stages"]
    ]
    assert weights == [-0.4, -0.8, -1.0]
    assert cfg.rewards["action_rate_l2"].weight == -0.6


def test_push_curriculum_ramps_from_zero():
    # Inherited pushes (±0.2 m/s), but ramped: a shove from step 0 disturbs the
    # bootstrap of the rise.
    cfg = make_microduck_roller_standup_env_cfg()
    assert "push_robot" in cfg.events
    stages = cfg.curriculum["push_magnitude"].params["push_stages"]
    assert cfg.curriculum["push_magnitude"].params["event_name"] == "push_robot"
    assert stages[0]["velocity_range"]["x"] == (0.0, 0.0)
    assert stages[-1]["velocity_range"]["x"] == (-0.2, 0.2)
    highs = [s["velocity_range"]["x"][1] for s in stages]
    assert highs == sorted(highs), "the push must GROW"


def test_inherited_dr_curricula_survive():
    # The DR inherited from the roller env must not have been lost along the way.
    cfg = make_microduck_roller_standup_env_cfg()
    for name in ("com_range", "head_com_range"):
        assert name in cfg.curriculum, f"DR curriculum lost: {name}"
    for name in (
        "randomize_com",
        "randomize_head_com",
        "randomize_armature",
        "randomize_joint_friction",
        "randomize_mass_inertia",
        "randomize_wheel_friction",
        "encoder_bias",
    ):
        assert name in cfg.events, f"DR event lost: {name}"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run --with pytest pytest tests/test_roller_standup_cfg.py -q
```
Expected: `test_wheel_friction_curriculum_is_decreasing` fails on `assert lows == sorted(lows, reverse=True)` (the roller env ramps 0 → 0.0015), `test_wheel_friction_event_starts_at_stage_zero` fails, `test_action_rate_ramp_is_the_standup_one_not_the_roller_one` fails on `[-1.0, -1.5, -2.0] != [-0.4, -0.8, -1.0]`, and `test_push_curriculum_ramps_from_zero` fails on `KeyError: 'push_magnitude'`. `test_inherited_dr_curricula_survive` already passes (a non-regression check).

- [ ] **Step 3 : Remplacer les curricula**

In `microduck_roller_standup_env_cfg.py`, insert this block **after** the `ground_state_mix` curriculum et **avant** le `return cfg` :

```python
    # ── REVERSED rolling friction: braked → free ─────────────────────────────
    # This is the only genuinely new piece of this env, and the heart of the
    # difficulty: the wheels roll, so there is NO longitudinal traction to push
    # against the ground. The roller env ramps this friction UP (0 → 0.0015);
    # here we ramp it DOWN, to bootstrap the gesture on an easy problem
    # (near-locked wheels ≈ feet) before imposing the real rolling physics.
    #
    # DIAGNOSTIC to watch: if Episode_Reward/standing_composite collapses at a
    # stage, the "sticky feet" gesture does not transfer to free wheels → we will
    # have to guide a skater technique (intermediate knee support, one skate at a
    # time). That is an actionable result, not a failure.
    #
    # sim2real WARNING: only checkpoints from AFTER the last stage (iter 4000+)
    # are deployment candidates. Before that, the policy relies on a rolling
    # friction that does not exist on the real robot.
    _WHEEL_FRICTION_STAGE0 = (0.0500, 0.0500)
    cfg.curriculum["wheel_friction"] = CurriculumTermCfg(
        func=microduck_mdp.wheel_friction_curriculum,
        params={
            "event_name": "randomize_wheel_friction",
            "ranges_stages": [
                {"step": 0,                        "ranges": _WHEEL_FRICTION_STAGE0},
                {"step": 1000 * NUM_STEPS_PER_ENV, "ranges": (0.0200, 0.0200)},
                {"step": 2000 * NUM_STEPS_PER_ENV, "ranges": (0.0080, 0.0080)},
                {"step": 3000 * NUM_STEPS_PER_ENV, "ranges": (0.0030, 0.0030)},
                {"step": 4000 * NUM_STEPS_PER_ENV, "ranges": (0.0015, 0.0015)},
            ],
        },
    )
    # The event's STARTING value must match stage 0: the curriculum is only
    # evaluated from the first step onward, otherwise the very first resets
    # would use the (0, 0) inherited from the roller env — FREE wheels during
    # le bootstrap, soit exactement l'inverse du but.
    cfg.events["randomize_wheel_friction"].params["ranges"] = _WHEEL_FRICTION_STAGE0

    # ── action_rate: standup's ramp, not the roller's ────────────────────────
    # The roller env ramps to -2.0 for a calm gait. That is a motion blocker: it
    # slows the fast action the rise from the back needs (standup documents that
    # too strong an action_rate killed that recovery). Smoothness is carried here
    # by joint_torque_rate_l2.
    cfg.rewards["action_rate_l2"].weight = -0.6
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0,                       "weight": -0.4},
                {"step": 250 * NUM_STEPS_PER_ENV, "weight": -0.8},
                {"step": 500 * NUM_STEPS_PER_ENV, "weight": -1.0},
            ],
        },
    )

    # ── Ramped pushes ───────────────────────────────────────────────────────
    # push_robot is inherited from the roller env (±0.2 m/s, every 3–6 s) but
    # without a curriculum. A shove from step 0 disturbs the bootstrap of the
    # rise: we ramp it up like standup does.
    cfg.curriculum["push_magnitude"] = CurriculumTermCfg(
        func=microduck_mdp.push_curriculum,
        params={
            "event_name": "push_robot",
            "push_stages": [
                {"step": 0, "velocity_range": {
                    "x": (0.0, 0.0), "y": (0.0, 0.0)}},
                {"step": 500 * NUM_STEPS_PER_ENV, "velocity_range": {
                    "x": (-0.08, 0.08), "y": (-0.08, 0.08)}},
                {"step": 1000 * NUM_STEPS_PER_ENV, "velocity_range": {
                    "x": (-0.2, 0.2), "y": (-0.2, 0.2)}},
            ],
        },
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run --with pytest pytest tests/test_roller_standup_cfg.py -q
```
Expected: 25 passed.

- [ ] **Step 5: Check that no other test regresses**

```bash
uv run --with pytest pytest tests/ -q
```
Expected: `4 failed, 71 passed` — only the 4 pre-existing failures in `tests/test_wheel_glide.py`.

- [ ] **Step 6 : Commit**

```bash
git add src/mjlab_microduck/tasks/microduck_roller_standup_env_cfg.py \
        tests/test_roller_standup_cfg.py
git commit -m "roller-standup: curriculum de friction de roulement inverse + pousses rampees"
```

---

## Task 5: End-to-end verification on GPU + handover doc

The Task 1–4 tests are **static**: they check the config, not execution. They cannot catch an out-of-bounds `joint_indices`, a wrong parameter name passed to an mdp function, or a missing sensor. This task is the only place where the env actually runs.

**Files:**
- Create: `docs/roller_standup_policy_summary.md`
- (aucune modification de code attendue si tout passe)

**Interfaces:**
- Consumes: the registered task `Mjlab-RollerStandUp-Flat-MicroDuck` (Task 1) and the complete env (Tasks 2–4).
- Produces: nothing programmatic — a handover doc and confirmation that the env runs.

- [ ] **Step 1: Run a very short training**

```bash
uv run train Mjlab-RollerStandUp-Flat-MicroDuck \
  --env.scene.num-envs 64 \
  --agent.max_iterations 3 \
  --agent.logger tensorboard
```

`--agent.logger tensorboard` avoids polluting wandb with a throwaway run.

Expected: `✓ RollerStandUp task registered: Mjlab-RollerStandUp-Flat-MicroDuck`, then 3 iterations that run without an exception, with a reward table showing the terms `pose_stand_legs`, `height_stand`, `standing_composite`, etc.

Erreurs plausibles et leur cause :
- `IndexError` on `joint_pos[:, joint_indices]` → the `_LEG_JOINTS` indices exceed the number de joints ; relire Task 2.
- `TypeError: ... unexpected keyword argument` → a parameter name does not match the mdp function's signature; compare with Task 2's **Interfaces** block.
- `KeyError` on a sensor name → a removed reward was the only user of a sensor, or a kept reward requires one that is missing.

- [ ] **Step 2: Check that the standup rewards are not all zero**

In the previous step's output, check that `Episode_Reward/standing_composite` and `Episode_Reward/height_stand` are **non-zero**. A value of exactly 0.0 across all three iterations signals a reward that never fires (wrong `asset_cfg`, wrong target height).

- [ ] **Step 3: Visually check the start on the ground**

```bash
uv run play Mjlab-RollerStandUp-Flat-MicroDuck --env.scene.num-envs 16
```

Expected: the robots appear **on the ground** (face down) or **standing on their wheels**, never in mid-air and never through the ground. No robot face up at this stage — that is normal: `face_up_prob = 0` at the curriculum's stage 0, and at play time the curriculum does not run.

If robots drop from a height, the `prone_z` ranges are misconfigured; if a robot goes through the ground, the start pose is spawning it below the plane.

- [ ] **Step 4: Write the handover doc**

Create `docs/roller_standup_policy_summary.md`, modeled on `docs/roller_slope_policy_summary.md` :

```markdown
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

`set_random_ground_state`: face down (`prone_z` 0.05–0.09) / face up / **already standing** (`standing_z` 0.134–0.144), ± 10° of pitch/roll noise. No "sitting" bucket. The "standing" bucket is necessary: without it the policy rises but does not hold.

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

**Sim2real**: only checkpoints from after iter 4000 are deployment candidates. Before that, the policy relies on a friction that does not exist on the real robot.

## Command

The `twist` slot is neutralized (± 0.01), and the `head_pose` / `body_pose` slots are **zero-padded** (roller convention). Intended deployment: in `--standing` alongside the roller policy in `--walking`, with the automatic switch on command magnitude (`infer_policy.py:262`, threshold 0.05); the twist slot is left at zero there (`infer_policy.py:239`).

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

## Out of scope

Integrating the standup into the rolling policy (the `velstand` recipe); side-lying start buckets; a rough variant; trunk/head impact penalties.
```

- [ ] **Step 5: Commit**

```bash
git add docs/roller_standup_policy_summary.md
git commit -m "roller-standup: handover doc"
```

---

## After the plan

Run a real training:

```bash
uv run train Mjlab-RollerStandUp-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max_iterations 15000
```

**The signal to read**: `Episode_Reward/standing_composite` must rise, and above all **its behavior at iters 1000 / 2000 / 3000 / 4000** (the rolling-friction stages) answers the question that motivated this whole design — is standing up on free wheels feasible with the "sticky feet" gesture, or must a skater technique be taught?
