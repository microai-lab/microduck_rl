# Roller Crouch-Glide Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "crouch while gliding, then stand back up" gesture triggered by button A, without modifying the Rust runtime, by training an mjlab policy loaded into the `--ground-pick` slot.

**Architecture:** A new mjlab task trained on the roller robot, driven by the `GroundPickPhaseCommand` phase command (the one the runtime's ground-pick slot sends). A new reward tracks a trapezoidal trunk-height target (high → low → 1 s plateau → high) along the phase. The same 61D obs layout as the roller policy → hot-swappable at runtime. ONNX export, loaded via `--ground-pick`.

**Tech Stack:** Python, PyTorch, mjlab 1.3.0, MuJoCo, uv, ONNX. Target runtime: `apirrone/microduck_runtime` (Rust, a binary — NOT modified).

## Global Constraints

- **No modification of the Rust runtime.** The gesture reuses the existing `--ground-pick` slot (button A, one-shot).
- **The unified 61D obs layout** is mandatory (`--new-cmd-obs`): `[twist(3), head(4), body(6)]`, head/body zero-padded. Every new policy MUST preserve this layout.
- **14 active joints** (passive wheels excluded via `SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))`), `action.scale = 1.0`, `kp_fw = 200`.
- **Training/deployment parity (sim2real):** at deployment, force `--ground-pick-kp-ratio 1.0` (default 0.6), `--ground-pick-action-scale` = the runtime action_scale, `--ground-pick-period 5.0`.
- **Phase encoding (imposed by the runtime):** `command = [cos(2π·φ), sin(2π·φ), 0]`, 4 s period. Glide plateau = 1 s → `hold_lo=0.375`, `hold_hi=0.625`.
- **Simple commits** (no `Co-Authored-By`).
- Run the tests via `uv run --with pytest pytest` (no pytest dependency added to the project).
- Reference spec: `docs/superpowers/specs/2026-07-17-roller-crouch-glide-design.md`.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/mjlab_microduck/tasks/mdp.py` | **Modify.** Add 3 functions: `crouch_height_target` (pure), `crouch_glide_reward_from_values` (pure), `crouch_glide_height_by_phase` (env wrapper), plus `forward_speed_reward`. |
| `tests/test_crouch_glide.py` | **Create.** Unit tests of the pure functions. |
| `src/mjlab_microduck/tasks/microduck_roller_crouch_env_cfg.py` | **Create.** The env (roller + phase hybrid) + `MicroduckRollerCrouchRlCfg`. |
| `src/mjlab_microduck/tasks/__init__.py` | **Modify.** Import + register `Mjlab-RollerCrouch-Flat-MicroDuck`. |
| `tests/test_roller_crouch_cfg.py` | **Create.** Smoke test: the env builds with the right command/rewards. |

---

## Task 1: Trapezoidal height target (pure function)

**Files:**
- Modify: `src/mjlab_microduck/tasks/mdp.py` (add the function, after `com_height_target` around line 737)
- Test: `tests/test_crouch_glide.py`

**Interfaces:**
- Produces: `crouch_height_target(phase: torch.Tensor, height_low: float, height_high: float, hold_lo: float = 0.375, hold_hi: float = 0.625) -> torch.Tensor` — takes the phase (B,) ∈ [0,1) and returns the target height (B,).

- [ ] **Step 1: Write the failing test**

Create `tests/test_crouch_glide.py`:

```python
import math
import torch
from mjlab_microduck.tasks import mdp


def test_crouch_height_target_endpoints_are_high():
    # phase 0 (start) and phase ~1 (end) → high stance (standing)
    phase = torch.tensor([0.0, 0.999])
    t = mdp.crouch_height_target(phase, height_low=0.075, height_high=0.11)
    assert torch.allclose(t, torch.tensor([0.11, 0.11]), atol=2e-3)


def test_crouch_height_target_plateau_is_low():
    # the whole plateau [0.375, 0.625] → constant low height
    phase = torch.tensor([0.375, 0.5, 0.624])
    t = mdp.crouch_height_target(phase, height_low=0.075, height_high=0.11)
    assert torch.allclose(t, torch.full((3,), 0.075), atol=1e-6)


def test_crouch_height_target_descent_midpoint():
    # midway through the descent (phase = hold_lo/2 = 0.1875) → midpoint of the two heights
    phase = torch.tensor([0.1875])
    t = mdp.crouch_height_target(phase, height_low=0.075, height_high=0.11)
    assert torch.allclose(t, torch.tensor([(0.11 + 0.075) / 2]), atol=1e-6)


def test_crouch_height_target_rise_midpoint():
    # midway through the rise (phase = 0.8125) → midpoint of the two heights
    phase = torch.tensor([0.8125])
    t = mdp.crouch_height_target(phase, height_low=0.075, height_high=0.11)
    assert torch.allclose(t, torch.tensor([(0.11 + 0.075) / 2]), atol=1e-6)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --with pytest pytest tests/test_crouch_glide.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'crouch_height_target'`

- [ ] **Step 3: Implement the function**

In `src/mjlab_microduck/tasks/mdp.py`, just after `com_height_target` (after line 737):

```python
def crouch_height_target(
    phase: torch.Tensor,
    height_low: float,
    height_high: float,
    hold_lo: float = 0.375,
    hold_hi: float = 0.625,
) -> torch.Tensor:
    """Trapezoidal trunk-height target along the phase [0,1).

    phase ∈ [0, hold_lo)      : descent   height_high -> height_low
    phase ∈ [hold_lo, hold_hi): plateau    height_low   (the crouched glide)
    phase ∈ [hold_hi, 1.0)    : rise       height_low  -> height_high

    Args:
        phase: (B,) per-env phase, in [0, 1).
        height_low: crouched trunk height (m).
        height_high: standing trunk height (m).
        hold_lo, hold_hi: bounds of the low plateau, as a fraction of phase.
    Returns:
        (B,) target height in meters.
    """
    descend = phase < hold_lo
    hold = (phase >= hold_lo) & (phase < hold_hi)

    frac_d = phase / hold_lo
    t_descend = height_high + (height_low - height_high) * frac_d

    t_hold = torch.full_like(phase, height_low)

    frac_r = (phase - hold_hi) / (1.0 - hold_hi)
    t_rise = height_low + (height_high - height_low) * frac_r

    return torch.where(descend, t_descend, torch.where(hold, t_hold, t_rise))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --with pytest pytest tests/test_crouch_glide.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/mjlab_microduck/tasks/mdp.py tests/test_crouch_glide.py
git commit -m "roller-crouch: trapezoidal height target (pure function + tests)"
```

---

## Task 2: The crouch-glide and forward-speed rewards

**Files:**
- Modify: `src/mjlab_microduck/tasks/mdp.py`
- Test: `tests/test_crouch_glide.py` (additions)

**Interfaces:**
- Consumes: `crouch_height_target` (Task 1).
- Produces:
  - `crouch_glide_reward_from_values(com_height, cmd_cos, cmd_sin, height_low, height_high, hold_lo=0.375, hold_hi=0.625, std=0.02) -> torch.Tensor` (pure).
  - `crouch_glide_height_by_phase(env, command_name="twist", height_low=0.075, height_high=0.11, hold_lo=0.375, hold_hi=0.625, std=0.02, asset_cfg=_DEFAULT_ASSET_CFG) -> torch.Tensor` (env wrapper).
  - `forward_speed_reward(env, vel_ref=0.2, asset_cfg=_DEFAULT_ASSET_CFG) -> torch.Tensor` — rewards forward speed (momentum), independent of the command.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_crouch_glide.py`:

```python
def test_reward_is_one_when_height_matches_target():
    # phase 0.5 (full plateau) → target = height_low; if com_height == height_low → reward 1
    cmd_cos = torch.tensor([math.cos(2 * math.pi * 0.5)])  # -1
    cmd_sin = torch.tensor([math.sin(2 * math.pi * 0.5)])  # ~0
    com_height = torch.tensor([0.075])
    r = mdp.crouch_glide_reward_from_values(
        com_height, cmd_cos, cmd_sin, height_low=0.075, height_high=0.11, std=0.02
    )
    assert torch.allclose(r, torch.tensor([1.0]), atol=1e-3)


def test_reward_decays_when_off_by_one_std():
    # at height_low + std from the target → exp(-1) ≈ 0.368
    cmd_cos = torch.tensor([math.cos(2 * math.pi * 0.5)])
    cmd_sin = torch.tensor([math.sin(2 * math.pi * 0.5)])
    com_height = torch.tensor([0.075 + 0.02])
    r = mdp.crouch_glide_reward_from_values(
        com_height, cmd_cos, cmd_sin, height_low=0.075, height_high=0.11, std=0.02
    )
    assert torch.allclose(r, torch.tensor([math.exp(-1.0)]), atol=1e-3)


def test_reward_at_phase_zero_expects_high_stance():
    # phase 0 → target = height_high; staying upright is rewarded, crouching is not
    cmd_cos = torch.tensor([1.0, 1.0])   # cos(0)
    cmd_sin = torch.tensor([0.0, 0.0])   # sin(0)
    com_height = torch.tensor([0.11, 0.075])  # standing vs crouched
    r = mdp.crouch_glide_reward_from_values(
        com_height, cmd_cos, cmd_sin, height_low=0.075, height_high=0.11, std=0.02
    )
    assert r[0] > 0.99          # standing at phase 0 → ~1
    assert r[1] < 0.2           # crouched at phase 0 → low
```

- [ ] **Step 2: Verify it fails**

Run: `uv run --with pytest pytest tests/test_crouch_glide.py -v`
Expected: FAIL — `crouch_glide_reward_from_values` does not exist.

- [ ] **Step 3: Implement the three functions**

In `src/mjlab_microduck/tasks/mdp.py`, following `crouch_height_target`:

```python
def crouch_glide_reward_from_values(
    com_height: torch.Tensor,
    cmd_cos: torch.Tensor,
    cmd_sin: torch.Tensor,
    height_low: float,
    height_high: float,
    hold_lo: float = 0.375,
    hold_hi: float = 0.625,
    std: float = 0.02,
) -> torch.Tensor:
    """Gaussian reward for tracking the height target (pure function).

    Decodes the phase from [cos, sin], then compares the measured height to the
    trapezoidal target. Returns exp(-((h - target)/std)^2) ∈ (0, 1].
    """
    phase = (torch.atan2(cmd_sin, cmd_cos) / (2 * torch.pi)) % 1.0
    target = crouch_height_target(phase, height_low, height_high, hold_lo, hold_hi)
    return torch.exp(-((com_height - target) / std) ** 2)


def crouch_glide_height_by_phase(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    height_low: float = 0.075,
    height_high: float = 0.11,
    hold_lo: float = 0.375,
    hold_hi: float = 0.625,
    std: float = 0.02,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Main reward: tracks the trunk-height target along the phase.

    The CoM height is computed as in `com_height_target` (world z minus the
    terrain origin, nan->0). The phase comes from the GroundPick command.
    """
    asset: Entity = env.scene[asset_cfg.name]
    com_height = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    cmd = env.command_manager.get_command(command_name)
    return crouch_glide_reward_from_values(
        com_height, cmd[:, 0], cmd[:, 1],
        height_low, height_high, hold_lo, hold_hi, std,
    )


def forward_speed_reward(
    env: ManagerBasedRlEnv,
    vel_ref: float = 0.2,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward the trunk's forward speed (keep the momentum / do not brake).

    Independent of the command (the command carries the phase, not the speed).
    tanh(clamp(vx, 0)/vel_ref) → saturates at ~1, never rewards going backward.
    """
    asset: Entity = env.scene[asset_cfg.name]
    vx = asset.data.root_link_lin_vel_b[:, 0]
    return torch.tanh(torch.clamp(vx, min=0.0) / vel_ref)
```

- [ ] **Step 4: Verify it passes**

Run: `uv run --with pytest pytest tests/test_crouch_glide.py -v`
Expected: PASS (7 tests in total)

- [ ] **Step 5: Commit**

```bash
git add src/mjlab_microduck/tasks/mdp.py tests/test_crouch_glide.py
git commit -m "roller-crouch: crouch-glide-height and forward-speed rewards"
```

---

## Task 3: The environment + task registration

**Files:**
- Create: `src/mjlab_microduck/tasks/microduck_roller_crouch_env_cfg.py`
- Modify: `src/mjlab_microduck/tasks/__init__.py`
- Test: `tests/test_roller_crouch_cfg.py`

**Interfaces:**
- Consumes: `crouch_glide_height_by_phase`, `forward_speed_reward`, `ground_pick_return_pose` (Task 2 + existing), `GroundPickPhaseCommandCfg`, `GroundPickPhaseCommand`, `MICRODUCK_WALK_ROLLERS_ROBOT_CFG`.
- Produces: `make_microduck_roller_crouch_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg`, `MicroduckRollerCrouchRlCfg`, and the `Mjlab-RollerCrouch-Flat-MicroDuck` task.

- [ ] **Step 1: Write the failing smoke test**

Create `tests/test_roller_crouch_cfg.py`:

```python
from mjlab_microduck.tasks.microduck_roller_crouch_env_cfg import (
    make_microduck_roller_crouch_env_cfg,
)
from mjlab_microduck.tasks import mdp as microduck_mdp


def test_cfg_uses_phase_command():
    cfg = make_microduck_roller_crouch_env_cfg()
    assert isinstance(
        cfg.commands["twist"], microduck_mdp.GroundPickPhaseCommandCfg
    )
    assert cfg.commands["twist"].period == 4.0


def test_cfg_has_crouch_and_forward_rewards():
    cfg = make_microduck_roller_crouch_env_cfg()
    assert "crouch_glide_height" in cfg.rewards
    assert "forward_speed" in cfg.rewards
    # active skating rewards removed (no stride during the trick)
    for gone in ("braking", "skating_air_time", "single_support", "glide", "wheel_speed"):
        assert gone not in cfg.rewards


def test_cfg_has_entry_velocity_event():
    cfg = make_microduck_roller_crouch_env_cfg()
    assert "entry_velocity" in cfg.events
```

- [ ] **Step 2: Verify it fails**

Run: `uv run --with pytest pytest tests/test_roller_crouch_cfg.py -v`
Expected: FAIL — `ModuleNotFoundError: ...microduck_roller_crouch_env_cfg`

- [ ] **Step 3: Create the environment file**

Create `src/mjlab_microduck/tasks/microduck_roller_crouch_env_cfg.py`:

```python
"""Microduck roller crouch-glide task.

One-shot gesture triggered by button A through the runtime's --ground-pick
slot: the robot crouches and glides on its momentum (~1 s plateau), then stands
back up and hands control back to the roller policy.

Hybrid:
  - roller physics / robot     ← microduck_velocity_rollers_env_cfg.py
  - one-shot phase machinery   ← microduck_ground_pick_env_cfg.py
    (GroundPickPhaseCommand: [cos(2πφ), sin(2πφ), 0], period 4 s)

Trapezoidal height target (high→low→1 s plateau→high) via
crouch_glide_height_by_phase. Unified 61D obs → hot-swappable at runtime.
"""

import math
from copy import deepcopy

ENABLE_SYMMETRY = False

# DR — taken from the roller env
ENABLE_COM_RANDOMIZATION             = True
ENABLE_HEAD_COM_RANDOMIZATION        = True
ENABLE_MASS_INERTIA_RANDOMIZATION    = True
ENABLE_JOINT_FRICTION_RANDOMIZATION  = True
ENABLE_ARMATURE_RANDOMIZATION        = True
ENABLE_WHEEL_FRICTION_RANDOMIZATION  = True
ENABLE_VELOCITY_PUSHES               = True
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS                  = True

COM_RANDOMIZATION_RANGE          = 0.003
HEAD_COM_RANDOMIZATION_RANGE     = 0.003
MASS_INERTIA_RANDOMIZATION_RANGE = (0.95, 1.05)
JOINT_FRICTION_RANDOMIZATION_RANGE = (0.9, 1.1)
ARMATURE_RANDOMIZATION_RANGE     = (0.9, 1.1)
VELOCITY_PUSH_INTERVAL_S         = (3.0, 6.0)
VELOCITY_PUSH_RANGE              = (-0.2, 0.2)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0
ENCODER_BIAS_RANGE               = (-0.015, 0.015)

# Gesture: target heights (m) and entry velocity (momentum)
CROUCH_HEIGHT_HIGH = 0.11    # standing trunk
CROUCH_HEIGHT_LOW  = 0.075   # crouched trunk (to be refined at play time)
CROUCH_STD         = 0.02
ENTRY_VELOCITY_X   = (0.2, 0.5)  # m/s: the robot arrives already rolling

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlModelCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_ROLLERS_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import HEAD_BODY_NAMES
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


def make_microduck_roller_crouch_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Crouch-glide-on-rollers env, driven by the ground-pick slot's phase."""

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(roller_blade|roller_blade_2)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    cfg = make_velocity_env_cfg()
    cfg.scene.entities = {"robot": MICRODUCK_WALK_ROLLERS_ROBOT_CFG}
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)
    cfg.viewer.body_name = "trunk_base"

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # === REWARDS ===
    keep = {"upright", "body_ang_vel", "angular_momentum", "action_rate_l2"}
    for name in list(cfg.rewards.keys()):
        if name not in keep:
            del cfg.rewards[name]

    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["upright"].weight = 2.0
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02
    cfg.rewards["action_rate_l2"].weight = -1.0

    # Main reward: trapezoidal height target along the phase
    cfg.rewards["crouch_glide_height"] = RewardTermCfg(
        func=microduck_mdp.crouch_glide_height_by_phase,
        weight=4.0,
        params={
            "command_name": "twist",
            "height_low": CROUCH_HEIGHT_LOW,
            "height_high": CROUCH_HEIGHT_HIGH,
            "hold_lo": 0.375,
            "hold_hi": 0.625,
            "std": CROUCH_STD,
        },
    )
    # Preserve the momentum (do not brake) — independent of the command
    cfg.rewards["forward_speed"] = RewardTermCfg(
        func=microduck_mdp.forward_speed_reward,
        weight=2.0,
        params={"vel_ref": 0.2},
    )
    # End of phase: converge to the standing roller pose for a clean handover
    _LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
    _NECK_JOINTS = [5, 6, 7, 8]
    cfg.rewards["return_pose_legs"] = RewardTermCfg(
        func=microduck_mdp.ground_pick_return_pose,
        weight=3.0,
        params={"std": 0.3, "command_name": "twist", "joint_indices": _LEG_JOINTS},
    )
    cfg.rewards["return_pose_neck"] = RewardTermCfg(
        func=microduck_mdp.ground_pick_return_pose,
        weight=3.0,
        params={"std": 0.15, "command_name": "twist", "joint_indices": _NECK_JOINTS},
    )
    # Glide stability
    cfg.rewards["feet_flat"] = RewardTermCfg(
        func=microduck_mdp.feet_flat_penalty,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
            "sensor_name": "feet_ground_contact",
        },
    )
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": "self_collision"},
    )
    cfg.rewards["neck_action_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.neck_action_rate_l2, weight=-0.5
    )
    cfg.rewards["joint_torques_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torques_l2, weight=-1e-3
    )

    # === TERMINATIONS ===
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan, time_out=False,
    )

    # === EVENTS ===
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history, mode="reset",
    )
    del cfg.events["foot_friction"]

    # Entry velocity: the robot starts rolling forward (momentum to preserve)
    cfg.events["entry_velocity"] = EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="reset",
        params={
            "velocity_range": {"x": ENTRY_VELOCITY_X, "y": (0.0, 0.0)},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    if ENABLE_VELOCITY_PUSHES:
        cfg.events["push_robot"] = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=VELOCITY_PUSH_INTERVAL_S,
            params={
                "velocity_range": {"x": VELOCITY_PUSH_RANGE, "y": VELOCITY_PUSH_RANGE},
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

    cfg.events["reset_base"].params["pose_range"]["z"] = (0.1335, 0.1435)

    if ENABLE_WHEEL_FRICTION_RANDOMIZATION:
        cfg.events["randomize_wheel_friction"] = EventTermCfg(
            func=dr.dof_frictionloss,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r"^passive_.*",)),
                "operation": "abs",
                "ranges": (0.000, 0.000),
            },
        )
    if ENABLE_COM_RANDOMIZATION:
        cfg.events["randomize_com"] = EventTermCfg(
            func=dr.body_ipos, mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "operation": "add",
                "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
            },
        )
    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.events["randomize_head_com"] = EventTermCfg(
            func=dr.body_ipos, mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES),
                "operation": "add",
                "ranges": (-HEAD_COM_RANDOMIZATION_RANGE, HEAD_COM_RANDOMIZATION_RANGE),
            },
        )
    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        _mi_lo, _mi_hi = MASS_INERTIA_RANDOMIZATION_RANGE
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia, mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "alpha_range": (math.log(_mi_lo) / 2.0, math.log(_mi_hi) / 2.0),
            },
        )
    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction, mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )
    if ENABLE_ARMATURE_RANDOMIZATION:
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature, mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",)),
                "operation": "scale",
                "ranges": ARMATURE_RANDOMIZATION_RANGE,
            },
        )

    # === OBSERVATIONS (unified 61D layout) ===
    del cfg.observations["actor"].terms["base_lin_vel"]
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel, scale=1.0,
    )

    gravity_term_name = "projected_gravity"
    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(
        cfg.observations["actor"].terms[gravity_term_name]
    )
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["actor"].terms["base_ang_vel"]
    )
    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 1
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64
    cfg.observations["actor"].terms["base_ang_vel"].noise = Unoise(n_min=-0.03, n_max=0.03)
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise = Unoise(n_min=-0.25, n_max=0.25)

    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        g = cfg.observations["actor"].terms[gravity_term_name]
        g.func = microduck_mdp.projected_gravity_imu_misaligned
        g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    cfg.observations["actor"].terms["joint_vel"] = deepcopy(
        cfg.observations["actor"].terms["joint_vel"]
    )
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    wheel_cfg = SceneEntityCfg("robot", joint_names=(r"^passive_.*",))
    cfg.observations["critic"].terms["wheel_vel"] = ObservationTermCfg(
        func=mdp.joint_vel_rel, scale=1.0, params={"asset_cfg": wheel_cfg},
    )

    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 4},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 6},
        )

    # === COMMAND: phase (like ground_pick) ===
    command: UniformVelocityCommandCfg = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    cfg.commands["twist"] = microduck_mdp.GroundPickPhaseCommandCfg(
        **{**vars(command), "class_type": microduck_mdp.GroundPickPhaseCommand, "period": 4.0}
    )

    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # === CURRICULUM ===
    del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.5},
                {"step": 250 * 24, "weight": -0.8},
                {"step": 500 * 24, "weight": -1.0},
            ],
        },
    )
    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    {"step": 0, "range": 0.003},
                    {"step": 500 * 24, "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                ],
            },
        )
    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_head_com",
                "range_stages": [
                    {"step": 0, "range": 0.003},
                    {"step": 500 * 24, "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                ],
            },
        )

    return cfg


MicroduckRollerCrouchRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
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
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="roller_crouch",
    run_name="roller_crouch",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=8_000,
)
```

- [ ] **Step 4: Register the task**

In `src/mjlab_microduck/tasks/__init__.py`, add the import after the rollers block (after line 54):

```python
from .microduck_roller_crouch_env_cfg import (
    make_microduck_roller_crouch_env_cfg,
    MicroduckRollerCrouchRlCfg,
)
```

and the registration after the rollers block (after line 175):

```python
register_mjlab_task(
    task_id="Mjlab-RollerCrouch-Flat-MicroDuck",
    env_cfg=make_microduck_roller_crouch_env_cfg(),
    play_env_cfg=make_microduck_roller_crouch_env_cfg(play=True),
    rl_cfg=MicroduckRollerCrouchRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ RollerCrouch task registered: Mjlab-RollerCrouch-Flat-MicroDuck")
```

- [ ] **Step 5: Verify the smoke test passes**

Run: `uv run --with pytest pytest tests/test_roller_crouch_cfg.py -v`
Expected: PASS (3 tests). (This test builds the env — it compiles the MuJoCo spec, so it is slower; that is normal.)

- [ ] **Step 6: Verify the task is properly registered**

Run: `uv run python -c "import mjlab_microduck.tasks"`
Expected: the line `✓ RollerCrouch task registered: Mjlab-RollerCrouch-Flat-MicroDuck` is printed without error.

- [ ] **Step 7: Commit**

```bash
git add src/mjlab_microduck/tasks/microduck_roller_crouch_env_cfg.py \
        src/mjlab_microduck/tasks/__init__.py tests/test_roller_crouch_cfg.py
git commit -m "roller-crouch: crouch-glide env + task registration"
```

---

## Task 4: Training smoke run (runtime verification)

**Files:** none (observational check).

**Interfaces:**
- Consumes: the `Mjlab-RollerCrouch-Flat-MicroDuck` task (Task 3).

- [ ] **Step 1: Run a very short training**

Run:
```bash
uv run train Mjlab-RollerCrouch-Flat-MicroDuck \
  --env.scene.num-envs 64 --agent.max_iterations 5
```
Expected: training starts, logs the rewards (including `crouch_glide_height`, `forward_speed`), runs 5 iterations without crashing, and writes a checkpoint.

- [ ] **Step 2: Check for obs-shape errors**

Inspect the startup log: the actor obs must be **61D** (like the other policies in the family). If the dimension differs, the head/body padding or the wheel exclusion is miswired — fix it before continuing.

- [ ] **Step 3: Commit (if a config file had to be adjusted)**

```bash
git add -A && git commit -m "roller-crouch: post smoke-run adjustment"
```
(If there is nothing to commit, skip this step.)

---

## Task 5: Full training + play verification

**Files:** possible iterations on `microduck_roller_crouch_env_cfg.py` (reward weights, `CROUCH_HEIGHT_LOW`).

- [ ] **Step 1: Run the full training**

Run:
```bash
uv run train Mjlab-RollerCrouch-Flat-MicroDuck \
  --env.scene.num-envs 4096 --agent.max_iterations 8000
```

- [ ] **Step 2: Visualize in play**

Run: `uv run scripts/play_latest.py` (or the project's play entry point for this task).
Watch the cycle: the robot **goes down**, **glides ~1 s** with the wheels still turning (it does not brake), then **stands back up** and the final pose rejoins the standing roller pose. It must not fall.

- [ ] **Step 3: Iterate if needed**

Typical adjustments (in `microduck_roller_crouch_env_cfg.py`):
- It does not go down far enough → lower `CROUCH_HEIGHT_LOW` (e.g. 0.07) and/or raise the `crouch_glide_height` weight.
- It brakes during the crouch → raise the `forward_speed` weight.
- It falls in the low position → raise `upright`, lower the entry velocity `ENTRY_VELOCITY_X`, or shorten the plateau (bring `hold_lo`/`hold_hi` closer together).
- The rise is brutal → raise `return_pose_*` and/or `action_rate_l2`.

After each change, retrain and re-visualize. Commit each adjustment you keep:
```bash
git add src/mjlab_microduck/tasks/microduck_roller_crouch_env_cfg.py
git commit -m "roller-crouch: tune <what changed>"
```

---

## Task 6: ONNX export + deployment on the robot

**Files:** none (manual / hardware).

- [ ] **Step 1: Export the policy to ONNX**

Run: `uv run scripts/export_latest.py` (the obs normalizer is baked into the graph by `scripts/export.py`).
Collect the `.onnx` file, rename it `roller_crouch.onnx`, and copy it onto the robot (e.g. `~/microduck/policies/roller_crouch.onnx`).

- [ ] **Step 2: Launch the runtime with the ground-pick slot**

On the robot:
```bash
microduck_runtime --variant pre-alpha --new-cmd-obs --roller \
  --model output.onnx \
  --new-dxl-imu --kp 200 --action-scale 0.8 \
  --max-linear-vel 0.6 --max-linear-vel-backward 0.5 --max-angular-vel 0.0 \
  --ground-pick ~/microduck/policies/roller_crouch.onnx \
  --ground-pick-period 5.0 \
  --ground-pick-kp-ratio 1.0 \
  --ground-pick-action-scale 0.8
```

**Critical parameters (sim2real parity):**
- `--ground-pick-kp-ratio 1.0` — the 0.6 default would lower kp to 120 while we train at 200.
- `--ground-pick-action-scale 0.8` — must match the training `action_scale`.
- `--ground-pick-period 5.0` — must match the trained period.

- [ ] **Step 3: Test the gesture**

Run the robot forward at low speed and press **A**. Check: it crouches, glides ~1 s, stands back up, and the roller policy cleanly takes back control. If unstable, go back to Task 5 (iterate on the weights / the height / the entry velocity).

---

## Verification notes (self-review)

- **Spec coverage:** 1 s trapezoidal target (Task 1); crouch + anti-braking + return-pose rewards (Task 2/3); roller robot + phase + 61D obs + DR (Task 3); entry velocity (Task 3, the `entry_velocity` event); deployment flags including the `kp-ratio` pitfall (Task 6). ✅
- **Phase vs velocity pitfall:** the roller env's `wheel_speed_reward`/`braking`/`coasting_reward` use `command[:,0]` as a *velocity* — invalid here, where `command[:,0]=cos(2πφ)`. They are therefore **removed** and replaced with `forward_speed_reward` (command-independent). Tested by `test_cfg_has_crouch_and_forward_rewards`.
- **Naming consistency:** `crouch_glide_height` (the reward key) vs `crouch_glide_height_by_phase` (the function) — intentional: the key is the term's name, the function goes in `func=`.
```
