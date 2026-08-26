# Kick task by pose following — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an RL task `Mjlab-Shoot-Flat-MicroDuck` that learns a one-shot kick gesture (right leg) by following a 4-keyframe pose trajectory (STAND → FOOT_BACK → FOOT_FORWARD → STAND) interpolated along the phase.

**Architecture:** The same mold as this branch's `ground_pick` task. A phase command (`GroundPickPhaseCommand`, `[cos,sin,0]`) drives a joint target interpolated between 3 poses; gaussian + L1 rewards pay for the tracking; a unified 61D obs for deployment in a runtime button slot. No simulated ball.

**Tech Stack:** Python, PyTorch, mjlab 1.3.0, MuJoCo, uv, pytest.

## Global Constraints

- **Unified 61D obs**, identical to the other microduck policies (`[gyro(3), projected_gravity(3), joint_pos(14), joint_vel(14), last_action(14), command(13)]`, head+body command zero-padded). Do not break this shape.
- Joints resolved **BY NAME** (`asset.find_joints([name])`), never by hardcoded index.
- **14 active joints** (mouth excluded). Robot `MICRODUCK_WALK_ROBOT_CFG`.
- Do not modify the Rust runtime, and do not change the command class in a breaking way: the added `randomize_phase` flag MUST default to `True` to preserve `ground_pick`.
- The **right** leg kicks, the **left** provides support.
- Tests: `uv run --with pytest pytest tests/ -q`.
- Commit convention: `feat:`/`docs:`/`test:` style messages.

---

## File Structure

- `src/mjlab_microduck/tasks/mdp.py` — MODIFY: add `kick_pose_target` (pure), `_kick_pose_error`, `kick_pose_track`, `kick_pose_track_l1`; add the `randomize_phase` flag to `GroundPickPhaseCommand` / `GroundPickPhaseCommandCfg`.
- `src/mjlab_microduck/tasks/microduck_shoot_env_cfg.py` — CREATE: `make_microduck_shoot_env_cfg`, `MicroduckShootRlCfg`, `STAND_POSE`/`KICK_BACK_POSE`/`KICK_FWD_POSE`, timings.
- `src/mjlab_microduck/tasks/__init__.py` — MODIFY: import + `register_mjlab_task("Mjlab-Shoot-Flat-MicroDuck", …)`.
- `tests/test_shoot.py` — CREATE: tests of the pure functions (`kick_pose_target`) + the rewards via a stub env.
- `tests/test_shoot_cfg.py` — CREATE: integration test (the env builds, with the right command/rewards).

---

### Task 1: `randomize_phase` flag on the phase command

**Files:**
- Modify: `src/mjlab_microduck/tasks/mdp.py:3618-3672` (`GroundPickPhaseCommand` + `GroundPickPhaseCommandCfg`)
- Test: `tests/test_shoot.py`

**Interfaces:**
- Produces: `GroundPickPhaseCommandCfg(randomize_phase: bool = True, period: float = 4.0, …)`; at runtime `reset()` sets φ=0 when `randomize_phase=False`, otherwise `rand()`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_shoot.py` with:

```python
from mjlab_microduck.tasks.mdp import GroundPickPhaseCommandCfg


def test_phase_cmd_randomize_flag_default_true():
    cfg = GroundPickPhaseCommandCfg()
    assert cfg.randomize_phase is True


def test_phase_cmd_randomize_flag_settable_false():
    cfg = GroundPickPhaseCommandCfg(randomize_phase=False)
    assert cfg.randomize_phase is False
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `uv run --with pytest pytest tests/test_shoot.py -q`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'randomize_phase'`.

- [ ] **Step 3: Add the field to the cfg + thread it through the class**

In `GroundPickPhaseCommandCfg` (dataclass, ~line 3667) add the field:

```python
@_dataclass(kw_only=True)
class GroundPickPhaseCommandCfg(UniformVelocityCommandCfg):
    class_type: type = GroundPickPhaseCommand
    period: float = 4.0  # cycle length in seconds; sitstand uses 8.0
    randomize_phase: bool = True  # False -> every episode starts at φ=0 (STAND)

    def build(self, env: ManagerBasedRlEnv) -> "GroundPickPhaseCommand":
        return GroundPickPhaseCommand(self, env)
```

In `GroundPickPhaseCommand.__init__` (~line 3634) read the flag:

```python
    def __init__(self, cfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._gp_phase = torch.zeros(self.num_envs, device=self.device)
        self._period = float(getattr(cfg, "period", self.PERIOD))
        self._randomize_phase = bool(getattr(cfg, "randomize_phase", True))
```

In `GroundPickPhaseCommand.reset` (~line 3649) honor the flag:

```python
    def reset(self, env_ids: torch.Tensor | None) -> dict:
        if env_ids is not None and len(env_ids) > 0:
            if self._randomize_phase:
                self._gp_phase[env_ids] = torch.rand(len(env_ids), device=self.device)
            else:
                self._gp_phase[env_ids] = 0.0
        return {}
```

- [ ] **Step 4: Run the test, verify it passes**

Run: `uv run --with pytest pytest tests/test_shoot.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/mjlab_microduck/tasks/mdp.py tests/test_shoot.py
git commit -m "feat: randomize_phase flag on GroundPickPhaseCommand (default True)"
```

---

### Task 2: Pure function `kick_pose_target`

**Files:**
- Modify: `src/mjlab_microduck/tasks/mdp.py` (add near `phase_pose_blend`, ~line 2062)
- Test: `tests/test_shoot.py`

**Interfaces:**
- Produces: `kick_pose_target(phase: Tensor(B,), stand, back, forward, windup_end: float, kick_end: float, return_end: float) -> Tensor(B,k)`. `stand/back/forward` are `(k,)` or `(1,k)` tensors. Segments: [0,windup_end) STAND→BACK, [windup_end,kick_end) BACK→FORWARD, [kick_end,return_end) FORWARD→STAND, [return_end,1) STAND.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_shoot.py`:

```python
import torch
from mjlab_microduck.tasks.mdp import kick_pose_target

W, K, R = 0.35, 0.45, 0.75  # windup_end, kick_end, return_end
STAND = torch.tensor([0.0, 0.0])
BACK = torch.tensor([1.0, -1.0])
FWD = torch.tensor([-1.0, 2.0])


def _t(phase):
    return kick_pose_target(torch.tensor([phase]), STAND, BACK, FWD, W, K, R)[0]


def test_kick_target_keypoints():
    assert torch.allclose(_t(0.0), STAND)          # start: STAND
    assert torch.allclose(_t(W), BACK)             # end of wind-up: BACK
    assert torch.allclose(_t(K), FWD)              # end of strike: FORWARD
    assert torch.allclose(_t(R), STAND)            # end of return: STAND
    assert torch.allclose(_t(0.9), STAND)          # rest: STAND


def test_kick_target_midsegments():
    assert torch.allclose(_t(W / 2), 0.5 * BACK)                    # mid wind-up
    assert torch.allclose(_t((W + K) / 2), 0.5 * (BACK + FWD))      # mid strike
    assert torch.allclose(_t((K + R) / 2), 0.5 * FWD)              # mid return


def test_kick_target_batch_shape():
    phase = torch.linspace(0.0, 1.0, 50)
    out = kick_pose_target(phase, STAND, BACK, FWD, W, K, R)
    assert out.shape == (50, 2)
    # each component stays within the envelope of the 3 poses
    lo = torch.minimum(torch.minimum(STAND, BACK), FWD)
    hi = torch.maximum(torch.maximum(STAND, BACK), FWD)
    assert (out >= lo - 1e-6).all() and (out <= hi + 1e-6).all()
```

- [ ] **Step 2: Run, verify it fails**

Run: `uv run --with pytest pytest tests/test_shoot.py -q`
Expected: FAIL — `ImportError: cannot import name 'kick_pose_target'`.

- [ ] **Step 3: Implement the pure function**

Add to `mdp.py` just after `phase_pose_blend` (~line 2062):

```python
def kick_pose_target(
    phase: torch.Tensor,
    stand: torch.Tensor,
    back: torch.Tensor,
    forward: torch.Tensor,
    windup_end: float,
    kick_end: float,
    return_end: float,
) -> torch.Tensor:
    """Interpolated joint target for a 4-keyframe kick gesture.

    phase (B,) ∈ [0,1). stand/back/forward (k,) or (1,k). Returns (B,k).

    [0, windup_end)        STAND   -> BACK     (wind-up)
    [windup_end, kick_end) BACK    -> FORWARD  (sharp strike)
    [kick_end, return_end) FORWARD -> STAND    (return)
    [return_end, 1.0)      STAND             (rest)
    """
    p = phase.unsqueeze(-1)  # (B,1)

    def interp(a, b, s):
        return a + s * (b - a)

    s1 = (p / windup_end).clamp(0.0, 1.0)
    s2 = ((p - windup_end) / (kick_end - windup_end)).clamp(0.0, 1.0)
    s3 = ((p - kick_end) / (return_end - kick_end)).clamp(0.0, 1.0)

    seg1 = interp(stand, back, s1)
    seg2 = interp(back, forward, s2)
    seg3 = interp(forward, stand, s3)  # at s3=1 (phase>=return_end) => STAND

    out = seg1
    out = torch.where(p >= windup_end, seg2, out)
    out = torch.where(p >= kick_end, seg3, out)
    return out
```

- [ ] **Step 4: Run, verify it passes**

Run: `uv run --with pytest pytest tests/test_shoot.py -q`
Expected: PASS (all the kick_target tests).

- [ ] **Step 5: Commit**

```bash
git add src/mjlab_microduck/tasks/mdp.py tests/test_shoot.py
git commit -m "feat: kick_pose_target - interpolated target of the kick gesture (4 keyframes)"
```

---

### Task 3: Tracking rewards `kick_pose_track` / `kick_pose_track_l1`

**Files:**
- Modify: `src/mjlab_microduck/tasks/mdp.py` (add after `kick_pose_target`)
- Test: `tests/test_shoot.py`

**Interfaces:**
- Consumes: `kick_pose_target` (Task 2).
- Produces:
  - `kick_pose_track(env, command_name="twist", stand_pose=None, back_pose=None, forward_pose=None, std=0.4, windup_end=0.35, kick_end=0.45, return_end=0.75, asset_cfg=_DEFAULT_ASSET_CFG) -> Tensor(B,)` — gaussian `exp(-((q-target)/std)²).mean`.
  - `kick_pose_track_l1(env, …same args without std) -> Tensor(B,)` — `-(|q-target|).mean`.
  - Helper `_kick_pose_error(env, asset_cfg, command_name, stand_pose, back_pose, forward_pose, windup_end, kick_end, return_end) -> (cur, target)`.

- [ ] **Step 1: Write the failing test (stub env)**

Add to `tests/test_shoot.py`:

```python
from mjlab_microduck.tasks.mdp import kick_pose_track, kick_pose_track_l1

STAND_D = {"a": 0.0, "b": 0.0}
BACK_D = {"a": 1.0, "b": -1.0}
FWD_D = {"a": -1.0, "b": 2.0}
_IDX = {"a": 0, "b": 1}


class _FakeData:
    def __init__(self, joint_pos):
        self.joint_pos = joint_pos
        self.default_joint_pos = torch.zeros_like(joint_pos)


class _FakeAsset:
    def __init__(self, joint_pos):
        self.data = _FakeData(joint_pos)

    def find_joints(self, names):
        return ([_IDX[names[0]]], names)


class _FakeScene:
    def __init__(self, asset):
        self._a = asset

    def __getitem__(self, name):
        return self._a


class _FakeCmdMgr:
    def __init__(self, cmd):
        self._cmd = cmd

    def get_command(self, name):
        return self._cmd


class _FakeEnv:
    def __init__(self, joint_pos, phase):
        self.scene = _FakeScene(_FakeAsset(joint_pos))
        # cmd = [cos, sin, 0]
        cmd = torch.stack(
            [torch.cos(2 * torch.pi * phase), torch.sin(2 * torch.pi * phase),
             torch.zeros_like(phase)], dim=-1)
        self.command_manager = _FakeCmdMgr(cmd)
        self.device = "cpu"
        self.num_envs = joint_pos.shape[0]


def test_kick_track_perfect_at_stand_phase():
    # phase=0 -> target STAND=[0,0]; joint_pos exactly STAND -> reward ~1
    env = _FakeEnv(torch.tensor([[0.0, 0.0]]), torch.tensor([0.0]))
    r = kick_pose_track(env, stand_pose=STAND_D, back_pose=BACK_D, forward_pose=FWD_D)
    assert torch.allclose(r, torch.tensor([1.0]), atol=1e-4)


def test_kick_track_lower_when_off_target():
    # phase=0.45 (kick_end) -> target FORWARD=[-1,2]; joint_pos=STAND -> reward < 0.5
    env = _FakeEnv(torch.tensor([[0.0, 0.0]]), torch.tensor([0.45]))
    r = kick_pose_track(env, stand_pose=STAND_D, back_pose=BACK_D, forward_pose=FWD_D)
    assert (r < 0.5).all()


def test_kick_track_l1_zero_when_perfect():
    env = _FakeEnv(torch.tensor([[0.0, 0.0]]), torch.tensor([0.0]))
    r = kick_pose_track_l1(env, stand_pose=STAND_D, back_pose=BACK_D, forward_pose=FWD_D)
    assert torch.allclose(r, torch.tensor([0.0]), atol=1e-6)
```

- [ ] **Step 2: Run, verify it fails**

Run: `uv run --with pytest pytest tests/test_shoot.py -q`
Expected: FAIL — `ImportError: cannot import name 'kick_pose_track'`.

- [ ] **Step 3: Implement the helper + rewards**

Add to `mdp.py` after `kick_pose_target`:

```python
def _kick_pose_error(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    stand_pose: dict,
    back_pose: dict,
    forward_pose: dict,
    windup_end: float,
    kick_end: float,
    return_end: float,
):
    """(cur, target) for the kick gesture, joints resolved BY NAME.

    The 3 poses share the same keys (14 joints). The name ordering comes from
    `stand_pose`.
    """
    if not stand_pose:
        raise ValueError("_kick_pose_error requires a non-empty stand_pose dict")
    asset: Entity = env.scene[asset_cfg.name]
    names = list(stand_pose.keys())
    ids = [int(asset.find_joints([n])[0][0]) for n in names]

    def vec(d):
        return torch.tensor([d[n] for n in names], device=env.device,
                            dtype=asset.data.joint_pos.dtype)

    stand_v, back_v, fwd_v = vec(stand_pose), vec(back_pose), vec(forward_pose)

    cmd = env.command_manager.get_command(command_name)
    phase = (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0  # (B,)
    target = kick_pose_target(phase, stand_v, back_v, fwd_v,
                              windup_end, kick_end, return_end)          # (B,k)
    cur = asset.data.joint_pos[:, ids]                                   # (B,k)
    return cur, target


def kick_pose_track(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    stand_pose: Optional[dict] = None,
    back_pose: Optional[dict] = None,
    forward_pose: Optional[dict] = None,
    std: float = 0.4,
    windup_end: float = 0.35,
    kick_end: float = 0.45,
    return_end: float = 0.75,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Gaussian on joint pose vs the interpolated kick target.

    A directive, symmetric reward: each phase prescribes the exact joint
    configuration. Resolution BY NAME.
    """
    cur, target = _kick_pose_error(
        env, asset_cfg, command_name, stand_pose or {}, back_pose or {},
        forward_pose or {}, windup_end, kick_end, return_end,
    )
    return torch.exp(-((cur - target) / std) ** 2).mean(dim=-1)


def kick_pose_track_l1(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    stand_pose: Optional[dict] = None,
    back_pose: Optional[dict] = None,
    forward_pose: Optional[dict] = None,
    windup_end: float = 0.35,
    kick_end: float = 0.45,
    return_end: float = 0.75,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """L1 bootstrap toward the interpolated target (constant gradient, penalty<=0)."""
    cur, target = _kick_pose_error(
        env, asset_cfg, command_name, stand_pose or {}, back_pose or {},
        forward_pose or {}, windup_end, kick_end, return_end,
    )
    return -(cur - target).abs().mean(dim=-1)
```

- [ ] **Step 4: Run, verify it passes**

Run: `uv run --with pytest pytest tests/test_shoot.py -q`
Expected: PASS (all tests, including the 3 new ones).

- [ ] **Step 5: Commit**

```bash
git add src/mjlab_microduck/tasks/mdp.py tests/test_shoot.py
git commit -m "feat: kick_pose_track + kick_pose_track_l1 rewards (kick gesture tracking)"
```

---

### Task 4: Env config `microduck_shoot_env_cfg.py`

**Files:**
- Create: `src/mjlab_microduck/tasks/microduck_shoot_env_cfg.py`
- Test: (via Task 5)

**Interfaces:**
- Consumes: `kick_pose_track`, `kick_pose_track_l1` (Task 3); `GroundPickPhaseCommandCfg(randomize_phase=…)` (Task 1); `feet_grounded_reward`, `feet_flat_penalty`, `neck_action_rate_l2`, `joint_torques_l2`, `zero_command_padding`, `robot_state_is_nan`, and the DR events (already in `mdp.py`).
- Produces: `make_microduck_shoot_env_cfg(play=False, rough=False) -> ManagerBasedRlEnvCfg`; `MicroduckShootRlCfg`; the constants `SHOOT_PERIOD`, `WINDUP_END`, `KICK_END`, `RETURN_END`, `STAND_POSE`, `KICK_BACK_POSE`, `KICK_FWD_POSE`.

- [ ] **Step 1: Start from the ground_pick file as a base**

```bash
cp src/mjlab_microduck/tasks/microduck_ground_pick_env_cfg.py \
   src/mjlab_microduck/tasks/microduck_shoot_env_cfg.py
```

That file already provides ALL the sim2real boilerplate to keep as-is: the DR (CoM, head CoM, mass/inertia, BAM friction, armature, obs-level IMU misalignment, encoder bias, pushes), the 61D obs block (`del base_lin_vel` on the actor, critic base_lin_vel, removal of `foot_height`/`height_scan`, delays/noise, `head_command`/`body_command` zero-padding), the `nan_state` termination, the `expand_bam_friction_fields` / `reset_action_history` events, and the action_rate/CoM curriculum. We modify only: the robot cfg, the sensors, the command, and the rewards block.

- [ ] **Step 2: Adapt the header, the function name and the constants**

Replace the top docstring with a kick description, and just before `def make_microduck_ground_pick_env_cfg`, add the constants + poses (placeholders — to be replaced with a `read_pose.py` reading). Rename the function to `make_microduck_shoot_env_cfg`.

```python
# ── Gesture timings (normalized phase [0,1)) ─────────────────────────────────
SHOOT_PERIOD = 2.5   # s — cycle duration (must match --ground-pick-period at deployment)
WINDUP_END = 0.35    # STAND -> BACK
KICK_END = 0.45      # BACK -> FORWARD (short segment = sharp strike)
RETURN_END = 0.75    # FORWARD -> STAND, then rest until 1.0

# ── Poses (rad, 14 joints, mouth excluded) ───────────────────────────────────
# Convention: the right leg kicks (right hip/knee active), the left provides support.
# STAND_POSE = the sim's HOME pose (HOME_FRAME / default_joint_pos) so that φ=0
# coincides with the reset configuration (the randomize_phase=False invariant).
# BACK/FWD are right-leg PLACEHOLDERS, to be refined via read_pose.py.
STAND_POSE = {
    "left_hip_yaw": 0.0, "left_hip_roll": -0.0873, "left_hip_pitch": -0.4579,
    "left_knee": -0.0049, "left_ankle": 0.4530,
    "neck_pitch": 0.3491, "head_pitch": 0.3491, "head_yaw": 0.0, "head_roll": 0.0,
    "right_hip_yaw": 0.0, "right_hip_roll": 0.0873, "right_hip_pitch": 0.4579,
    "right_knee": 0.0049, "right_ankle": -0.4530,
}
KICK_BACK_POSE = {  # wind-up: right hip in backward extension + knee flexed
    **STAND_POSE,
    "right_hip_pitch": -0.6,
    "right_knee": 0.8,
    "right_ankle": -0.2,
}
KICK_FWD_POSE = {  # strike: right hip flexed forward + knee extended
    **STAND_POSE,
    "right_hip_pitch": 0.7,
    "right_knee": -0.1,
    "right_ankle": 0.1,
}
```

> NOTE for whoever records the poses: replace these values with `read_pose.py` readings (torque off, robot placed by hand in each position). Keep the same 14 keys in all 3 dicts.

- [ ] **Step 3: Robot cfg and import**

In the imports, replace `MICRODUCK_GROUND_PICK_ROBOT_CFG` with `MICRODUCK_WALK_ROBOT_CFG`:

```python
from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_ROBOT_CFG
```

In the function, the entities line:

```python
    cfg.scene.entities = {"robot": MICRODUCK_WALK_ROBOT_CFG}
```

- [ ] **Step 4: Sensors — keep self_collision, replace the foot sensors**

Replace the `feet_ground_contact` sensor definition (2 feet) with a **left-foot-only** sensor (the support foot), and DELETE the `head_impact_cfg` sensor (useless here). The `self_collision_cfg` sensor stays.

```python
    left_foot_ground_cfg = ContactSensorCfg(
        name="left_foot_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^left_foot_collision$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
```

And the scene sensors line:

```python
    cfg.scene.sensors = (left_foot_ground_cfg, self_collision_cfg)
```

Delete the `head_impact_cfg` definition and every reference to it (the `head_impact_penalty` reward is removed in Step 6).

- [ ] **Step 5: Phase command (randomize_phase=False, kick period)**

Replace the command block (the one that creates `GroundPickPhaseCommandCfg`) with:

```python
    command: UniformVelocityCommandCfg = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    cfg.commands["twist"] = microduck_mdp.GroundPickPhaseCommandCfg(
        **{**vars(command), "class_type": microduck_mdp.GroundPickPhaseCommand}
    )
    cfg.commands["twist"].period = SHOOT_PERIOD
    cfg.commands["twist"].randomize_phase = False
```

- [ ] **Step 6: Rewards — remove ground_pick, add the kick**

Delete the ground_pick-specific rewards: `mouth_ground_proximity`, `mouth_perpendicular_to_ground`, `ground_pick_return_pose_legs`, `ground_pick_return_pose_neck`, `feet_grounded` (both feet), `head_impact_penalty`. Replace them with the kick block:

```python
    # ── Objective: follow the interpolated kick pose ──────────────────────────
    _pose_params = {
        "command_name": "twist",
        "stand_pose": STAND_POSE,
        "back_pose": KICK_BACK_POSE,
        "forward_pose": KICK_FWD_POSE,
        "windup_end": WINDUP_END,
        "kick_end": KICK_END,
        "return_end": RETURN_END,
    }
    cfg.rewards["kick_pose_track"] = RewardTermCfg(
        func=microduck_mdp.kick_pose_track,
        weight=6.0,
        params={**_pose_params, "std": 0.4},
    )
    cfg.rewards["kick_pose_l1"] = RewardTermCfg(
        func=microduck_mdp.kick_pose_track_l1,
        weight=2.0,
        params=dict(_pose_params),
    )

    # ── Balance / support (single leg) ────────────────────────────────────────
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["upright"].weight = 2.0
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05

    # LEFT foot planted (support). feet_grounded_reward with a single-foot sensor
    # -> found ∈ {0,1} -> reward ∈ {0,0.5}; weight 6.0 => max contribution ~3.0.
    cfg.rewards["support_foot_grounded"] = RewardTermCfg(
        func=microduck_mdp.feet_grounded_reward,
        weight=6.0,
        params={"sensor_name": left_foot_ground_cfg.name},
    )

    # Left foot flat.
    cfg.rewards["feet_flat_left"] = RewardTermCfg(
        func=microduck_mdp.feet_flat_penalty,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", site_names=("left_foot",))},
    )

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )
```

- [ ] **Step 7: Lighter regularization (let the snap through)**

The ground_pick file sets `action_rate_l2=-2.0`, `neck_action_rate_l2=-1.0`, `joint_torques_l2=-5e-3` plus an action_rate curriculum that ends at -2.0. For the kick we lighten it. Replace those 3 blocks with:

```python
    cfg.rewards["action_rate_l2"] = RewardTermCfg(
        func=mdp.action_rate_l2, weight=-0.5
    )
    cfg.rewards["neck_action_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.neck_action_rate_l2, weight=-0.5
    )
    cfg.rewards["joint_torques_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torques_l2, weight=-1e-3
    )
```

And lighten the action_rate curriculum (keep the structure, target -0.5):

```python
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0,        "weight": -0.2},
                {"step": 250 * 24, "weight": -0.4},
                {"step": 500 * 24, "weight": -0.5},
            ],
        },
    )
```

- [ ] **Step 8: Reset — standing stance height**

Keep the **standing height** `(0.12, 0.13)` — that is the value used by the velocity
(walking) env AND by ground_pick. ⚠️ This is NOT an additive "crouched stance" offset:
the default root `pos` of `InitialStateCfg` is (0,0,0), so the reset height is
z ∈ [0.12, 0.13] m **absolute** = standing (no fall). Check/set:

```python
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.12, 0.13)
```

(Do NOT inject an entry velocity — this is a standing kick, not a glide.)

- [ ] **Step 9: Rename the RlCfg**

At the bottom of the file, rename `MicroduckGroundPickRlCfg` to `MicroduckShootRlCfg` and change the experiment names:

```python
MicroduckShootRlCfg = RslRlOnPolicyRunnerCfg(
    # … (keep actor/critic/algorithm identical) …
    wandb_project="mjlab_microduck",
    experiment_name="shoot",
    run_name="shoot",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=20_000,
)
```

- [ ] **Step 10: Check that the module imports**

Run: `uv run python -c "from mjlab_microduck.tasks.microduck_shoot_env_cfg import make_microduck_shoot_env_cfg, MicroduckShootRlCfg; print('ok')"`
Expected: `ok` (no ImportError / NameError — in particular, no remaining reference to `head_impact_cfg`, `MICRODUCK_GROUND_PICK_ROBOT_CFG`, or the deleted ground_pick rewards).

- [ ] **Step 11: Commit**

```bash
git add src/mjlab_microduck/tasks/microduck_shoot_env_cfg.py
git commit -m "feat: Mjlab-Shoot env config (kick gesture by pose following)"
```

---

### Task 5: Registration + integration test

**Files:**
- Modify: `src/mjlab_microduck/tasks/__init__.py`
- Test: `tests/test_shoot_cfg.py`

**Interfaces:**
- Consumes: `make_microduck_shoot_env_cfg`, `MicroduckShootRlCfg` (Task 4).
- Produces: the registered task `Mjlab-Shoot-Flat-MicroDuck`.

- [ ] **Step 1: Write the failing integration test**

Create `tests/test_shoot_cfg.py`:

```python
from mjlab_microduck.tasks.microduck_shoot_env_cfg import (
    make_microduck_shoot_env_cfg,
    STAND_POSE, KICK_BACK_POSE, KICK_FWD_POSE, SHOOT_PERIOD,
)
from mjlab_microduck.tasks import mdp as microduck_mdp


def test_poses_have_same_14_keys():
    assert set(STAND_POSE) == set(KICK_BACK_POSE) == set(KICK_FWD_POSE)
    assert len(STAND_POSE) == 14
    assert "mouth" not in STAND_POSE


def test_shoot_cfg_builds_with_phase_command():
    cfg = make_microduck_shoot_env_cfg()
    twist = cfg.commands["twist"]
    assert isinstance(twist, microduck_mdp.GroundPickPhaseCommandCfg)
    assert twist.randomize_phase is False
    assert twist.period == SHOOT_PERIOD


def test_shoot_cfg_has_kick_rewards_and_no_walking():
    cfg = make_microduck_shoot_env_cfg()
    assert "kick_pose_track" in cfg.rewards
    assert "kick_pose_l1" in cfg.rewards
    assert "support_foot_grounded" in cfg.rewards
    for gone in ("track_linear_velocity", "track_angular_velocity",
                 "mouth_ground_proximity", "ground_pick_return_pose_legs"):
        assert gone not in cfg.rewards
```

- [ ] **Step 2: Run, verify it fails**

Run: `uv run --with pytest pytest tests/test_shoot_cfg.py -q`
Expected: the pose tests may PASS, but the whole file should only go green once the env builds without error; if `make_...` raises, FAIL. (At this stage importing the file already works thanks to Task 4.)

- [ ] **Step 3: Register the task**

In `src/mjlab_microduck/tasks/__init__.py`, after the ground_pick import block (~line 50), add:

```python
from .microduck_shoot_env_cfg import (
    make_microduck_shoot_env_cfg,
    MicroduckShootRlCfg,
)
```

After the GroundPick-Rough `register_mjlab_task` block (~line 161), add:

```python
register_mjlab_task(
    task_id="Mjlab-Shoot-Flat-MicroDuck",
    env_cfg=make_microduck_shoot_env_cfg(),
    play_env_cfg=make_microduck_shoot_env_cfg(play=True),
    rl_cfg=MicroduckShootRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ Shoot task registered: Mjlab-Shoot-Flat-MicroDuck")
```

- [ ] **Step 4: Run everything, verify it passes**

Run: `uv run --with pytest pytest tests/ -q`
Expected: PASS (test_shoot.py + test_shoot_cfg.py + the existing tests).

- [ ] **Step 5: Verify the task registration**

Run: `uv run python -c "import mjlab_microduck.tasks"`
Expected: the output contains `✓ Shoot task registered: Mjlab-Shoot-Flat-MicroDuck`.

- [ ] **Step 6: Commit**

```bash
git add src/mjlab_microduck/tasks/__init__.py tests/test_shoot_cfg.py
git commit -m "feat: register Mjlab-Shoot-Flat-MicroDuck + integration test"
```

---

## After implementation (outside the TDD plan)

1. **Record the real poses** with `read_pose.py` (STAND, FOOT_BACK, FOOT_FORWARD), and replace the placeholders in `microduck_shoot_env_cfg.py`.
2. **Train**: `uv run train Mjlab-Shoot-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max_iterations 8000`. Watch `Episode_Reward/kick_pose_track` (it must rise).
3. **Play**: the play_latest script; check the balance on the left foot during the strike.
4. **ONNX export** + deployment into a phase slot (`--ground-pick shoot.onnx --ground-pick-period 2.5 --ground-pick-kp-ratio 1.0`).
5. **Likely adjustments**: period/timings (snap), the `action_rate` weight, and possibly a "forward foot velocity" reward (strike segment) if the pose following lacks punch.

## Self-review — spec coverage

- File & registration → Tasks 4, 5. ✅
- 14-joint placeholder poses → Task 4 Step 2, tested in Task 5. ✅
- Phase command + `randomize_phase=False` + period → Tasks 1, 4 Step 5, tested in Task 5. ✅
- `kick_pose_target` + `kick_pose_track` + `kick_pose_track_l1` → Tasks 2, 3. ✅
- Balance/support (upright, left foot planted, left feet_flat, self_collisions, body_ang_vel) → Task 4 Step 6. ✅
- Lighter regularization → Task 4 Step 7. ✅
- 61D obs parity (inherited from ground_pick, preserved) → Task 4 Step 1. ✅
- Pure-function + cfg tests → Tasks 2, 3, 5. ✅
