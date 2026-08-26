# Ground-pick by pose following — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the `Mjlab-GroundPick-Flat-MicroDuck` task to drive the gesture through phase-interpolated joint-pose following (STAND→DOWN→STAND) instead of the current task-space objective (mouth-to-ground proximity + pose return).

**Architecture:** We add three pure/near-pure mdp functions (`phase_pose_blend`, `phase_pose_track`, `phase_pose_track_l1`) that compute a joint target interpolated between HOME (STAND) and a `DOWN_POSE` dict along a 4-segment phase profile, resolved **by name**. We add a `randomize_phase` flag to the existing phase command. We then rewrite the rewards block of `microduck_ground_pick_env_cfg.py`, keeping everything else (DR, 61D obs, curricula, RlCfg).

**Tech Stack:** Python, PyTorch, mjlab 1.3.0, MuJoCo, uv, pytest (via `uv run --with pytest`).

## Global Constraints

- Joints resolved **BY NAME** (`asset.find_joints([name])[0][0]`), never by hardcoded index.
- The unified 61D obs is **unchanged** (zero head/body padding) → the policy stays hot-swappable in the runtime slot.
- Task id unchanged: `Mjlab-GroundPick-Flat-MicroDuck` (+ the `-Rough-` variant).
- Phase period = **4.0 s** (the `--ground-pick-period` slot default).
- Phase profile (fractions): `DESCENT_END=0.15`, `HOLD_END=0.50`, `RISE_END=0.65`.
- `randomize_phase=False` for the ground_pick task (deployment parity with button A at φ=0); the cfg default stays `True` so as not to break sit/stand.
- STAND = HOME (`asset.data.default_joint_pos`, do not redefine). DOWN = the `DOWN_POSE` by-name dict.
- 14 active joints (mouth excluded). Robot `MICRODUCK_GROUND_PICK_ROBOT_CFG` (no wheels → indices 0-4 left leg, 5-8 neck/head, 9-13 right leg, but we still resolve by name).
- mdp file: the imports are already present (`torch`, `Optional`, `Entity`, `SceneEntityCfg`, `ManagerBasedRlEnv`, `_DEFAULT_ASSET_CFG`).

---

### Task 1: `phase_pose_blend` function (4-segment blend, pure)

**Files:**
- Modify: `src/mjlab_microduck/tasks/mdp.py` (add one function; insert it just before `phase_pose_match`, ~line 2041)
- Test: `tests/test_ground_pick_pose.py` (create)

**Interfaces:**
- Produces: `phase_pose_blend(phase: torch.Tensor, descent_end: float, hold_end: float, rise_end: float) -> torch.Tensor` — returns a blend ∈ [0,1] with the same shape as `phase` (0 = STAND, 1 = DOWN).

- [ ] **Step 1: Write the failing test**

Create `tests/test_ground_pick_pose.py`:

```python
import torch
from mjlab_microduck.tasks.mdp import phase_pose_blend

DESCENT_END, HOLD_END, RISE_END = 0.15, 0.50, 0.65


def test_phase_pose_blend_keypoints():
    phase = torch.tensor([0.0, 0.075, 0.15, 0.30, 0.50, 0.575, 0.65, 0.80])
    b = phase_pose_blend(phase, DESCENT_END, HOLD_END, RISE_END)
    expected = torch.tensor([0.0, 0.5, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0])
    assert torch.allclose(b, expected, atol=1e-6), b


def test_phase_pose_blend_range():
    phase = torch.linspace(0.0, 1.0, 101)
    b = phase_pose_blend(phase, DESCENT_END, HOLD_END, RISE_END)
    assert b.min() >= 0.0 and b.max() <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --with pytest pytest tests/test_ground_pick_pose.py -q`
Expected: FAIL — `ImportError: cannot import name 'phase_pose_blend'`

- [ ] **Step 3: Write minimal implementation**

In `src/mjlab_microduck/tasks/mdp.py`, just before `def phase_pose_match(` (~line 2041):

```python
def phase_pose_blend(
    phase: torch.Tensor,
    descent_end: float,
    hold_end: float,
    rise_end: float,
) -> torch.Tensor:
    """Blend 0..1 along the phase [0,1) — 0 = STAND pose, 1 = DOWN pose.

    [0, descent_end)       : 0 -> 1  (go down)
    [descent_end, hold_end): 1       (low)
    [hold_end, rise_end)   : 1 -> 0  (rise)
    [rise_end, 1.0)        : 0       (high / rest)
    """
    b = torch.zeros_like(phase)
    descend = phase < descent_end
    b = torch.where(descend, phase / descent_end, b)
    low = (phase >= descent_end) & (phase < hold_end)
    b = torch.where(low, torch.ones_like(phase), b)
    rise = (phase >= hold_end) & (phase < rise_end)
    b = torch.where(rise, 1.0 - (phase - hold_end) / (rise_end - hold_end), b)
    return b
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --with pytest pytest tests/test_ground_pick_pose.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add tests/test_ground_pick_pose.py src/mjlab_microduck/tasks/mdp.py
git commit -m "feat(mdp): phase_pose_blend - 4-segment STAND<->DOWN blend along the phase"
```

---

### Task 2: `phase_pose_track` / `phase_pose_track_l1` rewards (+ the `_phase_pose_error` helper)

**Files:**
- Modify: `src/mjlab_microduck/tasks/mdp.py` (add just after `phase_pose_blend`)
- Test: `tests/test_ground_pick_pose.py` (append)

**Interfaces:**
- Consumes: `phase_pose_blend` (Task 1).
- Produces:
  - `_phase_pose_error(env, asset_cfg, command_name, target_pose: dict, descent_end, hold_end, rise_end, source_pose: dict | None = None) -> (cur: Tensor, target: Tensor)` — (B, k) tensors resolved by name.
  - `phase_pose_track(env, command_name="twist", target_pose: dict | None = None, source_pose: dict | None = None, std=0.3, descent_end=0.15, hold_end=0.50, rise_end=0.65, asset_cfg=_DEFAULT_ASSET_CFG) -> Tensor` — gaussian `exp(-((cur-target)/std)²).mean(-1)`.
  - `phase_pose_track_l1(env, command_name="twist", target_pose=None, source_pose=None, descent_end=0.15, hold_end=0.50, rise_end=0.65, asset_cfg=_DEFAULT_ASSET_CFG) -> Tensor` — `-(cur-target).abs().mean(-1)`.

- [ ] **Step 1: Write the failing test**

Add a lightweight fake env + the assertions to `tests/test_ground_pick_pose.py`:

```python
from mjlab_microduck.tasks.mdp import phase_pose_track, phase_pose_track_l1


class _FakeData:
    def __init__(self, joint_pos, default_pos):
        self.joint_pos = joint_pos
        self.default_joint_pos = default_pos


class _FakeAsset:
    def __init__(self, names, joint_pos, default_pos):
        self._ids = {n: i for i, n in enumerate(names)}
        self.data = _FakeData(joint_pos, default_pos)

    def find_joints(self, query):
        # mjlab returns (ids, names); we only handle the [name] query form
        (name,) = query
        return ([self._ids[name]], [name])


class _FakeCmdMgr:
    def __init__(self, cmd):
        self._cmd = cmd

    def get_command(self, _name):
        return self._cmd


class _FakeEnv:
    def __init__(self, names, joint_pos, default_pos, phase):
        import math
        self.device = "cpu"
        self.scene = {"robot": _FakeAsset(names, joint_pos, default_pos)}
        ang = 2 * math.pi * phase
        cmd = torch.tensor([[math.cos(ang), math.sin(ang), 0.0]])
        self.command_manager = _FakeCmdMgr(cmd)


NAMES = ["j0", "j1"]
DOWN = {"j0": 1.0, "j1": -1.0}
# HOME (STAND source) = 0 for both joints
HOME = torch.tensor([[0.0, 0.0]])


def _env(cur, phase):
    return _FakeEnv(NAMES, torch.tensor([cur]), HOME.clone(), phase)


def test_phase_pose_track_perfect_at_down():
    # phase 0.30 -> blend 1 -> target = DOWN; cur == DOWN -> gaussian 1, l1 0
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    cfg = SceneEntityCfg("robot")
    env = _env([1.0, -1.0], phase=0.30)
    r = phase_pose_track(env, target_pose=DOWN, asset_cfg=cfg)
    assert torch.allclose(r, torch.tensor([1.0]), atol=1e-6), r
    env2 = _env([1.0, -1.0], phase=0.30)
    l1 = phase_pose_track_l1(env2, target_pose=DOWN, asset_cfg=cfg)
    assert torch.allclose(l1, torch.tensor([0.0]), atol=1e-6), l1


def test_phase_pose_track_l1_at_home_when_down_target():
    # phase 0.30 -> target DOWN=[1,-1]; cur=HOME=[0,0] -> l1 = -mean(|1|,|1|) = -1
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    cfg = SceneEntityCfg("robot")
    env = _env([0.0, 0.0], phase=0.30)
    l1 = phase_pose_track_l1(env, target_pose=DOWN, asset_cfg=cfg)
    assert torch.allclose(l1, torch.tensor([-1.0]), atol=1e-6), l1


def test_phase_pose_track_returns_to_stand():
    # phase 0.80 -> blend 0 -> target = HOME; cur=HOME -> gaussian 1
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    cfg = SceneEntityCfg("robot")
    env = _env([0.0, 0.0], phase=0.80)
    r = phase_pose_track(env, target_pose=DOWN, asset_cfg=cfg)
    assert torch.allclose(r, torch.tensor([1.0]), atol=1e-6), r
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --with pytest pytest tests/test_ground_pick_pose.py -q`
Expected: FAIL — `ImportError: cannot import name 'phase_pose_track'`

- [ ] **Step 3: Write minimal implementation**

In `src/mjlab_microduck/tasks/mdp.py`, just after `phase_pose_blend`:

```python
def _phase_pose_error(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    target_pose: dict,
    descent_end: float,
    hold_end: float,
    rise_end: float,
    source_pose: Optional[dict] = None,
):
    """(cur, target) for the phase-interpolated pose, resolved BY NAME.

    Target = source + blend(phase)·(target_pose - source), source = STAND
    (`source_pose` if given, otherwise the model's DEFAULT/HOME). blend ∈ [0,1]
    (0 = STAND, 1 = target_pose) via `phase_pose_blend`.
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    phase = (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0  # (B,)
    blend = phase_pose_blend(phase, descent_end, hold_end, rise_end)     # (B,)

    names = list(target_pose.keys())
    ids = [int(asset.find_joints([n])[0][0]) for n in names]
    default = asset.data.default_joint_pos[:, ids]                       # (B,k)

    source = default.clone()
    if source_pose:
        for j, n in enumerate(names):
            if n in source_pose:
                source[:, j] = source_pose[n]
    target_vec = torch.tensor(
        [target_pose[n] for n in names], device=env.device, dtype=default.dtype
    ).unsqueeze(0)                                                       # (1,k)

    target = source + blend.unsqueeze(-1) * (target_vec - source)        # (B,k)
    cur = asset.data.joint_pos[:, ids]                                   # (B,k)
    return cur, target


def phase_pose_track(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    target_pose: Optional[dict] = None,
    source_pose: Optional[dict] = None,
    std: float = 0.3,
    descent_end: float = 0.15,
    hold_end: float = 0.50,
    rise_end: float = 0.65,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Gaussian on joint pose vs the interpolated STAND<->DOWN target.

    A directive reward: it prescribes the exact joint configuration at each
    phase. Rising (target → STAND) is rewarded exactly like going down (target →
    DOWN) — symmetric by construction. Resolution BY NAME.
    """
    cur, target = _phase_pose_error(
        env, asset_cfg, command_name, target_pose or {},
        descent_end, hold_end, rise_end, source_pose,
    )
    return torch.exp(-((cur - target) / std) ** 2).mean(dim=-1)


def phase_pose_track_l1(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    target_pose: Optional[dict] = None,
    source_pose: Optional[dict] = None,
    descent_end: float = 0.15,
    hold_end: float = 0.50,
    rise_end: float = 0.65,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """L1 bootstrap toward the interpolated target (negative penalty).

    Constant gradient everywhere — gives a direction toward the target even when
    the Gaussian above has saturated to ~0 far from it.
    """
    cur, target = _phase_pose_error(
        env, asset_cfg, command_name, target_pose or {},
        descent_end, hold_end, rise_end, source_pose,
    )
    return -(cur - target).abs().mean(dim=-1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --with pytest pytest tests/test_ground_pick_pose.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add tests/test_ground_pick_pose.py src/mjlab_microduck/tasks/mdp.py
git commit -m "feat(mdp): phase_pose_track/_l1 - phase-interpolated pose following (by name)"
```

---

### Task 3: `randomize_phase` flag on `GroundPickPhaseCommandCfg`

**Files:**
- Modify: `src/mjlab_microduck/tasks/mdp.py` (the `GroundPickPhaseCommand` class ~3611/3626, cfg ~3644)
- Test: `tests/test_ground_pick_pose.py` (append)

**Interfaces:**
- Produces: `GroundPickPhaseCommandCfg.randomize_phase: bool = True`; `GroundPickPhaseCommand.reset()` sets the phase to 0 when `randomize_phase=False`, otherwise `torch.rand`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ground_pick_pose.py`:

```python
def test_ground_pick_cmd_cfg_has_randomize_phase_default_true():
    from mjlab_microduck.tasks.mdp import GroundPickPhaseCommandCfg
    from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
    # build a minimal cfg by copying a default velocity cfg
    base = UniformVelocityCommandCfg(
        asset_name="robot", resampling_time_range=(10.0, 10.0),
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0),
        ),
    )
    cfg = GroundPickPhaseCommandCfg(**{**vars(base)})
    assert cfg.randomize_phase is True
    assert cfg.period == 4.0
```

Note: if the local `UniformVelocityCommandCfg.Ranges` signature differs, adapt the fields — the key assertion is `cfg.randomize_phase is True`.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --with pytest pytest tests/test_ground_pick_pose.py::test_ground_pick_cmd_cfg_has_randomize_phase_default_true -q`
Expected: FAIL — `AttributeError: 'GroundPickPhaseCommandCfg' object has no attribute 'randomize_phase'`

- [ ] **Step 3: Write minimal implementation**

In `src/mjlab_microduck/tasks/mdp.py`, class `GroundPickPhaseCommand`, modify `__init__` and `reset`:

Replace (in `__init__`, ~line 3614):
```python
        self._period = float(getattr(cfg, "period", self.PERIOD))
```
with:
```python
        self._period = float(getattr(cfg, "period", self.PERIOD))
        self._randomize_phase = bool(getattr(cfg, "randomize_phase", True))
```

Replace the `reset` method (~line 3626):
```python
    def reset(self, env_ids: torch.Tensor | None) -> dict:
        if env_ids is not None and len(env_ids) > 0:
            self._gp_phase[env_ids] = torch.rand(len(env_ids), device=self.device)
        return {}
```
with:
```python
    def reset(self, env_ids: torch.Tensor | None) -> dict:
        if env_ids is not None and len(env_ids) > 0:
            if self._randomize_phase:
                self._gp_phase[env_ids] = torch.rand(len(env_ids), device=self.device)
            else:
                self._gp_phase[env_ids] = 0.0
        return {}
```

In the `GroundPickPhaseCommandCfg` cfg (~line 3644), add the field after `period`:
```python
@_dataclass(kw_only=True)
class GroundPickPhaseCommandCfg(UniformVelocityCommandCfg):
    class_type: type = GroundPickPhaseCommand
    period: float = 4.0  # cycle length in seconds; sitstand uses 8.0
    randomize_phase: bool = True  # False = every episode starts at φ=0 (button-A slot parity)

    def build(self, env: ManagerBasedRlEnv) -> "GroundPickPhaseCommand":
        return GroundPickPhaseCommand(self, env)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --with pytest pytest tests/test_ground_pick_pose.py::test_ground_pick_cmd_cfg_has_randomize_phase_default_true -q`
Expected: PASS. If constructing `UniformVelocityCommandCfg` fails for a local API reason, adjust the `base` fields in the test (the implementation itself is correct).

- [ ] **Step 5: Commit**

```bash
git add tests/test_ground_pick_pose.py src/mjlab_microduck/tasks/mdp.py
git commit -m "feat(mdp): randomize_phase flag on GroundPickPhaseCommandCfg (default True)"
```

---

### Task 4: Rewriting the rewards block + poses in the env cfg

**Files:**
- Modify: `src/mjlab_microduck/tasks/microduck_ground_pick_env_cfg.py`
- Test: `tests/test_ground_pick_cfg.py` (create)

**Interfaces:**
- Consumes: `phase_pose_track`, `phase_pose_track_l1` (Task 2); `randomize_phase` (Task 3).
- Produces: `make_microduck_ground_pick_env_cfg(play=False, rough=False)` returns a cfg whose: command is a `GroundPickPhaseCommand` with `randomize_phase=False`, `period=4.0`; rewards contain `phase_pose_track` (6.0) and `phase_pose_track_l1` (2.0), `mouth_ground_proximity` (1.0); and no longer contain `mouth_perpendicular_to_ground`, `ground_pick_return_pose_legs`, `ground_pick_return_pose_neck`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ground_pick_cfg.py`:

```python
from mjlab_microduck.tasks.microduck_ground_pick_env_cfg import (
    make_microduck_ground_pick_env_cfg,
)
from mjlab_microduck.tasks.mdp import GroundPickPhaseCommand


def test_ground_pick_cfg_builds_with_pose_rewards():
    cfg = make_microduck_ground_pick_env_cfg()
    rewards = cfg.rewards
    assert "phase_pose_track" in rewards
    assert "phase_pose_track_l1" in rewards
    assert rewards["phase_pose_track"].weight == 6.0
    assert rewards["phase_pose_track_l1"].weight == 2.0
    # mouth-to-ground safety net kept but lightened
    assert "mouth_ground_proximity" in rewards
    assert rewards["mouth_ground_proximity"].weight == 1.0
    # old mechanisms removed
    assert "mouth_perpendicular_to_ground" not in rewards
    assert "ground_pick_return_pose_legs" not in rewards
    assert "ground_pick_return_pose_neck" not in rewards


def test_ground_pick_cfg_command_is_phase_no_randomize():
    cfg = make_microduck_ground_pick_env_cfg()
    cmd = cfg.commands["twist"]
    assert cmd.class_type is GroundPickPhaseCommand
    assert cmd.period == 4.0
    assert cmd.randomize_phase is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --with pytest pytest tests/test_ground_pick_cfg.py -q`
Expected: FAIL — `assert 'phase_pose_track' in rewards` (KeyError/False).

- [ ] **Step 3: Write minimal implementation**

In `src/mjlab_microduck/tasks/microduck_ground_pick_env_cfg.py`:

(a) Add the pose/phase constants just before `def make_microduck_ground_pick_env_cfg(`:

```python
# ── Target poses of the gesture (rad, BY NAME) ────────────────────────────────
# STAND = HOME (the model's default_joint_pos) — do not redefine it here: it is
# the blend source. DOWN = deep forward fold (mouth toward the ground), initial
# values taken from the FOLD keyframe of scene_walk.xml. ⚠️ REPLACEABLE with a
# read_pose.py reading of the real robot placed mouth-to-ground when available.
DOWN_POSE = {
    "left_hip_yaw": 0.0, "left_hip_roll": 0.0, "left_hip_pitch": 1.57,
    "left_knee": 1.57, "left_ankle": 0.0,
    "neck_pitch": 1.0, "head_pitch": 1.0, "head_yaw": 0.0, "head_roll": 0.0,
    "right_hip_yaw": 0.0, "right_hip_roll": 0.0, "right_hip_pitch": -1.57,
    "right_knee": -1.57, "right_ankle": 0.0,
}

# Cycle timing (phase fractions), 4 s period:
#   descent [0, DESCENT_END) ~0.6s / low [DESCENT_END, HOLD_END) ~1.4s /
#   rise [HOLD_END, RISE_END) ~0.6s / rest [RISE_END, 1) ~1.4s
GP_PERIOD    = 4.0
DESCENT_END  = 0.15
HOLD_END     = 0.50
RISE_END     = 0.65
POSE_STD     = 0.3
```

(b) In the reward-removal loop (~lines 145-155), replace the gesture content. **Remove** the two `mouth_perpendicular_to_ground` blocks (~176-183) and the two `ground_pick_return_pose_*` blocks (~189-212), and **retune** `mouth_ground_proximity` to `weight=1.0` (~163-172, change `weight=2.0` → `weight=1.0`).

Concretely:
- Edit the `cfg.rewards["mouth_ground_proximity"]` block: `weight=2.0` → `weight=1.0`.
- Delete the `cfg.rewards["mouth_perpendicular_to_ground"] = RewardTermCfg(...)` block entirely.
- Delete the `_LEG_JOINTS = [...]` / `cfg.rewards["ground_pick_return_pose_legs"]` and `_NECK_JOINTS = [...]` / `cfg.rewards["ground_pick_return_pose_neck"]` blocks.
- Remove `"pose"` from the reward-removal list if present (unchanged) — but **also remove** the now-obsolete `# replaced by phase-conditioned ground_pick_return_pose` comment line (optional).

(c) Add the two new pose-following rewards (in place of the removed blocks, in the "main ground pick objectives" section):

```python
    # Phase-interpolated pose following (STAND<->DOWN<->STAND). Directive and
    # symmetric: the return to standing is rewarded exactly like the descent.
    cfg.rewards["phase_pose_track"] = RewardTermCfg(
        func=microduck_mdp.phase_pose_track,
        weight=6.0,
        params={
            "command_name": "twist",
            "target_pose": DOWN_POSE,
            "std": POSE_STD,
            "descent_end": DESCENT_END,
            "hold_end": HOLD_END,
            "rise_end": RISE_END,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["phase_pose_track_l1"] = RewardTermCfg(
        func=microduck_mdp.phase_pose_track_l1,
        weight=2.0,
        params={
            "command_name": "twist",
            "target_pose": DOWN_POSE,
            "descent_end": DESCENT_END,
            "hold_end": HOLD_END,
            "rise_end": RISE_END,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
```

(d) In the "Command" block (~line 368), pass the period and disable phase randomization:

Replace:
```python
    cfg.commands["twist"] = microduck_mdp.GroundPickPhaseCommandCfg(
        **{**vars(command), "class_type": microduck_mdp.GroundPickPhaseCommand}
    )
```
with:
```python
    cfg.commands["twist"] = microduck_mdp.GroundPickPhaseCommandCfg(
        **{
            **vars(command),
            "class_type": microduck_mdp.GroundPickPhaseCommand,
            "period": GP_PERIOD,
            "randomize_phase": False,
        }
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --with pytest pytest tests/test_ground_pick_cfg.py -q`
Expected: PASS (2 passed).

Then check that the whole suite passes:
Run: `uv run --with pytest pytest tests/ -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add tests/test_ground_pick_cfg.py src/mjlab_microduck/tasks/microduck_ground_pick_env_cfg.py
git commit -m "feat(ground_pick): phase-interpolated pose following (STAND->DOWN->STAND)"
```

---

### Task 5: End-to-end verification (runtime construction of the task)

**Files:**
- Test: `tests/test_ground_pick_cfg.py` (append)

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Write the failing/uncovered test**

Add to `tests/test_ground_pick_cfg.py`:

```python
def test_ground_pick_rough_variant_builds():
    cfg = make_microduck_ground_pick_env_cfg(rough=True)
    assert "phase_pose_track" in cfg.rewards


def test_ground_pick_play_variant_builds():
    cfg = make_microduck_ground_pick_env_cfg(play=True)
    assert cfg.commands["twist"].randomize_phase is False
```

- [ ] **Step 2: Run to verify**

Run: `uv run --with pytest pytest tests/test_ground_pick_cfg.py -q`
Expected: PASS.

- [ ] **Step 3: Verify the task registration (package import)**

Run: `uv run python -c "import mjlab_microduck.tasks; print('ok')"`
Expected: prints the `✓ ... registered` lines including `GroundPick`, then `ok`, without an exception.

- [ ] **Step 4: Commit**

```bash
git add tests/test_ground_pick_cfg.py
git commit -m "test(ground_pick): rough/play variants + package import"
```

---

## Self-Review

**1. Spec coverage:**
- §1 directive pose objective → Tasks 1,2,4. ✓
- §2 poses (STAND=HOME source, DOWN=FOLD by name) → Task 4 (a), Task 2 (`source_pose=None`→default). ✓
- §3 4-segment profile, 4 s period + `randomize_phase=False` → Task 1, Task 3, Task 4 (a,d). ✓
- §4 mdp functions `phase_pose_blend/track/_l1` by name → Tasks 1,2. ✓
- §5 rewards (additions + removals + mouth retune to 1.0) → Task 4 (b,c), Task 4 test. ✓
- §6 deployment (period 4, kp-ratio 1.0) → documented in the spec; period=4 verified in the Task 4 test. ✓
- §7 tests (pure functions + env construction) → Tasks 1,2,4,5. ✓
- §9 duplicate `pose_target_match` out of scope → not modified (compliant). ✓

**2. Placeholder scan:** no TODO/TBD; all the code is provided. ✓

**3. Type consistency:** `phase_pose_track(target_pose=..., std=..., asset_cfg=...)` and `phase_pose_track_l1(target_pose=..., asset_cfg=...)` are identical across Task 2 (def), Task 4 (call) and the tests. `randomize_phase` is consistent between Task 3 (def) and Task 4/the tests (usage). `GroundPickPhaseCommand`/`GroundPickPhaseCommandCfg` names unchanged. ✓
