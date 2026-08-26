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


# ── crouch_pose_blend: 4 segments (descent / low / rise / standing) ──────────
# test breakpoints: descent [0,0.1), low [0.1,0.5), rise [0.5,0.6),
# standing [0.6,1.0).
_BLEND = dict(descent_end=0.10, hold_end=0.50, rise_end=0.60)


def test_blend_zero_standing_at_start_and_top_hold():
    phase = torch.tensor([0.0, 0.6, 0.8, 0.999])  # start + high plateau
    b = mdp.crouch_pose_blend(phase, **_BLEND)
    assert torch.allclose(b, torch.zeros(4), atol=1e-6)


def test_blend_one_on_low_hold():
    phase = torch.tensor([0.10, 0.3, 0.499])  # low plateau
    b = mdp.crouch_pose_blend(phase, **_BLEND)
    assert torch.allclose(b, torch.ones(3), atol=1e-6)


def test_blend_descent_and_rise_midpoints():
    # descent midpoint (0.05 in [0,0.1)) → 0.5; rise midpoint (0.55 in [0.5,0.6)) → 0.5
    phase = torch.tensor([0.05, 0.55])
    b = mdp.crouch_pose_blend(phase, **_BLEND)
    assert torch.allclose(b, torch.tensor([0.5, 0.5]), atol=1e-6)


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
