import torch
from mjlab_microduck.tasks.mdp import slope_move_masks


def test_move_up_when_reached_bottom():
    # distance > size_x*0.4 (=3.2) → promote to a harder slope
    dist = torch.tensor([5.0, 4.1])
    up, down = slope_move_masks(dist, size_x=8.0)
    assert bool(up[0]) and bool(up[1])
    assert not bool(down[0]) and not bool(down[1])


def test_move_down_when_stuck_early():
    # distance < size_x*0.2 (=1.6) → demote to an easier slope
    dist = torch.tensor([0.5, 1.0])
    up, down = slope_move_masks(dist, size_x=8.0)
    assert not bool(up[0]) and not bool(up[1])
    assert bool(down[0]) and bool(down[1])


def test_stay_in_middle_band():
    # between 1.6 and 3.2 → neither up nor down
    dist = torch.tensor([2.5])
    up, down = slope_move_masks(dist, size_x=8.0)
    assert not bool(up[0]) and not bool(down[0])


def test_move_up_boundary_at_04():
    # promotion as soon as it has descended > 0.4*size_x (the robot covered a
    # good part of the ramp before reaching the flat runout).
    dist = torch.tensor([3.3])
    up, down = slope_move_masks(dist, size_x=8.0)
    assert bool(up[0])
    assert not bool(down[0])

    # 3.0 stays in the middle band (3.0 < 3.2 and 3.0 > 1.6)
    dist_mid = torch.tensor([3.0])
    up_mid, down_mid = slope_move_masks(dist_mid, size_x=8.0)
    assert not bool(up_mid[0]) and not bool(down_mid[0])
