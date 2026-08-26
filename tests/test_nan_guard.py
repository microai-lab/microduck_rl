"""robot_state_is_nan must catch a non-finite state anywhere (joints OR base OR
wheels), not just in joint_pos — otherwise a free joint that diverges to NaN
escapes the reset and corrupts the critic obs (base_lin_vel/wheel_vel), which
kills training through rsl_rl's global check_nan.
"""

import torch

from mjlab_microduck.tasks.mdp import robot_state_is_nan


class _Data:
    def __init__(self, n):
        self.joint_pos = torch.zeros(n, 4)
        self.joint_vel = torch.zeros(n, 4)
        self.root_link_pos_w = torch.zeros(n, 3)
        self.root_link_quat_w = torch.zeros(n, 4)
        self.root_link_lin_vel_w = torch.zeros(n, 3)
        self.root_link_ang_vel_w = torch.zeros(n, 3)


class _Asset:
    def __init__(self, data):
        self.data = data


class _Scene:
    def __init__(self, asset):
        self._a = asset

    def __getitem__(self, _key):
        return self._a


class _Env:
    def __init__(self, data):
        self.scene = _Scene(_Asset(data))


def test_catches_base_linear_velocity_nan():
    # env 1: NaN base velocity (diverged free joint) — joint_pos stays finite.
    d = _Data(3)
    d.root_link_lin_vel_w[1, 0] = float("nan")
    out = robot_state_is_nan(_Env(d))
    assert out.tolist() == [False, True, False]


def test_catches_base_velocity_inf():
    # inf in the base angular velocity (before it turns into NaN).
    d = _Data(2)
    d.root_link_ang_vel_w[0, 2] = float("inf")
    out = robot_state_is_nan(_Env(d))
    assert out.tolist() == [True, False]


def test_still_catches_joint_pos_nan():
    # historical behavior preserved.
    d = _Data(2)
    d.joint_pos[0, 1] = float("nan")
    out = robot_state_is_nan(_Env(d))
    assert out.tolist() == [True, False]


def test_clean_state_is_not_flagged():
    out = robot_state_is_nan(_Env(_Data(4)))
    assert out.tolist() == [False, False, False, False]
