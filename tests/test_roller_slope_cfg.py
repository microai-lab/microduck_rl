from mjlab_microduck.tasks.microduck_roller_slope_env_cfg import (
    make_microduck_roller_slope_env_cfg,
)
from mjlab_microduck.tasks.slope_terrain import FlatRampTerrainCfg


def test_terrain_is_flat_ramp_generator():
    cfg = make_microduck_roller_slope_env_cfg()
    assert cfg.scene.terrain.terrain_type == "generator"
    gen = cfg.scene.terrain.terrain_generator
    assert gen is not None and gen.curriculum is True
    assert any(isinstance(st, FlatRampTerrainCfg) for st in gen.sub_terrains.values())


def test_command_is_neutralised():
    cfg = make_microduck_roller_slope_env_cfg()
    cmd = cfg.commands["twist"]
    assert cmd.rel_standing_envs == 1.0
    assert cmd.rel_heading_envs == 0.0
    assert cmd.ranges.lin_vel_x == (0.0, 0.0)
    assert cmd.ranges.lin_vel_y == (0.0, 0.0)
    if getattr(cmd.ranges, "ang_vel_z", None) is not None:
        assert cmd.ranges.ang_vel_z == (0.0, 0.0)


def test_rolling_entry_no_base_push():
    # momentum given as ROLLING (reset_rolling_entry), not as a base push (base
    # alone + stationary wheels = skidding jolt). So reset_base sets no base
    # velocity at all.
    cfg = make_microduck_roller_slope_env_cfg()
    assert cfg.events["reset_base"].params["velocity_range"] == {}
    assert "reset_rolling_entry" in cfg.events
    lo, hi = cfg.events["reset_rolling_entry"].params["speed_range"]
    assert 0.0 < lo <= hi <= 0.6


def test_has_heading_hold_reward():
    # go straight: hold the spawn yaw
    cfg = make_microduck_roller_slope_env_cfg()
    assert "heading_hold" in cfg.rewards
    assert cfg.rewards["heading_hold"].weight > 0.0


def test_balance_rewards_no_fixed_pose():
    # free balance: upright/alive/glide present, but NO imposed fixed pose (it
    # must be able to move its center of gravity to hold the slope).
    cfg = make_microduck_roller_slope_env_cfg()
    for name in ("upright", "alive", "feet_flat", "wheel_glide", "neck_joint_pos_l2"):
        assert name in cfg.rewards
    assert "standing_pose" not in cfg.rewards
    assert "standing_pose_l1" not in cfg.rewards


def test_has_wheel_glide_reward_not_base_speed():
    # "letting itself glide" = rolling (wheels), not rewarding base speed (which
    # it reached by running). wheel_glide present, descent_speed absent.
    cfg = make_microduck_roller_slope_env_cfg()
    assert "wheel_glide" in cfg.rewards
    assert cfg.rewards["wheel_glide"].weight > 0.0
    assert "descent_speed" not in cfg.rewards


def test_no_roller_skating_rewards_survive():
    # the roller's SKATING rewards must not survive (heading_hold is deliberately
    # re-added to go straight, hence not in this list).
    cfg = make_microduck_roller_slope_env_cfg()
    for name in ("wheel_speed", "braking", "skating_air_time", "glide", "forward_lean"):
        assert name not in cfg.rewards


def test_spawn_yaw_faces_downhill():
    # yaw fixed at 0: always facing downhill (+x), not the inherited -pi/+pi
    cfg = make_microduck_roller_slope_env_cfg()
    assert cfg.events["reset_base"].params["pose_range"]["yaw"] == (0.0, 0.0)


def test_void_termination_present_no_edge_termination():
    cfg = make_microduck_roller_slope_env_cfg()
    assert "fell_into_void" in cfg.terminations
    assert "fell_over" in cfg.terminations
    # no more "terrain edge" termination (replaced by the flat runout)
    assert "reached_bottom" not in cfg.terminations
    assert "out_of_terrain_bounds" not in cfg.terminations


def test_obs_nan_policy_sanitize():
    # sanitized obs: a rare contact NaN must not kill training
    cfg = make_microduck_roller_slope_env_cfg()
    assert cfg.observations["actor"].nan_policy == "sanitize"
    assert cfg.observations["critic"].nan_policy == "sanitize"


def test_curriculum_present_and_starts_gentle():
    # gentle->steep curriculum: starts on the gentlest ramp, promotion enabled
    cfg = make_microduck_roller_slope_env_cfg()  # play=False (training)
    assert "terrain_levels" in cfg.curriculum
    assert cfg.scene.terrain.max_init_terrain_level == 0


def test_terrain_tile_fits_geometry():
    # the tile must fit flat + ramp_max + runout
    cfg = make_microduck_roller_slope_env_cfg()
    gen = cfg.scene.terrain.terrain_generator
    st = next(iter(gen.sub_terrains.values()))
    assert st.flat_length + st.ramp_length_range[1] + st.runout_length <= gen.size[0]
