import pytest

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
        # gentle_rise: POSITIVE weight. trunk_vertical_accel_penalty already
        # returns -|a_z|, so a negative weight turned it into a REWARD for
        # violence (measured bug: Episode_Reward/gentle_rise logged at +0.0118).
        "gentle_rise":         +0.02,
        "upright_linear":       6.0,
        "upright_sharp":        6.0,
        "standing_composite":  15.0,
        # -2e-3 contributed only -0.0002/step against +41.6 of task reward: nil.
        # -2.0 measured -0.255/step (run d8rnko6p) — not the freeze, but we drop
        # back to -0.2 to free up the damping budget while we isolate it.
        "joint_torque_rate_l2": -0.2,
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
    # Face down and face up share a single z range, but their contacts differ:
    # face down only clears the ground from 0.0752 up, face up rests at 0.0475.
    # prone_z_min = 0.076 to eliminate any interpenetration on the face-down side.
    assert (params["prone_z_min"], params["prone_z_max"]) == (0.076, 0.09)
    # Below 0.0752 (measured contact, HOME pose), a face-down start begins INSIDE
    # the ground — a contact pushout the policy would pay for through gentle_rise /
    # joint_torque_rate_l2. prone_z_min must stay above it.
    assert params["prone_z_min"] >= 0.0752
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


def test_wheel_friction_event_default_matches_stage_zero():
    # The curriculum manager runs BEFORE the reset events on every reset
    # (including the very first), and wheel_friction_curriculum itself defaults to
    # stage 0: this event default is therefore never read in practice. We just
    # check it stays consistent with the curriculum's stage 0 — defensive
    # redundancy, useful if the curriculum ever disappears while the event stays.
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


# ── Play override: force face-up starts ──────────────────────────────────────
# Without an override, a play run NEVER shows a face-up start: the play env is
# rebuilt from scratch, so common_step_counter restarts at 0 and the curriculum
# applies its stage 0, where face_up_prob = 0. Yet that is precisely the hardest
# case, the one we want to eyeball. STANDUP_PLAY_FACE_UP forces the mix, modeled
# on SLOPE_PLAY_DIFFICULTY in roller_slope.


def test_play_face_up_override_forces_back_starts(monkeypatch):
    monkeypatch.setenv("STANDUP_PLAY_FACE_UP", "1.0")
    cfg = make_microduck_roller_standup_env_cfg(play=True)
    params = cfg.events["set_ground_state"].params
    assert params["face_up_prob"] == 1.0
    assert params["face_down_prob"] == 0.0
    assert params["standing_prob"] == 0.0
    # Without this, the curriculum would rewrite the probabilities on the very
    # first reset (event_param_curriculum runs BEFORE the reset events).
    assert "ground_state_mix" not in cfg.curriculum


def test_play_face_up_override_splits_remainder_like_final_stage(monkeypatch):
    # 0.4 must reproduce the curriculum's LAST stage (0.40 face down / 0.20
    # standing / 0.40 face up): the remainder is split in that stage's 2:1 ratio.
    monkeypatch.setenv("STANDUP_PLAY_FACE_UP", "0.4")
    params = make_microduck_roller_standup_env_cfg(play=True).events["set_ground_state"].params
    assert params["face_up_prob"] == pytest.approx(0.40)
    assert params["face_down_prob"] == pytest.approx(0.40)
    assert params["standing_prob"] == pytest.approx(0.20)
    total = params["face_up_prob"] + params["face_down_prob"] + params["standing_prob"]
    assert total == pytest.approx(1.0)


def test_play_face_up_override_is_clamped(monkeypatch):
    monkeypatch.setenv("STANDUP_PLAY_FACE_UP", "3.0")
    params = make_microduck_roller_standup_env_cfg(play=True).events["set_ground_state"].params
    assert params["face_up_prob"] == 1.0


def test_play_face_up_override_ignored_during_training(monkeypatch):
    # Guard rail: the variable must NEVER touch training, otherwise we would
    # break the easy->hard curriculum without noticing.
    monkeypatch.setenv("STANDUP_PLAY_FACE_UP", "1.0")
    cfg = make_microduck_roller_standup_env_cfg(play=False)
    assert cfg.events["set_ground_state"].params["face_up_prob"] == 0.00
    assert "ground_state_mix" in cfg.curriculum


def test_play_without_override_keeps_curriculum_mix(monkeypatch):
    # Default behavior unchanged: stage 0, no face-up start.
    monkeypatch.delenv("STANDUP_PLAY_FACE_UP", raising=False)
    cfg = make_microduck_roller_standup_env_cfg(play=True)
    assert cfg.events["set_ground_state"].params["face_up_prob"] == 0.00
    assert "ground_state_mix" in cfg.curriculum


def test_play_face_up_override_invalid_value_falls_back(monkeypatch):
    monkeypatch.setenv("STANDUP_PLAY_FACE_UP", "pouet")
    cfg = make_microduck_roller_standup_env_cfg(play=True)
    assert cfg.events["set_ground_state"].params["face_up_prob"] == 0.00
    assert "ground_state_mix" in cfg.curriculum


def test_play_face_up_override_none_keyword_disables(monkeypatch):
    monkeypatch.setenv("STANDUP_PLAY_FACE_UP", "none")
    cfg = make_microduck_roller_standup_env_cfg(play=True)
    assert cfg.events["set_ground_state"].params["face_up_prob"] == 0.00
    assert "ground_state_mix" in cfg.curriculum


# ── Anti-violence: fixes after testing on the robot ──────────────────────────
# Symptoms observed (checkpoint 4000+, IN SIM TOO, so not a sim2real issue):
# very abrupt motions, the head banging the ground, failure to rise from the back
# on the real robot. Diagnosis measured in wandb (run vweolw91, iter 7500).


def test_already_negative_penalties_use_positive_weights():
    """Lock-in against the class of bug that made the policy violent.

    mdp.py mixes TWO sign conventions: some penalty functions return a positive
    magnitude (to be multiplied by a negative weight), others already return a
    negative value (to be multiplied by a POSITIVE weight).
    trunk_vertical_accel_penalty returns -|a_z|: with the -0.02 weight inherited
    from standup, the double negative REWARDED vertical acceleration — measured at
    Episode_Reward/gentle_rise = +0.0118, the only penalty term logged positive.
    """
    cfg = make_microduck_roller_standup_env_cfg()
    # These three terms call functions that already return negative values
    # (height_l1_penalty, pose_l1_penalty, trunk_vertical_accel_penalty).
    for name in ("height_stand_l1", "pose_stand_l1", "gentle_rise"):
        assert cfg.rewards[name].weight > 0, (
            f"{name} calls a function that already returns a negative value: "
            f"a negative weight would turn it into a reward"
        )
    # And these terms return a positive magnitude → negative weight.
    for name in ("joint_torques_l2", "joint_torque_rate_l2", "action_rate_l2"):
        assert cfg.rewards[name].weight < 0, f"{name} expects a negative weight"


def test_no_ungated_head_impact_penalty():
    """NO ungated head-impact penalty — it froze the policy.

    Tried at -1.0 (velstand's values): the policy converged to lying down, inert.
    Measured on run d8rnko6p: head_impact_penalty -1.01/step, the largest negative
    term, while standing_composite collapsed from +14.3 to +3.3.

    The reasoning error was believing that a "targeted" penalty does not restrict
    motion. False here: to get up from its back, this robot PIVOTS on its head and
    shoulders. The head is the fulcrum of the roll-over, not collateral damage —
    penalizing it means penalizing the only available mechanism.

    If the slam comes back once the gentle_rise sign is fixed, the reintroduction
    must be a HEIGHT-GATED penalty (as upright_sharp is), which spares the ground
    roll-over phase. Not this one.
    """
    cfg = make_microduck_roller_standup_env_cfg()
    assert "head_impact_penalty" not in cfg.rewards
    assert "head_impact_contact" not in [s.name for s in cfg.scene.sensors]


def test_inherited_sensors_intact():
    # The sensors inherited from the roller env are used by kept rewards
    # (self_collisions) and by the observations.
    cfg = make_microduck_roller_standup_env_cfg()
    names = [s.name for s in cfg.scene.sensors]
    assert "feet_ground_contact" in names
    assert "self_collision" in names


def test_lazy_prone_optimum_is_documented_risk():
    """The freeze comes from a lazy optimum: lying down, legs at HOME, still pays.

    pose_stand_legs stayed at +7.72 out of 8 while the robot was lying flat — the
    legs are at HOME in a prone position, so the pose reward is collected almost
    for free. That is the counterweight that makes "do nothing" viable as soon as
    a cost on motion is added. height_stand_l1 (weight +30) is the term meant to
    make "stay on the ground" net negative: it must stay strong.
    """
    cfg = make_microduck_roller_standup_env_cfg()
    assert cfg.rewards["height_stand_l1"].weight >= 30.0
    assert cfg.rewards["com_upward_velocity"].weight > 0.0


def test_damping_terms_are_not_numerically_negligible():
    """The dedicated dampers weighed literally nothing.

    Measured at convergence: joint_torque_rate_l2 -0.0002/step and joint_torques_l2
    -0.0001/step, against ~+41.6 of task reward (a ~35:1 ratio for all the dampers
    combined). joint_torque_rate_l2 is the SAFE lever to turn up: it penalizes
    torque RATE, not motion, so it does not act as a motion blocker — standup
    documents that body_ang_vel and action_rate did freeze the rise from the back.
    """
    cfg = make_microduck_roller_standup_env_cfg()
    assert abs(cfg.rewards["joint_torque_rate_l2"].weight) >= 0.1
    # The motion blockers stay at their "gets up from anywhere" values.
    assert cfg.rewards["body_ang_vel"].weight == -0.05
    weights = [s["weight"] for s in cfg.curriculum["action_rate_weight"].params["weight_stages"]]
    assert min(weights) >= -1.0, "action_rate beyond -1.0 froze the rise (standup)"
