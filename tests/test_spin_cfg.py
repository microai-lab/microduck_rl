from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_spin_env_cfg import (
    make_microduck_spin_env_cfg,
    MicroduckSpinRlCfg,
)


def test_cfg_uses_phase_command_with_runtime_default_period():
    cfg = make_microduck_spin_env_cfg()
    cmd = cfg.commands["twist"]
    assert isinstance(cmd, microduck_mdp.GroundPickPhaseCommandCfg)
    # 4.0 s = the default of --ground-pick-period: nothing to pass at runtime
    assert cmd.period == 4.0
    # every episode starts at phase 0 (standing), like the button at deployment
    assert cmd.randomize_phase is False


def test_cfg_has_the_spin_rewards():
    cfg = make_microduck_spin_env_cfg()
    for name in (
        "spin_rate_track",
        "spin_rate_l1",
        "spin_stay_in_place",
        "spin_wheel_differential",
        "spin_grounded",
        "leg_antisymmetry",
    ):
        assert name in cfg.rewards, name
    # main objective with a dominant weight
    assert cfg.rewards["spin_rate_track"].weight == 6.0
    # staying in place is a COST
    assert cfg.rewards["spin_stay_in_place"].weight < 0.0


def test_stay_in_place_is_attenuated_during_the_launch_ramp():
    # Strengthened to -3.0, this term would oppose the injection of angular
    # momentum if it were full price during the launch ramp: it must be attenuated
    # there.
    cfg = make_microduck_spin_env_cfg()
    params = cfg.rewards["spin_stay_in_place"].params
    assert 0.0 < params["launch_scale"] < 1.0
    assert params["accel_end"] == microduck_mdp.SPIN_ACCEL_END
    # positive target = counter-clockwise (the direction is carried by the envelope)
    assert microduck_mdp.SPIN_RATE_MAX > 0.0


def test_angular_momentum_reward_is_removed():
    # Regression: angular_momentum_penalty penalizes the 3D NORM of angular
    # momentum, so it would fight the spin head-on. It must be absent.
    cfg = make_microduck_spin_env_cfg()
    assert "angular_momentum" not in cfg.rewards
    # body_ang_vel only penalizes x/y -> it stays, it damps the wobble
    assert "body_ang_vel" in cfg.rewards


def test_head_yaw_is_free_to_act_as_a_flywheel():
    cfg = make_microduck_spin_env_cfg()
    pattern = cfg.rewards["neck_joint_pos_l2"].params["pattern"]
    assert "head_yaw" not in pattern


def test_entry_velocity_allows_standstill_and_slow_roll():
    cfg = make_microduck_spin_env_cfg()
    # never through a reset-mode push (the crouch's NaN regression)
    assert "entry_velocity" not in cfg.events
    lo, hi = cfg.events["reset_base"].params["velocity_range"]["x"]
    assert lo == 0.0 and hi > 0.0


def test_symmetry_augmentation_is_disabled():
    # L/R symmetry would turn a left spin into a right spin
    assert MicroduckSpinRlCfg.algorithm.symmetry_cfg is None


def test_leg_antisymmetry_shaping_decays():
    cfg = make_microduck_spin_env_cfg()
    stages = cfg.curriculum["leg_antisym_weight"].params["weight_stages"]
    weights = [s["weight"] for s in stages]
    assert weights[0] == cfg.rewards["leg_antisymmetry"].weight
    assert weights == sorted(weights, reverse=True)
    assert weights[-1] < weights[0]


def test_actor_observation_keeps_the_61d_slot_layout():
    # required for the ONNX to load in the runtime slot. Exact dimensional
    # equality with the crouch is checked by test_obs_parity_with_roller_crouch
    # below; here we check the structure.
    cfg = make_microduck_spin_env_cfg()
    terms = cfg.observations["actor"].terms
    assert "base_lin_vel" not in terms
    assert "height_scan" not in terms
    for padded in ("head_command", "body_command"):
        assert padded in terms
    assert terms["head_command"].params["dim"] == 4
    assert terms["body_command"].params["dim"] == 6


def test_obs_parity_with_roller_crouch():
    # Layout parity is mandatory: otherwise the exported ONNX will not load in the
    # runtime slot. Unlike the structural test above, this one compares the EXACT
    # ordering of the terms, group by group.
    from mjlab_microduck.tasks.microduck_roller_crouch_env_cfg import (
        make_microduck_roller_crouch_env_cfg,
    )

    spin = make_microduck_spin_env_cfg()
    crouch = make_microduck_roller_crouch_env_cfg()
    for grp in ("actor", "critic"):
        assert list(spin.observations[grp].terms.keys()) == list(
            crouch.observations[grp].terms.keys()
        ), f"observation layout diverged on group {grp}"
