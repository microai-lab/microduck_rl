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
import os

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

# ── Play override: force the share of FACE-UP starts ──────────────────────────
# At play time the env is rebuilt from scratch: common_step_counter restarts at
# 0, so the ground_state_mix curriculum applies its stage 0, where face_up_prob
# = 0. We therefore NEVER see a face-up start at play — yet that is the hardest
# case, the one we want to eyeball. This variable forces it.
#   STANDUP_PLAY_FACE_UP=1.0  -> 100% face-up starts
#   STANDUP_PLAY_FACE_UP=0.4  -> the mix of the curriculum's last stage
#   unset / "none" / "random" -> default behavior (stage 0)
# Has an effect ONLY when play=True. Same pattern as SLOPE_PLAY_DIFFICULTY in
# roller_slope.
PLAY_FACE_UP = None
# Face-down:standing ratio of the curriculum's LAST stage (0.40 / 0.20 = 2:1).
# The remainder (1 - face_up) is split in that ratio, so 0.4 reproduces the
# end-of-training mix exactly.
_PLAY_FACE_DOWN_SHARE = 2.0 / 3.0


def _resolve_play_face_up():
    """Share of face-up starts at play: env STANDUP_PLAY_FACE_UP, else the constant."""
    raw = os.environ.get("STANDUP_PLAY_FACE_UP")
    if raw is None:
        return PLAY_FACE_UP
    raw = raw.strip().lower()
    if raw in ("", "none", "random"):
        return None
    try:
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        print(f"[roller_standup] STANDUP_PLAY_FACE_UP='{raw}' invalid -> default {PLAY_FACE_UP}")
        return PLAY_FACE_UP

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
    #
    # ⚠️ POSITIVE WEIGHT, and that is not a typo. mdp.py mixes two sign
    # conventions: trunk_vertical_accel_penalty already returns -|a_z|
    # (mdp.py:2171), like height_l1_penalty and pose_l1_penalty — which are in fact
    # used here with weights +30 and +5. The -0.02 inherited from standup therefore
    # formed a double negative and REWARDED vertical acceleration: measured at
    # Episode_Reward/gentle_rise = +0.0118 (the only penalty term logged positive) on
    # run vweolw91. That is the cause of the "very violent" behavior, and it also
    # explains the unsuccessful damping attempts documented in standup, which were
    # fighting a term actively pushing the other way.
    #
    # We keep the magnitude 0.02 (the originally intended one) DELIBERATELY small:
    # |a_z| is necessarily high during a roll-over from the back, so a large weight
    # here would be a motion blocker. The real damping is carried by
    # joint_torque_rate_l2, which penalizes torque RATE rather than motion.
    cfg.rewards["gentle_rise"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=+0.02,
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
    # from the back, so it is THE safe lever to turn up.
    #
    # -2e-3 (the value inherited from standup) contributed only -0.0002/step
    # against ~+41.6 of task reward saturated at 95-99% — i.e. nothing at all.
    # Across all dampers the ratio was ~35:1 in favor of the task, so there was
    # no reason to be gentle. Measured on run vweolw91 at iteration 7500.
    #
    # Recalibration: the raw value of |Δτ|² is ~0.1 at convergence, so the
    # contribution ≈ 0.1 × |weight|. Measured at -0.255/step with a weight of
    # -2.0 (run d8rnko6p) — so NOT the cause of the freeze, but we drop back to
    # -0.2 to free up the damping budget while isolating the effect of the sign
    # bug alone. If it is still violent, raise THIS term (formula above) rather
    # than body_ang_vel or action_rate, which are motion blockers and were
    # freezing the rise from the back.
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torque_rate_l2,
        weight=-0.2,
    )

    # NO head-impact penalty. Tried with velstand's values (body_impact_cost,
    # `neck` subtree, weight -1.0, threshold 2.0): the policy converged to lying
    # down, INERT. Measured (run d8rnko6p): head_impact_penalty -1.01/step, the
    # largest negative term in the table, while standing_composite collapsed from
    # +14.3 to +3.3.
    #
    # The reasoning error was believing that a "targeted" penalty does not
    # restrict motion. False here: to get up from its back, this robot PIVOTS on
    # its head and shoulders. The head is the fulcrum of the roll-over, not
    # collateral damage — penalizing it blocks the only available mechanism, and
    # the face-up case was already the failing one.
    #
    # Hypothesis under test: head banging was a SYMPTOM of the violence (the
    # gentle_rise sign bug paid for brutality, and a brutal rise ends on the
    # head), not a separate defect. If the slam comes back once the sign is
    # fixed, the reintroduction must be a HEIGHT-GATED penalty — as upright_sharp
    # is — to spare the ground roll-over phase.
    #
    # ⚠️ Beware the lazy optimum that makes this freeze possible: pose_stand_legs
    # stayed at +7.72 out of 8 while the robot was lying flat (legs at HOME in a
    # prone position → reward collected almost for free). It is height_stand_l1
    # (weight +30) that must make "stay on the ground" net negative.

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
            # The two start poses (face down / face up) share a SINGLE z range,
            # yet their contacts have nothing in common: face down only clears the
            # ground from 0.0752 up, face up rests at 0.0475. A single floor
            # therefore cannot be ideal for both. We pick 0.076 to eliminate any
            # interpenetration on the face-down side (measured: at 0.05, +25 mm
            # into the ground), at the cost of a face-up start 28–42 mm above its
            # rest height — a far gentler artifact than a contact pushout.
            "prone_z_min":    0.076,
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

    # Play override: force face-up starts so they can be inspected. We write the
    # probabilities into the event AND remove the curriculum: without that,
    # event_param_curriculum (which runs BEFORE the reset events) would rewrite
    # them with its stage 0 on the very first reset. Play only, so training and
    # its easy → hard curriculum are untouched.
    if play:
        play_face_up = _resolve_play_face_up()
        if play_face_up is not None:
            remainder = 1.0 - play_face_up
            cfg.events["set_ground_state"].params.update({
                "face_up_prob":    play_face_up,
                "face_down_prob":  remainder * _PLAY_FACE_DOWN_SHARE,
                "standing_prob":   remainder * (1.0 - _PLAY_FACE_DOWN_SHARE),
                "sitting_prob":    0.00,
            })
            del cfg.curriculum["ground_state_mix"]

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
    # Defensive redundancy: the curriculum manager runs BEFORE the reset events
    # on every reset (including the very first), and wheel_friction_curriculum
    # itself defaults to stage 0 — so this line is never needed in practice. It
    # just keeps the event's DEFAULT value consistent with the curriculum's stage
    # 0, in case someone later removes the curriculum but leaves the event in
    # place.
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
