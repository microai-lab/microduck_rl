"""Microduck roller slope — balanced passive descent.

The robot spawns on the flat section (with a forward impulse), rolls onto a
downhill ramp and lets itself glide while staying upright. No steering: the
twist command is neutralized (rel_standing_envs=1.0). Custom flat+ramp terrain
(FlatRampTerrainCfg), steepness curriculum (terrain_levels_slope). Unified 61D
obs → hot-swappable at runtime (--new-cmd-obs) — inherited as-is from
make_microduck_velocity_rollers_env_cfg (DR/obs/reset untouched here).
"""

import math
import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as base_mdp
from mjlab.managers import CurriculumTermCfg, EventTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlModelCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.slope_terrain import FlatRampTerrainCfg, RAMP_DEG_MAX
from mjlab_microduck.tasks.microduck_velocity_rollers_env_cfg import (
    make_microduck_velocity_rollers_env_cfg,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg

# Geometry of the flat + ramp + runout terrain.
FLAT_LENGTH        = 2.0
RAMP_LENGTH_RANGE  = (3.0, 8.0)   # horizontal ramp length, drawn at random per tile
RUNOUT_LENGTH      = 4.0          # flat runout at the bottom
SPAWN_ON_RAMP      = 0.3          # spawn this many m onto the ramp (gravity -> rolling, no skidding)
ENTRY_VELOCITY_X   = (0.25, 0.45) # small initial forward/downhill momentum (m/s)
TILE_SIZE          = (15.0, 4.0)  # >= flat + ramp_max + runout (= 14) + margin
SPAWN_YAW          = (0.0, 0.0)   # facing downhill (+x), fixed

# Steepness at PLAY time: None = random (as during training). Set a value in
# 0..1 to force a specific slope (1.0 = steepest ~20°, 0.5 = medium).
# Overridable without editing the code through the SLOPE_PLAY_DIFFICULTY env var
# (e.g. SLOPE_PLAY_DIFFICULTY=1.0 uv run play ... ; "none"/"random" = random).
PLAY_DIFFICULTY    = None


def _resolve_play_difficulty():
    """Play difficulty: env SLOPE_PLAY_DIFFICULTY, else the constant."""
    raw = os.environ.get("SLOPE_PLAY_DIFFICULTY")
    if raw is None:
        return PLAY_DIFFICULTY
    raw = raw.strip().lower()
    if raw in ("", "none", "random"):
        return None
    try:
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        print(f"[roller_slope] SLOPE_PLAY_DIFFICULTY='{raw}' invalid -> default {PLAY_DIFFICULTY}")
        return PLAY_DIFFICULTY

# "Fell into the void" termination: below the lowest runout (steepest and
# longest ramp), with margin => never fires during a normal descent, only if the
# robot leaves the solid.
_MAX_DROP  = RAMP_LENGTH_RANGE[1] * math.tan(math.radians(RAMP_DEG_MAX))
VOID_FLOOR = -_MAX_DROP - 0.5


def make_microduck_roller_slope_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_velocity_rollers_env_cfg(play=play)

    # === TERRAIN: flat + ramp (random length) + flat runout ===
    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=TILE_SIZE,
            curriculum=True,
            num_rows=10,          # 10 steepness levels
            num_cols=1,
            difficulty_range=(0.0, 1.0),
            sub_terrains={
                "flat_ramp": FlatRampTerrainCfg(
                    flat_length=FLAT_LENGTH,
                    ramp_length_range=RAMP_LENGTH_RANGE,
                    runout_length=RUNOUT_LENGTH,
                    spawn_on_ramp=SPAWN_ON_RAMP,
                )
            },
        ),
        max_init_terrain_level=0,  # curriculum: start on the gentlest ramp
    )

    # At play time: show a variety of slopes. difficulty None -> random
    # steepness (level drawn across all rows); a value in 0..1 forces a specific
    # steepness (1.0 = the steepest). Controlled via SLOPE_PLAY_DIFFICULTY.
    if play:
        play_difficulty = _resolve_play_difficulty()
        if play_difficulty is not None:
            cfg.scene.terrain.terrain_generator.difficulty_range = (play_difficulty, play_difficulty)
        else:
            cfg.scene.terrain.max_init_terrain_level = None

    # === Neutralized COMMAND (pure balance) ===
    command = cfg.commands["twist"]
    command.rel_standing_envs = 1.0
    command.rel_heading_envs = 0.0
    command.ranges.lin_vel_x = (0.0, 0.0)
    command.ranges.lin_vel_y = (0.0, 0.0)
    if getattr(command.ranges, "ang_vel_z", None) is not None:
        command.ranges.ang_vel_z = (0.0, 0.0)

    # === RESET: always facing downhill (+x), NO base push ===
    # The inherited yaw is random (-180°/+180°) -> we fix it at 0 (facing the
    # bottom of the slope). No base velocity injected: the robot spawns on the
    # ramp (see spawn_on_ramp) and gravity spins the wheels (momentum in the
    # wheels, no slip). The old base push (fast base, stationary wheels) skidded
    # -> contact spike -> NaN divergence, and the robot "walked to a stop"
    # instead of rolling.
    cfg.events["reset_base"].params["pose_range"]["yaw"] = SPAWN_YAW
    # NO base push here (moving base + stationary wheels = skidding jolt on the
    # first step). The initial momentum is given as consistent ROLLING (base +
    # wheels, ω·r = v) by reset_rolling_entry below -> clean start.
    cfg.events["reset_base"].params["velocity_range"] = {}

    # === REWARDS: FREE balance (it places its own center of gravity) ===
    # NO fixed pose reward: we no longer dictate the flat-ground standing posture
    # (which prevented it from flexing/leaning). It is free to move its CoM
    # (hips/knees, lean) to hold the slope. We only reward: staying upright,
    # being alive, gliding, going straight — and not falling (terminations).
    keep = {"action_rate_l2"}
    for name in list(cfg.rewards.keys()):
        if name not in keep:
            del cfg.rewards[name]

    cfg.rewards["upright"] = RewardTermCfg(
        func=microduck_mdp.body_upright_gaussian,
        weight=3.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)), "std": 0.2},
    )
    cfg.rewards["alive"] = RewardTermCfg(func=microduck_mdp.is_alive, weight=1.0)
    # LET IT GLIDE (roll), do NOT accelerate/run: rewards downhill WHEEL ROLLING,
    # capped at cap_speed. Capped => no incentive to push faster; wheel-based =>
    # "running" (pushing the base without rolling) pays nothing. Without a glide
    # reward the optimum would be to stand still; with it, the robot lets itself
    # roll as long as it can hold its balance.
    cfg.rewards["wheel_glide"] = RewardTermCfg(
        func=microduck_mdp.wheel_glide_reward, weight=2.0, params={"cap_speed": 0.35},
    )
    # GO STRAIGHT: hold the spawn yaw (= 0 = facing downhill). Being corrective
    # (the robot can recover), this is the right way to go straight. NB: the PPO
    # symmetry (SYMMETRY_CFG) is coded for the old 51D obs -> unusable here.
    cfg.rewards["heading_hold"] = RewardTermCfg(
        func=microduck_mdp.heading_hold_reward, weight=1.5, params={"std": 0.4},
    )
    cfg.rewards["feet_flat"] = RewardTermCfg(
        func=microduck_mdp.feet_flat_penalty,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
            "sensor_name": "feet_ground_contact",
        },
    )
    cfg.rewards["neck_action_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.neck_action_rate_l2, weight=-0.5,
    )
    # KEEP THE HEAD UPRIGHT: penalizes neck/head joint deviation from the home
    # position. We removed the fixed LEG pose (for free balance), but nothing was
    # holding the head -> it drifted anywhere. This constrains the head/neck
    # ONLY, not the legs.
    cfg.rewards["neck_joint_pos_l2"] = RewardTermCfg(
        func=microduck_mdp.neck_joint_pos_l2, weight=-0.75,
    )
    cfg.rewards["joint_torques_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torques_l2, weight=-1e-3,
    )
    cfg.rewards["action_rate_l2"].weight = -1.0

    # === TERMINATIONS: fall + fell into the void ===
    # The runout provides solid ground at the bottom of the ramp, so terminating
    # "at the edge" is no longer needed (terrain_edge_reached cut long ramps off
    # too early). We keep: fall (bad_orientation), NaN, and "fell into the void"
    # (trunk below the lowest runout) in case the robot leaves the solid.
    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=base_mdp.bad_orientation,
        params={"limit_angle": 1.0, "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    if "out_of_terrain_bounds" in cfg.terminations:
        del cfg.terminations["out_of_terrain_bounds"]
    cfg.terminations["fell_into_void"] = TerminationTermCfg(
        func=microduck_mdp.root_height_below,
        params={"min_height": VOID_FLOOR, "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan, time_out=False,
    )

    # === OBS: sanitize NaN/Inf (robustness to rare contact divergences) ===
    # A rare contact (~1 in 25M env-steps) makes the free joint diverge to NaN.
    # Because of a one-substep offset, the nan_state termination only catches it
    # on the NEXT STEP (reset), but the NaN already reaches the current step's obs
    # -> rsl_rl's check_nan kills training. nan_policy="sanitize" replaces NaN/Inf
    # with 0 in the returned obs (no crash); nan_state then resets the offending
    # env.
    for grp in ("actor", "critic"):
        cfg.observations[grp].nan_policy = "sanitize"

    # === EVENTS ===
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history, mode="reset",
    )
    # Rolling start (momentum in the wheels, no skidding). AFTER reset_base.
    cfg.events["reset_rolling_entry"] = EventTermCfg(
        func=microduck_mdp.reset_rolling_entry, mode="reset",
        params={"speed_range": ENTRY_VELOCITY_X},
    )

    # === CURRICULUM: steepness gentle -> steep ===
    # Starts on the gentlest slope (2°) and promotes to steeper ones (up to 20°)
    # once the robot has descended far enough (terrain_levels_slope, based on the
    # distance travelled). Viable now that descent_speed makes it MOVE (before it
    # stood still -> never promoted). It learns balance progressively instead of
    # being thrown straight onto 20° (where it nose-dives).
    for name in list(cfg.curriculum.keys()):
        del cfg.curriculum[name]
    cfg.curriculum["terrain_levels"] = CurriculumTermCfg(func=microduck_mdp.terrain_levels_slope)

    return cfg


MicroduckRollerSlopeRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
        entropy_coef=0.01, num_learning_epochs=5, num_mini_batches=4,
        learning_rate=1.0e-3, schedule="adaptive", gamma=0.99, lam=0.95,
        desired_kl=0.01, max_grad_norm=1.0, symmetry_cfg=None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="roller_slope",
    run_name="roller_slope",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=8_000,
)
