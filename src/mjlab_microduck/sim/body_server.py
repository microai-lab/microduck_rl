"""A microduck body in MuJoCo, served to the real `robotd` over TCP.

    uv run duck-body                      # one duck, a viewer
    uv run duck-body --ducks 4            # four, sharing one world and one window
    uv run duck-body --port 7801
    uv run duck-body --headless           # no window, for tests and for many ducks

Then, on the daemon side:

    robotd --sim 127.0.0.1:7801

One duck per TCP port, from `--port` upwards, and one `mj_step` for all of them — so ducks share
a floor and can bump into each other, which is the difference between a room with four robots in
it and four robots on the same screen.

Everything above `duck_control::io::RobotIo` is the code that runs on a real robot — the 50 Hz loop,
the ONNX policies, safety, fall detection, odometry, kinematics, every IPC call. This process is the
only part that knows there is no robot.

**Why this repo.** It already owns the scenes, the BAM actuator models fitted to the real XL330s and
mjlab; serving a body to a daemon is the mirror of the sim2real it does today. The daemon-side half
lives in `microduck` because it implements an in-repo trait against an in-repo protocol.

## The protocol

Newline-delimited JSON over TCP, one request and one answer per line, `protocol` checked in the
handshake — the two halves live in two repositories, so "your simulator is old" and "your daemon is
old" must not be the same symptom. `duck_control::sim` is the other end and carries the reasoning
for TCP-not-a-unix-socket and JSON-not-a-packed-struct.

## Two mappings this side owns, on purpose

**Fifteen joints out here, fourteen in the model.** The daemon indexes joints as `JOINT_NAMES`,
which includes `mouth` at index 9; no alpha policy drives it and the walking model does not have it.
The daemon must not learn that, so this inserts and drops it. Where the knowledge about a model's
own shape lives is the whole reason the protocol carries the robot's units rather than MuJoCo's.

**Gravity, not just orientation.** The policy observes projected gravity in the trunk frame. MuJoCo
gives an orientation quaternion, so this does the rotation — the same arithmetic the IMU's SFLP
filter does on the robot, on the other side of the same wire.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import socket
import socketserver
import threading
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from mjlab_microduck.sim.tof import COLS, ROWS, Tof

PROTOCOL = 1

# What the policies were trained at, and what `scripts/infer_policy.py` sets. The scenes ship 0.002;
# with that script's decimation of 4 this is exactly the 50 Hz the daemon's control loop runs at.
# Not a performance knob: the BAM actuator fit, the contact solref and the joint armature are all
# tuned at this step, so 0.002 gives a duck whose legs reach the right angles and still cannot hold
# itself up.
TIMESTEP = 0.005

# Where `scripts/infer_policy.py` puts a duck before it starts: trunk this high, upright, every joint
# at the home pose. Not a keyframe — the keyframes are poses and this is a *placement*.
HOME_TRUNK_Z = 0.125

# `duck_ipc_proto::JOINT_NAMES`, which is protocol: every positional array on the wire is indexed by
# it. Duplicated here rather than shared, because the two repositories cannot share a constant — and
# checked against the model at startup, which is the next best thing.
JOINT_NAMES = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll", "mouth",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)
MOUTH_INDEX = JOINT_NAMES.index("mouth")

# `duck_control::DEFAULT_POSITION`, and `DEFAULT_POSE` in `infer_policy.py` with the mouth put back.
# The right leg is mirrored, not symmetric — worth reading rather than assuming.
HOME_POSE = (
    0.0, -0.0873, -0.4579, -0.0049, 0.4530,
    0.3491, 0.3491, 0.0, 0.0, 0.0,
    0.0, 0.0873, 0.4579, 0.0049, -0.4530,
)

SCENES = Path(__file__).resolve().parents[1] / "robot" / "microduck"
# `scene.xml`, not `scene_walk.xml`: the walking scene includes the model the RL work trains
# against, whose actuator default classes carry `contype="0" conaffinity="0"`, so the robot collides
# with nothing and sinks through a floor the scene really does contain.
DEFAULT_SCENE = SCENES / "scene.xml"
# The robot with no floor and no scenery — what extra ducks are attached from.
ROBOT_ONLY = SCENES / "robot_allcollisions.xml"

# Far enough apart not to touch at rest, close enough to be in one screenful.
SPACING = 0.5

# What the robot reports and nothing here simulates. Constants rather than omissions, so
# `robotctl health` shows a plausible robot instead of an alarming one.
NOMINAL_VOLTS = 7.4
NOMINAL_TEMP_C = 32.0

DEFAULT_RENDER_WIDTH = 1280
DEFAULT_RENDER_HEIGHT = 720
DEFAULT_RENDER_FPS = 24
DEFAULT_RENDER_QUALITY = 95
DEFAULT_RENDER_FORMAT = "jpeg"


def duck_prefix(index: int) -> str:
    """`""` for the first duck — it is the scene's own — and `d1_`, `d2_` … for attached ones."""
    return "" if index == 0 else f"d{index}_"


def build_world(scene: Path, count: int) -> mujoco.MjModel:
    """One model holding `count` ducks, so they share a floor and can bump into each other.

    The scene already contains one duck; the rest are attached to it under a name prefix, which is
    what `MjSpec` is for. Sharing a world rather than running N simulators is the whole point: two
    ducks in separate physics can be beside each other on a screen and never touch, and "beside each
    other" is what every social behaviour is about.
    """
    if count == 1:
        return mujoco.MjModel.from_xml_path(str(scene))

    spec = mujoco.MjSpec.from_file(str(scene))
    for index in range(1, count):
        # A fresh child every time: `attach` renames the spec it is given, so reusing one gives the
        # third duck names like `d2_d1_left_hip_yaw` and a compile error about incompatible ids.
        robot = mujoco.MjSpec.from_file(str(ROBOT_ONLY))
        frame = spec.worldbody.add_frame(pos=[0.0, index * SPACING, 0.0])
        spec.attach(robot, prefix=duck_prefix(index), frame=frame)
    return spec.compile()


def pose_table(scene: Path, keyframe: str) -> tuple[dict[str, float] | None, float]:
    """A named pose from the scene's keyframes, as joint name → angle.

    Read from a single-duck model and applied by name, because attaching a robot does not bring the
    scene's keyframes with it — and a pose is a fact about the robot, not about how many of them are
    in the room.
    """
    if keyframe.upper() == "HOME":
        return None, HOME_TRUNK_Z
    model = mujoco.MjModel.from_xml_path(str(scene))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i) for i in range(model.nkey)]
    if keyframe not in names:
        raise SystemExit(
            f"no keyframe {keyframe!r} in {scene.name}. It has: {', '.join(n for n in names if n)}"
        )
    qpos = model.key_qpos[names.index(keyframe)]
    table = {}
    for joint in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if name in JOINT_NAMES:
            table[name] = float(qpos[model.jnt_qposadr[joint]])
    # The keyframe's own trunk height, not the home one: a seated duck placed at standing height is
    # a duck hovering above the floor, which drops the moment anybody enables it.
    return table, float(qpos[2])


def gravity_in_trunk(quat: np.ndarray) -> np.ndarray:
    """World gravity in the trunk frame. Upright is `[0, 0, -1]`.

    What the policy actually observes, and it is here rather than in the daemon because the daemon's
    IMU delivers exactly this, already rotated, from the sensor's own filter.
    """
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, quat)
    return -rotation.reshape(3, 3).T[:, 2]


class World:
    """The physics, shared by every duck in it.

    One `mj_step` advances all of them, which is what makes a shove real rather than decorative. The
    lock is held only to copy numbers in or out, never across a step: a daemon asking for sensors
    must not wait on the solver, for the same reason a real bus read does not wait on a servo.
    """

    def __init__(self, scene: Path, count: int = 1):
        self.model = build_world(scene, count)
        self.model.opt.timestep = TIMESTEP
        self.data = mujoco.MjData(self.model)
        self.lock = threading.Lock()
        self.bodies: list[Body] = []

    def step(self, times: int = 1) -> None:
        """Advance the world, taking the lock once for the whole batch.

        **Batched because the lock is the bottleneck, not the solver.** Four daemons at 50 Hz make
        400 requests a second, each of which wants this lock, and Python hands the GIL around
        between every one of them — so a physics loop taking and releasing it 200 times a second
        loses. Four steps at 5 ms is 20 ms of world, which is exactly one control tick, so nothing
        sees a sensor older than the tick it belongs to.
        """
        with self.lock:
            for _ in range(times):
                mujoco.mj_step(self.model, self.data)
                # A duck nobody has enabled yet is put back where it was. Physics is shared, so
                # it cannot simply not be stepped — and a hand steadying one robot while another
                # walks about is an ordinary thing for a room to contain.
                for body in self.bodies:
                    if not body.released:
                        body.restore()


class Body:
    """One duck's view of the world: its joints, its actuators, its trunk."""

    def __init__(self, world: World, index: int, limp: bool = False, kp: float = 200.0):
        self.world = world
        self.index = index
        self.prefix = duck_prefix(index)
        model = world.model

        def ident(kind, name):
            found = mujoco.mj_name2id(model, kind, self.prefix + name)
            if found < 0:
                raise SystemExit(f"the model has no {name!r} for duck {index}")
            return found

        # By name, never by index: an MJCF edit reorders silently.
        self.actuators = []
        self.to_wire = []
        for wire_index, name in enumerate(JOINT_NAMES):
            found = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.prefix + name)
            if found >= 0:
                self.actuators.append(found)
                self.to_wire.append(wire_index)
        if not self.actuators:
            raise SystemExit(f"no actuated joints for duck {index} (prefix {self.prefix!r})")

        self.qpos_adr = np.array(
            [model.jnt_qposadr[model.actuator_trnid[a, 0]] for a in self.actuators]
        )
        self.qvel_adr = np.array(
            [model.jnt_dofadr[model.actuator_trnid[a, 0]] for a in self.actuators]
        )
        # The depth sensor, on the model's own `tof` site — so a head that turns takes it along,
        # which is what makes `robot.look` a way to scan a room.
        self.tof = Tof(model, ident(mujoco.mjtObj.mjOBJ_SITE, "tof"), seed=index)
        self.trunk = int(model.jnt_qposadr[ident(mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")])
        self.trunk_dof = int(model.jnt_dofadr[ident(mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")])

        self.actuator_slice = np.array(self.actuators)
        self._gain = model.actuator_gainprm[self.actuator_slice, 0].copy()

        # **Held until the daemon takes it.** A biped at a static pose is not stable: holding the
        # home pose with position control alone puts this duck on the ground in under a second, at
        # any timestep, from any placement — `infer_policy.py` never does it, because it has the
        # policy balancing from step zero. `robotd` deliberately does not enable torque when it
        # starts, so the seconds before it would be spent falling over.
        self.released = limp
        self.torque_on = not limp
        self.kp = kp
        self.held = None

    # ── placement ─────────────────────────────────────────────────────────

    def place(self, pose: dict[str, float] | None, trunk_z: float, offset_y: float) -> None:
        data = self.world.data
        data.qpos[self.trunk + 0] = 0.0
        data.qpos[self.trunk + 1] = offset_y
        data.qpos[self.trunk + 2] = trunk_z
        data.qpos[self.trunk + 3 : self.trunk + 7] = [1.0, 0.0, 0.0, 0.0]
        data.qvel[self.trunk_dof : self.trunk_dof + 6] = 0.0
        for slot, wire_index in enumerate(self.to_wire):
            name = JOINT_NAMES[wire_index]
            value = HOME_POSE[wire_index] if pose is None else pose.get(name, HOME_POSE[wire_index])
            data.qpos[self.qpos_adr[slot]] = value
            data.qvel[self.qvel_adr[slot]] = 0.0
        data.ctrl[self.actuator_slice] = data.qpos[self.qpos_adr]
        self._apply_torque()
        self.remember()

    def remember(self) -> None:
        data = self.world.data
        self.held = (
            data.qpos[self.trunk : self.trunk + 7].copy(),
            data.qpos[self.qpos_adr].copy(),
        )

    def restore(self) -> None:
        """Put this duck back where it was, for the one that has not been enabled yet."""
        if self.held is None:
            return
        trunk, joints = self.held
        data = self.world.data
        data.qpos[self.trunk : self.trunk + 7] = trunk
        data.qpos[self.qpos_adr] = joints
        data.qvel[self.trunk_dof : self.trunk_dof + 6] = 0.0
        data.qvel[self.qvel_adr] = 0.0

    # ── what the daemon sees ──────────────────────────────────────────────

    def sensors(self) -> dict:
        data = self.world.data
        with self.world.lock:
            sim_time = float(data.time)
            positions = data.qpos[self.qpos_adr].copy()
            velocities = data.qvel[self.qvel_adr].copy()
            force = data.actuator_force[self.actuator_slice].copy()
            quat = data.qpos[self.trunk + 3 : self.trunk + 7].copy()
            gyro = data.qvel[self.trunk_dof + 3 : self.trunk_dof + 6].copy()
            trunk = [float(v) for v in data.qpos[self.trunk : self.trunk + 3]]
            trunk_z = trunk[2]

        wire_pos = [0.0] * len(JOINT_NAMES)
        wire_vel = [0.0] * len(JOINT_NAMES)
        wire_cur = [0.0] * len(JOINT_NAMES)
        for slot, wire_index in enumerate(self.to_wire):
            wire_pos[wire_index] = float(positions[slot])
            wire_vel[wire_index] = float(velocities[slot])
            # Not calibrated against a real servo: a stand-in with the right shape, so a consumer
            # watching load sees load. Amps from a simulated torque would be a fiction with a unit.
            wire_cur[wire_index] = abs(float(force[slot])) * 100.0

        return {
            "positions": wire_pos,
            "velocities": wire_vel,
            "currents_ma": wire_cur,
            # **Not part of the protocol, and deliberately extra.** No robot can measure how high its
            # own trunk is, and serde ignores these on the daemon side. They are here because a tool
            # asking "did it stand up?" has no other way to know — a duck sitting on its bottom with
            # a vertical trunk has gravity [0, 0, -1] too — and because a simulator whose seconds are
            # not seconds ruins a policy silently.
            "trunk_z": trunk_z,
            # Where this duck is in the room. No robot knows that either — it is here so a simulated
            # radio can decide who is close enough to hear whom, which is the one thing a real BLE
            # advertisement gets for free and a faked one has to be told.
            "trunk": trunk,
            "sim_time": sim_time,
            "imu": {
                "gyro": [float(v) for v in gyro],
                "gravity": [float(v) for v in gravity_in_trunk(quat)],
                "quat": [float(v) for v in quat],
            },
        }

    def slow_sensors(self) -> dict:
        return {"volts": NOMINAL_VOLTS, "temps_c": [NOMINAL_TEMP_C] * len(JOINT_NAMES)}

    def depth(self) -> dict:
        """One 8x8 depth frame, in the units `tofd` publishes.

        Sixty-four ray casts, so this is the most expensive thing here — asked for at the sensor's
        own 15 Hz rather than the control loop's 50, exactly as the hardware is.
        """
        with self.world.lock:
            distance_mm, status = self.tof.frame(self.world.data)
        return {"rows": ROWS, "cols": COLS, "distance_mm": distance_mm, "status": status}

    # ── what the daemon commands ──────────────────────────────────────────

    def set_targets(self, wire_targets: list[float]) -> None:
        if len(wire_targets) != len(JOINT_NAMES):
            raise ValueError(f"expected {len(JOINT_NAMES)} targets, got {len(wire_targets)}")
        with self.world.lock:
            for slot, wire_index in enumerate(self.to_wire):
                self.world.data.ctrl[self.actuators[slot]] = wire_targets[wire_index]

    def set_gain(self, kp: int) -> None:
        with self.world.lock:
            self.kp = float(kp)
            self._apply_torque()

    def set_torque(self, on: bool) -> None:
        with self.world.lock:
            self.torque_on = bool(on)
            if on:
                self.released = True
                # Torque arriving must not fling the robot at a stale target.
                self.world.data.ctrl[self.actuator_slice] = self.world.data.qpos[self.qpos_adr]
            self._apply_torque()

    def _apply_torque(self) -> None:
        """Torque off means limp, not frozen.

        Refusing to command a fallen robot only freezes it in the pose it fell in — which is why
        `RobotIo::set_gain` exists at all. Zero gain is the simulated equivalent of cutting power.
        The daemon's kp is a Dynamixel register value whose 200 is what BAM fitted the model's own
        gain to, so it is a ratio against 200 rather than a number in the same units.
        """
        scale = (self.kp / 200.0) if self.torque_on else 0.0
        model = self.world.model
        model.actuator_gainprm[self.actuator_slice, 0] = self._gain * scale
        model.actuator_biasprm[self.actuator_slice, 1] = -self._gain * scale


class RenderStream:
    """Render cached JPEGs from snapshots of the authoritative MuJoCo world.

    Snapshotting is the only work performed under ``world.lock``. OpenGL rendering and JPEG
    encoding happen against a private ``MjData`` after the lock is released, so a slow software
    renderer cannot stall the 50 Hz physics and robot control path.
    """

    def __init__(
        self,
        world: World,
        *,
        width: int = DEFAULT_RENDER_WIDTH,
        height: int = DEFAULT_RENDER_HEIGHT,
        fps: int = DEFAULT_RENDER_FPS,
        quality: int = DEFAULT_RENDER_QUALITY,
        image_format: str = DEFAULT_RENDER_FORMAT,
        exit_idle_seconds: float | None = None,
    ):
        self._validate_settings(width, height, fps, quality, image_format)
        if exit_idle_seconds is not None and exit_idle_seconds <= 0:
            raise ValueError("render idle exit must be positive")
        self.world = world
        self.width = width
        self.height = height
        self.fps = fps
        self.quality = quality
        self.image_format = image_format
        self.backend = os.getenv("MUJOCO_GL", "default")
        self.exit_idle_seconds = exit_idle_seconds
        self.exit_requested = threading.Event()
        self._condition = threading.Condition()
        self._frame: dict | None = None
        self._error: str | None = None
        self._stopping = threading.Event()
        self._clients = 0
        self._had_client = False
        self._idle_timer: threading.Timer | None = None
        self._camera_azimuth = 135.0
        self._camera_elevation = -18.0
        self._camera_distance = max(0.85, 0.65 + len(self.world.bodies) * 0.2)
        self._thread = threading.Thread(
            target=self._run, name="mujoco-render", daemon=True
        )

    @staticmethod
    def _validate_settings(
        width: int, height: int, fps: int, quality: int, image_format: str
    ) -> None:
        if width < 1 or height < 1 or width * height > 3840 * 2160:
            raise ValueError("render dimensions must be positive and at most 3840x2160 pixels")
        if not 1 <= fps <= 60:
            raise ValueError("render FPS must be between 1 and 60")
        if not 1 <= quality <= 100:
            raise ValueError("JPEG quality must be between 1 and 100")
        if image_format not in {"jpeg", "png"}:
            raise ValueError("render format must be jpeg or png")

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stopping.set()
        with self._condition:
            if self._idle_timer is not None:
                self._idle_timer.cancel()
                self._idle_timer = None
            self._condition.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def client_connected(self) -> None:
        with self._condition:
            self._clients += 1
            self._had_client = True
            if self._idle_timer is not None:
                self._idle_timer.cancel()
                self._idle_timer = None

    def client_disconnected(self) -> None:
        with self._condition:
            self._clients = max(0, self._clients - 1)
            if (
                self._clients == 0
                and self._had_client
                and self.exit_idle_seconds is not None
                and not self._stopping.is_set()
            ):
                if self._idle_timer is not None:
                    self._idle_timer.cancel()
                self._idle_timer = threading.Timer(
                    self.exit_idle_seconds, self._exit_if_still_idle
                )
                self._idle_timer.daemon = True
                self._idle_timer.start()

    def _exit_if_still_idle(self) -> None:
        with self._condition:
            self._idle_timer = None
            if self._clients == 0 and not self._stopping.is_set():
                self.exit_requested.set()

    def camera(self, action: str, dx: float = 0.0, dy: float = 0.0) -> dict:
        if not math.isfinite(dx) or not math.isfinite(dy):
            raise ValueError("camera movement must be finite")
        with self._condition:
            if action == "orbit":
                self._camera_azimuth = (self._camera_azimuth - dx * 0.3) % 360
                self._camera_elevation = min(
                    85.0, max(-85.0, self._camera_elevation - dy * 0.25)
                )
            elif action == "zoom":
                self._camera_distance = min(
                    5.0, max(0.25, self._camera_distance * math.exp(dy * 0.0015))
                )
            elif action == "reset":
                self._camera_azimuth = 135.0
                self._camera_elevation = -18.0
                self._camera_distance = max(0.85, 0.65 + len(self.world.bodies) * 0.2)
            else:
                raise ValueError(f"unknown camera action {action!r}")
            return {
                "azimuth": self._camera_azimuth,
                "elevation": self._camera_elevation,
                "distance": self._camera_distance,
            }

    def configure(
        self,
        *,
        width: int,
        height: int,
        fps: int,
        quality: int,
        image_format: str,
    ) -> dict:
        self._validate_settings(width, height, fps, quality, image_format)
        with self._condition:
            self.width = width
            self.height = height
            self.fps = fps
            self.quality = quality
            self.image_format = image_format
            self._condition.notify_all()
            return {
                "width": width,
                "height": height,
                "fps": fps,
                "quality": quality,
                "format": image_format,
            }

    def get(self, after: int = 0, timeout_ms: int = 1000) -> dict:
        if after < 0:
            raise ValueError("render sequence must not be negative")
        if not 0 <= timeout_ms <= 30_000:
            raise ValueError("render timeout_ms must be between 0 and 30000")
        deadline = time.monotonic() + timeout_ms / 1000
        with self._condition:
            while self._frame is None or self._frame["seq"] <= after:
                if self._error is not None:
                    raise RuntimeError(f"MuJoCo renderer failed: {self._error}")
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self._stopping.is_set():
                    return {"seq": after, "timeout": True}
                self._condition.wait(remaining)
            return dict(self._frame)

    def _run(self) -> None:
        renderer = None
        try:
            render_data = mujoco.MjData(self.world.model)
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            next_frame = time.monotonic()
            seq = 0
            active_size = None
            while not self._stopping.is_set():
                with self._condition:
                    width = self.width
                    height = self.height
                    fps = self.fps
                    quality = self.quality
                    image_format = self.image_format
                    camera.azimuth = self._camera_azimuth
                    camera.elevation = self._camera_elevation
                    camera.distance = self._camera_distance
                if active_size != (width, height):
                    if renderer is not None:
                        renderer.close()
                    visual_global = getattr(
                        getattr(self.world.model, "vis", None), "global_", None
                    )
                    if visual_global is not None:
                        visual_global.offwidth = max(visual_global.offwidth, width)
                        visual_global.offheight = max(visual_global.offheight, height)
                    renderer = mujoco.Renderer(
                        self.world.model, height=height, width=width
                    )
                    active_size = (width, height)
                period = 1.0 / fps
                delay = next_frame - time.monotonic()
                if delay > 0 and self._stopping.wait(delay):
                    break

                with self.world.lock:
                    mujoco.mj_copyData(render_data, self.world.model, self.world.data)
                sim_time = float(render_data.time)
                if self.world.bodies:
                    trunk = self.world.bodies[0].trunk
                    camera.lookat[:] = render_data.qpos[trunk : trunk + 3]
                    camera.lookat[2] = max(0.08, float(render_data.qpos[trunk + 2]) * 0.6)
                renderer.update_scene(render_data, camera=camera)
                rgb = renderer.render()
                output = io.BytesIO()
                if image_format == "png":
                    Image.fromarray(rgb).save(output, format="PNG", compress_level=1)
                    mime = "image/png"
                else:
                    Image.fromarray(rgb).save(output, format="JPEG", quality=quality)
                    mime = "image/jpeg"
                seq += 1
                frame = {
                    "seq": seq,
                    "sim_time": sim_time,
                    "width": width,
                    "height": height,
                    "mime": mime,
                    "backend": self.backend,
                    "image_b64": base64.b64encode(output.getvalue()).decode("ascii"),
                }
                with self._condition:
                    self._frame = frame
                    self._condition.notify_all()
                next_frame = max(next_frame + period, time.monotonic())
        except Exception as error:  # noqa: BLE001 - renderer failures are surfaced to clients
            with self._condition:
                self._error = str(error)
                self._condition.notify_all()
        finally:
            if renderer is not None:
                renderer.close()


class Handler(socketserver.StreamRequestHandler):
    """One duck's daemon. One connection at a time, which is the real relationship too."""

    def handle(self) -> None:
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        body: Body = self.server.body
        print(f"== duck {body.index}: daemon connected from {self.client_address}", flush=True)
        render_client = False
        try:
            for raw in self.rfile:
                try:
                    request = json.loads(raw)
                    renderer = self.server.renderer
                    if (
                        request.get("op") == "render"
                        and renderer is not None
                        and not render_client
                    ):
                        renderer.client_connected()
                        render_client = True
                    answer = self.dispatch(body, request)
                except Exception as error:  # noqa: BLE001 - isolate malformed client frames
                    answer = {"error": str(error)}
                self.wfile.write((json.dumps(answer) + "\n").encode())
                self.wfile.flush()
        finally:
            if render_client:
                self.server.renderer.client_disconnected()
            print(f"== duck {body.index}: daemon disconnected", flush=True)

    def dispatch(self, body: Body, request: dict) -> dict:
        op = request.get("op")
        if op == "hello":
            asked = request.get("protocol")
            if asked != PROTOCOL:
                raise ValueError(
                    f"the daemon speaks protocol {asked} and this simulator speaks {PROTOCOL}"
                )
            return {"protocol": PROTOCOL}
        if op == "read":
            return body.sensors()
        if op == "write":
            body.set_targets(request["targets"])
            return {}
        if op == "gain":
            body.set_gain(int(request["kp"]))
            return {}
        if op == "torque":
            body.set_torque(bool(request["on"]))
            return {}
        if op == "slow":
            return body.slow_sensors()
        if op == "tof":
            return body.depth()
        if op == "render":
            renderer: RenderStream | None = self.server.renderer
            if renderer is None:
                raise RuntimeError("MuJoCo rendering is disabled; start duck-body with --render")
            return renderer.get(
                after=int(request.get("after", 0)),
                timeout_ms=int(request.get("timeout_ms", 1000)),
            )
        if op == "camera":
            renderer: RenderStream | None = self.server.renderer
            if renderer is None:
                raise RuntimeError("MuJoCo rendering is disabled; start duck-body with --render")
            return renderer.camera(
                action=str(request.get("action", "")),
                dx=float(request.get("dx", 0.0)),
                dy=float(request.get("dy", 0.0)),
            )
        if op == "render_config":
            renderer: RenderStream | None = self.server.renderer
            if renderer is None:
                raise RuntimeError("MuJoCo rendering is disabled; start duck-body with --render")
            return renderer.configure(
                width=int(request["width"]),
                height=int(request["height"]),
                fps=int(request.get("fps", DEFAULT_RENDER_FPS)),
                quality=int(request.get("quality", DEFAULT_RENDER_QUALITY)),
                image_format=str(request.get("format", DEFAULT_RENDER_FORMAT)),
            )
        raise ValueError(f"unknown op {op!r}")


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def run(world: World, headless: bool, stop_event: threading.Event | None = None) -> None:
    """Step in real time.

    **Real time, not as fast as possible.** The daemon's loop is wall-clock and its health gate
    fails below 45 of 50 Hz, so a simulator running at its own pace does not merely look wrong — it
    makes every duck report unhealthy and the updater start rolling releases back.
    """
    viewer = None
    if not headless:
        try:
            import mujoco.viewer

            # No side panels: this window is for watching ducks, and everything the panels would
            # drive belongs to the daemons.
            viewer = mujoco.viewer.launch_passive(
                world.model, world.data, show_left_ui=False, show_right_ui=False
            )
        except Exception as error:  # noqa: BLE001 - viewer fallback must include GL failures
            print(f"== no viewer ({error}); running headless", flush=True)

    dt = world.model.opt.timestep
    # One control tick of world per pass: 20 ms, the same decimation `infer_policy.py` uses.
    batch = max(1, round(0.020 / dt))
    period = batch * dt
    # A frame every N passes, counted — not `data.time % 0.033`, which is float arithmetic on an
    # accumulating value and fires when it feels like it. 30 rather than 60: `viewer.sync()` copies
    # the scene on this thread, and with several ducks in it that is the difference between keeping
    # real time and not.
    passes_per_frame = max(1, round((1.0 / 30.0) / period))
    step = 0
    next_step = time.perf_counter()
    behind = 0
    try:
        while stop_event is None or not stop_event.is_set():
            world.step(batch)
            if viewer is not None and not viewer.is_running():
                break
            next_step += period
            slack = next_step - time.perf_counter()
            # Only sleep when there is something worth sleeping for: `time.sleep` on a few
            # milliseconds overshoots by more than it waits.
            if slack > 0.002:
                time.sleep(slack)
            elif slack < -0.25:
                behind += 1
                print(
                    f"== behind real time by {-slack:.2f}s (x{behind}) — fewer ducks, or --headless",
                    flush=True,
                )
                next_step = time.perf_counter()
            step += 1
            if viewer is not None and step % passes_per_frame == 0:
                viewer.sync()
    except KeyboardInterrupt:
        pass
    finally:
        if viewer is not None:
            viewer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--ducks", type=int, default=1, help="how many, sharing one world")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7801, help="the first duck's port; +1 each")
    parser.add_argument("--headless", action="store_true", help="no viewer window")
    parser.add_argument("--render", action="store_true", help="serve cached JPEG frames")
    parser.add_argument("--render-width", type=int, default=DEFAULT_RENDER_WIDTH)
    parser.add_argument("--render-height", type=int, default=DEFAULT_RENDER_HEIGHT)
    parser.add_argument("--render-fps", type=int, default=DEFAULT_RENDER_FPS)
    parser.add_argument("--render-quality", type=int, default=DEFAULT_RENDER_QUALITY)
    parser.add_argument(
        "--render-format", choices=("jpeg", "png"), default=DEFAULT_RENDER_FORMAT
    )
    parser.add_argument(
        "--exit-on-render-idle",
        type=float,
        metavar="SECONDS",
        help="exit after the last render client stays disconnected for this long",
    )
    parser.add_argument(
        "--limp",
        action="store_true",
        help="start with no torque, so a duck collapses where it stands — a robot found on the "
        "floor, which is what `robotd`'s seated-boot path is for",
    )
    parser.add_argument(
        "--keyframe",
        default="SIT",
        help="where to start. SIT is a duck folded on the floor, which is stable while it waits and "
        "which the standing policy rises from on its own. HOME is infer_policy.py's placement — "
        "home pose, trunk 0.125 m, upright — and STAND and FOLD are the scene's other poses",
    )
    args = parser.parse_args()

    if not args.scene.exists():
        raise SystemExit(
            f"no scene at {args.scene}. Available:\n  "
            + "\n  ".join(sorted(p.name for p in SCENES.glob("scene*.xml")))
        )
    if args.ducks < 1:
        raise SystemExit("--ducks needs at least one duck")
    if args.exit_on_render_idle is not None and not args.render:
        raise SystemExit("--exit-on-render-idle requires --render")

    world = World(args.scene, args.ducks)
    pose, trunk_z = pose_table(args.scene, args.keyframe)
    servers = []
    renderer = None
    for index in range(args.ducks):
        body = Body(world, index, limp=args.limp)
        body.place(pose, trunk_z, offset_y=index * SPACING)
        world.bodies.append(body)
        server = Server((args.host, args.port + index), Handler)
        server.body = body
        server.renderer = None
        servers.append(server)

    mujoco.mj_forward(world.model, world.data)
    if args.render:
        renderer = RenderStream(
            world,
            width=args.render_width,
            height=args.render_height,
            fps=args.render_fps,
            quality=args.render_quality,
            image_format=args.render_format,
            exit_idle_seconds=args.exit_on_render_idle,
        )
        renderer.start()
        for server in servers:
            server.renderer = renderer
    for server in servers:
        threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"== {args.scene.name}: {args.ducks} duck(s), starting at {args.keyframe}", flush=True)
    for index in range(args.ducks):
        print(f"==   duck {index}: robotd --sim {args.host}:{args.port + index}", flush=True)

    try:
        run(
            world,
            headless=args.headless,
            stop_event=renderer.exit_requested if renderer is not None else None,
        )
    finally:
        if renderer is not None:
            renderer.close()
        for server in servers:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    main()
