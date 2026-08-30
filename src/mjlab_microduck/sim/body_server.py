"""A microduck body in MuJoCo, served to the real `robotd` over TCP.

    uv run duck-body                      # one duck, the walking scene, a viewer
    uv run duck-body --port 7801
    uv run duck-body --headless           # no window, for tests and for many ducks

Then, on the daemon side:

    robotd --sim 127.0.0.1:7801

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
import json
import math
import socket
import socketserver
import threading
import time
from pathlib import Path

import mujoco
import numpy as np

PROTOCOL = 1

# `duck_ipc_proto::JOINT_NAMES`, which is protocol: every positional array on the wire is indexed by
# it. Duplicated here rather than derived, because the two repositories cannot share a constant —
# and asserted against the model at startup, which is the next best thing.
JOINT_NAMES = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "mouth",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)
MOUTH_INDEX = JOINT_NAMES.index("mouth")

SCENES = Path(__file__).resolve().parents[1] / "robot" / "microduck"
# **`scene.xml`, not `scene_walk.xml`.** The walking scene includes `robot_walk.xml`, the model the
# RL work trains against — and its actuator default classes carry `contype="0" conaffinity="0"`, so
# the robot's geoms collide with nothing. Started there, the duck sinks straight through a floor that
# is present in the scene and does nothing: trunk z went 0.120 → -0.105 in one second, and the daemon
# read it, quite correctly, as a robot lying on its back. `scene.xml` includes
# `robot_allcollisions.xml`, which is the one with a body that touches things.
DEFAULT_SCENE = SCENES / "scene.xml"

# What the robot reports and nothing here simulates. Stated as constants rather than omitted, so
# `robotctl health` shows a plausible robot instead of an alarming one.
NOMINAL_VOLTS = 7.4
NOMINAL_TEMP_C = 32.0


class Body:
    """The simulated robot: physics on one thread, the socket on another.

    The lock is held only to copy numbers in or out, never across a physics step — a daemon that
    asks for sensors must never be waiting on the solver, for the same reason a real bus read does
    not wait for a servo's control loop.
    """

    def __init__(self, scene: Path, keyframe: str = "STAND", limp: bool = False, kp: float = 200.0):
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.lock = threading.Lock()
        self.reset_to(keyframe)

        # Model joint order, once, by name — never by index, because an MJCF edit reorders silently.
        self.actuated = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.model.nu)
        ]
        missing = [n for n in self.actuated if n not in JOINT_NAMES]
        if missing:
            raise SystemExit(
                f"{scene.name} actuates joints the daemon does not know: {missing}.\n"
                "  The wire is positional and indexed by JOINT_NAMES, so a name only this side "
                "knows about cannot be sent anywhere."
            )
        self.to_wire = [JOINT_NAMES.index(n) for n in self.actuated]

        self.qpos_adr = np.array(
            [
                self.model.jnt_qposadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                ]
                for n in self.actuated
            ]
        )
        self.qvel_adr = np.array(
            [
                self.model.jnt_dofadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                ]
                for n in self.actuated
            ]
        )

        # **Torque on, holding the pose it started in** — because that is what a daemon finds.
        # `robotd` deliberately never enables torque at startup: "a robotd restarted by an update
        # must leave a standing robot standing", and on a real robot the servos are already holding
        # when the process comes up. Starting limp instead means the duck collapses before the first
        # read, and the daemon quite correctly reports a seated boot and tries to stand a robot that
        # is already on the floor. `--limp` is the other case, for when that is what you want.
        self.torque_on = not limp
        self.kp = kp
        self._gain = self.model.actuator_gainprm[:, 0].copy()
        self.data.ctrl[:] = self.data.qpos[self.qpos_adr]
        self._apply_torque()
        mujoco.mj_forward(self.model, self.data)

    def reset_to(self, keyframe: str) -> None:
        """Start in a pose the daemon recognises.

        **`qpos0` is every joint at zero, which is not a pose this robot is ever in.** Started
        there, `robotd` measured 0.41 rad of deviation from its home frame, correctly called it a
        seated boot, tried to rise with the sitstand policy and ended on its back — a completely
        reasonable response to a robot found folded in a way no robot folds. The scene carries the
        real poses as keyframes; `STAND` is the one that matches `duck_control::DEFAULT_POSITION`,
        which its own comment says must match `HOME_FRAME` in the training env.

        Naming a keyframe the scene does not have is fatal rather than a shrug: the alternative is
        starting somewhere arbitrary and spending an afternoon on the consequences.
        """
        available = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_KEY, i)
            for i in range(self.model.nkey)
        ]
        if keyframe in available:
            mujoco.mj_resetDataKeyframe(self.model, self.data, available.index(keyframe))
            return
        if not available:
            mujoco.mj_resetData(self.model, self.data)
            print("== this scene has no keyframes; starting at qpos0, which is not a real pose")
            return
        raise SystemExit(
            f"no keyframe {keyframe!r} in this scene. It has: {', '.join(available)}"
        )

    # ── what the daemon sees ──────────────────────────────────────────────

    def sensors(self) -> dict:
        with self.lock:
            positions = self.data.qpos[self.qpos_adr].copy()
            velocities = self.data.qvel[self.qvel_adr].copy()
            force = self.data.actuator_force.copy()
            quat = self.data.qpos[3:7].copy()  # the free joint's orientation, scalar-first
            trunk_z = float(self.data.qpos[2])

        wire_pos = [0.0] * len(JOINT_NAMES)
        wire_vel = [0.0] * len(JOINT_NAMES)
        wire_cur = [0.0] * len(JOINT_NAMES)
        for model_index, wire_index in enumerate(self.to_wire):
            wire_pos[wire_index] = float(positions[model_index])
            wire_vel[wire_index] = float(velocities[model_index])
            # Not calibrated against a real servo: a stand-in with the right shape, so a consumer
            # watching load sees load. Amps from a simulated torque would be a fiction with a unit.
            wire_cur[wire_index] = abs(float(force[model_index])) * 100.0

        return {
            "positions": wire_pos,
            "velocities": wire_vel,
            "currents_ma": wire_cur,
            # **Not part of the protocol, and deliberately extra.** No robot can measure how high its
            # own trunk is, so nothing above `RobotIo` may use this — serde ignores it on the daemon
            # side. It is here because a tool asking "did it stand up?" has no other way to know:
            # a duck sitting on its bottom with a vertical trunk has gravity [0, 0, -1] too, which
            # is exactly how a check on orientation alone reported a seated duck as upright.
            "trunk_z": trunk_z,
            "imu": {
                "gyro": [float(v) for v in self._gyro()],
                "gravity": [float(v) for v in gravity_in_trunk(quat)],
                "quat": [float(v) for v in quat],
            },
        }

    def _gyro(self) -> np.ndarray:
        # Body-frame angular velocity: the free joint's rotational DOFs, which MuJoCo already
        # expresses in the body frame.
        with self.lock:
            return self.data.qvel[3:6].copy()

    def slow_sensors(self) -> dict:
        return {
            "volts": NOMINAL_VOLTS,
            "temps_c": [NOMINAL_TEMP_C] * len(JOINT_NAMES),
        }

    # ── what the daemon commands ──────────────────────────────────────────

    def set_targets(self, wire_targets: list[float]) -> None:
        if len(wire_targets) != len(JOINT_NAMES):
            raise ValueError(f"expected {len(JOINT_NAMES)} targets, got {len(wire_targets)}")
        with self.lock:
            for model_index, wire_index in enumerate(self.to_wire):
                self.data.ctrl[model_index] = wire_targets[wire_index]

    def set_gain(self, kp: int) -> None:
        with self.lock:
            self.kp = float(kp)
            self._apply_torque()

    def set_torque(self, on: bool) -> None:
        with self.lock:
            self.torque_on = bool(on)
            if on:
                # Torque arriving must not fling the robot at a stale target. The real robot avoids
                # this by ramping to the home pose after enabling; this makes the first instant
                # after `set_torque(true)` a hold rather than a lunge.
                self.data.ctrl[:] = self.data.qpos[self.qpos_adr]
            self._apply_torque()

    def _apply_torque(self) -> None:
        """Torque off means limp, not frozen.

        Refusing to command a fallen robot only freezes it in the pose it fell in — which is why
        `RobotIo::set_gain` exists at all. Zero gain here is the simulated equivalent of cutting
        power to the servos.
        """
        scale = (self.kp / 200.0) if self.torque_on else 0.0
        self.model.actuator_gainprm[:, 0] = self._gain * scale
        self.model.actuator_biasprm[:, 1] = -self._gain * scale

    # ── physics ───────────────────────────────────────────────────────────

    def step(self) -> None:
        with self.lock:
            mujoco.mj_step(self.model, self.data)


def gravity_in_trunk(quat: np.ndarray) -> np.ndarray:
    """World gravity, expressed in the trunk frame. Upright is `[0, 0, -1]`.

    What the policy actually observes — and the reason this is here rather than in the daemon is
    that the daemon's IMU delivers exactly this, already rotated, from the sensor's own filter.
    """
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, quat)
    # The world→body rotation is the transpose; gravity is -z in the world.
    return -rotation.reshape(3, 3).T[:, 2]


class Handler(socketserver.StreamRequestHandler):
    """One duck's daemon. One connection at a time, which is the real relationship too."""

    def handle(self) -> None:  # noqa: D102
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        body: Body = self.server.body  # type: ignore[attr-defined]
        print(f"== daemon connected from {self.client_address}")
        for raw in self.rfile:
            try:
                request = json.loads(raw)
                answer = self.dispatch(body, request)
            except Exception as error:  # a bad frame must not take the simulator down with it
                answer = {"error": str(error)}
            self.wfile.write((json.dumps(answer) + "\n").encode())
            self.wfile.flush()
        print("== daemon disconnected")

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
        raise ValueError(f"unknown op {op!r}")


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7801)
    parser.add_argument("--headless", action="store_true", help="no viewer window")
    parser.add_argument(
        "--limp",
        action="store_true",
        help="start with no torque, so the duck collapses where it stands — a robot found on the "
        "floor, which is what `robotd`'s seated-boot path is for",
    )
    parser.add_argument(
        "--keyframe",
        default="STAND",
        help="pose to start in. STAND matches the daemon's home frame; SIT is what a duck "
        "left folded looks like, and is how to exercise standing up",
    )
    args = parser.parse_args()

    if not args.scene.exists():
        raise SystemExit(f"no scene at {args.scene}. Available:\n  " + "\n  ".join(
            sorted(p.name for p in SCENES.glob("scene*.xml"))
        ))

    body = Body(args.scene, keyframe=args.keyframe, limp=args.limp)
    server = Server((args.host, args.port), Handler)
    server.body = body  # type: ignore[attr-defined]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"== {args.scene.name}: {len(body.actuated)} actuated joints, starting at {args.keyframe}")
    print(f"== serving a body on {args.host}:{args.port} — robotd --sim {args.host}:{args.port}")

    run(body, headless=args.headless)


def run(body: Body, headless: bool) -> None:
    """Step in real time.

    **Real time, not as fast as possible.** The daemon's loop is wall-clock and its health gate
    fails below 45 of 50 Hz, so a simulator running at its own pace does not merely look wrong — it
    makes every duck report unhealthy and the updater start rolling releases back.
    """
    viewer = None
    if not headless:
        try:
            import mujoco.viewer

            viewer = mujoco.viewer.launch_passive(body.model, body.data)
        except Exception as error:
            print(f"== no viewer ({error}); running headless")

    dt = body.model.opt.timestep
    # A frame every N steps, counted — not `data.time % 0.033`, which is float arithmetic on an
    # accumulating value and fires when it feels like it. A viewer that renders the first frame and
    # then rarely again is a window showing a pose the robot left seconds ago.
    steps_per_frame = max(1, round((1.0 / 60.0) / dt))
    step = 0
    next_step = time.perf_counter()
    behind = 0
    try:
        while True:
            body.step()
            if viewer is not None and not viewer.is_running():
                break
            next_step += dt
            slack = next_step - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            elif slack < -0.25:
                # Said once per lapse rather than per step: a simulator that cannot keep real time
                # is the one thing that breaks the daemon's health gate, and it should say so.
                behind += 1
                print(f"== behind real time by {-slack:.2f}s (x{behind}) — fewer ducks, or --headless")
                next_step = time.perf_counter()
            step += 1
            if viewer is not None and step % steps_per_frame == 0:
                viewer.sync()
    except KeyboardInterrupt:
        pass
    finally:
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    main()
