"""Interactive crouch pose editor (roller robot).

Opens the MuJoCo viewer with the roller robot standing. In the viewer's
"Control" panel, move the sliders (knees/hips/ankles…) to compose the CROUCH
pose you want. Gravity is turned off and the base is held upright and lowered so
that the lowest point stays on the ground (so you see the trunk descend as you
bend the knees). When the window is closed, the pose is printed as a CROUCH_POSE
dict  {joint_name: angle_rad}  ready to paste.

Usage:
    uv run python scripts/crouch_pose_editor.py
"""

import re
import time

import mujoco
import mujoco.viewer

from mjlab_microduck.robot.microduck_constants import (
    get_walk_rollers_spec,
    HOME_FRAME,
)


def home_value(joint_name: str):
    for pattern, val in HOME_FRAME.joint_pos.items():
        if re.search(pattern, joint_name):
            return float(val)
    return 0.0


# Model built directly from the robot spec (14 <position> actuators in the XML).
model = get_walk_rollers_spec().compile()
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)
model.opt.gravity[:] = [0, 0, 0]  # nothing collapses: only the sliders move it

has_free = model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE

# Actuated joints (excluding passive wheels), with their qpos address.
joints = []
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    if not name or "freejoint" in name or "passive_" in name:
        continue
    joints.append((name, model.jnt_qposadr[i]))

# Initial ctrl = HOME pose (the position actuators hold that target).
for a in range(model.nu):
    aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
    data.ctrl[a] = home_value(aname or "")

if has_free:
    data.qpos[0:3] = [0.0, 0.0, 0.14]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    base_xy = data.qpos[0:2].copy()
    base_quat = data.qpos[3:7].copy()

robot_geoms = [g for g in range(model.ngeom)
               if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE]

mujoco.mj_forward(model, data)

print("=== Crouch Pose Editor (rollers) ===")
print(f"actuators: {model.nu} | floating base: {has_free}")
print("Open the viewer's 'Control' panel and move the sliders to compose")
print("the CROUCH pose. Close the window when you are happy with it.\n")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        if has_free:
            data.qpos[0:2] = base_xy
            data.qpos[3:7] = base_quat
            data.qvel[0:6] = 0.0
        mujoco.mj_step(model, data)  # position actuators -> the joints follow ctrl
        if has_free:
            data.qpos[0:2] = base_xy
            data.qpos[3:7] = base_quat
            data.qvel[0:6] = 0.0
            mujoco.mj_forward(model, data)
            try:
                zmin = min(float(data.geom_xpos[g, 2] - model.geom_rbound[g])
                           for g in robot_geoms)
                data.qpos[2] -= zmin
                mujoco.mj_forward(model, data)
            except Exception:
                pass
        viewer.sync()
        time.sleep(1.0 / 60.0)

print("\n=== Captured crouch pose ===\n")
print("CROUCH_POSE = {")
for name, adr in joints:
    print(f'    "{name}": {float(data.qpos[adr]):.4f},')
print("}")
if has_free:
    print(f"\n# final base height (info): z = {float(data.qpos[2]):.4f}")
print("# Paste CROUCH_POSE here and hand it to Claude to wire up the reward.")
