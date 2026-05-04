import pybullet as p
import pybullet_data
import time
import os
import math
import numpy as np

# -------------------- SIM --------------------
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0)
p.setRealTimeSimulation(True)
p.loadURDF("plane.urdf")

# -------------------- UR5 --------------------
ROBOT_URDF_PATH = "./ur_e_description/urdf/ur5e.urdf"

ur5 = p.loadURDF(
    ROBOT_URDF_PATH,
    basePosition=[0, -1.0, 0],
    useFixedBase=True
)

end_effector_index = 7

# -------------------- JOINT INIT --------------------
num_joints = p.getNumJoints(ur5)
for i in range(num_joints):
    info = p.getJointInfo(ur5, i)
    if info[2] == p.JOINT_REVOLUTE:
        p.setJointMotorControl2(ur5, info[0], p.VELOCITY_CONTROL, targetVelocity=0, force=0)

joint_targets = [0, -1.57, 1.57, -1.57, -1.57, 0]

for i in range(6):
    p.resetJointState(ur5, i + 1, joint_targets[i])

def apply_joints():
    p.setJointMotorControlArray(
        ur5,
        [1, 2, 3, 4, 5, 6],
        p.POSITION_CONTROL,
        targetPositions=joint_targets,
        targetVelocities=[0] * 6,
        positionGains=[0.04] * 6,
        forces=[500] * 6
    )

# -------------------- TURNTABLE --------------------
disk_radius = 0.2
disk_height = 0.02

disk_visual = p.createVisualShape(
    p.GEOM_CYLINDER,
    radius=disk_radius,
    length=disk_height,
    rgbaColor=[0.3, 0.3, 0.3, 1]
)
disk_collision = p.createCollisionShape(
    p.GEOM_CYLINDER,
    radius=disk_radius,
    height=disk_height
)
disk_id = p.createMultiBody(
    baseMass=0,
    baseCollisionShapeIndex=disk_collision,
    baseVisualShapeIndex=disk_visual,
    basePosition=[0, 0, disk_height / 2]
)

# -------------------- OBJECT --------------------
object_folder = "object_files"
obj_path = os.path.join(object_folder, "demetra_200k.obj")
scale = [0.3, 0.3, 0.3]

base_offset_pos = np.array([0, 0, disk_height])
base_quat = p.getQuaternionFromEuler([math.radians(87), 0, 0])

visual    = p.createVisualShape(p.GEOM_MESH, fileName=obj_path, meshScale=scale)
collision = p.createCollisionShape(p.GEOM_MESH, fileName=obj_path, meshScale=scale)

obj_id = p.createMultiBody(
    baseMass=0,
    baseCollisionShapeIndex=collision,
    baseVisualShapeIndex=visual,
    basePosition=base_offset_pos.tolist(),
    baseOrientation=base_quat
)

def attach_object(disk_pos, disk_quat):
    obj_pos, obj_quat = p.multiplyTransforms(
        disk_pos, disk_quat,
        base_offset_pos, base_quat
    )
    p.resetBasePositionAndOrientation(obj_id, obj_pos, obj_quat)

# -------------------- CAMERA --------------------
width, height = 1280, 720

# Offset της κάμερας σε σχέση με το end effector (local frame)
# [x=μπροστά, y=αριστερά, z=πάνω] — ρύθμισε αν χρειαστεί
cam_local_offset = np.array([0.05, 0.0, 0.0])   # λίγο μπροστά από την άκρη
cam_look_distance = 0.3                           # πόσο μακριά κοιτάει

# -------------------- DEBUG IDs (persistent) --------------------
pyramid_line_ids = []
cam_text_id = -1   # ✅ αποθηκεύουμε το ID για replace κάθε frame

# -------------------- FRUSTUM --------------------
def draw_camera_frustum(c_pos, t_pos, existing_ids=[]):
    color     = [1, 0, 0]
    near_dist = 0.15
    size      = 0.08

    forward = (t_pos - c_pos)
    norm = np.linalg.norm(forward)
    if norm < 1e-6:
        return existing_ids
    forward = forward / norm

    world_up = np.array([0, 0, 1])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 0.001:
        right = np.array([1, 0, 0])
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)

    base_center = c_pos + forward * near_dist
    p1 = base_center + right * size + up * size
    p2 = base_center - right * size + up * size
    p3 = base_center - right * size - up * size
    p4 = base_center + right * size - up * size

    lines = [
        (c_pos, p1), (c_pos, p2), (c_pos, p3), (c_pos, p4),
        (p1, p2), (p2, p3), (p3, p4), (p4, p1)
    ]

    new_ids = []
    for i, (s, e) in enumerate(lines):
        lid = p.addUserDebugLine(
            s, e, color, 2,
            replaceItemUniqueId=existing_ids[i] if i < len(existing_ids) else -1
        )
        new_ids.append(lid)
    return new_ids

# -------------------- HELPER --------------------
def key_down(keys, k):
    return bool(keys.get(k, 0) & p.KEY_IS_DOWN)

# -------------------- LOOP --------------------
time.sleep(0.5)
disk_yaw    = 0.0
disk_speed  = 15.0
joint_speed = 0.25

while True:
    keys = p.getKeyboardEvents()

    # ---------------- DISK ----------------
    if key_down(keys, ord('j')): disk_yaw += disk_speed
    if key_down(keys, ord('l')): disk_yaw -= disk_speed

    disk_quat = p.getQuaternionFromEuler([0, 0, math.radians(disk_yaw)])
    disk_pos  = [0, 0, disk_height / 2]
    p.resetBasePositionAndOrientation(disk_id, disk_pos, disk_quat)
    attach_object(disk_pos, disk_quat)

    # ---------------- UR5 ----------------
    if key_down(keys, p.B3G_LEFT_ARROW):  joint_targets[0] -= joint_speed
    if key_down(keys, p.B3G_RIGHT_ARROW): joint_targets[0] += joint_speed
    if key_down(keys, p.B3G_UP_ARROW):    joint_targets[1] += joint_speed
    if key_down(keys, p.B3G_DOWN_ARROW):  joint_targets[1] -= joint_speed

    if key_down(keys, ord('q')): joint_targets[2] += joint_speed
    if key_down(keys, ord('a')): joint_targets[2] -= joint_speed
    if key_down(keys, ord('w')): joint_targets[3] += joint_speed
    if key_down(keys, ord('s')): joint_targets[3] -= joint_speed
    if key_down(keys, ord('e')): joint_targets[4] += joint_speed
    if key_down(keys, ord('d')): joint_targets[4] -= joint_speed
    if key_down(keys, ord('r')): joint_targets[5] += joint_speed
    if key_down(keys, ord('f')): joint_targets[5] -= joint_speed

    apply_joints()

    # ---------------- CAMERA ΠΡΟΣΑΡΤΗΜΕΝΗ ΣΤΗΝ ΑΚΡΗ ΤΟΥ ΒΡΑΧΙΟΝΑ ----------------
    link_state = p.getLinkState(ur5, end_effector_index, computeForwardKinematics=True)
    ee_pos     = np.array(link_state[0])   # world position του EE
    ee_ori     = link_state[1]             # world orientation του EE

    # Rotation matrix από quaternion (world frame)
    rot = np.array(p.getMatrixFromQuaternion(ee_ori)).reshape(3, 3)

    # Axes του EE στο world frame
    ee_forward = rot[:, 0]   # X axis — μπροστά
    ee_right   = rot[:, 1]   # Y axis — δεξιά
    ee_up      = rot[:, 2]   # Z axis — πάνω

    # ✅ Η κάμερα είναι ακριβώς στην άκρη του βραχίονα (EE position)
    # + μικρό offset στην κατεύθυνση του EE αν θέλεις
    cam_pos = ee_pos + ee_forward * cam_local_offset[0] \
                     + ee_right   * cam_local_offset[1] \
                     + ee_up      * cam_local_offset[2]

    # Το target είναι μπροστά από την κάμερα στην κατεύθυνση του EE
    target = cam_pos + ee_forward * cam_look_distance

    # Up vector της κάμερας = Z axis του EE
    up_vec = ee_up.tolist()

    # ---------------- FRUSTUM (ακολουθεί την κάμερα) ----------------
    pyramid_line_ids = draw_camera_frustum(cam_pos, target, pyramid_line_ids)

    # ✅ replaceItemUniqueId για να μην αφήνει ίχνη το κείμενο
    cam_text_id = p.addUserDebugText(
        "Camera POV",
        cam_pos.tolist(),
        [1, 0, 0],
        textSize=1.2,
        replaceItemUniqueId=cam_text_id
    )

    # ---------------- RENDER ----------------
    view = p.computeViewMatrix(cam_pos.tolist(), target.tolist(), up_vec)
    proj = p.computeProjectionMatrixFOV(60, width / height, 0.01, 10)
    p.getCameraImage(width, height, view, proj)

    time.sleep(1 / 60)