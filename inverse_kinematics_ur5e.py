import pybullet as p
import pybullet_data
import time
import os
import math
import numpy as np
import tkinter as tk
import threading

# ============================================================
# SHARED STATE
# ============================================================
disk_yaw_shared = 0.0
blue_angle_deg = 0.0
mouse_pressed = False
lock = threading.Lock()

# ============================================================
# TKINTER (2D DISPLAY ONLY)
# ============================================================
WIDTH, HEIGHT = 800, 800
CENTER = (WIDTH // 2, HEIGHT // 2)
SCALE = 150
current_point = None

def to_canvas(x, y):
    cx = CENTER[0] + x * SCALE
    cy = CENTER[1] - y * SCALE
    return cx, cy

def update_angle_from_point(x, y):
    global disk_yaw_shared, blue_angle_deg
    if y < 0:
        angle = math.degrees(math.atan2(-y, x))
        with lock:
            disk_yaw_shared = 2 * angle
            blue_angle_deg = -angle
    else:
        with lock:
            disk_yaw_shared = 0.0
            blue_angle_deg = 0.0

def draw():
    canvas.delete("all")
    canvas.create_line(0, CENTER[1], WIDTH, CENTER[1], fill="red", width=2, dash=(4, 4))
    canvas.create_line(CENTER[0], 0, CENTER[0], HEIGHT, fill="green", width=2, dash=(4, 4))

    r = 2
    x0, y0 = to_canvas(-r, -r)
    x1, y1 = to_canvas(r, r)
    canvas.create_oval(x0, y0, x1, y1, outline="gray")

    with lock:
        total_yaw = disk_yaw_shared
        start_arc = blue_angle_deg
        is_pressed = mouse_pressed
        pt = current_point

    if is_pressed and pt is not None:
        x, y = pt
        v2 = (x, -y)
        bbox = (CENTER[0]-50, CENTER[1]-50, CENTER[0]+50, CENTER[1]+50)
        canvas.create_arc(bbox, start=start_arc, extent=total_yaw,
                          fill="#FFEB3B", outline="#FBC02D", width=2)

        canvas.create_line(*to_canvas(0, 0), *to_canvas(x, y),
                           fill="blue", width=3, arrow=tk.LAST)

        canvas.create_line(*to_canvas(0, 0), *to_canvas(*v2),
                           fill="red", width=3, arrow=tk.LAST)

        canvas.create_text(WIDTH-120, 40,
                           text=f"Rotation: {total_yaw:.1f}°",
                           font=("Courier", 14, "bold"),
                           fill="darkred")
    else:
        canvas.create_text(WIDTH-120, 40,
                           text="Rotation: 0.0°",
                           font=("Courier", 14),
                           fill="gray")

def tk_loop():
    draw()
    root.after(33, tk_loop)

def run_tk():
    global canvas, root
    root = tk.Tk()
    root.title("Control Panel")
    canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="white")
    canvas.pack()
    tk_loop()
    root.mainloop()

threading.Thread(target=run_tk, daemon=True).start()

# ============================================================
# PYBULLET SETUP
# ============================================================
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setRealTimeSimulation(True)
p.loadURDF("plane.urdf")

# -------------------- CAMERA FRUSTUM --------------------
def draw_camera_frustum(c_pos, t_pos, existing_ids=[]):
    color = [1, 0, 0]
    near_dist = 0.15
    size = 0.08

    c_pos = np.array(c_pos)
    t_pos = np.array(t_pos)

    forward = (t_pos - c_pos) / (np.linalg.norm(t_pos - c_pos) + 1e-6)
    world_up = np.array([0, 0, 1])

    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 0.001:
        right = np.array([1, 0, 0])
    right /= np.linalg.norm(right)

    up = np.cross(right, forward)
    base_center = c_pos + forward * near_dist

    p1 = base_center + right*size + up*size
    p2 = base_center - right*size + up*size
    p3 = base_center - right*size - up*size
    p4 = base_center + right*size - up*size

    lines = [(c_pos,p1),(c_pos,p2),(c_pos,p3),(c_pos,p4),
             (p1,p2),(p2,p3),(p3,p4),(p4,p1)]

    new_ids = []
    for i,(s,e) in enumerate(lines):
        l_id = p.addUserDebugLine(
            s.tolist(), e.tolist(), color, 2,
            replaceItemUniqueId=existing_ids[i] if i < len(existing_ids) else -1
        )
        new_ids.append(l_id)

    return new_ids

# -------------------- forward vector -> quaternion για UR5e ----
def camera_forward_to_ee_quat(forward_vec):
    """
    Το tool0 του UR5e έχει Z-axis = forward του εργαλείου.
    Χτίζουμε rotation matrix έτσι ώστε:
      - Z_tool = forward_vec (κατεύθυνση κάμερας)
      - Y_tool = up της κάμερας (κατά προσέγγιση)
      - X_tool = right
    """
    z_axis = np.array(forward_vec, dtype=float)
    z_axis /= np.linalg.norm(z_axis) + 1e-9

    # world up hint
    up_hint = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(z_axis, up_hint)) > 0.99:
        up_hint = np.array([0.0, 1.0, 0.0])

    x_axis = np.cross(up_hint, z_axis)
    x_axis /= np.linalg.norm(x_axis) + 1e-9

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis) + 1e-9

    # Rotation matrix (columns = x, y, z axes of tool in world frame)
    R = np.array([
        [x_axis[0], y_axis[0], z_axis[0]],
        [x_axis[1], y_axis[1], z_axis[1]],
        [x_axis[2], y_axis[2], z_axis[2]]
    ])

    # R -> quaternion [x, y, z, w] (Shepperd)
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2,1] - R[1,2]) * s
        y = (R[0,2] - R[2,0]) * s
        z = (R[1,0] - R[0,1]) * s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w = (R[2,1] - R[1,2]) / s
        x = 0.25 * s
        y = (R[0,1] + R[1,0]) / s
        z = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = 2.0 * math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w = (R[0,2] - R[2,0]) / s
        x = (R[0,1] + R[1,0]) / s
        y = 0.25 * s
        z = (R[1,2] + R[2,1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w = (R[1,0] - R[0,1]) / s
        x = (R[0,2] + R[2,0]) / s
        y = (R[1,2] + R[2,1]) / s
        z = 0.25 * s

    return [x, y, z, w]

# -------------------- UR5e --------------------
ur5 = p.loadURDF("./ur_e_description/urdf/ur5e.urdf",
                 basePosition=[0, 0.5, 0],
                 useFixedBase=True)

num_joints = p.getNumJoints(ur5)
for i in range(num_joints):
    info = p.getJointInfo(ur5, i)
    print(f"Joint {i}: {info[1].decode()} | Link: {info[12].decode()}")

# tool0 = joint index 8
ee_link = 8

joint_targets = [0, -1.57, 1.57, -1.57, -1.57, 0]
for i in range(6):
    p.resetJointState(ur5, i+1, joint_targets[i])

ik_active      = False
ik_target_pos  = [0.0, 0.0, 0.5]
ik_target_quat = [0.0, 0.0, 0.0, 1.0]

# -------------------- TURNTABLE --------------------
disk_radius, disk_height = 0.2, 0.02

disk_id = p.createMultiBody(
    0,
    p.createCollisionShape(p.GEOM_CYLINDER, radius=disk_radius, height=disk_height),
    p.createVisualShape(p.GEOM_CYLINDER, radius=disk_radius, length=disk_height,
                        rgbaColor=[0.3,0.3,0.3,1]),
    [0,0,disk_height/2]
)

# -------------------- OBJECT --------------------
object_folder = "object_files"
obj_path = os.path.join(object_folder, "demetra_200k.obj")

base_offset_pos = np.array([0,0,disk_height])
base_quat = p.getQuaternionFromEuler([math.radians(87),0,0])

try:
    obj_id = p.createMultiBody(
        0,
        p.createCollisionShape(p.GEOM_MESH, fileName=obj_path, meshScale=[0.3]*3),
        p.createVisualShape(p.GEOM_MESH, fileName=obj_path, meshScale=[0.3]*3),
        base_offset_pos.tolist(),
        base_quat
    )
except:
    obj_id = p.createMultiBody(
        0,
        p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05]*3),
        p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05]*3,
                            rgbaColor=[0.8,0.2,0.2,1]),
        [0,0,0.1]
    )

# -------------------- CAMERA --------------------
cam_eye    = [0.5, 0.5, 0.5]
cam_target = [0, 0, 0]

yaw   = 135.0
pitch = -30.0
speed = 0.02
sens  = 1.5

frustum_ids    = []
space_was_down = False

# Υπολογισμός forward για να έχουμε τιμή πριν το loop
ry, rp = math.radians(yaw), math.radians(pitch)
forward = np.array([
    math.cos(ry)*math.cos(rp),
    math.sin(ry)*math.cos(rp),
    math.sin(rp)
])
new_cam_eye    = np.array(cam_eye)
new_cam_target = new_cam_eye + forward

# ============================================================
# MAIN LOOP
# ============================================================
while True:
    keys  = p.getKeyboardEvents()
    moved = False

    if keys.get(p.B3G_LEFT_ARROW,0)  & p.KEY_IS_DOWN: yaw   -= sens; moved=True
    if keys.get(p.B3G_RIGHT_ARROW,0) & p.KEY_IS_DOWN: yaw   += sens; moved=True
    if keys.get(p.B3G_UP_ARROW,0)    & p.KEY_IS_DOWN: pitch += sens; moved=True
    if keys.get(p.B3G_DOWN_ARROW,0)  & p.KEY_IS_DOWN: pitch -= sens; moved=True

    pitch = max(-89, min(89, pitch))

    ry, rp = math.radians(yaw), math.radians(pitch)

    forward = np.array([
        math.cos(ry)*math.cos(rp),
        math.sin(ry)*math.cos(rp),
        math.sin(rp)
    ])

    right = np.array([
        math.sin(ry),
        -math.cos(ry),
        0
    ])

    if keys.get(ord('w'),0) & p.KEY_IS_DOWN: cam_eye += forward*speed; moved=True
    if keys.get(ord('s'),0) & p.KEY_IS_DOWN: cam_eye -= forward*speed; moved=True
    if keys.get(ord('a'),0) & p.KEY_IS_DOWN: cam_eye -= right*speed;   moved=True
    if keys.get(ord('d'),0) & p.KEY_IS_DOWN: cam_eye += right*speed;   moved=True
    if keys.get(ord('q'),0) & p.KEY_IS_DOWN: cam_eye[2] -= speed;      moved=True
    if keys.get(ord('e'),0) & p.KEY_IS_DOWN: cam_eye[2] += speed;      moved=True

    cam_target = (np.array(cam_eye) + forward).tolist()

    # ── SPACE ───────────────────────────────────────────────
    space_down = bool(keys.get(ord(' '), 0) & p.KEY_IS_DOWN)

    if space_down and not space_was_down:
        current_point = (cam_eye[0], cam_eye[1])
        ik_active = True
        with lock:
            mouse_pressed = True
        update_angle_from_point(cam_eye[0], cam_eye[1])

    elif space_down and space_was_down:
        current_point = (cam_eye[0], cam_eye[1])
        update_angle_from_point(cam_eye[0], cam_eye[1])

    elif not space_down and space_was_down:
        ik_active = False
        with lock:
            mouse_pressed = False
            disk_yaw_shared = 0.0
            blue_angle_deg  = 0.0
        current_point = None

    space_was_down = space_down
    # ────────────────────────────────────────────────────────

    # ---------------- T MATRICES ----------------
    Tpoint = np.array([
        [1, 0, 0, cam_eye[0]],
        [0, 1, 0, cam_eye[1]],
        [0, 0, 1, cam_eye[2]],
        [0, 0, 0, 1]
    ])

    Tcenter = np.eye(4)

    with lock:
        current_yaw = disk_yaw_shared

    rad_yaw = math.radians(current_yaw)

    T2th = np.array([
        [np.cos(rad_yaw), -np.sin(rad_yaw), 0, 0],
        [np.sin(rad_yaw),  np.cos(rad_yaw), 0, 0],
        [0,                0,               1, 0],
        [0,                0,               0, 1]
    ])

    T_final     = Tcenter @ T2th @ np.linalg.inv(Tcenter) @ Tpoint
    new_cam_eye = T_final[:3, 3]

    target_vec         = np.array([cam_target[0], cam_target[1], cam_target[2], 1.0])
    new_cam_target_vec = Tcenter @ T2th @ np.linalg.inv(Tcenter) @ target_vec
    new_cam_target     = new_cam_target_vec[:3]

    # Forward vector της κάμερας μετά τον μετασχηματισμό
    cam_fwd = new_cam_target - new_cam_eye
    cam_fwd_norm = np.linalg.norm(cam_fwd)
    if cam_fwd_norm > 1e-6:
        cam_fwd /= cam_fwd_norm

    print("\nT_final (transformed matrix):")
    print(T_final)

    # ---------------- ROBOT (IK ή default) ----------------
    if ik_active:
        ik_target_pos  = new_cam_eye.tolist()
        ik_target_quat = camera_forward_to_ee_quat(cam_fwd)

        ik_solution = p.calculateInverseKinematics(
            ur5,
            ee_link,          # tool0 = 8
            ik_target_pos,
            ik_target_quat,
            maxNumIterations=200,
            residualThreshold=1e-5
        )

        # Τα 6 κινητά joints είναι indices 1-6
        for i in range(6):
            p.setJointMotorControl2(
                ur5,
                i + 1,
                p.POSITION_CONTROL,
                targetPosition=ik_solution[i],
                force=500,
                maxVelocity=1.0
            )
    else:
        p.setJointMotorControlArray(
            ur5, [1,2,3,4,5,6],
            p.POSITION_CONTROL,
            targetPositions=joint_targets,
            forces=[500]*6
        )

    # ---------------- TURNTABLE & OBJECT ----------------
    disk_quat = p.getQuaternionFromEuler([0, 0, math.radians(current_yaw)])
    p.resetBasePositionAndOrientation(disk_id, [0,0,disk_height/2], disk_quat)

    obj_pos, obj_quat = p.multiplyTransforms(
        [0,0,disk_height/2],
        disk_quat,
        base_offset_pos,
        base_quat
    )
    p.resetBasePositionAndOrientation(obj_id, obj_pos, obj_quat)

    # ---------------- FRUSTUM ----------------
    frustum_ids = draw_camera_frustum(new_cam_eye, new_cam_target, frustum_ids)

    time.sleep(1/60)