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
# TKINTER (2D CONTROL)
# ============================================================
WIDTH, HEIGHT = 800, 800
CENTER = (WIDTH // 2, HEIGHT // 2)
SCALE = 150
current_point = None

def to_canvas(x, y):
    cx = CENTER[0] + x * SCALE
    cy = CENTER[1] - y * SCALE
    return cx, cy

def to_math(cx, cy):
    x = (cx - CENTER[0]) / SCALE
    y = (CENTER[1] - cy) / SCALE
    return x, y

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
    
    # Άξονες με χρώματα PyBullet (X=Κόκκινο, Y=Πράσινο)
    canvas.create_line(0, CENTER[1], WIDTH, CENTER[1], fill="red", width=2, dash=(4, 4))
    canvas.create_line(CENTER[0], 0, CENTER[0], HEIGHT, fill="green", width=2, dash=(4, 4))
    
    # Circle (Όριο τραπεζιού)
    r = 2
    x0, y0 = to_canvas(-r, -r)
    x1, y1 = to_canvas(r, r)
    canvas.create_oval(x0, y0, x1, y1, outline="gray")

    with lock:
        total_yaw = disk_yaw_shared
        start_arc = blue_angle_deg

    if mouse_pressed and current_point is not None:
        x, y = current_point
        v2 = (x, -y)
        bbox = (CENTER[0]-50, CENTER[1]-50, CENTER[0]+50, CENTER[1]+50)
        
        # Τόξο περιστροφής
        canvas.create_arc(bbox, start=start_arc, extent=total_yaw, 
                          fill="#FFEB3B", outline="#FBC02D", width=2)
        
        # Μπλε γραμμή (Ποντίκι) - Κόκκινη (Mirror/Target)
        canvas.create_line(*to_canvas(0, 0), *to_canvas(x, y), fill="blue", width=3, arrow=tk.LAST)
        canvas.create_line(*to_canvas(0, 0), *to_canvas(*v2), fill="red", width=3, arrow=tk.LAST)
        
        canvas.create_text(WIDTH-120, 40, text=f"Rotation: {total_yaw:.1f}°", 
                           font=("Courier", 14, "bold"), fill="darkred")
    else:
        canvas.create_text(WIDTH-120, 40, text="Rotation: 0.0°", font=("Courier", 14), fill="gray")

def on_press(event):
    global current_point, mouse_pressed
    x, y = to_math(event.x, event.y)
    current_point = (x, y)
    mouse_pressed = True
    update_angle_from_point(x, y)
    draw()

def on_drag(event):
    global current_point
    if not mouse_pressed: return
    x, y = to_math(event.x, event.y)
    current_point = (x, y)
    update_angle_from_point(x, y)
    draw()

def on_release(event):
    global mouse_pressed, disk_yaw_shared
    mouse_pressed = False
    with lock: disk_yaw_shared = 0.0
    draw()

def run_tk():
    global canvas
    root = tk.Tk()
    root.title("Control Panel: Blue to Red (Mirror Logic)")
    canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="white")
    canvas.pack()
    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    draw()
    root.mainloop()

# Έναρξη Tkinter σε ξεχωριστό Thread
tk_thread = threading.Thread(target=run_tk)
tk_thread.daemon = True
tk_thread.start()

# ============================================================
# PYBULLET SETUP
# ============================================================
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setRealTimeSimulation(True)
p.loadURDF("plane.urdf")

# -------------------- UR5 ROBOT --------------------
ROBOT_URDF_PATH = "./ur_e_description/urdf/ur5e.urdf"
ur5 = p.loadURDF(ROBOT_URDF_PATH, basePosition=[0, 1.0, 0], useFixedBase=True)
end_effector_index = 7

# Joint Initialization
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
        ur5, [1, 2, 3, 4, 5, 6],
        p.POSITION_CONTROL,
        targetPositions=joint_targets,
        forces=[500] * 6
    )

# -------------------- TURNTABLE --------------------
disk_radius, disk_height = 0.2, 0.02
disk_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=disk_radius, length=disk_height, rgbaColor=[0.3, 0.3, 0.3, 1])
disk_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=disk_radius, height=disk_height)
disk_id = p.createMultiBody(0, disk_collision, disk_visual, [0, 0, disk_height / 2])

# -------------------- OBJECT ON DISK --------------------
object_folder = "object_files"
obj_path = os.path.join(object_folder, "demetra_200k.obj")
scale = [0.3, 0.3, 0.3]
base_offset_pos = np.array([0, 0, disk_height])
base_quat = p.getQuaternionFromEuler([math.radians(87), 0, 0])

try:
    visual = p.createVisualShape(p.GEOM_MESH, fileName=obj_path, meshScale=scale)
    collision = p.createCollisionShape(p.GEOM_MESH, fileName=obj_path, meshScale=scale)
    obj_id = p.createMultiBody(0, collision, visual, base_offset_pos.tolist(), base_quat)
except:
    print("Object file not found. Creating a box placeholder.")
    obj_id = p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.1]), 
                               p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.1], rgbaColor=[0.8, 0.2, 0.2, 1]),
                               [0, 0, 0.1])

def attach_object(disk_pos, disk_quat):
    obj_pos, obj_quat = p.multiplyTransforms(disk_pos, disk_quat, base_offset_pos, base_quat)
    p.resetBasePositionAndOrientation(obj_id, obj_pos, obj_quat)

# -------------------- INDICATOR ARROW --------------------
arrow_length, arrow_radius = 0.15, 0.010
arrow_z = disk_height + 0.015
local_rot_arrow = p.getQuaternionFromEuler([0, math.pi / 2, 0])

shaft_v = p.createVisualShape(p.GEOM_CYLINDER, radius=arrow_radius, length=arrow_length, rgbaColor=[1, 0, 0, 1])
shaft_id = p.createMultiBody(0, -1, shaft_v, [0,0,0])
head_v = p.createVisualShape(p.GEOM_CYLINDER, radius=0.025, length=0.05, rgbaColor=[1, 0, 0, 1])
head_id = p.createMultiBody(0, -1, head_v, [0,0,0])

def update_arrow(yaw_deg):
    quat = p.getQuaternionFromEuler([0, 0, math.radians(yaw_deg)])
    # Shaft
    l_shaft = [arrow_length / 2, 0, arrow_z]
    ws_s_p, ws_s_r = p.multiplyTransforms([0,0,0], quat, l_shaft, local_rot_arrow)
    p.resetBasePositionAndOrientation(shaft_id, ws_s_p, ws_s_r)
    # Head
    l_head = [arrow_length, 0, arrow_z]
    ws_h_p, ws_h_r = p.multiplyTransforms([0,0,0], quat, l_head, local_rot_arrow)
    p.resetBasePositionAndOrientation(head_id, ws_h_p, ws_h_r)

# -------------------- CAMERA FRUSTUM --------------------
pyramid_line_ids = []
cam_text_id = -1

def draw_camera_frustum(c_pos, t_pos, existing_ids=[]):
    color, near_dist, size = [1, 0, 0], 0.15, 0.08
    forward = (t_pos - c_pos)
    norm = np.linalg.norm(forward)
    if norm < 1e-6: return existing_ids
    forward /= norm
    world_up = np.array([0, 0, 1])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 0.001: right = np.array([1, 0, 0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    base_center = c_pos + forward * near_dist
    p1, p2 = base_center + right * size + up * size, base_center - right * size + up * size
    p3, p4 = base_center - right * size - up * size, base_center + right * size - up * size
    lines = [(c_pos, p1), (c_pos, p2), (c_pos, p3), (c_pos, p4), (p1, p2), (p2, p3), (p3, p4), (p4, p1)]
    new_ids = []
    for i, (s, e) in enumerate(lines):
        lid = p.addUserDebugLine(s, e, color, 2, replaceItemUniqueId=existing_ids[i] if i < len(existing_ids) else -1)
        new_ids.append(lid)
    return new_ids

# -------------------- MAIN LOOP --------------------
cam_local_offset = np.array([0.05, 0.0, 0.0])
cam_look_distance = 0.3

while True:
    # Λήψη δεδομένων από το Tkinter (Thread-safe)
    with lock:
        current_yaw = disk_yaw_shared

    # Έλεγχος βραχίονα με πληκτρολόγιο
    keys = p.getKeyboardEvents()
    joint_speed = 0.05
    if keys.get(p.B3G_LEFT_ARROW, 0) & p.KEY_IS_DOWN:  joint_targets[0] -= joint_speed
    if keys.get(p.B3G_RIGHT_ARROW, 0) & p.KEY_IS_DOWN: joint_targets[0] += joint_speed
    if keys.get(p.B3G_UP_ARROW, 0) & p.KEY_IS_DOWN:    joint_targets[1] += joint_speed
    if keys.get(p.B3G_DOWN_ARROW, 0) & p.KEY_IS_DOWN:  joint_targets[1] -= joint_speed
    
    apply_joints()

    # Ενημέρωση Τραπεζιού και Αντικειμένου
    disk_quat = p.getQuaternionFromEuler([0, 0, math.radians(current_yaw)])
    disk_pos = [0, 0, disk_height / 2]
    p.resetBasePositionAndOrientation(disk_id, disk_pos, disk_quat)
    attach_object(disk_pos, disk_quat)
    update_arrow(current_yaw)

    # Υπολογισμός Κάμερας EE
    link_state = p.getLinkState(ur5, end_effector_index, computeForwardKinematics=True)
    ee_pos, ee_ori = np.array(link_state[0]), link_state[1]
    rot = np.array(p.getMatrixFromQuaternion(ee_ori)).reshape(3, 3)
    ee_forward, ee_right, ee_up = rot[:, 0], rot[:, 1], rot[:, 2]

    cam_pos = ee_pos + ee_forward * cam_local_offset[0]
    target = cam_pos + ee_forward * cam_look_distance
    
    # Debug visuals
    pyramid_line_ids = draw_camera_frustum(cam_pos, target, pyramid_line_ids)
    cam_text_id = p.addUserDebugText("Camera POV", cam_pos.tolist(), [1, 0, 0], textSize=1.2, replaceItemUniqueId=cam_text_id)

    # Render POV
    view = p.computeViewMatrix(cam_pos.tolist(), target.tolist(), ee_up.tolist())
    proj = p.computeProjectionMatrixFOV(60, 1280/720, 0.01, 10)
    p.getCameraImage(1280, 720, view, proj)

    time.sleep(1 / 60)