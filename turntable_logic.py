import pybullet as p
import pybullet_data
import time
import os
import math
import numpy as np
import tkinter as tk
import threading
import shutil
from PIL import Image

# ============================================================
# SHARED STATE
# ============================================================
disk_yaw_shared = 0.0
blue_angle_deg  = 0.0
mouse_pressed   = False
lock            = threading.Lock()

# ============================================================
# FOLDERS
# ============================================================
object_folder  = "object_files"
save_folder    = "scans"
results_folder = "results"

error_flag_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "error_flag.npy")
next_pose_file  = "next_position.npy"
last_mtime      = None
frame_count     = 0

def setup_folders(folder):
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
        print(f"Folder '{folder}' cleared.")
    else:
        os.makedirs(folder)
        print(f"Created folder: {folder}")

setup_folders(save_folder)
setup_folders(results_folder)

# ============================================================
# TKINTER (2D DISPLAY ONLY)
# ============================================================
WIDTH, HEIGHT = 800, 800
CENTER        = (WIDTH // 2, HEIGHT // 2)
SCALE         = 150
current_point = None

def to_canvas(x, y):
    cx = CENTER[0] + x * SCALE
    cy = CENTER[1] - y * SCALE
    return cx, cy

def update_angle_from_point(x, y):
    """Περιστροφή δίσκου ΜΟΝΟ όταν Y < 0."""
    global disk_yaw_shared, blue_angle_deg
    if y < 0:
        angle = math.degrees(math.atan2(-y, x))
        with lock:
            disk_yaw_shared = 2 * angle
            blue_angle_deg  = -angle
    else:
        with lock:
            disk_yaw_shared = 0.0
            blue_angle_deg  = 0.0

def draw():
    canvas.delete("all")
    canvas.create_line(0, CENTER[1], WIDTH, CENTER[1], fill="red",   width=2, dash=(4,4))
    canvas.create_line(CENTER[0], 0, CENTER[0], HEIGHT, fill="green", width=2, dash=(4,4))

    r = 2
    x0, y0 = to_canvas(-r, -r)
    x1, y1 = to_canvas( r,  r)
    canvas.create_oval(x0, y0, x1, y1, outline="gray")

    with lock:
        total_yaw  = disk_yaw_shared
        start_arc  = blue_angle_deg
        is_pressed = mouse_pressed
        pt         = current_point

    if is_pressed and pt is not None:
        x, y = pt
        v2   = (x, -y)
        bbox = (CENTER[0]-50, CENTER[1]-50, CENTER[0]+50, CENTER[1]+50)
        canvas.create_arc(bbox, start=start_arc, extent=total_yaw,
                          fill="#FFEB3B", outline="#FBC02D", width=2)
        canvas.create_line(*to_canvas(0,0), *to_canvas(x,y),
                           fill="blue", width=3, arrow=tk.LAST)
        canvas.create_line(*to_canvas(0,0), *to_canvas(*v2),
                           fill="red",  width=3, arrow=tk.LAST)
        canvas.create_text(WIDTH-120, 40,
                           text=f"Rotation: {total_yaw:.1f}°",
                           font=("Courier", 14, "bold"), fill="darkred")
    else:
        canvas.create_text(WIDTH-120, 40,
                           text="Rotation: 0.0°",
                           font=("Courier", 14), fill="gray")

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

# -------------------- CAMERA SETTINGS --------------------
width_img, height_img = 1920, 1080
fov = 60

# -------------------- CAMERA FRUSTUM --------------------
def draw_camera_frustum(c_pos, t_pos, existing_ids=[]):
    color     = [1, 0, 0]
    near_dist = 0.15
    size      = 0.08

    c_pos = np.array(c_pos)
    t_pos = np.array(t_pos)

    forward  = (t_pos - c_pos) / (np.linalg.norm(t_pos - c_pos) + 1e-6)
    world_up = np.array([0, 0, 1])
    right    = np.cross(forward, world_up)
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
    for i, (s, e) in enumerate(lines):
        l_id = p.addUserDebugLine(
            s.tolist(), e.tolist(), color, 2,
            replaceItemUniqueId=existing_ids[i] if i < len(existing_ids) else -1
        )
        new_ids.append(l_id)
    return new_ids

# -------------------- UR5 --------------------
#ur5 = p.loadURDF("./ur_e_description/urdf/ur5e.urdf",
#                 basePosition=[0, 1.0, 0],
#                 useFixedBase=True)

#joint_targets = [0, -1.57, 1.57, -1.57, -1.57, 0]
#for i in range(6):
#    p.resetJointState(ur5, i+1, joint_targets[i])

# -------------------- TURNTABLE --------------------
disk_radius, disk_height = 0.2, 0.02

disk_id = p.createMultiBody(
    0,
    p.createCollisionShape(p.GEOM_CYLINDER, radius=disk_radius, height=disk_height),
    p.createVisualShape(p.GEOM_CYLINDER, radius=disk_radius, length=disk_height,
                        rgbaColor=[0.3, 0.3, 0.3, 1]),
    [0, 0, disk_height/2]
)

# -------------------- OBJECT --------------------
obj_path = os.path.join(object_folder, "demetra_200k.obj")
img_path = os.path.join(object_folder, "demetra_200k.png")

base_offset_pos = np.array([0, 0, disk_height])
base_quat       = p.getQuaternionFromEuler([math.radians(87), 0, 0])

try:
    visual_id    = p.createVisualShape(p.GEOM_MESH, fileName=obj_path, meshScale=[0.3]*3)
    collision_id = p.createCollisionShape(p.GEOM_MESH, fileName=obj_path, meshScale=[0.3]*3)
    obj_id = p.createMultiBody(0, collision_id, visual_id,
                               base_offset_pos.tolist(), base_quat)
except Exception as e:
    print(f"Mesh not found, loading default box. Error: {e}")
    obj_id = p.createMultiBody(
        0,
        p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05]*3),
        p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05]*3, rgbaColor=[0.8,0.2,0.2,1]),
        [0, 0, 0.1]
    )

# Texture
try:
    if os.path.exists(img_path):
        raw_img       = Image.open(img_path).convert('RGB')
        temp_tex_path = os.path.join(object_folder, "fixed_texture.png")
        raw_img.save(temp_tex_path)
        tex_id = p.loadTexture(temp_tex_path)
        if tex_id >= 0:
            p.changeVisualShape(obj_id, -1, textureUniqueId=tex_id, rgbaColor=[1,1,1,1])
except Exception as e:
    print(f"Texture Error: {e}")

# -------------------- CAMERA STATE --------------------
CAM_EYE_INIT = np.array([0.7, -0.7, 0.7])
YAW_INIT     = 135.0
PITCH_INIT   = -30.0

cam_eye = CAM_EYE_INIT.copy()
yaw     = YAW_INIT
pitch   = PITCH_INIT
speed   = 0.02
sens    = 1.5

space_was_down = False

# ── Frustum εμφανίζεται αμέσως στην αρχική θέση ──────────
_ry0  = math.radians(YAW_INIT)
_rp0  = math.radians(PITCH_INIT)
_fwd0 = np.array([
    math.cos(_ry0) * math.cos(_rp0),
    math.sin(_ry0) * math.cos(_rp0),
    math.sin(_rp0)
])
frustum_ids = draw_camera_frustum(CAM_EYE_INIT, CAM_EYE_INIT + _fwd0)

# ============================================================
# CAPTURE FUNCTION
# ============================================================
def capture_ellipse_sequence(c_pos_np, target_pos_np, old_pos, old_target):
    global frame_count

    # 1. Υπολογισμός forward vector για τις ελλειπτικές λήψεις (βασισμένο στη θέση rendering)
    forward_vec = target_pos_np - c_pos_np
    norm = np.linalg.norm(forward_vec)
    if norm < 1e-6:
        return
    forward_vec /= norm

    a          = 0.1
    b          = 0.05
    num_points = 8

    world_up  = np.array([0, 0, 1])
    right_vec = np.cross(forward_vec, world_up)
    if np.linalg.norm(right_vec) < 1e-6:
        right_vec = np.array([1, 0, 0])
    else:
        right_vec /= np.linalg.norm(right_vec)
    up_vec = np.cross(right_vec, forward_vec)
    up_vec /= np.linalg.norm(up_vec)

    # ── Ελλειπτικές λήψεις (Rendering από c_pos_np) ──────────────────────────────
    for i in range(num_points):
        theta   = (2 * np.pi * i) / num_points
        offset  = a * np.cos(theta) * right_vec + b * np.sin(theta) * up_vec
        cam_pos = c_pos_np + offset

        view_matrix       = p.computeViewMatrix(cam_pos.tolist(), target_pos_np.tolist(), [0,0,1])
        projection_matrix = p.computeProjectionMatrixFOV(fov, width_img/height_img, 0.01, 10.0)

        _, _, rgb_img, _, _ = p.getCameraImage(
            width_img, height_img, view_matrix, projection_matrix,
            lightDirection=[1,1,1], lightColor=[1,1,1], lightDistance=5,
            lightAmbientCoeff=0.3, lightDiffuseCoeff=0.7, lightSpecularCoeff=0.4,
            shadow=1, renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        # Μετατροπή σε uint8 για την PIL
        rgb_array = np.array(rgb_img, dtype=np.uint8).reshape(height_img, width_img, 4)[:,:,:3]
        Image.fromarray(rgb_array).save(
            os.path.join(results_folder, f"image_{frame_count}_ellipse_{i}.png")
        )
        time.sleep(0.01)

    # ── Κεντρική λήψη (Rendering από c_pos_np) ────────────────────────
    view_matrix       = p.computeViewMatrix(c_pos_np.tolist(), target_pos_np.tolist(), [0,0,1])
    projection_matrix = p.computeProjectionMatrixFOV(fov, width_img/height_img, 0.01, 10.0)

    _, _, rgb_img, depth_img, _ = p.getCameraImage(
        width_img, height_img, view_matrix, projection_matrix,
        lightDirection=[1,1,1], lightColor=[1,1,1], lightDistance=5,
        lightAmbientCoeff=0.3, lightDiffuseCoeff=0.7, lightSpecularCoeff=0.4,
        shadow=1, renderer=p.ER_BULLET_HARDWARE_OPENGL
    )

    rgb_array    = np.array(rgb_img, dtype=np.uint8).reshape(height_img, width_img, 4)[:,:,:3]
    depth_buffer = np.array(depth_img).reshape(height_img, width_img)

    near, far  = 0.01, 10.0
    real_depth = far * near / (far - (far - near) * depth_buffer)

    Image.fromarray(rgb_array).save(
        os.path.join(results_folder, f"image_{frame_count}.png")
    )

    # ── Metadata Calculation (Βασισμένο στην old_pos/old_target) ────────
    # Υπολογίζουμε ένα view matrix που αντιστοιχεί στην αρχική (αρνητική Y) θέση
    # ώστε το cam_rot_mat να είναι σωστό για το metadata.
    meta_view_matrix = p.computeViewMatrix(
        np.array(old_pos).tolist(), 
        np.array(old_target).tolist(), 
        [0, 0, 1]
    )
    vm_meta = np.array(meta_view_matrix).reshape(4, 4).T

    save_data = {
        'depth_map':        real_depth,
        'rgb_map':          rgb_array,
        'cam_pos':          np.array(old_pos),    # Αποθήκευση της αρχικής θέσης
        'cam_target':       np.array(old_target), # Αποθήκευση του αρχικού target
        'cam_rot_mat':      vm_meta[:3, :3].T,    # Rotation matrix της αρχικής θέσης
        'full_view_matrix': np.array(meta_view_matrix)
    }

    # Αποθήκευση του scan
    np.save(os.path.join(save_folder, f"scan_{frame_count}.npy"), save_data)

    if os.path.exists(error_flag_file):
        
        print("✅ Error flag cleared after retake.")
       

    print(f"📸 Scan {frame_count} SAVED. Metadata Pos: {old_pos}")
    frame_count += 1

# ============================================================
# MAIN LOOP
# ============================================================
print(f"\nREADY. Scans → /{save_folder}  |  Images → /{results_folder}")
print("ARROWS: Rotate | WASD/QE: Move | SPACE: Capture (always) | R: Reset\n")

while True:
    keys         = p.getKeyboardEvents()
    moved        = False
    auto_capture = False

    # ============================================================
    # ERROR FLAG → RETAKE (Simulate Space Logic)
    # ============================================================
    if os.path.exists(error_flag_file):
        print("⚠️ Error flag detected → Preparing Retake")
        auto_capture = True
        
        # Αν η κάμερα είναι στο Y < 0, ενεργοποιούμε την περιστροφή του δίσκου
        if cam_eye[1] < 0:
            current_point = (cam_eye[0], cam_eye[1])
            with lock:
                mouse_pressed = True
            update_angle_from_point(cam_eye[0], cam_eye[1])

    # ============================================================
    # NPY INPUT → EXACT SPACE SIMULATION
    # ============================================================
    npy_trigger = False

    if os.path.exists(next_pose_file):
        try:
            mtime = os.path.getmtime(next_pose_file)
            if last_mtime is None or mtime != last_mtime:
                last_mtime = mtime
                data_in = np.load(next_pose_file, allow_pickle=True).item()

                # 1. Θέτουμε τη θέση της κάμερας
                cam_eye = np.array(data_in["position"])
                view_dir = np.array(data_in["view_direction"])
                view_dir /= (np.linalg.norm(view_dir) + 1e-6)

                yaw   = math.degrees(math.atan2(-view_dir[1], -view_dir[0]))
                pitch = math.degrees(math.asin(max(-1.0, min(1.0, -view_dir[2]))))

                # 2. Ενεργοποιούμε τη λογική περιστροφής
                if cam_eye[1] < 0:
                    current_point = (cam_eye[0], cam_eye[1])
                    with lock:
                        mouse_pressed = True
                    update_angle_from_point(cam_eye[0], cam_eye[1])
                
                npy_trigger = True
                print(f"📍 NPY Trigger: Moved to {cam_eye} and rotating disk.")

        except Exception as e:
            print(f"Error loading npy: {e}")

    # ============================================================
    # RESET
    # ============================================================
    if keys.get(ord('r'), 0) & p.KEY_WAS_TRIGGERED:
        cam_eye = CAM_EYE_INIT.copy()
        yaw     = YAW_INIT
        pitch   = PITCH_INIT
        moved   = True
        print("🔄 Camera reset")

    # ============================================================
    # CAMERA ROTATION
    # ============================================================
    if keys.get(p.B3G_LEFT_ARROW,  0) & p.KEY_IS_DOWN: yaw -= sens; moved = True
    if keys.get(p.B3G_RIGHT_ARROW, 0) & p.KEY_IS_DOWN: yaw += sens; moved = True
    if keys.get(p.B3G_UP_ARROW,    0) & p.KEY_IS_DOWN: pitch += sens; moved = True
    if keys.get(p.B3G_DOWN_ARROW,  0) & p.KEY_IS_DOWN: pitch -= sens; moved = True

    pitch = max(-89, min(89, pitch))

    ry, rp = math.radians(yaw), math.radians(pitch)
    forward = np.array([
        math.cos(ry) * math.cos(rp),
        math.sin(ry) * math.cos(rp),
        math.sin(rp)
    ])
    right = np.array([math.sin(ry), -math.cos(ry), 0])

    # ============================================================
    # CAMERA MOVEMENT
    # ============================================================
    if keys.get(ord('w'), 0) & p.KEY_IS_DOWN: cam_eye += forward * speed; moved = True
    if keys.get(ord('s'), 0) & p.KEY_IS_DOWN: cam_eye -= forward * speed; moved = True
    if keys.get(ord('a'), 0) & p.KEY_IS_DOWN: cam_eye -= right   * speed; moved = True
    if keys.get(ord('d'), 0) & p.KEY_IS_DOWN: cam_eye += right   * speed; moved = True
    if keys.get(ord('q'), 0) & p.KEY_IS_DOWN: cam_eye[2] -= speed; moved = True
    if keys.get(ord('e'), 0) & p.KEY_IS_DOWN: cam_eye[2] += speed; moved = True

    cam_target = (cam_eye + forward).tolist()

    # ============================================================
    # SPACE LOGIC (MANUAL)
    # ============================================================
    space_down = bool(keys.get(ord(' '), 0) & p.KEY_IS_DOWN)

    if space_down and not space_was_down:
        current_point = (cam_eye[0], cam_eye[1])
        with lock: 
            mouse_pressed = True
        update_angle_from_point(cam_eye[0], cam_eye[1])
    elif space_down and space_was_down:
        update_angle_from_point(cam_eye[0], cam_eye[1])
    elif not space_down and space_was_down:
        with lock:
            mouse_pressed   = False
            disk_yaw_shared = 0.0
            blue_angle_deg  = 0.0
        current_point = None

    space_was_down = space_down

    # ============================================================
    # CALCULATE FINAL TRANSFORMED POSITIONS
    # ============================================================
    with lock:
        current_yaw = disk_yaw_shared

    rad_yaw = math.radians(current_yaw)
    T2th = np.array([
        [np.cos(rad_yaw), -np.sin(rad_yaw), 0, 0],
        [np.sin(rad_yaw),  np.cos(rad_yaw), 0, 0],
        [0,                0,               1, 0],
        [0,                0,               0, 1]
    ])

    Tpoint = np.eye(4)
    Tpoint[:3, 3] = cam_eye
    
    T_final = T2th @ Tpoint
    new_cam_eye = T_final[:3, 3]

    target_vec = np.append(cam_target, 1.0)
    new_cam_target = (T2th @ target_vec)[:3]

    # ============================================================
    # UPDATE VISUALS
    # ============================================================
    disk_quat = p.getQuaternionFromEuler([0, 0, rad_yaw])
    p.resetBasePositionAndOrientation(disk_id, [0, 0, disk_height/2], disk_quat)
    obj_pos, obj_quat = p.multiplyTransforms([0, 0, disk_height/2], disk_quat, base_offset_pos, base_quat)
    p.resetBasePositionAndOrientation(obj_id, obj_pos, obj_quat)

    frustum_ids = draw_camera_frustum(new_cam_eye, new_cam_target, frustum_ids)

    # ============================================================
    # CAPTURE TRIGGER (SPACE OR NPY OR AUTO)
    # ============================================================
    space_triggered = bool(keys.get(ord(' '), 0) & p.KEY_WAS_TRIGGERED)

    if space_triggered or npy_trigger or auto_capture:
        # Step A: Capture
        time.sleep(0.1) 
        capture_ellipse_sequence(new_cam_eye, new_cam_target, cam_eye, cam_target)
        
        # Step B: Reset state αν η λήψη ήταν αυτόματη (NPY ή Error Flag)
        if npy_trigger or auto_capture:
            with lock:
                mouse_pressed   = False
                disk_yaw_shared = 0.0
                blue_angle_deg  = 0.0
            current_point = None
            if auto_capture:
                print("🔄 Error Retake Complete: Disk Reset.")
            else:
                print("🔄 NPY Sequence Complete: Disk Reset.")

    time.sleep(1/60)