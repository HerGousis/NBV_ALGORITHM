import pybullet as p
import pybullet_data
import time
import os
import math
import numpy as np
from PIL import Image
import shutil

# -------------------- 1. Σύνδεση στον simulator --------------------
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")

# -------------------- 2. ΡΥΘΜΙΣΕΙΣ ΦΑΚΕΛΩΝ --------------------
object_folder = "object_files"
save_folder = "scans"
results_folder = "results"

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

# -------------------- 3. ΦΟΡΤΩΣΗ ΑΝΤΙΚΕΙΜΕΝΟΥ --------------------
obj_path = os.path.join(object_folder, "3.obj")
img_path = os.path.join(object_folder, "3.png")
current_scale = [0.3, 0.3, 0.3]
pos = [0, 0, 0.0]
quat = p.getQuaternionFromEuler([math.radians(87), 0, 0])

try:
    visual_id = p.createVisualShape(p.GEOM_MESH, fileName=obj_path, meshScale=current_scale)
    collision_id = p.createCollisionShape(p.GEOM_MESH, fileName=obj_path, meshScale=current_scale)
    obj_id = p.createMultiBody(0, collision_id, visual_id, pos, quat)
except Exception as e:
    print(f"Mesh not found, loading default cube. Error: {e}")
    obj_id = p.loadURDF("cube_small.urdf", pos)

# Texture
try:
    if os.path.exists(img_path):
        raw_img = Image.open(img_path).convert('RGB')
        temp_tex_path = os.path.join(object_folder, "fixed_texture.png")
        raw_img.save(temp_tex_path)
        tex_id = p.loadTexture(temp_tex_path)
        if tex_id >= 0:
            p.changeVisualShape(obj_id, -1, textureUniqueId=tex_id, rgbaColor=[1, 1, 1, 1])
except Exception as e:
    print(f"Texture Error: {e}")

# -------------------- 4. CAMERA SETTINGS --------------------
width, height = 1280, 720
fov = 60

cam_x, cam_y, cam_z = 0.7, -0.7, 0.7
yaw, pitch = 135.0, -30.0
speed = 0.02
sens = 1.2
frame_count = 0

pyramid_line_ids = []

def draw_camera_frustum(c_pos, t_pos, existing_ids=[]):
    color = [1, 0, 0]
    near_dist = 0.15
    size = 0.08
    forward = (t_pos - c_pos) / (np.linalg.norm(t_pos - c_pos) + 1e-6)
    world_up = np.array([0, 0, 1])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 0.001: right = np.array([1, 0, 0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    base_center = c_pos + forward * near_dist
    p1 = base_center + (right * size) + (up * size)
    p2 = base_center - (right * size) + (up * size)
    p3 = base_center - (right * size) - (up * size)
    p4 = base_center + (right * size) - (up * size)

    lines = [(c_pos, p1), (c_pos, p2), (c_pos, p3), (c_pos, p4),
             (p1, p2), (p2, p3), (p3, p4), (p4, p1)]

    new_ids = []
    for i, (start, end) in enumerate(lines):
        l_id = p.addUserDebugLine(start, end, color, 2,
                                  replaceItemUniqueId=existing_ids[i] if i < len(existing_ids) else -1)
        new_ids.append(l_id)
    return new_ids

target_x = cam_x + math.cos(math.radians(yaw)) * math.cos(math.radians(pitch))
target_y = cam_y + math.sin(math.radians(yaw)) * math.cos(math.radians(pitch))
target_z = cam_z + math.sin(math.radians(pitch))
camera_target_pos = np.array([target_x, target_y, target_z])

pyramid_line_ids = draw_camera_frustum(np.array([cam_x, cam_y, cam_z]), camera_target_pos)
text_id = p.addUserDebugText("Camera POV", [cam_x, cam_y, cam_z], [1, 0, 0])

# -------------------- 5. next_position.npy --------------------
next_pose_file = "next_position.npy"
last_mtime = None

print(f"\nREADY. Saving Data to: /{save_folder} and Images to: /{results_folder}")
print("ARROWS: Rotate | WASD: Move | SPACE: Save Manually\n")

# -------------------- 6. HELPER FUNCTION --------------------
def capture_cross_sequence():
    global frame_count, c_pos_np, forward_vec

    # offset για μικρές μετακινήσεις
    offset = 0.1

    # Πίνακας μετατοπίσεων γύρω από το center
    positions = [
        c_pos_np + np.array([ offset, 0, 0]),  # +X
        c_pos_np + np.array([-offset, 0, 0]),  # -X
        c_pos_np + np.array([0,  0, offset]),  # +Y
        c_pos_np + np.array([0,  0, -offset])   # -Y
    ]

    # 1. SIDE CAPTURES (μόνο εικόνες)
    for i, pos in enumerate(positions):
        temp_target = pos + forward_vec

        view_matrix = p.computeViewMatrix(pos.tolist(), temp_target.tolist(), [0,0,1])
        projection_matrix = p.computeProjectionMatrixFOV(fov, width/height, 0.01, 10.0)

        _, _, rgb_img, _, _ = p.getCameraImage(width, height, view_matrix, projection_matrix)
        rgb_array = np.array(rgb_img).reshape(height, width, 4)[:, :, :3].astype(np.uint8)

        img_pil = Image.fromarray(rgb_array)
        img_pil.save(os.path.join(results_folder, f"image_{frame_count}_side_{i}.png"))
        time.sleep(0.05)

    # 2. BACK TO CENTER & FINAL CAPTURE
    center_target = c_pos_np + forward_vec
    view_matrix = p.computeViewMatrix(c_pos_np.tolist(), center_target.tolist(), [0,0,1])
    projection_matrix = p.computeProjectionMatrixFOV(fov, width/height, 0.01, 10.0)

    _, _, rgb_img, depth_img, _ = p.getCameraImage(width, height, view_matrix, projection_matrix)
    rgb_array = np.array(rgb_img).reshape(height, width, 4)[:, :, :3].astype(np.uint8)
    depth_buffer = np.array(depth_img).reshape(height, width)
    real_depth = 10.0 * 0.01 / (10.0 - (10.0 - 0.01) * depth_buffer)

    # save image
    img_pil = Image.fromarray(rgb_array)
    img_pil.save(os.path.join(results_folder, f"image_{frame_count}.png"))

    # save scan
    vm = np.array(view_matrix).reshape(4,4).T
    save_data = {
        'depth_map': real_depth,
        'rgb_map': rgb_array,
        'cam_pos': c_pos_np,
        'cam_rot_mat': vm[:3, :3].T,
        'full_view_matrix': np.array(view_matrix)
    }
    np.save(os.path.join(save_folder, f"scan_{frame_count}.npy"), save_data)

    print(f"📸 Cross capture {frame_count} DONE (scan μόνο στο center)")
    frame_count += 1

# -------------------- 7. MAIN LOOP --------------------
while True:
    keys = p.getKeyboardEvents()
    moved = False
    auto_capture = False

    # ----------- AUTO CAMERA UPDATE -----------
    if os.path.exists(next_pose_file):
        try:
            mtime = os.path.getmtime(next_pose_file)
            if last_mtime is None or mtime != last_mtime:
                last_mtime = mtime
                data_in = np.load(next_pose_file, allow_pickle=True).item()

                new_pos = np.array(data_in["position"])
                view_dir = np.array(data_in["view_direction"])
                view_dir /= (np.linalg.norm(view_dir) + 1e-6)

                cam_x, cam_y, cam_z = new_pos
                yaw = math.degrees(math.atan2(-view_dir[1], -view_dir[0]))
                pitch = math.degrees(math.asin(-view_dir[2]))

                moved = True
                auto_capture = True
                print("📍 New position received!")
        except Exception as e:
            print(f"Error loading npy: {e}")

    # ----------- KEYBOARD -----------
    if keys.get(p.B3G_LEFT_ARROW, 0) & p.KEY_IS_DOWN: yaw -= sens; moved = True
    if keys.get(p.B3G_RIGHT_ARROW, 0) & p.KEY_IS_DOWN: yaw += sens; moved = True
    if keys.get(p.B3G_UP_ARROW, 0) & p.KEY_IS_DOWN: pitch += sens; moved = True
    if keys.get(p.B3G_DOWN_ARROW, 0) & p.KEY_IS_DOWN: pitch -= sens; moved = True
    pitch = max(-89, min(89, pitch))

    rad_yaw, rad_pitch = math.radians(yaw), math.radians(pitch)

    forward_vec = np.array([
        math.cos(rad_yaw)*math.cos(rad_pitch),
        math.sin(rad_yaw)*math.cos(rad_pitch),
        math.sin(rad_pitch)
    ])

    right_vec = np.array([
        math.sin(rad_yaw),
        -math.cos(rad_yaw),
        0
    ])

    if keys.get(ord('w'),0) & p.KEY_IS_DOWN: cam_x+=forward_vec[0]*0.02; cam_y+=forward_vec[1]*0.02; cam_z+=forward_vec[2]*0.02; moved=True
    if keys.get(ord('s'),0) & p.KEY_IS_DOWN: cam_x-=forward_vec[0]*0.02; cam_y-=forward_vec[1]*0.02; cam_z-=forward_vec[2]*0.02; moved=True
    if keys.get(ord('a'),0) & p.KEY_IS_DOWN: cam_x-=right_vec[0]*0.02; cam_y-=right_vec[1]*0.02; moved=True
    if keys.get(ord('d'),0) & p.KEY_IS_DOWN: cam_x+=right_vec[0]*0.02; cam_y+=right_vec[1]*0.02; moved=True
    if keys.get(ord('e'),0) & p.KEY_IS_DOWN: cam_z+=0.02; moved=True
    if keys.get(ord('q'),0) & p.KEY_IS_DOWN: cam_z-=0.02; moved=True

    c_pos_np = np.array([cam_x, cam_y, cam_z])
    camera_target_pos = c_pos_np + forward_vec

    if moved:
        pyramid_line_ids = draw_camera_frustum(c_pos_np, camera_target_pos, pyramid_line_ids)
        text_id = p.addUserDebugText("Camera POV", [cam_x, cam_y, cam_z], [1, 0, 0], replaceItemUniqueId=text_id)

    # ----------- CAPTURE -----------
    if (keys.get(p.B3G_SPACE, 0) & p.KEY_WAS_TRIGGERED) or auto_capture:
        time.sleep(0.05)
        capture_cross_sequence()

    p.stepSimulation()
    time.sleep(1/240)