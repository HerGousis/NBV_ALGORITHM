import os
import numpy as np
import pyvista as pv
import imageio
from sklearn.neighbors import NearestNeighbors

# -------------------- Ρυθμίσεις --------------------
folder = "global_models"
output_video = "planning_progress.mp4"
frame_folder = "frames_temp"
fps = 4  # frames per second βίντεο

if not os.path.exists(frame_folder):
    os.makedirs(frame_folder)

# -------------------- 1. Φόρτωση και κανονικά frames --------------------
files = [f for f in os.listdir(folder) if f.startswith("global_model_") and f.endswith(".npy")]
files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
frame_paths = []

for i, f in enumerate(files):
    data = np.load(os.path.join(folder, f), allow_pickle=True).item()
    points = data['points']
    metadata = data.get('metadata', {})
    voxel_size = metadata.get('voxel_size', 0.005)

    # Υπολογισμός interest scores
    nbrs = NearestNeighbors(radius=voxel_size*2.5).fit(points)
    adj_matrix = nbrs.radius_neighbors_graph(points)
    neighbor_counts = np.array(adj_matrix.sum(axis=1)).flatten()
    scores = 1.0 - (neighbor_counts / (np.max(neighbor_counts)+1e-6))

    # PyVista plot
    plotter = pv.Plotter(off_screen=True)
    plotter.set_background("black")

    # Voxels
    pc = pv.PolyData(points)
    pc["Interest"] = scores
    voxels = pc.glyph(
        geom=pv.Cube(x_length=voxel_size, y_length=voxel_size, z_length=voxel_size),
        scale=False
    )
    plotter.add_mesh(voxels, scalars="Interest", cmap="turbo", opacity=0.6)

    # Ρομπότ
    last_pose = metadata.get('last_camera_pose', None)
    curr_pos = np.array(last_pose['position']) if last_pose else np.array([0,0,0])
    plotter.add_mesh(pv.Sphere(radius=0.015, center=curr_pos), color="cyan")

    # Ιστορικό κάμερας
    camera_history = metadata.get('camera_history', [])
    for pose in camera_history:
        cam_pos = np.array(pose['position'])
        rot = pose.get('rotation', np.eye(3))
        view_dir = rot @ np.array([0,0,-1])
        plotter.add_mesh(pv.Arrow(start=cam_pos, direction=view_dir, scale=0.03),
                         color="gray", opacity=0.4)

    # Κείμενο
    plotter.add_text(f"Global Model ID: {i}", position='upper_left', font_size=12, color='white')

    # Αποθήκευση frame
    frame_path = os.path.join(frame_folder, f"frame_{i:03d}.png")
    plotter.screenshot(frame_path)
    frame_paths.append(frame_path)
    plotter.close()

# -------------------- 2. Δημιουργία βίντεο --------------------
with imageio.get_writer(output_video, mode='I', fps=fps) as writer:
    for fp in frame_paths:
        image = imageio.imread(fp)
        writer.append_data(image)

print(f"🎬 Video saved as {output_video}")