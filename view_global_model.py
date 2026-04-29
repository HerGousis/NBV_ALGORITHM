import numpy as np
import pyvista as pv
import os

def visualize_global_model(file_path="/home/hercules/mpambis_diplomatiki/global_models/global_model_81.npy"):
    if not os.path.exists(file_path):
        print(f"Error: Το αρχείο {file_path} δεν βρέθηκε!")
        return

    # Φόρτωση δεδομένων
    data = np.load(file_path, allow_pickle=True).item()
    points = data['points']
    colors = data['colors']
    metadata = data.get('metadata', {})
    voxel_size = metadata.get('voxel_size', 0.005)
    camera_history = metadata.get('camera_history', [])
    last_pose = metadata.get('last_camera_pose', None)

    print(f"--- Global Model Viewer ---")
    print(f"Συνολικά σημεία: {len(points)}")
    print(f"Αρχεία σάρωσης στο ιστορικό: {len(camera_history)}")

    # Δημιουργία Plotter
    plotter = pv.Plotter(title="Global Model Viewer - Camera Path & Current Pose")
    plotter.set_background("#FBFDFD")

    # Προβολή Voxel Grid
    point_cloud = pv.PolyData(points)
    point_cloud["RGB"] = colors
    cube = pv.Cube(x_length=voxel_size, y_length=voxel_size, z_length=voxel_size)
    voxel_mesh = point_cloud.glyph(geom=cube, scale=False, orient=False)
    plotter.add_mesh(voxel_mesh, rgb=True, scalars="RGB", preference="point", opacity=1.0)

    # Προβολή ιστορικού κινήσεων
    if len(camera_history) > 1:
        for i in range(len(camera_history) - 1):
            pose = camera_history[i]
            pos = pose['position']
            rot = pose['rotation']
            direction = rot @ np.array([0, 0, -1])
            old_arrow = pv.Arrow(start=pos, direction=direction, scale=0.04)
            plotter.add_mesh(old_arrow, color="gray", opacity=0.5)

    # Προβολή τελευταίας θέσης
    if last_pose:
        curr_pos = last_pose['position']
        curr_rot = last_pose['rotation']
        curr_direction = curr_rot @ np.array([0, 0, -1])
        last_arrow = pv.Arrow(start=curr_pos, direction=curr_direction, scale=0.08)
        plotter.add_mesh(last_arrow, color="cyan", label="")
        plotter.add_mesh(pv.Sphere(radius=0.012, center=curr_pos), color="cyan")
        plotter.add_point_labels([curr_pos], ["CURRENT POSE"], font_size=14, text_color="blue")

    # Ρυθμίσεις εμφάνισης
    plotter.add_axes()
    plotter.add_legend()
    print("Το παράθυρο άνοιξε. Τα γκρι βέλη δείχνουν τη διαδρομή, το Cyan τη θέση σου.")
    plotter.show()

if __name__ == "__main__":
    visualize_global_model()