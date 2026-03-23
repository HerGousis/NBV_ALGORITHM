import numpy as np
import pyvista as pv
import os

def visualize_global_model(file_path="global_model.npy"):
    if not os.path.exists(file_path):
        print(f"Error: Το αρχείο {file_path} δεν βρέθηκε!")
        return

    # 1. Φόρτωση των δεδομένων
    data = np.load(file_path, allow_pickle=True).item()
    points = data['points']
    colors = data['colors']
    metadata = data.get('metadata', {})
    voxel_size = metadata.get('voxel_size', 0.005)
    
    # Ανάκτηση ιστορικού και τελευταίας θέσης
    camera_history = metadata.get('camera_history', [])
    last_pose = metadata.get('last_camera_pose', None)

    print(f"--- Global Model Viewer ---")
    print(f"Συνολικά σημεία: {len(points)}")
    print(f"Αρχεία σάρωσης στο ιστορικό: {len(camera_history)}")

    # 2. Δημιουργία Plotter
    plotter = pv.Plotter(title="Global Model Viewer - Camera Path & Current Pose")
    plotter.set_background("#FBFDFD")

    # 3. Προβολή Voxel Grid (Το μοντέλο)
    point_cloud = pv.PolyData(points)
    point_cloud["RGB"] = colors
    cube = pv.Cube(x_length=voxel_size, y_length=voxel_size, z_length=voxel_size)
    
    # Χρήση glyph για voxel οπτικοποίηση
    voxel_mesh = point_cloud.glyph(geom=cube, scale=False, orient=False)
    plotter.add_mesh(voxel_mesh, rgb=True, scalars="RGB", preference="point", opacity=1.0)

    # 4. Προβολή Ιστορικού Κινήσεων (Παλιά βέλη)
    # Προβάλλουμε όλα τα βέλη εκτός από το τελευταίο με γκρι χρώμα
    if len(camera_history) > 1:
        for i in range(len(camera_history) - 1):
            pose = camera_history[i]
            pos = pose['position']
            rot = pose['rotation']
            # Υπολογισμός κατεύθυνσης (Rotation @ [0,0,-1])
            direction = rot @ np.array([0, 0, -1])
            
            old_arrow = pv.Arrow(start=pos, direction=direction, scale=0.04)
            plotter.add_mesh(old_arrow, color="gray", opacity=0.5)

    # 5. Προβολή Τελευταίας Θέσης (Το τρέχον βέλος)
    if last_pose:
        curr_pos = last_pose['position']
        curr_rot = last_pose['rotation']
        curr_direction = curr_rot @ np.array([0, 0, -1])
        
        # Το τρέχον βέλος είναι μεγαλύτερο και Cyan
        last_arrow = pv.Arrow(start=curr_pos, direction=curr_direction, scale=0.08)
        plotter.add_mesh(last_arrow, color="cyan", label="")
        
        # Προσθήκη σφαίρας και Label
        plotter.add_mesh(pv.Sphere(radius=0.012, center=curr_pos), color="cyan")
        plotter.add_point_labels([curr_pos], ["CURRENT POSE"], font_size=14, text_color="blue")

    # 6. Ρυθμίσεις εμφάνισης
    plotter.add_axes()
    plotter.add_legend()
    
    print("Το παράθυρο άνοιξε. Τα γκρι βέλη δείχνουν τη διαδρομή, το Cyan τη θέση σου.")
    plotter.show()

if __name__ == "__main__":
    visualize_global_model("global_model.npy")