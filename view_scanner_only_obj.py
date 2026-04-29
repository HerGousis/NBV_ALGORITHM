import numpy as np
import pyvista as pv
import os
from sklearn.linear_model import RANSACRegressor
import glob


class ScanMonitor:
    def __init__(self, scan_folder="scans", voxel_size=0.005, output_folder="global_models"):
        self.scan_folder = scan_folder
        self.voxel_size = voxel_size
        self.output_folder = output_folder
        self.loaded_files = set() 
        
        # Μετρητής για την ονοματοδοσία των αρχείων (versioning)
        self.model_counter = 0
        
        self.global_points = np.empty((0, 3))
        self.global_colors = np.empty((0, 3))
        
        # Λίστα για όλα τα camera poses (position & rotation)
        self.camera_history = [] 
        
        # Δημιουργία φακέλων αν δεν υπάρχουν
        if not os.path.exists(self.scan_folder):
            os.makedirs(self.scan_folder)
            
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            print(f"Created output folder: {self.output_folder}")
        
      
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            print(f"Created output folder: {self.output_folder}")
        else:
    # Διαγραφή παλιών .npy αρχείων
            old_files = glob.glob(os.path.join(self.output_folder, "*.npy"))
            for f in old_files:
                os.remove(f)
                print(f"Cleared {len(old_files)} old .npy files from {self.output_folder}")

        self.plotter = pv.Plotter(title="Real-Time Mapping (Floor Filtered)")
        self.plotter.set_background("#FFFFFF")
        self.plotter.camera_position = [(1.5, 1.5, 1.5), (0, 0, 0.25), (0, 0, 1)]
        
        self.plotter.add_timer_event(max_steps=1000000, duration=500, callback=self.check_for_new_scans)
        
        print(f"Monitoring folder: {self.scan_folder}...")
        self.check_for_new_scans()
        self.plotter.show()

    def filter_floor(self, points, colors):
        if len(points) < 50: return points, colors
        try:
            xy = points[:, :2]
            z = points[:, 2]
            ransac = RANSACRegressor(residual_threshold=0.01)
            ransac.fit(xy, z)
            z_pred = ransac.predict(xy)
            mask = z > (z_pred + 0.02)
            return points[mask], colors[mask]
        except: return points, colors

    def apply_voxel_filter(self, points, colors):
        if len(points) == 0: return points, colors
        coords = (points / self.voxel_size).astype(int)
        _, indices = np.unique(coords, axis=0, return_index=True)
        return points[indices], colors[indices]

    def save_global_model(self):
        # Αυξάνουμε τον μετρητή και ορίζουμε το νέο όνομα αρχείου
        self.model_counter += 1
        file_name = f"global_model_{self.model_counter}.npy"
        file_path = os.path.join(self.output_folder, file_name)
        
        last_pose = self.camera_history[-1] if self.camera_history else None
        
        combined_data = {
            'points': self.global_points,
            'colors': self.global_colors,
            'metadata': {
                'voxel_size': self.voxel_size, 
                'total_points': len(self.global_points),
                'camera_history': self.camera_history,  # Όλα τα βέλη
                'last_camera_pose': last_pose           # Το τελευταίο βέλος ξεχωριστά
            }
        }
        np.save(file_path, combined_data)
        print(f"Global model saved to: {file_path}")

    def process_file(self, filename):
        file_path = os.path.join(self.scan_folder, filename)
        try:
            data = np.load(file_path, allow_pickle=True).item()
            rgb = data['rgb_map'] / 255.0
            depth = data['depth_map']
            cam_pos = data['cam_pos']
            cam_rot_mat = data['cam_rot_mat']
            
            # Αποθήκευση του τρέχοντος pose στη λίστα ιστορικού
            current_pose = {
                'position': cam_pos,
                'rotation': cam_rot_mat
            }
            self.camera_history.append(current_pose)
            
            h, w = depth.shape
            f = h / (2 * np.tan(np.radians(60) / 2))
            u, v = np.meshgrid(np.arange(w), np.arange(h))
            
            mask = (depth.flatten() < 1.5) & (depth.flatten() > 0.1)
            z_local = depth.flatten()[mask]
            x_local = (u.flatten()[mask] - w/2) * z_local / f
            y_local = (v.flatten()[mask] - h/2) * z_local / f
            
            points_local = np.vstack((x_local, -y_local, -z_local)).T
            colors_new = rgb.reshape(-1, 3)[mask]
            points_world = points_local @ cam_rot_mat.T + cam_pos

            points_clean, colors_clean = self.filter_floor(points_world, colors_new)

            self.global_points = np.vstack((self.global_points, points_clean))
            self.global_colors = np.vstack((self.global_colors, colors_clean))
            
            self.global_points, self.global_colors = self.apply_voxel_filter(
                self.global_points, self.global_colors
            )
            
            # Αποθήκευση του μοντέλου σε νέο αρχείο στον φάκελο global_models
            self.save_global_model()

            # Οπτικοποίηση
            pc = pv.PolyData(points_clean)
            pc["RGB"] = colors_clean
            cube = pv.Cube(x_length=self.voxel_size, y_length=self.voxel_size, z_length=self.voxel_size)
            vox_mesh = pc.glyph(geom=cube, scale=False, orient=False)
            self.plotter.add_mesh(vox_mesh, rgb=True, scalars="RGB", preference="point")
            
            # Προβολή βέλους
            direction = cam_rot_mat @ np.array([0, 0, -1])
            cam_arrow = pv.Arrow(start=cam_pos, direction=direction, scale=0.05)
            self.plotter.add_mesh(cam_arrow, color="cyan")
            
            print(f"File: {filename} | History size: {len(self.camera_history)}")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    def check_for_new_scans(self, *args):
        try:
            current_files = sorted([f for f in os.listdir(self.scan_folder) if f.startswith("scan_") and f.endswith(".npy")])
            for f in current_files:
                if f not in self.loaded_files:
                    self.process_file(f)
                    self.loaded_files.add(f)
        except Exception:
            pass

if __name__ == "__main__":
    # Μπορείς να αλλάξεις το output_folder αν θες άλλο όνομα φακέλου
    monitor = ScanMonitor(voxel_size=0.005, output_folder="global_models")