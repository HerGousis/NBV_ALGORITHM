import numpy as np
import pyvista as pv
import os

class ScanMonitor:
    def __init__(self, scan_folder="scans", voxel_size=0.005, output_file="global_model.npy"):
        self.scan_folder = scan_folder
        self.voxel_size = voxel_size
        self.output_file = output_file
        self.loaded_files = set() 
        
        # Αποθήκες για το συνολικό μοντέλο
        self.global_points = np.empty((0, 3))
        self.global_colors = np.empty((0, 3))
        
        if not os.path.exists(self.scan_folder):
            os.makedirs(self.scan_folder)

        self.plotter = pv.Plotter(title="Real-Time Autonomous Mapping (with Downsampling)")
        self.plotter.set_background("#FFFFFF")
        self.plotter.camera_position = [(1.5, 1.5, 1.5), (0, 0, 0.25), (0, 0, 1)]
        
        # Timer για έλεγχο νέων αρχείων κάθε 500ms
        self.plotter.add_timer_event(max_steps=1000000, duration=500, callback=self.check_for_new_scans)
        
        print(f"Monitoring folder: {self.scan_folder}...")
        print(f"Downsampling resolution: {self.voxel_size}m")
        print(f"Global model saved as: {self.output_file}")
        
        self.check_for_new_scans()
        self.plotter.show()

    def apply_voxel_filter(self, points, colors):
        """
        Κρατάει μόνο ένα σημείο ανά voxel για να διατηρεί το μοντέλο ελαφρύ.
        """
        if len(points) == 0:
            return points, colors

        # 1. Μετατροπή συντεταγμένων σε ακέραια "κουτιά" (voxels)
        # Χρησιμοποιούμε το voxel_size ως οδηγό
        coords = (points / self.voxel_size).astype(int)
        
        # 2. Εύρεση μοναδικών κουτιών
        # Η return_index μας δίνει το πρώτο σημείο που "έπεσε" σε κάθε κουτί
        _, indices = np.unique(coords, axis=0, return_index=True)
        
        return points[indices], colors[indices]

    def save_global_model(self):
        """Αποθηκεύει το ενοποιημένο και καθαρισμένο μοντέλο."""
        combined_data = {
            'points': self.global_points,
            'colors': self.global_colors,
            'metadata': {'voxel_size': self.voxel_size, 'total_points': len(self.global_points)}
        }
        np.save(self.output_file, combined_data)

    def process_file(self, filename):
        file_path = os.path.join(self.scan_folder, filename)
        try:
            data = np.load(file_path, allow_pickle=True).item()
            
            rgb = data['rgb_map'] / 255.0
            depth = data['depth_map']
            cam_pos = data['cam_pos']
            cam_rot_mat = data['cam_rot_mat']
            
            # Camera Intrinsics (fov_deg=60)
            h, w = depth.shape
            f = h / (2 * np.tan(np.radians(60) / 2))
            u, v = np.meshgrid(np.arange(w), np.arange(h))
            
            # Masking (Κρατάμε μόνο χρήσιμα βάθη)
            mask = (depth.flatten() < 1.5) & (depth.flatten() > 0.1)
            z_local = depth.flatten()[mask]
            x_local = (u.flatten()[mask] - w/2) * z_local / f
            y_local = (v.flatten()[mask] - h/2) * z_local / f
            
            points_local = np.vstack((x_local, -y_local, -z_local)).T
            colors_new = rgb.reshape(-1, 3)[mask]
            
            # Μετατροπή σε World Coordinates
            points_world = points_local @ cam_rot_mat.T + cam_pos

            # --- ΕΝΣΩΜΑΤΩΣΗ & DOWNSAMPLING ---
            # Προσθέτουμε τα νέα σημεία στα παλιά
            self.global_points = np.vstack((self.global_points, points_world))
            self.global_colors = np.vstack((self.global_colors, colors_new))
            
            # Φιλτράρουμε το συνολικό μοντέλο για να σβήσουμε τα διπλότυπα
            self.global_points, self.global_colors = self.apply_voxel_filter(
                self.global_points, self.global_colors
            )
            
            self.save_global_model()

            # Οπτικοποίηση στον Plotter (μόνο του νέου scan για ταχύτητα)
            # Ή αν θες να βλέπεις το καθαρισμένο μοντέλο, βάλε self.global_points
            pc = pv.PolyData(points_world)
            pc["RGB"] = colors_new
            cube = pv.Cube(x_length=self.voxel_size, y_length=self.voxel_size, z_length=self.voxel_size)
            vox_mesh = pc.glyph(geom=cube, scale=False, orient=False)
            
            self.plotter.add_mesh(vox_mesh, rgb=True, scalars="RGB", preference="point")
            
            # Σχεδίαση βέλους κάμερας
            direction = cam_rot_mat @ np.array([0, 0, -1])
            cam_arrow = pv.Arrow(start=cam_pos, direction=direction, scale=0.05)
            self.plotter.add_mesh(cam_arrow, color="cyan")
            
            print(f"File: {filename} | Total Unique Points: {len(self.global_points)}")
            
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
    # Μπορείς να ορίσεις voxel_size=0.002 για μεγαλύτερη λεπτομέρεια
    monitor = ScanMonitor(voxel_size=0.005) 