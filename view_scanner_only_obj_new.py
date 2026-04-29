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
        self.model_counter = 0

        self.global_points = np.empty((0, 3), dtype=np.float32)
        self.camera_history = []
        self._viz_mesh_actor = None

        # 🔴 Error flag path
        self.error_flag_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "error_flag.npy")

        os.makedirs(self.scan_folder, exist_ok=True)

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        else:
            for f in glob.glob(os.path.join(self.output_folder, "*.npy")):
                os.remove(f)

        self.plotter = pv.Plotter(title="Real-Time Mapping (Floor Filtered)")
        self.plotter.set_background("#FFFFFF")
        self.plotter.camera_position = [(1.5, 1.5, 1.5), (0, 0, 0.25), (0, 0, 1)]
        self.plotter.add_timer_event(max_steps=1000000, duration=500, callback=self.check_for_new_scans)

        print(f"Monitoring folder: {self.scan_folder}...")
        self.check_for_new_scans()
        self.plotter.show()

    # 🔴 Create error flag
    def create_error_flag(self, message="Unknown error"):
        np.save(self.error_flag_path, {
            "error": True,
            "message": message,
            "timestamp": str(np.datetime64('now'))
        })
        print("⚠️ Error flag created")

    # 🔴 Clear error flag
    def clear_error_flag(self):
        if os.path.exists(self.error_flag_path):
            os.remove(self.error_flag_path)
            print("✅ Error flag cleared")

    def filter_floor(self, points):
        if len(points) < 50:
            return points
        try:
            ransac = RANSACRegressor(residual_threshold=0.01)
            ransac.fit(points[:, :2], points[:, 2])
            z_pred = ransac.predict(points[:, :2])
            return points[points[:, 2] > (z_pred + 0.02)]
        except:
            return points

    def apply_voxel_filter(self, points):
        if len(points) == 0:
            return points
        coords = (points / self.voxel_size).astype(np.int32)
        dtype = np.dtype([('x', np.int32), ('y', np.int32), ('z', np.int32)])
        structured = np.ascontiguousarray(coords).view(dtype).reshape(-1)
        _, indices = np.unique(structured, return_index=True)
        return points[indices]

    def save_global_model(self):
        self.model_counter += 1
        file_path = os.path.join(self.output_folder, f"global_model_{self.model_counter}.npy")
        np.save(file_path, {
            'points': self.global_points,
            'metadata': {
                'voxel_size': self.voxel_size,
                'total_points': len(self.global_points),
                'camera_history': self.camera_history,
                'last_camera_pose': self.camera_history[-1] if self.camera_history else None
            }
        })
        print(f"Saved: global_model_{self.model_counter}.npy ({len(self.global_points):,} points)")

    def process_file(self, filename):
        file_path = os.path.join(self.scan_folder, filename)
        try:
            data = np.load(file_path, allow_pickle=True).item()

            depth = data['depth_map'].astype(np.float32)
            cam_pos = np.array(data['cam_pos'], dtype=np.float32)
            cam_rot_mat = np.array(data['cam_rot_mat'], dtype=np.float32)

            self.camera_history.append({'position': cam_pos, 'rotation': cam_rot_mat})

            h, w = depth.shape
            f = np.float32(h / (2 * np.tan(np.radians(60) / 2)))

            depth_flat = depth.flatten()
            indices = np.where((depth_flat > 0.1) & (depth_flat < 1.5))[0]
            z_local = depth_flat[indices]
            x_local = (indices % w - w / 2).astype(np.float32) * z_local / f
            y_local = -(indices // w - h / 2).astype(np.float32) * z_local / f

            points_local = np.column_stack((x_local, y_local, -z_local))
            points_world = (points_local @ cam_rot_mat.T + cam_pos).astype(np.float32)

            points_clean = self.filter_floor(points_world)

            self.global_points = np.vstack((self.global_points, points_clean))
            self.global_points = self.apply_voxel_filter(self.global_points)

            # Αποθήκευση σε κάθε scan
            self.save_global_model()

            # 🔴 Αν όλα πήγαν καλά, σβήσε το error flag
            self.clear_error_flag()

            # Visualization
            if self._viz_mesh_actor is not None:
                self.plotter.remove_actor(self._viz_mesh_actor)

            pc = pv.PolyData(self.global_points)
            cube = pv.Cube(x_length=self.voxel_size, y_length=self.voxel_size, z_length=self.voxel_size)
            vox_mesh = pc.glyph(geom=cube, scale=False, orient=False)
            self._viz_mesh_actor = self.plotter.add_mesh(vox_mesh, color="lightblue")

            direction = cam_rot_mat @ np.array([0, 0, -1], dtype=np.float32)
            self.plotter.add_mesh(
                pv.Arrow(start=cam_pos, direction=direction, scale=0.05),
                color="cyan"
            )

            print(f"File: {filename} | Points: {len(self.global_points):,} | Scans: {len(self.camera_history)}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            # 🔴 Δημιουργία error flag
            self.create_error_flag(str(e))

    def check_for_new_scans(self, *args):
        try:
            current_files = sorted([
                f for f in os.listdir(self.scan_folder)
                if f.startswith("scan_") and f.endswith(".npy")
            ])
            for f in current_files:
                if f not in self.loaded_files:
                    self.process_file(f)
                    self.loaded_files.add(f)
        except Exception:
            pass


if __name__ == "__main__":
    monitor = ScanMonitor(voxel_size=0.005, output_folder="global_models")