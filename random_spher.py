import numpy as np
import pyvista as pv
import os
import time
import psutil
import datetime


NUM_POSITIONS   = 65       
SPHERE_RADIUS   = 0.65     
Z_FLOOR         = 0.15     
FOLDER          = "global_models"




def random_sphere_point(radius: float,
                         center: np.ndarray,
                         z_floor: float = 0.0,
                         rng: np.random.Generator = None) -> np.ndarray:
   
    if rng is None:
        rng = np.random.default_rng()

    
    vec = rng.standard_normal(3)
    vec /= np.linalg.norm(vec) + 1e-12
    point = center + radius * vec

    
    if point[2] < z_floor:
        dz = z_floor - center[2]
        r2 = radius**2 - dz**2
        if r2 > 0:
            r_circle = np.sqrt(r2)
            xy      = point[:2] - center[:2]
            xy_norm = np.linalg.norm(xy)
            if xy_norm > 1e-6:
                xy_unit = xy / xy_norm
                point = np.array([
                    center[0] + xy_unit[0] * r_circle,
                    center[1] + xy_unit[1] * r_circle,
                    z_floor
                ])
            else:
                
                angle = rng.uniform(0, 2 * np.pi)
                point = np.array([
                    center[0] + r_circle * np.cos(angle),
                    center[1] + r_circle * np.sin(angle),
                    z_floor
                ])

    return point


def view_direction(cam_pos: np.ndarray, obj_center: np.ndarray) -> np.ndarray:
    """Κανονικοποιημένο διάνυσμα κάμερα → obj_center."""
    d = obj_center - cam_pos
    return d / (np.linalg.norm(d) + 1e-6)




class RandomSpherePlanner:
    

    def __init__(self, folder: str = FOLDER,
                 num_positions: int = NUM_POSITIONS,
                 radius: float = SPHERE_RADIUS,
                 z_floor: float = Z_FLOOR):
        self.folder         = folder
        self.num_positions  = num_positions
        self.radius         = radius
        self.z_floor        = z_floor
        self.last_file_id   = -1

        
        self.obj_center     = None
        self.generated_pts  = []   
        self.rng            = np.random.default_rng()   

       
        self.start_time     = time.time()
        self.ram_samples    = []
        self._process       = psutil.Process(os.getpid())
        self._last_hist_pos = np.empty((0, 3))

        os.makedirs(self.folder, exist_ok=True)

        self.plotter = pv.Plotter(title="Random Sphere Planner")
        self.plotter.set_background("#111111")
        self.plotter.add_timer_event(max_steps=1_000_000, duration=500,
                                     callback=self.update)
        self.update()

    
    def get_latest_file(self):
        files = [f for f in os.listdir(self.folder)
                 if f.startswith("global_model_") and f.endswith(".npy")]
        if not files:
            return None, -1
        latest = max(files,
                     key=lambda x: int(x.split('_')[-1].split('.')[0]))
        fid = int(latest.split('_')[-1].split('.')[0])
        return os.path.join(self.folder, latest), fid

    
    def update(self, *args):
        ram_mb = self._process.memory_info().rss / 1024 / 1024
        self.ram_samples.append(ram_mb)

        file_path, file_id = self.get_latest_file()
        if file_id > self.last_file_id:
            self.last_file_id = file_id
            self.run_planning(file_path)

    
    def save_stats(self, num_visited: int):
        elapsed   = time.time() - self.start_time
        ram_avg   = float(np.mean(self.ram_samples))  if self.ram_samples else 0.0
        ram_peak  = float(np.max(self.ram_samples))   if self.ram_samples else 0.0
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        hours, rem   = divmod(int(elapsed), 3600)
        minutes, sec = divmod(rem, 60)
        elapsed_str  = f"{hours:02d}h {minutes:02d}m {sec:02d}s"

        lines = [
            "=" * 52,
            "  RANDOM SPHERE PLANNER – ΣΤΑΤΙΣΤΙΚΑ ΕΚΤΕΛΕΣΗΣ",
            "=" * 52,
            f"  Ημερομηνία / Ώρα    : {timestamp}",
            f"  Αριθμός θέσεων      : {self.num_positions}",
            f"  Επισκεφτηκαν        : {num_visited}/{self.num_positions}",
            f"  Ακτίνα σφαίρας      : {self.radius:.3f} m",
            f"  Z-floor             : {self.z_floor:.3f} m",
            "-" * 52,
            f"  Συνολικός χρόνος    : {elapsed_str}  ({elapsed:.1f} s)",
            f"  RAM μέση χρήση      : {ram_avg:.1f} MB",
            f"  RAM μέγιστη χρήση   : {ram_peak:.1f} MB",
            "=" * 52,
            "",
        ]

        os.makedirs("metrics/dimitra", exist_ok=True)
        out_path = "metrics/dimitra/metrics_random_sphere.txt"
        with open(out_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Στατιστικά αποθηκεύτηκαν → {out_path}")
        for l in lines:
            print(l)

    
    def run_planning(self, file_path):
        try:
            data           = np.load(file_path, allow_pickle=True).item()
            points         = data['points']
            metadata       = data.get('metadata', {})
            camera_history = metadata.get('camera_history', [])
            last_pose      = metadata.get('last_camera_pose', None)

            obj_center = np.mean(points, axis=0)

            
            if (self.obj_center is None
                    or not np.allclose(self.obj_center, obj_center, atol=0.05)):
                self.obj_center   = obj_center
                self.generated_pts = []
                print(f"Νέο αντικείμενο – τυχαία δειγματοληψία σφαίρας.")

            
            num_visited = len(camera_history)

            if camera_history:
                self._last_hist_pos = np.array(
                    [p['position'] for p in camera_history])
            else:
                self._last_hist_pos = np.empty((0, 3))

            
            if num_visited >= self.num_positions:
                print(" Όλες οι θέσεις έχουν επισκεφτεί!")
                self.save_stats(num_visited=num_visited)
                self._draw_scene(points, metadata, camera_history,
                                 last_pose, next_cam=None,
                                 num_visited=num_visited)
                return

           
            next_cam = random_sphere_point(
                self.radius, obj_center, self.z_floor, self.rng)
            next_dir = view_direction(next_cam, obj_center)

            
            self.generated_pts.append(next_cam.copy())

            np.save("next_position.npy", {
                "position":       next_cam,
                "view_direction": -next_dir   
            })

            print(f"Τυχαία θέση {num_visited + 1}/{self.num_positions}  "
                  f"XYZ=({next_cam[0]:.3f}, {next_cam[1]:.3f}, {next_cam[2]:.3f})")

            self._draw_scene(points, metadata, camera_history,
                             last_pose, next_cam, num_visited)

        except Exception as e:
            print(f"Error: {e}")
            import traceback; traceback.print_exc()

    
    def _draw_scene(self, points, metadata, camera_history,
                    last_pose, next_cam, num_visited):
        obj_center = self.obj_center
        self.plotter.clear()
        self.plotter.add_axes()

       
        pc         = pv.PolyData(points)
        voxel_size = metadata.get('voxel_size', 0.005)
        self.plotter.add_mesh(
            pc.glyph(geom=pv.Cube(x_length=voxel_size,
                                   y_length=voxel_size,
                                   z_length=voxel_size),
                      scale=False),
            color="#44aaff", opacity=0.35)

        
        self.plotter.add_mesh(
            pv.Sphere(radius=self.radius, center=obj_center,
                      theta_resolution=24, phi_resolution=24),
            color="white", opacity=0.04, style="wireframe")

       
        dz = self.z_floor - obj_center[2]
        r2 = self.radius**2 - dz**2
        if r2 > 0:
            r_circle  = np.sqrt(r2)
            theta_arr = np.linspace(0, 2 * np.pi, 80)
            floor_ring = np.column_stack([
                obj_center[0] + r_circle * np.cos(theta_arr),
                obj_center[1] + r_circle * np.sin(theta_arr),
                np.full(80, self.z_floor)
            ])
            self.plotter.add_mesh(pv.PolyData(floor_ring),
                                  color="yellow", point_size=3,
                                  render_points_as_spheres=True)

        
        for sp in self.generated_pts[:-1] if next_cam is not None else self.generated_pts:
            vd = view_direction(sp, obj_center)
            self.plotter.add_mesh(
                pv.Sphere(radius=0.010, center=sp), color="gray")
            self.plotter.add_mesh(
                pv.Line(sp, obj_center), color="gray", line_width=1)
            self.plotter.add_mesh(
                pv.Arrow(start=sp, direction=vd, scale=0.05), color="gray")

        
        if next_cam is not None:
            next_dir = view_direction(next_cam, obj_center)
            self.plotter.add_mesh(
                pv.Sphere(radius=0.025, center=next_cam), color="lime")
            self.plotter.add_mesh(
                pv.Line(next_cam, obj_center), color="lime", line_width=2)
            self.plotter.add_mesh(
                pv.Arrow(start=next_cam, direction=next_dir, scale=0.05),
                color="lime")
            self.plotter.add_point_labels(
                [next_cam], [f"NEXT ({num_visited + 1})"],
                font_size=11, text_color="lime",
                always_visible=True, shape_opacity=0.0)

        
        if last_pose:
            curr_pos = np.array(last_pose['position'])
            self.plotter.add_mesh(
                pv.Sphere(radius=0.020, center=curr_pos), color="cyan")

        
        if camera_history and len(camera_history) > 1:
            hist_pos = np.array([p['position'] for p in camera_history])
            for i in range(len(hist_pos) - 1):
                self.plotter.add_mesh(
                    pv.Line(hist_pos[i], hist_pos[i + 1]),
                    color="deepskyblue", line_width=2)

        
        self.plotter.add_text(
            f"Θέσεις: {num_visited}/{self.num_positions} επισκεφτηκαν  |  Random Sphere",
            position="upper_left", font_size=12, color="white")

    def show(self):
        self.plotter.show()
        visited_count = len(self._last_hist_pos) if hasattr(self, '_last_hist_pos') else 0




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Random Sphere Viewpoint Planner")
    parser.add_argument("--n",      type=int,   default=NUM_POSITIONS,
                        help="Συνολικές τυχαίες θέσεις (default: %(default)s)")
    parser.add_argument("--radius", type=float, default=SPHERE_RADIUS,
                        help="Ακτίνα σφαίρας σε μέτρα (default: %(default)s)")
    parser.add_argument("--zfloor", type=float, default=Z_FLOOR,
                        help="Ελάχιστο Z (floor) σε μέτρα (default: %(default)s)")
    parser.add_argument("--folder", type=str,   default=FOLDER,
                        help="Φάκελος global_model_*.npy (default: %(default)s)")
    args = parser.parse_args()

    monitor = RandomSpherePlanner(
        folder        = args.folder,
        num_positions = args.n,
        radius        = args.radius,
        z_floor       = args.zfloor
    )
    monitor.show()