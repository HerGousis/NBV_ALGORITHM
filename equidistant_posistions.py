import numpy as np
import pyvista as pv
import os
import time
import psutil
import datetime

# ============================================================
#  ΠΑΡΑΜΕΤΡΟΙ  –  ΑΛΛΑΞΕ ΜΟΝΟ ΑΥΤΑ
# ============================================================
NUM_POSITIONS   = 65       # πλήθος ισαπεχουσών θέσεων
SPHERE_RADIUS   = 0.65     # απόσταση κάμερας από obj_center  (m)
Z_FLOOR         = 0.15     # ελάχιστο επιτρεπτό Z  (floor constraint)
FOLDER          = "global_models"
# ============================================================


# -------------------- 1. ΓΕΩΜΕΤΡΙΚΕΣ ΣΥΝΑΡΤΗΣΕΙΣ --------------------

def fibonacci_sphere(n: int, radius: float,
                     center: np.ndarray,
                     z_floor: float = 0.0) -> np.ndarray:
    """
    Παράγει n ισαπέχοντα σημεία πάνω σε σφαίρα ακτίνας `radius`
    γύρω από `center` χρησιμοποιώντας το Fibonacci / golden-angle spiral.

    Σημεία κάτω από z_floor προβάλλονται στο δαπεδικό δακτύλιο
    (σταθερό Z = z_floor, ίδια ακτίνα από Z-άξονα).
    """
    golden = (1 + np.sqrt(5)) / 2
    indices = np.arange(n)

    # polar angle  θ ∈ [0, π]
    theta = np.arccos(1 - 2 * (indices + 0.5) / n)
    # azimuthal   φ ∈ [0, 2π]
    phi   = 2 * np.pi * indices / golden

    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    points = np.column_stack([x, y, z]) + center

    # floor constraint
    for i, p in enumerate(points):
        if p[2] < z_floor:
            # προβολή στο δαπεδικό δακτύλιο
            dz   = z_floor - center[2]
            r2   = radius**2 - dz**2
            if r2 > 0:
                r_circle = np.sqrt(r2)
                xy       = p[:2] - center[:2]
                xy_norm  = np.linalg.norm(xy)
                if xy_norm > 1e-6:
                    xy_unit   = xy / xy_norm
                    points[i] = np.array([
                        center[0] + xy_unit[0] * r_circle,
                        center[1] + xy_unit[1] * r_circle,
                        z_floor
                    ])
                else:
                    points[i, 2] = z_floor

    return points


def viewpoints_from_positions(positions: np.ndarray,
                               obj_center: np.ndarray):
    """
    Για κάθε θέση επιστρέφει τη view_direction (κανονικοποιημένο
    διάνυσμα από κάμερα → obj_center).
    """
    dirs = obj_center - positions
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs / (norms + 1e-6)


# -------------------- 2. MONITORING & VISUALIZATION --------------------

class UniformSpherePlanner:
    """
    Παρακολουθεί τον φάκελο `folder` για νέα global_model_*.npy,
    υπολογίζει NUM_POSITIONS ισαπέχουσες θέσεις γύρω από το αντικείμενο
    και προτείνει τη **επόμενη αδείαστη** θέση διαδοχικά.
    """

    def __init__(self, folder: str = FOLDER,
                 num_positions: int = NUM_POSITIONS,
                 radius: float = SPHERE_RADIUS,
                 z_floor: float = Z_FLOOR):
        self.folder         = folder
        self.num_positions  = num_positions
        self.radius         = radius
        self.z_floor        = z_floor
        self.last_file_id   = -1

        # Κατάσταση διαδρομής
        self.sphere_points  = None   # (N, 3)  –  θέσεις σφαίρας
        self.view_dirs      = None   # (N, 3)
        self.current_idx    = 0      # επόμενη θέση προς επίσκεψη
        self.obj_center     = None

        # ── Στατιστικά εκτέλεσης ──────────────────────────────────────
        self.start_time     = time.time()   # ώρα εκκίνησης αλγορίθμου
        self.ram_samples    = []            # δειγματοληψία RAM (MB) ανά update
        self._process       = psutil.Process(os.getpid())

        os.makedirs(self.folder, exist_ok=True)

        self.plotter = pv.Plotter(title="Uniform Sphere Planner")
        self.plotter.set_background("#111111")
        self.plotter.add_timer_event(max_steps=1_000_000, duration=500,
                                     callback=self.update)
        self.update()

    # ── file helpers ──────────────────────────────────────────────────
    def get_latest_file(self):
        files = [f for f in os.listdir(self.folder)
                 if f.startswith("global_model_") and f.endswith(".npy")]
        if not files:
            return None, -1
        latest = max(files,
                     key=lambda x: int(x.split('_')[-1].split('.')[0]))
        fid = int(latest.split('_')[-1].split('.')[0])
        return os.path.join(self.folder, latest), fid

    # ── main loop ─────────────────────────────────────────────────────
    def update(self, *args):
        # Δειγματοληψία RAM σε κάθε tick (MB)
        ram_mb = self._process.memory_info().rss / 1024 / 1024
        self.ram_samples.append(ram_mb)

        file_path, file_id = self.get_latest_file()
        if file_id > self.last_file_id:
            self.last_file_id = file_id
            self.run_planning(file_path)

    # ── αποθήκευση στατιστικών ────────────────────────────────────────
    def save_stats(self, num_visited: int):
        elapsed   = time.time() - self.start_time
        ram_avg   = float(np.mean(self.ram_samples))   if self.ram_samples else 0.0
        ram_peak  = float(np.max(self.ram_samples))    if self.ram_samples else 0.0
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        hours, rem   = divmod(int(elapsed), 3600)
        minutes, sec = divmod(rem, 60)
        elapsed_str  = f"{hours:02d}h {minutes:02d}m {sec:02d}s"

        lines = [
            "=" * 52,
            "  UNIFORM SPHERE PLANNER – ΣΤΑΤΙΣΤΙΚΑ ΕΚΤΕΛΕΣΗΣ",
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

        out_path = "metrics/dimitra/metrics_equidistant.txt"
        with open(out_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        print(f"📊 Στατιστικά αποθηκεύτηκαν → {out_path}")
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

            # ── (Επαν)υπολόγισε σφαιρικές θέσεις αν αλλάξαμε αντικείμενο ──
            if (self.sphere_points is None
                    or self.obj_center is None
                    or not np.allclose(self.obj_center, obj_center, atol=0.05)):
                self.obj_center    = obj_center
                self.sphere_points = fibonacci_sphere(
                    self.num_positions, self.radius,
                    obj_center, self.z_floor)
                self.view_dirs     = viewpoints_from_positions(
                    self.sphere_points, obj_center)
                self.current_idx   = 0
                print(f"🔄 Νέο αντικείμενο – {self.num_positions} θέσεις υπολογίστηκαν.")

            # ── Ποιές θέσεις έχουν ήδη επισκεφτεί; ──────────────────────
            visited = set()
            if camera_history:
                hist_pos = np.array([p['position'] for p in camera_history])
                self._last_hist_pos = hist_pos          # ← αποθήκευση για on-close
                for idx, sp in enumerate(self.sphere_points):
                    dists = np.linalg.norm(hist_pos - sp, axis=1)
                    if dists.min() < 0.02:
                        visited.add(idx)
            else:
                self._last_hist_pos = np.empty((0, 3))

            # ── Βρες την επόμενη αδείαστη θέση ──────────────────────────
            next_idx = None
            for offset in range(self.num_positions):
                candidate = (self.current_idx + offset) % self.num_positions
                if candidate not in visited:
                    next_idx = candidate
                    break

            # ── Αποθήκευση next_position.npy ─────────────────────────────
            if next_idx is not None:
                self.current_idx = (next_idx + 1) % self.num_positions
                next_cam = self.sphere_points[next_idx]
                next_dir = self.view_dirs[next_idx]

                np.save("next_position.npy", {
                    "position":       next_cam,
                    "view_direction": -next_dir
                })
                print(f"🎯 Θέση {next_idx + 1}/{self.num_positions}  "
                      f"XYZ=({next_cam[0]:.3f}, {next_cam[1]:.3f}, {next_cam[2]:.3f})  "
                      f"(επισκεφτηκαν: {len(visited)})")
            else:
                print("✅ Όλες οι θέσεις έχουν επισκεφτεί!")
                self.save_stats(num_visited=len(visited))
                next_idx = None

            # ── VISUALIZATION ─────────────────────────────────────────────
            self.plotter.clear()
            self.plotter.add_axes()

            # Point cloud
            pc = pv.PolyData(points)
            voxel_size = metadata.get('voxel_size', 0.005)
            self.plotter.add_mesh(
                pc.glyph(geom=pv.Cube(x_length=voxel_size,
                                       y_length=voxel_size,
                                       z_length=voxel_size),
                          scale=False),
                color="#44aaff", opacity=0.35)

            # Wireframe σφαίρα
            self.plotter.add_mesh(
                pv.Sphere(radius=self.radius, center=obj_center,
                          theta_resolution=24, phi_resolution=24),
                color="white", opacity=0.04, style="wireframe")

            # Floor ring
            dz = self.z_floor - obj_center[2]
            r2 = self.radius**2 - dz**2
            if r2 > 0:
                r_circle = np.sqrt(r2)
                theta_arr = np.linspace(0, 2 * np.pi, 80)
                floor_ring = np.column_stack([
                    obj_center[0] + r_circle * np.cos(theta_arr),
                    obj_center[1] + r_circle * np.sin(theta_arr),
                    np.full(80, self.z_floor)
                ])
                self.plotter.add_mesh(pv.PolyData(floor_ring),
                                      color="yellow", point_size=3,
                                      render_points_as_spheres=True)

            # Ζωγράφισε όλες τις θέσεις σφαίρας
            for idx, (sp, vd) in enumerate(
                    zip(self.sphere_points, self.view_dirs)):
                is_next    = (idx == next_idx)
                is_visited = (idx in visited)

                if is_next:
                    color, size = "lime",    0.025
                elif is_visited:
                    color, size = "gray",    0.010
                else:
                    color, size = "orange",  0.015

                self.plotter.add_mesh(
                    pv.Sphere(radius=size, center=sp), color=color)

                # Γραμμή κάμερα → obj_center
                self.plotter.add_mesh(
                    pv.Line(sp, obj_center),
                    color=color,
                    line_width=2 if is_next else 1)

                # Βέλος view direction
                self.plotter.add_mesh(
                    pv.Arrow(start=sp, direction=vd, scale=0.05),
                    color=color)

                # Label αριθμού
                self.plotter.add_point_labels(
                    [sp], [str(idx + 1)],
                    font_size=10,
                    text_color="white" if not is_visited else "gray",
                    always_visible=True,
                    shape_opacity=0.0)

            # Τρέχουσα θέση κάμερας
            if last_pose:
                curr_pos = np.array(last_pose['position'])
                self.plotter.add_mesh(
                    pv.Sphere(radius=0.020, center=curr_pos), color="cyan")

            # Ιστορικό διαδρομής
            if camera_history and len(camera_history) > 1:
                hist_pos = np.array([p['position'] for p in camera_history])
                for i in range(len(hist_pos) - 1):
                    self.plotter.add_mesh(
                        pv.Line(hist_pos[i], hist_pos[i + 1]),
                        color="deepskyblue", line_width=2)

            # Τίτλος με πρόοδο
            done  = len(visited)
            total = self.num_positions
            self.plotter.add_text(
                f"Θέσεις: {done}/{total} επισκεφτηκαν",
                position="upper_left", font_size=12, color="white")

        except Exception as e:
            print(f"⚠️ Error: {e}")
            import traceback; traceback.print_exc()

    def show(self):
        self.plotter.show()
        # Αποθήκευση στατιστικών και κατά το κλείσιμο του παραθύρου
        visited_count = (
            sum(1 for idx, sp in enumerate(self.sphere_points)
                if hasattr(self, '_last_hist_pos')
                   and len(self._last_hist_pos) > 0
                   and np.linalg.norm(self._last_hist_pos - sp, axis=1).min() < 0.12)
            if self.sphere_points is not None else 0
        )
        


# -------------------- 3. ENTRY POINT --------------------
# Εγκατάσταση εξαρτήσεων αν χρειαστεί:  pip install psutil

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Uniform Sphere Viewpoint Planner")
    parser.add_argument("--n",      type=int,   default=NUM_POSITIONS,
                        help="Πλήθος ισαπεχουσών θέσεων (default: %(default)s)")
    parser.add_argument("--radius", type=float, default=SPHERE_RADIUS,
                        help="Ακτίνα σφαίρας σε μέτρα (default: %(default)s)")
    parser.add_argument("--zfloor", type=float, default=Z_FLOOR,
                        help="Ελάχιστο Z (floor) σε μέτρα (default: %(default)s)")
    parser.add_argument("--folder", type=str,   default=FOLDER,
                        help="Φάκελος global_model_*.npy (default: %(default)s)")
    args = parser.parse_args()

    monitor = UniformSpherePlanner(
        folder        = args.folder,
        num_positions = args.n,
        radius        = args.radius,
        z_floor       = args.zfloor
    )
    monitor.show()