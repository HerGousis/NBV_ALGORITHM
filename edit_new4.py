import numpy as np
import pyvista as pv
import os
import time
import psutil
import datetime
from sklearn.neighbors import NearestNeighbors, KDTree
from sklearn.cluster import DBSCAN

# -------------------- 1. ΥΠΟΛΟΓΙΣΤΙΚΕΣ ΣΥΝΑΡΤΗΣΕΙΣ (NBS) --------------------

def calculate_interest_score(points, radius=0.03):
    if len(points) == 0:
        return np.array([])
    tree = KDTree(points)
    counts = tree.query_radius(points, r=radius, count_only=True).astype(float)
    counts -= 1
    scores = 1.0 - (counts / (counts.max() + 1e-6))
    return scores


def estimate_cluster_normal(cluster_points):
    centroid = np.mean(cluster_points, axis=0)
    centered = cluster_points - centroid
    cov = centered.T @ centered
    _, eigenvectors = np.linalg.eigh(cov)
    normal = eigenvectors[:, 0]
    return normal / (np.linalg.norm(normal) + 1e-6)


def orient_normal_outward(normal, cluster_center, object_center):
    vec_out = cluster_center - object_center
    vec_out /= np.linalg.norm(vec_out) + 1e-6
    if np.dot(normal, vec_out) < 0:
        return -normal
    return normal


def build_occlusion_tree(points, voxel_size):
    return NearestNeighbors(radius=voxel_size * 0.85,
                            algorithm='ball_tree').fit(points)


def is_occluded(target_xyz, camera_xyz, nbrs_tree, num_samples=60):
    line_points = np.linspace(target_xyz, camera_xyz, num_samples)[3:-3]
    distances, _ = nbrs_tree.radius_neighbors(line_points, return_distance=True)
    return any(len(d) > 0 for d in distances)


def calculate_efficiency_score(alignment, move_dist, cluster_priority):
    benefit     = (alignment * 0.7) + (cluster_priority * 0.5)
    energy_cost = (move_dist * 0.3) + 0.05
    return benefit / (energy_cost + 0.1)


def is_too_close_to_history(candidate_xyz, history_positions, min_dist=0.18):
    if history_positions is None or len(history_positions) == 0:
        return False
    diffs = history_positions - candidate_xyz
    return bool(np.any(np.linalg.norm(diffs, axis=1) < min_dist))


def _project_to_z_floor(camera_xyz, obj_center, d_s, z_floor=0.1):
    dz = z_floor - obj_center[2]
    r2 = d_s**2 - dz**2
    if r2 <= 0:
        return None
    r_circle = np.sqrt(r2)
    xy = camera_xyz[:2] - obj_center[:2]
    xy_norm = np.linalg.norm(xy)
    if xy_norm < 1e-6:
        return None
    xy_unit = xy / xy_norm
    return np.array([
        obj_center[0] + xy_unit[0] * r_circle,
        obj_center[1] + xy_unit[1] * r_circle,
        z_floor
    ])


# ── ΝΕΟ: Ανίχνευση κενών στο ιστορικό ────────────────────────────────────────
def find_history_gap_targets(camera_history, obj_center, points,
                              gap_threshold=0.35, min_gap_angle_deg=40.0):
    if len(camera_history) < 2:
        return []

    positions = np.array([p['position'] for p in camera_history])

    dirs = positions - obj_center
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs /= (norms + 1e-6)

    cos_min = np.cos(np.radians(min_gap_angle_deg))
    gap_targets = []

    for i in range(len(positions) - 1):
        p1, p2   = positions[i], positions[i + 1]
        d1, d2   = dirs[i],      dirs[i + 1]

        lin_dist  = np.linalg.norm(p2 - p1)
        cos_angle = np.clip(np.dot(d1, d2), -1.0, 1.0)

        if lin_dist < gap_threshold or cos_angle > cos_min:
            continue

        angle_deg = np.degrees(np.arccos(cos_angle))

        mid_dir = d1 + d2
        mid_norm = np.linalg.norm(mid_dir)
        if mid_norm < 1e-6:
            continue
        mid_dir /= mid_norm

        rel       = points - obj_center
        proj      = rel @ mid_dir
        perp      = rel - np.outer(proj, mid_dir)
        perp_dist = np.linalg.norm(perp, axis=1)

        valid_mask = proj > 0
        if not np.any(valid_mask):
            continue

        best_pt_idx = np.argmin(
            np.where(valid_mask, perp_dist, np.inf)
        )
        gap_center = points[best_pt_idx]
        gap_normal = gap_center - obj_center
        gap_normal /= np.linalg.norm(gap_normal) + 1e-6

        priority = angle_deg / 180.0

        gap_targets.append({
            'center':   gap_center,
            'normal':   gap_normal,
            'priority': priority,
            'is_gap':   True,
            'gap_deg':  angle_deg
        })

    if gap_targets:
        max_p = max(t['priority'] for t in gap_targets)
        for t in gap_targets:
            t['priority'] /= max_p + 1e-6

    return gap_targets


def find_top_k_candidate_views(target_info, nbrs_tree, d_s, curr_pos,
                                obj_center, r_circle_floor,
                                history_positions, z_floor=0.1,
                                normal_weight=0.7, top_k=3,
                                min_angular_sep_deg=25.0):
    target_xyz = target_info['center']
    normal     = target_info['normal']
    t_priority = target_info['priority']

    geom_dir  = target_xyz - obj_center
    geom_norm = np.linalg.norm(geom_dir)
    if geom_norm < 1e-6:
        return []
    geom_dir /= geom_norm

    cone_axis      = (1.0 - normal_weight) * geom_dir + normal_weight * normal
    cone_axis_norm = np.linalg.norm(cone_axis)
    cone_axis      = (cone_axis / cone_axis_norm
                      if cone_axis_norm > 1e-6 else geom_dir)

    def make_cone_dirs(axis, elevations, azimuths):
        ref_ = [1, 0, 0] if abs(axis[0]) < 0.9 else [0, 1, 0]
        xa   = np.cross(axis, ref_);  xa /= np.linalg.norm(xa)
        ya   = np.cross(axis, xa)
        al   = np.repeat(elevations,  len(azimuths))
        ph   = np.tile(azimuths,      len(elevations))
        dirs = (np.cos(al)[:, None] * axis
                + np.sin(al)[:, None] * (np.cos(ph)[:, None] * xa
                                        + np.sin(ph)[:, None] * ya))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        return dirs

    dirs_narrow = make_cone_dirs(cone_axis,
                                 np.linspace(0.0,  0.25, 4),
                                 np.linspace(0, 2*np.pi, 5, endpoint=False))
    dirs_wide   = make_cone_dirs(cone_axis,
                                 np.linspace(0.30, 0.65, 4),
                                 np.linspace(0, 2*np.pi, 5, endpoint=False))
    dirs_geom   = make_cone_dirs(geom_dir,
                                 np.linspace(0.0,  0.30, 5),
                                 np.linspace(0, 2*np.pi, 5, endpoint=False))
    all_dirs = np.vstack([dirs_narrow, dirs_wide, dirs_geom])

    cos_thresh = np.cos(np.radians(5))
    keep = np.ones(len(all_dirs), dtype=bool)
    for i in range(1, len(all_dirs)):
        if not keep[i]: continue
        if np.any((all_dirs[:i][keep[:i]] @ all_dirs[i]) > cos_thresh):
            keep[i] = False
    all_dirs   = all_dirs[keep]
    candidates = obj_center + all_dirs * d_s

    scored = []
    for camera_xyz in candidates:
        if camera_xyz[2] < z_floor:
            if r_circle_floor is None: continue
            projected = _project_to_z_floor(camera_xyz, obj_center, d_s, z_floor)
            if projected is None: continue
            camera_xyz = projected

        if is_too_close_to_history(camera_xyz, history_positions): continue
        if is_occluded(target_xyz, camera_xyz, nbrs_tree):         continue

        view_dir = target_xyz - camera_xyz
        vn = np.linalg.norm(view_dir)
        if vn < 1e-6: continue
        view_dir /= vn

        alignment = max(0.0, np.dot(view_dir, -normal))
        move_dist = np.linalg.norm(camera_xyz - curr_pos)
        score     = calculate_efficiency_score(alignment, move_dist, t_priority)
        scored.append((score, camera_xyz.copy(), view_dir.copy()))

    if not scored:
        return []

    scored.sort(key=lambda x: x[0], reverse=True)

    cos_sep  = np.cos(np.radians(min_angular_sep_deg))
    selected = []
    sel_dirs = []

    for score, cam_xyz, view_dir in scored:
        if len(selected) >= top_k:
            break
        cam_dir  = cam_xyz - obj_center
        cam_dir /= np.linalg.norm(cam_dir) + 1e-6
        if any(np.dot(cam_dir, sd) > cos_sep for sd in sel_dirs):
            continue
        selected.append({'camera': cam_xyz, 'view_dir': view_dir, 'score': score})
        sel_dirs.append(cam_dir)

    return selected


def find_prioritized_targets(points, scores, voxel_size, threshold=0.65):
    indices = np.where(scores > threshold)[0]
    if len(indices) == 0:
        return []
    f_points   = points[indices]
    clustering = DBSCAN(eps=voxel_size * 5, min_samples=10).fit(f_points)
    labels     = clustering.labels_
    results    = []
    for label in set(labels):
        if label == -1:
            continue
        mask  = labels == label
        c_pts = f_points[mask]
        results.append({
            'center':   np.mean(c_pts, axis=0),
            'normal':   estimate_cluster_normal(c_pts),
            'priority': len(c_pts) * float(np.mean(scores[indices][mask])),
            'is_gap':   False
        })
    if results:
        max_p = max(t['priority'] for t in results)
        for t in results:
            t['priority'] /= max_p + 1e-6
    return results


# -------------------- 2. MONITORING & VISUALIZATION --------------------

class PlanningMonitor:
    def __init__(self, folder="global_models"):
        self.folder = folder
        self.last_processed_id = -1
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        # ── Στατιστικά εκτέλεσης ──────────────────────────────────────
        self.start_time  = time.time()
        self.ram_samples = []
        self._process    = psutil.Process(os.getpid())

        self.plotter = pv.Plotter(title="NBS - Normal-Guided + Gap Detection")
        self.plotter.set_background("#111111")
        self.plotter.add_timer_event(max_steps=1_000_000, duration=500,
                                     callback=self.update)
        self.update()

    def get_latest_file(self):
        files = [f for f in os.listdir(self.folder)
                 if f.startswith("global_model_") and f.endswith(".npy")]
        if not files:
            return None, -1
        latest = max(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        return os.path.join(self.folder, latest), \
               int(latest.split('_')[-1].split('.')[0])

    def update(self, *args):
        # Δειγματοληψία RAM σε κάθε tick (MB)
        ram_mb = self._process.memory_info().rss / 1024 / 1024
        self.ram_samples.append(ram_mb)

        file_path, file_id = self.get_latest_file()
        if file_id > self.last_processed_id:
            self.last_processed_id = file_id
            self.run_planning(file_path)

    # ── Αποθήκευση στατιστικών ────────────────────────────────────────
    def save_stats(self):
        elapsed   = time.time() - self.start_time
        ram_avg   = float(np.mean(self.ram_samples))  if self.ram_samples else 0.0
        ram_peak  = float(np.max(self.ram_samples))   if self.ram_samples else 0.0
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        hours, rem   = divmod(int(elapsed), 3600)
        minutes, sec = divmod(rem, 60)
        elapsed_str  = f"{hours:02d}h {minutes:02d}m {sec:02d}s"

        lines = [
            "=" * 52,
            "  NBS PLANNER – ΣΤΑΤΙΣΤΙΚΑ ΕΚΤΕΛΕΣΗΣ",
            "=" * 52,
            f"  Ημερομηνία / Ώρα    : {timestamp}",
            f"  Αριθμός επεξ. files : {self.last_processed_id + 1}",
            "-" * 52,
            f"  Συνολικός χρόνος    : {elapsed_str}  ({elapsed:.1f} s)",
            f"  RAM μέση χρήση      : {ram_avg:.1f} MB",
            f"  RAM μέγιστη χρήση   : {ram_peak:.1f} MB",
            "=" * 52,
            "",
        ]

        out_path = "metrics/dimitra/metrics_nbv_hercules.txt"
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
            voxel_size     = metadata.get('voxel_size', 0.005)
            camera_history = metadata.get('camera_history', [])
            last_pose      = metadata.get('last_camera_pose', None)

            obj_center      = np.mean(points, axis=0)
            interest_scores = calculate_interest_score(points,
                                                       radius=voxel_size * 2.5)

            # ── Targets: NBS + gap targets ─────────────────────────────────
            nbs_targets = find_prioritized_targets(points, interest_scores,
                                                   voxel_size)
            gap_targets = find_history_gap_targets(
                camera_history, obj_center, points,
                gap_threshold=0.35,
                min_gap_angle_deg=40.0
            )
            all_targets = nbs_targets + gap_targets

            

            # ── Προ-υπολογισμοί κοινοί ────────────────────────────────────
            d_s     = 0.65 #0.65
            z_floor = 0.15
            dz      = z_floor - obj_center[2]
            r2      = d_s**2 - dz**2
            r_circle_floor = float(np.sqrt(r2)) if r2 > 0 else None

            nbrs_tree = build_occlusion_tree(points, voxel_size)

            history_positions = (np.array([p['position'] for p in camera_history])
                                 if camera_history else None)
            curr_pos = (np.array(last_pose['position']) if last_pose
                        else np.zeros(3))

            # ── Visualization setup ────────────────────────────────────────
            self.plotter.clear()
            self.plotter.add_axes()

            pc = pv.PolyData(points)
            pc["Interest"] = interest_scores
            self.plotter.add_mesh(
                pc.glyph(geom=pv.Cube(x_length=voxel_size,
                                       y_length=voxel_size,
                                       z_length=voxel_size), scale=False),
                scalars="Interest", cmap="turbo", opacity=0.3)

            self.plotter.add_mesh(pv.Sphere(radius=0.015, center=curr_pos),
                                  color="cyan")
            self.plotter.add_mesh(
                pv.Sphere(radius=d_s, center=obj_center,
                          theta_resolution=20, phi_resolution=20),
                color="white", opacity=0.04, style="wireframe")

            if r_circle_floor is not None:
                theta      = np.linspace(0, 2 * np.pi, 60)
                floor_ring = np.column_stack([
                    obj_center[0] + r_circle_floor * np.cos(theta),
                    obj_center[1] + r_circle_floor * np.sin(theta),
                    np.full(60, z_floor)
                ])
                self.plotter.add_mesh(pv.PolyData(floor_ring), color="yellow",
                                      point_size=3, render_points_as_spheres=True)

            # Οπτικοποίηση ιστορικού θέσεων
            if history_positions is not None and len(history_positions) > 1:
                for i in range(len(history_positions) - 1):
                    p1, p2 = history_positions[i], history_positions[i + 1]
                    d1 = (p1 - obj_center) / (np.linalg.norm(p1 - obj_center) + 1e-6)
                    d2 = (p2 - obj_center) / (np.linalg.norm(p2 - obj_center) + 1e-6)
                    cos_a = np.clip(np.dot(d1, d2), -1, 1)
                    ang   = np.degrees(np.arccos(cos_a))
                    dist  = np.linalg.norm(p2 - p1)
                    color = "red" if (dist > 0.35 and ang > 40) else "gray"
                    self.plotter.add_mesh(pv.Line(p1, p2),
                                          color=color, line_width=2)

            # ── Planning loop ──────────────────────────────────────────────
            processed = []
            for t in all_targets:
                t['normal'] = orient_normal_outward(t['normal'],
                                                    t['center'], obj_center)
                views = find_top_k_candidate_views(
                    t, nbrs_tree, d_s, curr_pos,
                    obj_center, r_circle_floor,
                    history_positions, z_floor,
                    top_k=3,
                    min_angular_sep_deg=25.0
                )
                for v in views:
                    processed.append({**t, **v})

            if not processed:
                print("⚠️  Δεν βρέθηκαν έγκυρες θέσεις.")
                self.save_stats()
                return

            best_idx = max(range(len(processed)),
                           key=lambda i: processed[i]['score'])

            # ── Visualization + αποθήκευση ─────────────────────────────────
            for i, t in enumerate(processed):
                is_best = (i == best_idx)
                is_gap  = t.get('is_gap', False)

                if is_best:
                    color = "lime"
                elif is_gap:
                    color = "magenta"
                else:
                    color = "orange"

                dist = np.linalg.norm(t['camera'] - obj_center)

                self.plotter.add_mesh(
                    pv.Arrow(start=t['center'], direction=t['normal'],
                             scale=0.05),
                    color="cyan" if is_gap else "blue")
                self.plotter.add_mesh(
                    pv.Line(t['center'], t['camera']),
                    color=color, line_width=3)
                self.plotter.add_mesh(
                    pv.Cone(center=t['camera'],
                            direction=t['center'] - t['camera'],
                            height=0.04, radius=0.015), color=color)

                if is_best:
                    np.save("next_position.npy", {
                        "position":       t['camera'],
                        "view_direction": -t['view_dir']
                    })
                    gap_tag = " [GAP]" if is_gap else ""
                    print(f"🎯 Best{gap_tag}  "
                          f"Z={t['camera'][2]:.3f}  "
                          f"dist={dist:.3f}m  "
                          f"score={t['score']:.3f}")

        except Exception as e:
            print(f"⚠️ Error: {e}")
            import traceback; traceback.print_exc()

    def show(self):
        self.plotter.show()
        # Αποθήκευση στατιστικών όταν κλείσει το παράθυρο
        self.save_stats()


if __name__ == "__main__":
    monitor = PlanningMonitor(folder="global_models")
    monitor.show()