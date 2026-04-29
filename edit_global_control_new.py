import numpy as np
import pyvista as pv
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

# -------------------- 1. ΥΠΟΛΟΓΙΣΤΙΚΕΣ ΣΥΝΑΡΤΗΣΕΙΣ (NBS) --------------------

def calculate_angle(vector_a, vector_b):
    unit_a = vector_a / (np.linalg.norm(vector_a) + 1e-6)
    unit_b = vector_b / (np.linalg.norm(vector_b) + 1e-6)
    dot_product = np.clip(np.dot(unit_a, unit_b), -1.0, 1.0)
    return np.degrees(np.arccos(dot_product))

def calculate_interest_score(points, radius=0.03):
    if len(points) == 0: return np.array([])
    nbrs = NearestNeighbors(radius=radius).fit(points)
    adj_matrix = nbrs.radius_neighbors_graph(points)
    neighbor_counts = np.array(adj_matrix.sum(axis=1)).flatten()
    scores = 1.0 - (neighbor_counts / (np.max(neighbor_counts) + 1e-6))
    return scores

def estimate_cluster_normal(cluster_points):
    centroid = np.mean(cluster_points, axis=0)
    centered = cluster_points - centroid
    cov = np.dot(centered.T, centered)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    normal = eigenvectors[:, 0]
    return normal

def is_occluded(target_xyz, camera_xyz, points, voxel_size):
    num_samples = 65
    line_points = np.linspace(target_xyz, camera_xyz, num_samples)
    nbrs = NearestNeighbors(radius=voxel_size * 0.9).fit(points)
    for sample in line_points[5:-5]:
        distances, _ = nbrs.radius_neighbors([sample])
        if len(distances[0]) > 0: return True
    return False

def is_too_close_to_history(candidate_xyz, history, min_dist=0.18):
    if not history: return False
    for pose in history:
        past_pos = np.array(pose['position'])
        if np.linalg.norm(candidate_xyz - past_pos) < min_dist: return True
    return False

def calculate_efficiency_score(alignment, move_dist, cluster_priority):
    benefit = (alignment * 0.7) + (cluster_priority * 0.5)
    energy_cost = (move_dist * 0.3) + 0.05
    return benefit / (energy_cost + 0.1)

def find_best_candidate_view_dual(target_info, points, voxel_size, d_s, curr_pos, camera_history):
    target_xyz = target_info['center']
    normals = [target_info['normal'], -target_info['normal']]
    
    valid_sides = []
    for norm in normals:
        camera_xyz = target_xyz + (norm * d_s)
        
        # Έλεγχος Ιστορικού
        if is_too_close_to_history(camera_xyz, camera_history, min_dist=0.18):
            continue
            
        # Έλεγχος Occlusion (αν τρυπάει το grid)
        if not is_occluded(target_xyz, camera_xyz, points, voxel_size):
            view_dir = target_xyz - camera_xyz
            view_dir /= (np.linalg.norm(view_dir) + 1e-6)
            
            vec_to_robot = curr_pos - target_xyz
            angle = calculate_angle(norm, vec_to_robot)
            
            valid_sides.append({'camera': camera_xyz, 'view_dir': view_dir, 'norm': norm, 'angle': angle})
            
    if not valid_sides: return None, None, None
    
    # Επιλογή της πλευράς με τη μικρότερη γωνία προς το ρομπότ
    best = min(valid_sides, key=lambda x: x['angle'])
    return best['camera'], best['view_dir'], best['norm']

def find_prioritized_targets(points, scores, voxel_size, threshold=0.65):
    frontier_indices = np.where(scores > threshold)[0]
    if len(frontier_indices) == 0: return []
    frontier_points = points[frontier_indices]
    clustering = DBSCAN(eps=voxel_size * 5, min_samples=10).fit(frontier_points)
    labels = clustering.labels_
    results = []
    for label in set(labels):
        if label == -1: continue
        mask = (labels == label)
        cluster_pts = frontier_points[mask]
        avg_interest = np.mean(scores[frontier_indices][mask])
        results.append({'center': np.mean(cluster_pts, axis=0), 'normal': estimate_cluster_normal(cluster_pts), 'priority': len(cluster_pts) * avg_interest})
    if results:
        max_p = max(t['priority'] for t in results)
        for t in results: t['priority'] /= (max_p + 1e-6)
    return results

# -------------------- 2. MONITORING & VISUALIZATION CLASS --------------------

class PlanningMonitor:
    def __init__(self, folder="global_models"):
        self.folder = folder
        self.last_processed_id = -1
        if not os.path.exists(self.folder): os.makedirs(self.folder)
        self.plotter = pv.Plotter(title="NBS - Dual Side History Planning")
        self.plotter.set_background("#0f0f0f")
        self.plotter.add_timer_event(max_steps=1000000, duration=500, callback=self.update)
        print(f"Watching folder: {self.folder}...")

    def get_latest_file(self):
        files = [f for f in os.listdir(self.folder) if f.startswith("global_model_") and f.endswith(".npy")]
        if not files: return None, -1
        def get_id(name):
            try: return int(name.split('_')[-1].split('.')[0])
            except: return -1
        latest_file = max(files, key=get_id)
        return os.path.join(self.folder, latest_file), get_id(latest_file)

    def update(self, *args):
        file_path, file_id = self.get_latest_file()
        if file_id > self.last_processed_id:
            self.last_processed_id = file_id
            self.run_planning(file_path)

    def run_planning(self, file_path):
        try:
            data = np.load(file_path, allow_pickle=True).item()
            points, metadata = data['points'], data.get('metadata', {})
            voxel_size = metadata.get('voxel_size', 0.005)
            camera_history = metadata.get('camera_history', [])
            last_pose = metadata.get('last_camera_pose', None)
            curr_pos = np.array(last_pose['position']) if last_pose else np.array([0, 0, 0])

            interest_scores = calculate_interest_score(points, radius=voxel_size * 3.0)
            targets = find_prioritized_targets(points, interest_scores, voxel_size)

            self.plotter.clear()
            self.plotter.add_axes()
            self.plotter.add_mesh(pv.PolyData(points), scalars=interest_scores, cmap="viridis", point_size=3, opacity=0.4)
            self.plotter.add_mesh(pv.Sphere(radius=0.015, center=curr_pos), color="cyan")

            # Εμφάνιση Ιστορικού (Γκρι)
            for pose in camera_history:
                p, rot = pose['position'], pose.get('rotation', np.eye(3))
                d = rot @ np.array([0, 0, -1]) 
                self.plotter.add_mesh(pv.Arrow(start=p, direction=d, scale=0.03), color="gray", opacity=0.4)

            d_s, processed, max_eff, best_idx = 0.45, [], -float('inf'), -1
            for i, t in enumerate(targets):
                cam_xyz, view_dir, chosen_norm = find_best_candidate_view_dual(t, points, voxel_size, d_s, curr_pos, camera_history)
                t['cam_xyz'], t['view_dir'], t['chosen_norm'] = cam_xyz, view_dir, chosen_norm
                if cam_xyz is not None:
                    eff = calculate_efficiency_score(1.0, np.linalg.norm(cam_xyz - curr_pos), t['priority'])
                    if eff > max_eff: max_eff, best_idx = eff, i
                processed.append(t)

            for i, t in enumerate(processed):
                # Μπλε (Θετικό) / Ματζέντα (Αρνητικό)
                self.plotter.add_mesh(pv.Arrow(start=t['center'], direction=t['normal'], scale=0.06), color="blue")
                self.plotter.add_mesh(pv.Arrow(start=t['center'], direction=-t['normal'], scale=0.04), color="magenta")

                if t['cam_xyz'] is not None:
                    color = "lime" if i == best_idx else "orange"
                    self.plotter.add_mesh(pv.Line(t['center'], t['cam_xyz']), color=color, line_width=4 if i==best_idx else 2)
                    self.plotter.add_mesh(pv.Cone(center=t['cam_xyz'], direction=t['center']-t['cam_xyz'], height=0.04, radius=0.015), color=color)
                    if i == best_idx:
                        np.save("next_position.npy", {"position": t['cam_xyz'].tolist(), "view_direction": (-t['view_dir']).tolist()})
                        self.plotter.add_mesh(pv.Line(curr_pos, t['cam_xyz']), color="white", line_width=2)
                else:
                    self.plotter.add_mesh(pv.Line(t['center'], t['center'] + t['normal']*0.1), color="red")

        except Exception as e: print(f"⚠️ Error: {e}")

    def show(self): self.plotter.show()

if __name__ == "__main__":
    monitor = PlanningMonitor(folder="global_models")
    monitor.show()