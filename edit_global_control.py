import numpy as np
import pyvista as pv
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

# -------------------- 1. ΥΠΟΛΟΓΙΣΤΙΚΕΣ ΣΥΝΑΡΤΗΣΕΙΣ (NBS) --------------------

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
    if normal[2] < 0: normal = -normal
    return normal

def is_occluded(target_xyz, camera_xyz, points, voxel_size):
    num_samples = 60
    line_points = np.linspace(target_xyz, camera_xyz, num_samples)
    nbrs = NearestNeighbors(radius=voxel_size * 0.85).fit(points)
    for sample in line_points[3:-3]:
        distances, _ = nbrs.radius_neighbors([sample])
        if len(distances[0]) > 0: return True
    return False

def calculate_efficiency_score(alignment, move_dist, cluster_priority):
    benefit = (alignment * 0.7) + (cluster_priority * 0.5)
    energy_cost = (move_dist * 0.3) + 0.05
    return benefit / (energy_cost + 0.1)

def is_too_close_to_history(candidate_xyz, history, min_dist=0.18):
    if not history: return False
    for pose in history:
        past_pos = np.array(pose['position'])
        if np.linalg.norm(candidate_xyz - past_pos) < min_dist: return True
    return False

def find_best_candidate_view(target_info, points, voxel_size, d_s, curr_pos, camera_history=[]):
    target_xyz, normal, t_priority = target_info['center'], target_info['normal'], target_info['priority']
    best_camera_xyz, best_direction, max_score = None, None, -float('inf')
    z_surf = normal / (np.linalg.norm(normal) + 1e-6)
    azimuths = np.linspace(0, 2 * np.pi, 18)
    elevations = [0.0, 0.2, 0.4]
    for elev in elevations:
        for phi in azimuths:
            direction = np.array([np.cos(phi), np.sin(phi), elev])
            direction /= np.linalg.norm(direction)
            camera_xyz = target_xyz + direction * d_s
            if is_too_close_to_history(camera_xyz, camera_history, min_dist=0.15): continue
            if is_occluded(target_xyz, camera_xyz, points, voxel_size): continue
            alignment = np.dot(-direction, z_surf)
            move_dist = np.linalg.norm(camera_xyz - curr_pos)
            score = calculate_efficiency_score(alignment, move_dist, t_priority)
            if score > max_score:
                max_score, best_camera_xyz, best_direction = score, camera_xyz, direction
    return best_camera_xyz, best_direction

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
        
        # Δημιουργία του φακέλου αν δεν υπάρχει για αποφυγή σφαλμάτων
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        self.plotter = pv.Plotter(title="NBS - Live Planning Monitor")
        self.plotter.set_background("#111111")
        self.plotter.add_axes()
        
        # Χρήση μεγάλου ακεραίου αντί για None στο max_steps για αποφυγή TypeError
        self.plotter.add_timer_event(max_steps=1000000, duration=500, callback=self.update)
        
        print(f"Watching folder: {self.folder} for new global models...")
        self.update() # Πρώτος έλεγχος αμέσως

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
        
        # Φόρτωση μόνο αν βρεθεί νέο ID
        if file_id > self.last_processed_id:
            print(f"🆕 New model detected: ID {file_id}. Processing...")
            self.last_processed_id = file_id
            self.run_planning(file_path)

    def run_planning(self, file_path):
        try:
            data = np.load(file_path, allow_pickle=True).item()
            points = data['points']
            metadata = data.get('metadata', {})
            voxel_size = metadata.get('voxel_size', 0.005)
            camera_history = metadata.get('camera_history', [])
            last_pose = metadata.get('last_camera_pose', None)

            # NBS Logic
            interest_scores = calculate_interest_score(points, radius=voxel_size * 2.5)
            targets = find_prioritized_targets(points, interest_scores, voxel_size)

            # Ανανέωση Plotter
            self.plotter.clear()
            self.plotter.add_axes()
            self.plotter.add_text(f"Global Model ID: {self.last_processed_id}", 
                                position='upper_left', font_size=10, color="white")

            # Εμφάνιση Point Cloud
            pc = pv.PolyData(points)
            pc["Interest"] = interest_scores
            voxels = pc.glyph(geom=pv.Cube(x_length=voxel_size, y_length=voxel_size, z_length=voxel_size), scale=False)
            self.plotter.add_mesh(voxels, scalars="Interest", cmap="turbo", opacity=0.3, name="cloud")

            # Θέση Ρομπότ
            curr_pos = np.array(last_pose['position']) if last_pose else np.array([0, 0, 0])
            self.plotter.add_mesh(pv.Sphere(radius=0.015, center=curr_pos), color="cyan", name="robot")

            # Εμφάνιση Ιστορικού Κάμερας (Γκρι Βέλη)
            for i, pose in enumerate(camera_history):
                p = pose['position']
                rot = pose.get('rotation', np.eye(3))
                d = rot @ np.array([0, 0, -1]) # Κατεύθυνση θέασης
                self.plotter.add_mesh(pv.Arrow(start=p, direction=d, scale=0.03), color="gray", opacity=0.4)

            # Επιλογή Βέλτιστου Στόχου
            d_s, best_idx, max_eff, processed = 0.55, -1, -float('inf'), []
            for i, t in enumerate(targets):
                cam_xyz, view_dir = find_best_candidate_view(t, points, voxel_size, d_s, curr_pos, camera_history)
                t['camera'], t['view_dir'] = cam_xyz, view_dir
                
                if cam_xyz is not None:
                    dist = np.linalg.norm(cam_xyz - curr_pos)
                    # Υπολογισμός efficiency (alignment=1.0 για τον υπολογισμό προτεραιότητας)
                    eff = calculate_efficiency_score(1.0, dist, t['priority'])
                    if eff > max_eff:
                        max_eff, best_idx = eff, i
                processed.append(t)

            # Σχεδίαση Στόχων στην οθόνη
            for i, t in enumerate(processed):
                if t['camera'] is None: continue
                color = "lime" if (i == best_idx) else "orange"
                
                # Γραμμή από τον στόχο στην προτεινόμενη θέση κάμερας
                self.plotter.add_mesh(pv.Line(t['center'], t['camera']), color=color, line_width=3)
                # Κώνος που δείχνει την κατεύθυνση θέασης
                self.plotter.add_mesh(pv.Cone(center=t['camera'], direction=t['center'] - t['camera'], 
                                              height=0.04, radius=0.015), color=color)
                
                if i == best_idx:
                    # Γραμμή κίνησης ρομπότ
                    self.plotter.add_mesh(pv.Line(curr_pos, t['camera']), color="white", line_width=2)
                    # Αποθήκευση για το επόμενο βήμα
                    output = {"position": t['camera'], "view_direction": t['view_dir']}
                    np.save("next_position.npy", output)
                    print(f"🎯 Target updated! Next position saved for Model {self.last_processed_id}")

        except Exception as e:
            print(f"⚠️ Error in run_planning: {e}")

    def show(self):
        self.plotter.show()

if __name__ == "__main__":
    monitor = PlanningMonitor(folder="global_models")
    monitor.show()