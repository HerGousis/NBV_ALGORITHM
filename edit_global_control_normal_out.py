import numpy as np
import pyvista as pv
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

# -------------------- UTILS --------------------

def calculate_interest_score(points, radius=0.03):
    if len(points) == 0:
        return np.array([])

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

    return normal / (np.linalg.norm(normal) + 1e-6)

# -------------------- 🔥 OUTWARD NORMAL --------------------

def orient_normal_outward(normal, cluster_center, object_center):
    vec_out = cluster_center - object_center
    vec_out /= (np.linalg.norm(vec_out) + 1e-6)

    if np.dot(normal, vec_out) < 0:
        return -normal

    return normal

# -------------------- --------------------

def is_occluded(target_xyz, camera_xyz, points, voxel_size):
    num_samples = 60
    line_points = np.linspace(target_xyz, camera_xyz, num_samples)

    nbrs = NearestNeighbors(radius=voxel_size * 0.85).fit(points)

    for sample in line_points[3:-3]:
        distances, _ = nbrs.radius_neighbors([sample])
        if len(distances[0]) > 0:
            return True

    return False

def calculate_efficiency_score(alignment, move_dist, cluster_priority):
    benefit = (alignment * 0.7) + (cluster_priority * 0.5)
    energy_cost = (move_dist * 0.3) + 0.05
    return benefit / (energy_cost + 0.1)

def find_best_candidate_view(target_info, points, voxel_size, d_s, curr_pos):
    target_xyz = target_info['center']
    normal = target_info['normal']
    t_priority = target_info['priority']

    best_camera_xyz, best_direction = None, None
    max_score = -float('inf')

    azimuths = np.linspace(0, 2 * np.pi, 18)
    elevations = [0.0, 0.2, 0.4]

    for elev in elevations:
        for phi in azimuths:
            direction = np.array([np.cos(phi), np.sin(phi), elev])
            direction /= np.linalg.norm(direction)

            camera_xyz = target_xyz + direction * d_s

            if is_occluded(target_xyz, camera_xyz, points, voxel_size):
                continue

            alignment = np.dot(-direction, normal)
            move_dist = np.linalg.norm(camera_xyz - curr_pos)

            score = calculate_efficiency_score(alignment, move_dist, t_priority)

            if score > max_score:
                max_score = score
                best_camera_xyz = camera_xyz
                best_direction = direction

    return best_camera_xyz, best_direction

def find_prioritized_targets(points, scores, voxel_size, threshold=0.65):
    frontier_indices = np.where(scores > threshold)[0]

    if len(frontier_indices) == 0:
        return []

    frontier_points = points[frontier_indices]

    clustering = DBSCAN(eps=voxel_size * 5, min_samples=10).fit(frontier_points)
    labels = clustering.labels_

    results = []

    for label in set(labels):
        if label == -1:
            continue

        mask = (labels == label)
        cluster_pts = frontier_points[mask]

        avg_interest = np.mean(scores[frontier_indices][mask])

        results.append({
            'center': np.mean(cluster_pts, axis=0),
            'normal': estimate_cluster_normal(cluster_pts),
            'priority': len(cluster_pts) * avg_interest
        })

    if results:
        max_p = max(t['priority'] for t in results)
        for t in results:
            t['priority'] /= (max_p + 1e-6)

    return results

# -------------------- VISUALIZATION --------------------

class PlanningMonitor:
    def __init__(self, folder="global_models"):
        self.folder = folder
        self.last_processed_id = -1

        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        self.plotter = pv.Plotter(title="ONLY OUTWARD NORMALS")
        self.plotter.set_background("#111111")

        self.plotter.add_timer_event(
            max_steps=1000000,
            duration=500,
            callback=self.update
        )

        self.update()

    def get_latest_file(self):
        files = [f for f in os.listdir(self.folder)
                 if f.startswith("global_model_") and f.endswith(".npy")]

        if not files:
            return None, -1

        def get_id(name):
            try:
                return int(name.split('_')[-1].split('.')[0])
            except:
                return -1

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
            points = data['points']
            metadata = data.get('metadata', {})

            voxel_size = metadata.get('voxel_size', 0.005)
            last_pose = metadata.get('last_camera_pose', None)

            curr_pos = np.array(last_pose['position']) if last_pose else np.array([0, 0, 0])

            object_center = np.mean(points, axis=0)

            interest_scores = calculate_interest_score(points, radius=voxel_size * 2.5)
            targets = find_prioritized_targets(points, interest_scores, voxel_size)

            self.plotter.clear()
            self.plotter.add_axes()

            # point cloud
            pc = pv.PolyData(points)
            pc["Interest"] = interest_scores

            self.plotter.add_mesh(
                pc.glyph(geom=pv.Cube(
                    x_length=voxel_size,
                    y_length=voxel_size,
                    z_length=voxel_size),
                    scale=False),
                scalars="Interest",
                cmap="turbo",
                opacity=0.3
            )

            # robot
            self.plotter.add_mesh(
                pv.Sphere(radius=0.015, center=curr_pos),
                color="cyan"
            )

            # object center
            self.plotter.add_mesh(
                pv.Sphere(radius=0.02, center=object_center),
                color="yellow"
            )

            d_s = 0.55

            for t in targets:

                outward_normal = orient_normal_outward(
                    t['normal'],
                    t['center'],
                    object_center
                )

                cam_xyz, view_dir = find_best_candidate_view(
                    {
                        'center': t['center'],
                        'normal': outward_normal,
                        'priority': t['priority']
                    },
                    points, voxel_size, d_s, curr_pos
                )

                # 🔵 ONLY OUTWARD NORMAL ARROW
                self.plotter.add_mesh(
                    pv.Arrow(
                        start=t['center'],
                        direction=outward_normal,
                        scale=0.06
                    ),
                    color="blue"
                )

        except Exception as e:
            print(f"⚠️ Error: {e}")

    def show(self):
        self.plotter.show()


# -------------------- RUN --------------------

if __name__ == "__main__":
    monitor = PlanningMonitor(folder="global_models")
    monitor.show()