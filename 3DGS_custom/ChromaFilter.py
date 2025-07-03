# Chroma LiDAR feature Subsampling
# Hansol Lim 2024-03-13
# hansol.lim@stonybrook.edu

import open3d as o3d
import numpy as np
from collections import defaultdict

def load_point_cloud(file_path):
    return o3d.io.read_point_cloud(file_path)

def remove_noise(pcd, nb_neighbors=100, std_ratio=2.0):
    clean_pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return clean_pcd

def limit_rgb_occurrences(pcd, max_points_per_color=1):
    colors_scaled = np.asarray(pcd.colors) * 255
    colors_int = np.round(colors_scaled).astype(int)
    colors_tuples = [tuple(color) for color in colors_int]
    
    color_occurrences = defaultdict(list)
    
    for idx, color in enumerate(colors_tuples):
        color_occurrences[color].append(idx)
    
    limited_indices = []
    for color, indices in color_occurrences.items():
        if len(indices) > max_points_per_color:
            limited_indices.extend(np.random.choice(indices, size=max_points_per_color, replace=False))
        else:
            limited_indices.extend(indices)
    
    limited_pcd = o3d.geometry.PointCloud()
    limited_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[limited_indices])
    limited_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[limited_indices])
    
    return limited_pcd

file_path = r'[\path\to\your\ply\file.ply]'
pcd = load_point_cloud(file_path)
o3d.visualization.draw_geometries([pcd], window_name="Reduced and Cleaned Point Cloud")

clean_pcd = remove_noise(pcd)

reduced_pcd = limit_rgb_occurrences(clean_pcd, max_points_per_color=1) #! 1, 5, 10, 20

output_path = r'[\path\to\filtered\file.ply]'
o3d.io.write_point_cloud(output_path, reduced_pcd)
#o3d.io.write_point_cloud(output_path, clean_pcd)

