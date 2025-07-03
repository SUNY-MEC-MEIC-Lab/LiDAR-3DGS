import numpy as np
import open3d as o3d
import os
from tqdm import tqdm

def reorder_images(output_folder):  
    image_files = sorted([f for f in os.listdir(output_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    for i, file in enumerate(tqdm(image_files, desc="Resorting Images"), start=1):
        os.rename(os.path.join(output_folder, file), os.path.join(output_folder, f"{i:04}.jpg"))
    print("Image Resorting Complete!")

def subsample_and_convert_ply(input_file, output_folder, subsample_ratio=0.003):  # Set default subsample ratio to 10%
    print("Loading LiDAR data from PLY...")
    point_cloud = o3d.io.read_point_cloud(input_file)  # Directly read .ply file
    
    points = np.asarray(point_cloud.points)
    has_colors = point_cloud.has_colors()
    if has_colors:
        colors = np.asarray(point_cloud.colors)
    
    print(f"Subsampling to {subsample_ratio*100}% of original size...")
    sampled_indices = np.random.choice(len(points), int(len(points) * subsample_ratio), replace=False)
    
    sampled_points = points[sampled_indices]
    subsampled_cloud = o3d.geometry.PointCloud()
    subsampled_cloud.points = o3d.utility.Vector3dVector(sampled_points)
    if has_colors:
        sampled_colors = colors[sampled_indices]
        subsampled_cloud.colors = o3d.utility.Vector3dVector(sampled_colors)

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_file_path = os.path.join(output_folder, "sampled_ply_file.ply")
    print(f"Saving subsampled PLY to: {output_file_path}")
    o3d.io.write_point_cloud(output_file_path, subsampled_cloud, write_ascii=True, compressed=False)
    print(f"Subsampled PLY data saved to {output_file_path}")

def load_and_visualize_ply(filepath):
    point_cloud = o3d.io.read_point_cloud(filepath)
    o3d.visualization.draw_geometries([point_cloud])    

# Default sampling percentage is set to 10%
input_file = r'[\gaussian-splatting\.ply]'
output_folder = r'[\gaussian-splatting]'  # Correct path to a folder

# Call the subsampling function with a 10% sampling rate
subsample_and_convert_ply(input_file, output_folder)
output_file_path = os.path.join(output_folder, "sampled_ply_file.ply")
load_and_visualize_ply(output_file_path)
