# Convert PLY file to a COLMAP-readable TXT format
# Hansol Lim 2024-03-13
# hansol.lim@stonybrook.edu

import numpy as np
import open3d as o3d 

class Point3D:
    def __init__(self, id, xyz, rgb, error, image_ids, point2D_idxs):
        self.id = id
        self.xyz = xyz
        self.rgb = rgb
        self.error = error
        self.image_ids = image_ids
        self.point2D_idxs = point2D_idxs

def ply_to_colmap_txt(ply_path, colmap_txt_path):
    pcd = o3d.io.read_point_cloud(ply_path)

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.colors else np.zeros((len(points), 3))
    colors = (colors * 255).astype(np.uint8)
    
    error = 0
    image_ids = [0] 
    point2D_idxs = [0] 

    points3D = {}
    for i in range(len(points)):
        point_id = i + 1 
        xyz = points[i]
        rgb = colors[i] if colors.size else [0, 0, 0]
        points3D[point_id] = Point3D(point_id, xyz, rgb, error, image_ids, point2D_idxs)

    write_points3D_text(points3D, colmap_txt_path)

def write_points3D_text(points3D, path):
    if not points3D:
        mean_track_length = 0
    else:
        mean_track_length = sum(len(pt.image_ids) for pt in points3D.values()) / len(points3D)
    
    header = (
        "# 3D point list with one line of data per point:\n"
        "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        "# Number of points: {}, mean track length: {}\n".format(len(points3D), mean_track_length)
    )

    with open(path, "w") as fid:
        fid.write(header)
        for pt in points3D.values():
            point_header = [pt.id, *pt.xyz, *pt.rgb, pt.error]
            track_strings = [" ".join(map(str, [image_id, point2D_idx]))
                             for image_id, point2D_idx in zip(pt.image_ids, pt.point2D_idxs)]
            fid.write(" ".join(map(str, point_header)) + " " + " ".join(track_strings) + "\n")


if __name__ == "__main__":
    ply_path = r'[\gaussian-splatting\your_data\.ply]'
    colmap_txt_path = r'[\gaussian-splatting\your_data\.txt]'
    ply_to_colmap_txt(ply_path, colmap_txt_path)
