# Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import argparse
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
from open3d.visualization.gui import Label3D

from read_write_model import read_model, write_model, qvec2rotmat, rotmat2qvec


def draw_camera(K, R, t, w, h, scale=1, color=[0.8, 0.2, 0.8]):
    """Create axis, plane and pyramid geometries in Open3D format."""
    K = K.copy() / scale
    Kinv = np.linalg.inv(K)

    T = np.column_stack((R, t))
    T = np.vstack((T, (0, 0, 0, 1)))

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5 * scale)
    axis.transform(T)

    points_pixel = [[0, 0, 0], [0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]]
    points = [Kinv @ p for p in points_pixel]

    width = abs(points[1][0]) + abs(points[3][0])
    height = abs(points[1][1]) + abs(points[3][1])
    plane = o3d.geometry.TriangleMesh.create_box(width, height, depth=1e-6)
    plane.paint_uniform_color(color)
    plane.translate([points[1][0], points[1][1], scale])
    plane.transform(T)

    points_in_world = [(R @ p + t) for p in points]
    lines = [[0, 1], [0, 2], [0, 3], [0, 4]]
    colors = [color for _ in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points_in_world),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    direction_vector = R @ np.array([0, 0, 5])
    direction_points = [t, t + direction_vector]
    direction_lines = [[0, 1]]
    direction_line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(direction_points),
        lines=o3d.utility.Vector2iVector(direction_lines),
    )
    direction_line_set.colors = o3d.utility.Vector3dVector([color])

    return [axis, plane, line_set, direction_line_set]


class Model:
    def __init__(self):
        self.cameras = []
        self.images = []
        self.points3D = []
        self.__vis = None

    def read_model(self, path, ext=""):
        self.cameras, self.images, self.points3D = read_model(path, ext)

    def add_points(self, min_track_len=3, remove_statistical_outlier=True):
        pcd = o3d.geometry.PointCloud()

        xyz = []
        rgb = []
        for point3D in self.points3D.values():
            track_len = len(point3D.point2D_idxs)
            if track_len < min_track_len:
                continue
            xyz.append(point3D.xyz)
            rgb.append(point3D.rgb / 255)

        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)

        if remove_statistical_outlier:
            [pcd, _] = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        self.__vis.add_geometry("points", pcd)

    def add_cameras(self, scale=1):
        for idx, (image_id, img) in enumerate(self.images.items()):
            R = qvec2rotmat(img.qvec)
            t = img.tvec
            t = -R.T @ t
            R = R.T

            cam = self.cameras[img.camera_id]
            if cam.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
                fx = fy = cam.params[0]
                cx, cy = cam.params[1], cam.params[2]
            elif cam.model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"):
                fx, fy = cam.params[0], cam.params[1]
                cx, cy = cam.params[2], cam.params[3]
            else:
                raise Exception("Camera model not supported")

            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

            color = [0.8, 0.2, 0.8] if idx == 0 else [1.0, 0.0, 0.0]
            cam_model = draw_camera(K, R, t, cam.width, cam.height, scale, color)
            
            for i, geom in enumerate(cam_model):
                self.__vis.add_geometry(f"camera_{idx}_{i}", geom)

            self.__vis.add_3d_label(t, f"Cam {idx+1}")

    def create_window(self):
        gui.Application.instance.initialize()
        self.__vis = o3d.visualization.O3DVisualizer("COLMAP Model Viewer", 1024, 768)
        gui.Application.instance.add_window(self.__vis)

    def show(self):
        self.__vis.reset_camera_to_default()
        self.__vis.show_settings = True

    def load_geometry(self, file_path):
        if file_path.endswith(".ply"):
            geometry = o3d.io.read_triangle_mesh(file_path)
        elif file_path.endswith(".obj"):
            geometry = o3d.io.read_triangle_mesh(file_path)
        else:
            raise ValueError("Unsupported file format. Only PLY and OBJ are supported.")

        if not geometry.has_vertex_normals():
            geometry.compute_vertex_normals()

        self.__vis.add_geometry("loaded_geometry", geometry)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize COLMAP binary and text models")
    parser.add_argument("--input_model", required=True, help="path to input model folder")
    parser.add_argument("--input_format", choices=[".bin", ".txt"], help="input model format", default="")
    parser.add_argument("--geometry_file", help="path to the PLY or OBJ file to visualize")
    return parser.parse_args()

def main():
    args = parse_args()

    model = Model()
    model.read_model(args.input_model, ext=args.input_format)

    print("num_cameras:", len(model.cameras))
    print("num_images:", len(model.images))
    print("num_points3D:", len(model.points3D))

    model.create_window()
    model.add_points()
    model.add_cameras(scale=0.25)

    if args.geometry_file:
        model.load_geometry(args.geometry_file)

    model.show()
    gui.Application.instance.run()

if __name__ == "__main__":
    main()