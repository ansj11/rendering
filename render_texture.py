import argparse

import cv2
import numpy as np
from vispy import app, scene
from vispy.scene import visuals
from vispy.io import imread, load_data_file, read_mesh
from vispy.scene.visuals import Mesh
from vispy.scene import transforms
from vispy.visuals.filters import TextureFilter
from vispy.visuals.filters import Alpha
import transforms3d
from moviepy.editor import ImageSequenceClip


class Canvas_view():
    def __init__(self,
                 fov,
                 verts,
                 faces,
                 colors,
                 canvas_size,
                 factor=1,
                 bgcolor='gray',
                 proj='perspective',
                 ):
        self.canvas = scene.SceneCanvas(bgcolor=bgcolor, size=(canvas_size*factor, canvas_size*factor), )
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'perspective' # 'arcball' # 'perspective'
        self.view.camera.fov = fov
        self.view.camera.depth_value = 10 * (vertices.max() - vertices.min())
        self.mesh = visuals.Mesh(vertices, faces, shading=None, color='white')
        self.mesh.transform = transforms.MatrixTransform()
        self.mesh.transform.rotate(90, (1, 0, 0))
        # self.mesh.transform.rotate(135, (0, 0, 1))
        self.mesh.attach(Alpha(1.0))
        self.view.add(self.mesh)
        self.tr = self.view.camera.transform
        # self.mesh.set_data(vertices=verts, faces=faces, vertex_colors=colors[:, :3])
        self.translate([0,0,0])
        self.rotate(axis=[1,0,0], angle=180)
        self.view_changed()

    def translate(self, trans=[0,0,0]):
        self.tr.translate(trans)

    def rotate(self, axis=[1,0,0], angle=0):
        self.tr.rotate(axis=axis, angle=angle)

    def view_changed(self):
        self.view.camera.view_changed()

    def render(self):
        return self.canvas.render()

    def reinit_mesh(self, verts, faces, colors):
        self.mesh.set_data(vertices=verts, faces=faces) # , vertex_colors=colors[:, :3])

    def reinit_camera(self, fov):
        self.view.camera.fov = fov
        self.view.camera.view_changed()

    def attach(self, texture, texcoords):
        texture_filter = TextureFilter(texture, texcoords)
        self.mesh.attach(texture_filter)


def get_pose(i, video_traj_type, focus_point, mean_loc_depth):
    upper, lower = 50, -5
    height = abs(np.tan(53 / 180 * np.pi / 2) * focus_point[2])
    dmax = height * np.tan((53 - lower) / 180 * np.pi / 2)
    dmin = height * np.tan((53 + upper) / 180 * np.pi / 2)
    zrange = dmax - dmin
    x_range, y_range = focus_point[0] / 2, focus_point[1]/2
    xs, ys, zs = [], [], []
    num_frames = 82
    pose = np.eye(4)
    xs = -x_range * np.exp(-4.0 * i / num_frames) #- x_range / 3
    ys = y_range * np.exp(-4.0 * i / num_frames) #+ y_range / 3
    zs = zrange * np.exp(-5.0 * i / num_frames)  # 近 - 远
    xt = x_range * np.exp(-4.0 * i / num_frames) * 1/3 - x_range/6
    yt = -y_range * np.exp(-4.0 * i / num_frames) * 1/3 + y_range/6
    zt = focus_point[2] + np.exp(-5.0 * i / num_frames)*(mean_loc_depth - focus_point[2])

    def normalize(x):
        return x / np.sqrt(np.sum(x ** 2) + 1e-6)

    pose[:3, -1] = np.array([xs, ys, zs])  # -z 指向屏幕外
    z_axis = np.array([xt - xs, yt - ys, zt - zs])
    z_axis = normalize(z_axis)
    up = np.array([0,1,0])
    x_axis = normalize(np.cross(up, z_axis))
    y_axis = normalize(np.cross(z_axis, x_axis))

    R = np.stack((x_axis, y_axis, z_axis), axis=0)
    pose[:3,:3] = R.transpose(1, 0)
    return pose


if __name__ == "__main__":
    mesh_path = load_data_file('spot/spot.obj.gz')
    texture_path = load_data_file('spot/spot.png')
    vertices, faces, normals, texcoords = read_mesh(mesh_path)
    texture = np.flipud(imread(texture_path))
    colors = None

    fov = 68.0
    canvas_size = 960
    init_factor = 3

    canvas = Canvas_view(fov,
                         vertices,
                         faces,
                         colors,
                         canvas_size=canvas_size,
                         factor=init_factor,
                         bgcolor='gray',
                         proj='perspective')
    canvas.attach(texture, texcoords)

    video_traj_type = "circle"

    focus_point = [vertices[:,0].mean(), vertices[:,1].mean(), vertices[:,2].mean()]
    mean_loc_depth = focus_point[2]
    ref_pose = np.eye(4)

    stereos = []
    for tp_id in range(82):
        tp = get_pose(tp_id, video_traj_type, focus_point, mean_loc_depth)
        rel_pose = np.linalg.inv(np.dot(tp, np.linalg.inv(ref_pose)))
        axis, angle = transforms3d.axangles.mat2axangle(rel_pose[0:3, 0:3])
        canvas.rotate(axis=axis, angle=(angle * 180) / np.pi)
        canvas.translate(rel_pose[:3, 3])
        new_mean_loc_depth = mean_loc_depth - float(rel_pose[2, 3])
        if 'dolly' in video_traj_type:
            new_fov = float((np.arctan2(plane_width, np.array([np.abs(new_mean_loc_depth)])) * 180. / np.pi) * 2)
            canvas.reinit_camera(new_fov)
        else:
            canvas.reinit_camera(fov)
        canvas.view_changed()
        img = canvas.render()
        img = cv2.GaussianBlur(img, (int(init_factor // 2 * 2 + 1), int(init_factor // 2 * 2 + 1)), 0)
        img = cv2.resize(img, (int(img.shape[1] / init_factor), int(img.shape[0] / init_factor)),
                         interpolation=cv2.INTER_AREA)

        stereos.append(img[..., :3])
        canvas.translate(-rel_pose[:3, 3])  # 再转会原来角度
        canvas.rotate(axis=axis, angle=-(angle * 180) / np.pi)
        canvas.view_changed()

    clip = ImageSequenceClip(stereos, fps=25)
    clip.write_videofile('demo.mp4', fps=25)
