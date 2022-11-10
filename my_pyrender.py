"""Examples of using pyrender for viewing and offscreen rendering.
"""
import pyglet
pyglet.options['shadow_window'] = False
import os
import numpy as np
import trimesh

from pyrender import PerspectiveCamera,\
                     DirectionalLight, SpotLight, PointLight,\
                     MetallicRoughnessMaterial,\
                     Primitive, Mesh, Node, Scene,\
                     Viewer, OffscreenRenderer, RenderFlags

#==============================================================================
# Mesh creation
#==============================================================================

#------------------------------------------------------------------------------
# Creating textured meshes from trimeshes
#------------------------------------------------------------------------------

# Drill trimesh
cow_trimesh = trimesh.load('./data/cow.obj')
cow_mesh = Mesh.from_trimesh(cow_trimesh)
cow_pose = np.eye(4)
cow_pose[0,3] = 0.1
cow_pose[2,3] = -np.min(cow_trimesh.vertices[:,2])
# cow_pose = np.array([
#     [1.0, 0.0,  0.0, 0.1],
#     [0.0, 0.0, -1.0, -0.16],
#     [0.0, 1.0,  0.0, 0.13],
#     [0.0, 0.0,  0.0, 1.0],
# ])

#==============================================================================
# Light creation
#==============================================================================

direc_l = DirectionalLight(color=np.ones(3), intensity=1.0) # 方向光
spot_l = SpotLight(color=np.ones(3), intensity=10.0,
                   innerConeAngle=np.pi/16, outerConeAngle=np.pi/6) # 聚光灯
point_l = PointLight(color=np.ones(3), intensity=10.0) # 点光源

#==============================================================================
# Camera creation
#==============================================================================

cam = PerspectiveCamera(yfov=(np.pi / 3.0)) # 透视投影相机fov
cam_pose = np.array([
    [0.0,  -np.sqrt(2)/2, np.sqrt(2)/2, 0.5],
    [1.0, 0.0,           0.0,           0.0],
    [0.0,  np.sqrt(2)/2,  np.sqrt(2)/2, 0.4],
    [0.0,  0.0,           0.0,          1.0]
]) # 相机位姿

#==============================================================================
# Scene creation
#==============================================================================

scene = Scene(ambient_light=np.array([0.02, 0.02, 0.02, 1.0])) # 创建场景，环境光

#==============================================================================
# Adding objects to the scene
#==============================================================================

#------------------------------------------------------------------------------
# By manually creating nodes
#------------------------------------------------------------------------------
cow_node = Node(mesh=cow_mesh, translation=np.array([0.1, 0.15, -np.min(cow_trimesh.vertices[:,2])]))
scene.add_node(cow_node) # 加入子mesh，只有平移

#------------------------------------------------------------------------------
# By using the add() utility function
#------------------------------------------------------------------------------
# cow_node = scene.add(cow_mesh, pose=cow_pose) # 用add加入带位姿变换的mesh
direc_l_node = scene.add(direc_l, pose=cam_pose) # 加入光
spot_l_node = scene.add(spot_l, pose=cam_pose)

#==============================================================================
# Using the viewer with a default camera
#==============================================================================

v = Viewer(scene, shadows=True) # viewer用默认相机

#==============================================================================
# Using the viewer with a pre-specified camera
#==============================================================================
cam_node = scene.add(cam, pose=cam_pose)
v = Viewer(scene, central_node=cow_node) # 加入相机位姿，确定中心位置

#==============================================================================
# Rendering offscreen from that camera
#==============================================================================

r = OffscreenRenderer(viewport_width=640*2, viewport_height=480*2)
color, depth = r.render(scene)

import matplotlib.pyplot as plt
plt.figure()
plt.imshow(color)
plt.show()

#==============================================================================
# Segmask rendering
#==============================================================================

nm = {node: 20*(i + 1) for i, node in enumerate(scene.mesh_nodes)}
seg = r.render(scene, RenderFlags.SEG, nm)[0]
plt.figure()
plt.imshow(seg)
plt.show()

r.delete()