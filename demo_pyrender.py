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

# Fuze trimesh
fuze_trimesh = trimesh.load('/Users/anshijie/gitlab/pyrender/examples/models/fuze.obj')
fuze_mesh = Mesh.from_trimesh(fuze_trimesh)

# Drill trimesh
drill_trimesh = trimesh.load('/Users/anshijie/gitlab/pyrender/examples/models/drill.obj')
drill_mesh = Mesh.from_trimesh(drill_trimesh)
drill_pose = np.eye(4)
drill_pose[0,3] = 0.1
drill_pose[2,3] = -np.min(drill_trimesh.vertices[:,2])

# Wood trimesh
wood_trimesh = trimesh.load('/Users/anshijie/gitlab/pyrender/examples/models/wood.obj')
wood_mesh = Mesh.from_trimesh(wood_trimesh)

# Water bottle trimesh
bottle_gltf = trimesh.load('/Users/anshijie/gitlab/pyrender/examples/models/WaterBottle.glb')
bottle_trimesh = bottle_gltf.geometry[list(bottle_gltf.geometry.keys())[0]] # ??effect
bottle_mesh = Mesh.from_trimesh(bottle_trimesh)
bottle_pose = np.array([
    [1.0, 0.0,  0.0, 0.1],
    [0.0, 0.0, -1.0, -0.16],
    [0.0, 1.0,  0.0, 0.13],
    [0.0, 0.0,  0.0, 1.0],
])

#------------------------------------------------------------------------------
# Creating meshes with per-vertex colors
#------------------------------------------------------------------------------
boxv_trimesh = trimesh.creation.box(extents=0.1*np.ones(3))
boxv_vertex_colors = np.random.uniform(size=(boxv_trimesh.vertices.shape)) # 随机顶点颜色box mesh
boxv_trimesh.visual.vertex_colors = boxv_vertex_colors
boxv_mesh = Mesh.from_trimesh(boxv_trimesh, smooth=False)

#------------------------------------------------------------------------------
# Creating meshes with per-face colors
#------------------------------------------------------------------------------
boxf_trimesh = trimesh.creation.box(extents=0.1*np.ones(3))
boxf_face_colors = np.random.uniform(size=boxf_trimesh.faces.shape)
boxf_trimesh.visual.face_colors = boxf_face_colors ## 随机面片颜色的box mesh
boxf_mesh = Mesh.from_trimesh(boxf_trimesh, smooth=False)

#------------------------------------------------------------------------------
# Creating meshes from point clouds
#------------------------------------------------------------------------------
points = trimesh.creation.icosphere(radius=0.05).vertices
point_colors = np.random.uniform(size=points.shape)
points_mesh = Mesh.from_points(points, colors=point_colors) # 从点云创建一个球

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
fuze_node = Node(mesh=fuze_mesh, translation=np.array([0.1, 0.15, -np.min(fuze_trimesh.vertices[:,2])]))
scene.add_node(fuze_node) # 加入子mesh，只有平移
boxv_node = Node(mesh=boxv_mesh, translation=np.array([-0.1, 0.10, 0.05]))
scene.add_node(boxv_node)
boxf_node = Node(mesh=boxf_mesh, translation=np.array([-0.1, -0.10, 0.05]))
scene.add_node(boxf_node)
# points_node = Node(mesh=points_mesh, translation=np.array([-0.1, -0.10, 0.05]))
# scene.add_node(points_node)

#------------------------------------------------------------------------------
# By using the add() utility function
#------------------------------------------------------------------------------
drill_node = scene.add(drill_mesh, pose=drill_pose) # 用add加入带位姿变换的mesh
bottle_node = scene.add(bottle_mesh, pose=bottle_pose)
wood_node = scene.add(wood_mesh)
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
v = Viewer(scene, central_node=drill_node) # 加入相机位姿，确定中心位置
# v = Viewer(scene, central_node=points_node) # 加入相机位姿，确定中心位置

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