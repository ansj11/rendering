import os
import sys
import time
import torch
import argparse

import numpy as np

import pytorch3d

from pytorch3d.io import load_obj
import matplotlib.pyplot as plt

# Data structures and functions for rendering
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.renderer import (
    BlendParams,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    SoftPhongShader,
    Textures,
    Materials,
    PointLights,
    PerspectiveCameras,
    FoVPerspectiveCameras,
    look_at_view_transform,
    FoVOrthographicCameras,
    RasterizationSettings,
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor
)
from pytorch3d.io import load_objs_as_meshes, load_obj
from IPython import embed

t = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

obj_filename = 'cow.obj'
mesh = load_objs_as_meshes([obj_filename], device=device)

plt.figure(figsize=(7,7))
texture_image=mesh.textures.maps_padded()
plt.imsave('texture.png', texture_image.squeeze().cpu().numpy())
# plt.imshow(texture_image.squeeze().cpu().numpy())
# plt.axis("off");
print('time: ', time.time() - t)

# Initialize an OpenGL perspective camera.
R, T = look_at_view_transform(2.7, 10, 20)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

# Define the settings for rasterization and shading. Here we set the output image to be of size
# 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
# and blur_radius=0.0. Refer to rasterize_meshes.py for explanations of these parameters.
raster_settings = RasterizationSettings(
    image_size=(960, 720),
    blur_radius=0.0,
    faces_per_pixel=1,
)

# Place a point light in front of the object. As mentioned above, the front of the cow is facing the
# -z direction. z direction oppose to colab
lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

# Create a Phong renderer by composing a rasterizer and a shader. Here we can use a predefined
# PhongShader, passing in the device on which to initialize the default parameters
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    shader=HardPhongShader(device=device, cameras=cameras)
)

images = renderer(mesh)
plt.imsave('images.png', images[0, ..., :3].squeeze().cpu().numpy())
# plt.figure(figsize=(10, 10))
# plt.imshow(images[0, ..., :3].cpu().numpy())
# plt.axis("off");

# Now move the light so it is on the +Z axis which will be behind the cow.
lights.location = torch.tensor([0.0, 0.0, +1.0], device=device)[None]
images = renderer(mesh, lights=lights)

plt.imsave('images1.png', images[0, ..., :3].squeeze().cpu().numpy())


# Rotate the object by increasing the elevation and azimuth angles
R, T = look_at_view_transform(dist=2.7, elev=10, azim=-150)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

# Move the light location so the light is shining on the cow's face.
lights.location = torch.tensor([[2.0, 2.0, -2.0]], device=device)

# Change specular color to green and change material shininess
materials = Materials(
    device=device,
    specular_color=[[0.0, 1.0, 0.0]],
    shininess=10.0
)

# Re render the mesh, passing in keyword arguments for the modified components.
images = renderer(mesh, lights=lights, materials=materials, cameras=cameras)

plt.imsave('images2.png', images[0, ..., :3].squeeze().cpu().numpy())
print('time: ', time.time() - t)

# Set batch size - this is the number of different viewpoints from which we want to render the mesh.
batch_size = 20

t = time.time()
# Create a batch of meshes by repeating the cow mesh and associated textures.
# Meshes has a useful `extend` method which allows us do this very easily.
# This also extends the textures.
meshes = mesh.extend(batch_size)

# Get a batch of viewing angles.
elev = torch.linspace(0, 180, batch_size)
azim = torch.linspace(-180, 180, batch_size)

# All the cameras helper methods support mixed type inputs and broadcasting. So we can
# view the camera from the same distance and specify dist=2.7 as a float,
# and then specify elevation and azimuth angles for each viewpoint as tensors.
R, T = look_at_view_transform(dist=2.7, elev=elev, azim=azim)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

# Move the light back in front of the cow which is facing the -z direction.
lights.location = torch.tensor([[0.0, 0.0, -3.0]], device=device)


# We can pass arbitrary keyword arguments to the rasterizer/shader via the renderer
# so the renderer does not need to be reinitialized if any of the settings change.
images = renderer(meshes, cameras=cameras, lights=lights)
print('time: ', time.time() - t)

for i, image in enumerate(images):
    plt.imsave('image-%0d.png' % i, image[..., :3].squeeze().cpu().numpy())

