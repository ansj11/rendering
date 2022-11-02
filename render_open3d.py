import copy
import numpy as np
import open3d as o3d

if __name__ == "__main__":
    print("Testing mesh in open3d ...")
    mesh = o3d.io.read_triangle_mesh("data/a.obj")
    print(mesh)
    print(np.asarray(mesh.vertices))
    print(np.asarray(mesh.triangles))
    verts = np.asarray(mesh.vertices)

    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.6, 0.1, 0.5])

    # cords = o3d.geometry.TriangleMesh.create_coordinate_frame()
    # o3d.visualization.draw_geometries([mesh])

    R = mesh.get_rotation_matrix_from_xyz((np.pi/2, 0, 0))
    mesh.rotate(R, center=(0, 0, 0))

    mat_box = o3d.visualization.rendering.MaterialRecord()
    # mat_box.shader = 'defaultLitTransparency'
    mat_box.shader = 'defaultLitSSR'
    mat_box.base_color = [0.467, 0.467, 0.467, 0.2]
    mat_box.base_roughness = 0.0
    mat_box.base_reflectance = 0.0
    mat_box.base_clearcoat = 1.0
    mat_box.thickness = 1.0
    mat_box.transmission = 1.0
    mat_box.absorption_distance = 10
    mat_box.absorption_color = [0.5, 0.5, 0.5]

    o3d.visualization.draw([{'name': 'box', 'geometry': mesh, 'material': mat_box}])