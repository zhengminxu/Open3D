# yapf: disable

import open3d as o3d
import open3d.core as o3c
import numpy as np

device = o3c.Device("CUDA:0")

# TriangleMesh V2
mesh = o3d.geometry.TrianglMesh(device)
mesh.vertex["vertices"]    = o3c.Tensor(..., device)
mesh.vertex["normals"]     = o3c.Tensor(..., device)
mesh.vertex["colors"]      = o3c.Tensor(..., device)
mesh.triangle["triangles"] = o3c.Tensor(..., device)

# TriangleMesh V1
mesh = o3d.geometry.TrianglMesh()
mesh.vertices       = o3d.utility.Vector3dVector(np.array(...))
mesh.vertex_normals = o3d.utility.Vector3dVector(np.array(...))
mesh.vertex_colors  = o3d.utility.Vector3dVector(np.array(...))
mesh.triangles      = o3d.utility.Vector3dVector(np.array(...))

# yapf: enable
