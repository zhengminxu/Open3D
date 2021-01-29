import open3d.core as o3c
import numpy as np

# Creation
a = o3c.Tensor([[0, 1], [2, 3]],
                dtype=o3c.Dtype.Float32
                device=o3c.Device("CUDA:0"))
a = o3c.Tensor(np.array([0, 1, 2]))

# Basic operations
b = a[:, 0] * a - 1  # Broadcast
b = a.sum(dim=(0,))  # Reduction

# Slicing, advanced indexing
a = o3c.Tensor.ones((2, 3, 4))
b = a[1, 0:2, [1, 2]]

# Linear algebra
o3d.inv, svd, solve, lstsq, det
