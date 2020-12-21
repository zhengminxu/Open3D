# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/python/reconstruction_system/integrate_scene.py

import numpy as np
import time
import open3d as o3d

if __name__ == '__main__':
    device = o3d.core.Device("CPU:0")

    intrinsic = o3d.core.Tensor(
        [[525.0, 0.0, 319.5], [0.0, 525.0, 239.5], [0.0, 0.0, 1.0]],
        o3d.core.Dtype.Float32, device)
    extrinsic = o3d.core.Tensor.eye(4, o3d.core.Dtype.Float32, device)

    depth = o3d.io.read_image(
        "/home/yixing/data/stanford/lounge/lounge_png/depth/000001.png")
    depth = o3d.t.geometry.Image.from_legacy_image(depth, device=device)

    for i in range(15):
        np.linalg.inv(np.eye(4))

        start = time.time()
        o3d.t.geometry.PointCloud.create_from_depth_image(
            depth, intrinsic, extrinsic, 1000.0, 3.0, 4)
        end = time.time()
        print('iter {}: create_from_depth_image takes {:.3f} ms'.format(
            i, (end - start) * 1000.0))
