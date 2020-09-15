import open3d.core as o3c
import numpy as np
import pytest
from open3d.ml.contrib import knn_search


def test_knn_search():
    while True:
        query_points = np.random.rand(100, 3).astype(np.float32)
        dataset_points = np.random.rand(100, 3).astype(np.float32)
        knn = 3
        indices = knn_search(o3c.Tensor.from_numpy(query_points),
                             o3c.Tensor.from_numpy(dataset_points), knn)
