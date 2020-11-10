# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2020 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

import open3d as o3d
import open3d.ml.torch
import numpy as np
import pytest
import mltest
import torch

# skip all tests if the ml ops were not built
pytestmark = mltest.default_marks


def nms_gpu(boxes, scores, thresh):
    """
    :param boxes: (N, 5) [x1, y1, x2, y2, ry]
    :param scores: (N)
    :param thresh:
    :return:
    """
    import open3d.ml.torch
    order = scores.sort(0, descending=True)[1]
    boxes = boxes[order].contiguous()
    keep = torch.LongTensor(boxes.size(0)).cuda()
    num_out = open3d.ml.torch.ops.nms(boxes, keep, thresh)
    return order[keep[:num_out].cuda()].contiguous()


def test_nms():
    ref_out_selected = np.array([0])
    boxes = torch.tensor([[15.0811, -7.9803, 15.6721, -6.8714, 0.5152],
                          [15.1166, -7.9261, 15.7060, -6.8137, 0.6501],
                          [15.1304, -7.8129, 15.7069, -6.8903, 0.7296],
                          [15.2050, -7.8447, 15.8311, -6.7437, 1.0506],
                          [15.1343, -7.8136, 15.7121, -6.8479, 1.0352],
                          [15.0931, -7.9552, 15.6675, -7.0056, 0.5979]],
                         dtype=torch.float32,
                         device=torch.device('cuda:0'))
    scores = torch.tensor([0.1616, 0.1556, 0.1520, 0.1501, 0.1336, 0.1298],
                          dtype=torch.float32,
                          device=torch.device('cuda:0'))
    thresh = 0.01
    out = nms_gpu(boxes, scores, thresh)
    print(out)
