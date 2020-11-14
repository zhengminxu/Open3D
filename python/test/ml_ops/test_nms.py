# ----------------------------------------------------------------------------
# -                        Open3D: www.o3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2020 www.o3d.org
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
import numpy as np
import torch
import iou3d_cuda
import time
import open3d.ml.torch


def nms_cpu(boxes, scores, thresh):
    return o3d.ml.torch.ops.nms(boxes, scores, thresh)


def nms_cuda(boxes, scores, thresh):
    return o3d.ml.torch.ops.nms(boxes, scores, thresh)


def nms_author(boxes, scores, thresh):
    order = scores.sort(0, descending=True)[1]
    boxes = boxes[order].contiguous()
    keep = torch.LongTensor(boxes.size(0))
    num_out = iou3d_cuda.nms_gpu(boxes, keep, thresh)
    return order[keep[:num_out].cuda()].contiguous()


def run_impls(np_boxes, np_scores, thresh):
    cpu_boxes = torch.Tensor(np_boxes)
    cpu_scores = torch.Tensor(np_scores)
    cuda_boxes = cpu_boxes.cuda()
    cuda_scores = cpu_scores.cuda()

    print()
    result_cpu = nms_cpu(cpu_boxes, cpu_scores, thresh).cpu().numpy()
    result_cuda = nms_cuda(cuda_boxes, cuda_scores, thresh).cpu().numpy()
    result_author = nms_author(cuda_boxes, cuda_scores, thresh).cpu().numpy()

    np.testing.assert_equal(result_cpu, result_author)
    np.testing.assert_equal(result_cuda, result_author)


def test_rand():
    np.random.seed(0)
    num_boxes = 65
    np_boxes = np.random.rand(num_boxes, 5)
    np_scores = np.random.rand(num_boxes,)
    thresh = np.random.rand()
    run_impls(np_boxes, np_scores, thresh)
