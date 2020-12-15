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
import numpy as np
import open3d.ml.tf.ops as ml3d_ops
import tensorflow as tf
import os
pwd = os.path.dirname(os.path.realpath(__file__))


def test_nms_tf():
    thresh = 0.01

    boxes0 = np.load(os.path.join(pwd, "nms", "bboxes0.npy"))
    scores0 = np.load(os.path.join(pwd, "nms", "scores0.npy"))
    boxes1 = np.load(os.path.join(pwd, "nms", "bboxes0.npy"))
    scores1 = np.load(os.path.join(pwd, "nms", "scores0.npy"))
    boxes2 = np.load(os.path.join(pwd, "nms", "bboxes0.npy"))
    scores2 = np.load(os.path.join(pwd, "nms", "scores0.npy"))

    num_iters = 10000
    with tf.device("GPU:0"):
        for i in range(num_iters):
            ml3d_ops.nms(boxes0, scores0, thresh)
            ml3d_ops.nms(boxes1, scores1, thresh)
            ml3d_ops.nms(boxes2, scores2, thresh)
            if i % 100 == 0:
                print(f"{i} out of {num_iters} done.")
