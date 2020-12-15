import open3d as o3d
import numpy as np
import open3d.ml.tf.ops as ml3d_ops
import tensorflow as tf
import os
pwd = os.path.dirname(os.path.realpath(__file__))


def test_nms_tf():
    thresh = 0.01
    gpu_devices = tf.config.list_physical_devices('GPU')
    for gpu_device in gpu_devices:
        tf.config.experimental.set_memory_growth(gpu_device, True)

    num_iters = 100000
    with tf.device("GPU:0"):
        boxes0 = tf.constant(np.load(os.path.join(pwd, "nms", "bboxes0.npy")))
        scores0 = tf.constant(np.load(os.path.join(pwd, "nms", "scores0.npy")))
        boxes1 = tf.constant(np.load(os.path.join(pwd, "nms", "bboxes0.npy")))
        scores1 = tf.constant(np.load(os.path.join(pwd, "nms", "scores0.npy")))
        boxes2 = tf.constant(np.load(os.path.join(pwd, "nms", "bboxes0.npy")))
        scores2 = tf.constant(np.load(os.path.join(pwd, "nms", "scores0.npy")))
        for i in range(num_iters):
            ml3d_ops.nms(boxes0, scores0, thresh)
            ml3d_ops.nms(boxes1, scores1, thresh)
            ml3d_ops.nms(boxes2, scores2, thresh)
            if i % 100 == 0:
                print(f"{i} out of {num_iters} done.")
