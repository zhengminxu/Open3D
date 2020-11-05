import numpy as np
import torch


def test_voxelize():
    from mmdet3d.ops import Voxelization

    voxel_layer = Voxelization(
        voxel_size=[0.16, 0.16, 4],
        point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
        max_num_points=32,
        max_voxels=(16000, 40000))

    ref_out_voxels = np.load("res_voxels.npy")
    ref_out_coors = np.load("res_coors.npy")
    ref_out_num_points = np.load("res_num_points.npy")

    in_points = torch.tensor(np.load("res.npy"),
                             dtype=torch.float32,
                             device=torch.device('cuda:0'))
    out0, out1, out2 = voxel_layer(in_points)

    np.testing.assert_allclose(out0.cpu(), ref_out_voxels)
    np.testing.assert_allclose(out1.cpu(), ref_out_coors)
    np.testing.assert_allclose(out2.cpu(), ref_out_num_points)


def test_nms():
    from iou3d.iou3d_utils import nms_gpu

    ref_out_selected = np.array([0])

    in_boxes = torch.tensor([[15.0811, -7.9803, 15.6721, -6.8714, 0.5152],
                             [15.1166, -7.9261, 15.7060, -6.8137, 0.6501],
                             [15.1304, -7.8129, 15.7069, -6.8903, 0.7296],
                             [15.2050, -7.8447, 15.8311, -6.7437, 1.0506],
                             [15.1343, -7.8136, 15.7121, -6.8479, 1.0352],
                             [15.0931, -7.9552, 15.6675, -7.0056, 0.5979]],
                            dtype=torch.float32,
                            device=torch.device('cuda:0'))
    in_scores = torch.tensor([0.1616, 0.1556, 0.1520, 0.1501, 0.1336, 0.1298],
                             dtype=torch.float32,
                             device=torch.device('cuda:0'))
    in_thrs = 0.01

    out0 = nms_gpu(in_boxes, in_scores, in_thrs)
    np.testing.assert_allclose(out0.cpu(), ref_out_selected)
    print("test_nms() passes")


if __name__ == '__main__':
    # test_voxelize()
    test_nms()
