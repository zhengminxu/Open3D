# import open3d
# import numpy as np

import numpy as np
import open3d


def align(pcloud1, pcloud2):
    pc1 = open3d.geometry.PointCloud()
    pc1.points = open3d.utility.Vector3dVector(pcloud1)
    pc2 = open3d.geometry.PointCloud()
    pc2.points = open3d.utility.Vector3dVector(pcloud2)

    voxel_size = 0.2
    normal_radius = 4 * voxel_size
    open3d.geometry.PointCloud.estimate_normals(
        pc1,
        search_param=open3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius, max_nn=30))
    open3d.geometry.PointCloud.estimate_normals(
        pc2,
        search_param=open3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius, max_nn=30))

    feature_radius = 10 * voxel_size
    pc1_fpfh = open3d.pipelines.registration.compute_fpfh_feature(
        pc1,
        open3d.geometry.KDTreeSearchParamHybrid(radius=feature_radius,
                                                max_nn=100))
    pc2_fpfh = open3d.pipelines.registration.compute_fpfh_feature(
        pc2,
        open3d.geometry.KDTreeSearchParamHybrid(radius=feature_radius,
                                                max_nn=100))

    # 3D feature based registration.
    dist_threshold = 3 * voxel_size
    result = open3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source=pc1,
        target=pc2,
        source_feature=pc1_fpfh,
        target_feature=pc2_fpfh,
        max_correspondence_distance=dist_threshold,
        estimation_method=open3d.pipelines.registration.
        TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            open3d.pipelines.registration.
            CorrespondenceCheckerBasedOnEdgeLength(0.9),
            open3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                dist_threshold)
        ],
        criteria=open3d.pipelines.registration.RANSACConvergenceCriteria(
            4000000, 500))

    result_icp2 = open3d.pipelines.registration.registration_icp(
        pc1, pc2, 0.25, result.transformation,
        open3d.pipelines.registration.TransformationEstimationPointToPlane())

    return result_icp2.transformation, result_icp2.fitness, result_icp2.inlier_rmse


if __name__ == '__main__':
    pc1 = np.load('pc1.npy')
    pc2 = np.load('pc2.npy')

    for i in range(200):
        print(i)
        align(pc1, pc2)
