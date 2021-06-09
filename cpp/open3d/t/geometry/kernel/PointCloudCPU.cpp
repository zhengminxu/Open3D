// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "open3d/core/Dispatch.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/kernel/CPULauncher.h"
#include "open3d/t/geometry/kernel/GeometryIndexer.h"
#include "open3d/t/geometry/kernel/GeometryMacros.h"
#include "open3d/t/geometry/kernel/PointCloud.h"
#include "open3d/t/geometry/kernel/PointCloudImpl.h"
#include "open3d/utility/Console.h"
#include "open3d/utility/Eigen.h"

namespace open3d {
namespace t {
namespace geometry {
namespace kernel {
namespace pointcloud {

void ProjectCPU(
        core::Tensor& depth,
        utility::optional<std::reference_wrapper<core::Tensor>> image_colors,
        const core::Tensor& points,
        utility::optional<std::reference_wrapper<const core::Tensor>> colors,
        const core::Tensor& intrinsics,
        const core::Tensor& extrinsics,
        float depth_scale,
        float depth_max) {
    const bool has_colors = image_colors.has_value();

    int64_t n = points.GetLength();

    const float* points_ptr = points.GetDataPtr<float>();
    const float* point_colors_ptr =
            has_colors ? colors.value().get().GetDataPtr<float>() : nullptr;

    TransformIndexer transform_indexer(intrinsics, extrinsics, 1.0f);
    NDArrayIndexer depth_indexer(depth, 2);

    NDArrayIndexer color_indexer;
    if (has_colors) {
        color_indexer = NDArrayIndexer(image_colors.value().get(), 2);
    }

    core::kernel::CPULauncher::LaunchGeneralKernel(
            n, [&](int64_t workload_idx) {
                float x = points_ptr[3 * workload_idx + 0];
                float y = points_ptr[3 * workload_idx + 1];
                float z = points_ptr[3 * workload_idx + 2];

                // coordinate in camera (in voxel -> in meter)
                float xc, yc, zc, u, v;
                transform_indexer.RigidTransform(x, y, z, &xc, &yc, &zc);

                // coordinate in image (in pixel)
                transform_indexer.Project(xc, yc, zc, &u, &v);
                if (!depth_indexer.InBoundary(u, v) || zc <= 0 ||
                    zc > depth_max) {
                    return;
                }

                float* depth_ptr = depth_indexer.GetDataPtr<float>(
                        static_cast<int64_t>(u), static_cast<int64_t>(v));
                float d = zc * depth_scale;
#pragma omp critical
                {
                    if (*depth_ptr == 0 || *depth_ptr >= d) {
                        *depth_ptr = d;

                        if (has_colors) {
                            uint8_t* color_ptr =
                                    color_indexer.GetDataPtr<uint8_t>(
                                            static_cast<int64_t>(u),
                                            static_cast<int64_t>(v));

                            color_ptr[0] = static_cast<uint8_t>(
                                    point_colors_ptr[3 * workload_idx + 0] *
                                    255.0);
                            color_ptr[1] = static_cast<uint8_t>(
                                    point_colors_ptr[3 * workload_idx + 1] *
                                    255.0);
                            color_ptr[2] = static_cast<uint8_t>(
                                    point_colors_ptr[3 * workload_idx + 2] *
                                    255.0);
                        }
                    }
                }
            });
}

void EstimatePointWiseColorGradientCPU(const core::Tensor& points,
                                       const core::Tensor& normals,
                                       const core::Tensor& colors,
                                       const core::Tensor neighbour_indices,
                                       core::Tensor& color_gradients,
                                       const int min_knn_threshold /*= 4*/) {
    int64_t n = points.GetLength();
    int64_t max_knn = neighbour_indices.GetShape()[1];

    const float* points_ptr = points.GetDataPtr<float>();
    const float* normals_ptr = normals.GetDataPtr<float>();
    const float* colors_ptr = colors.GetDataPtr<float>();
    const int64_t* neighbour_indices_ptr =
            neighbour_indices.GetDataPtr<int64_t>();

    float* color_gradients_ptr = color_gradients.GetDataPtr<float>();

#pragma omp parallel for schedule(static)
    for (int k = 0; k < static_cast<int>(n); k++) {
        const Eigen::Vector3d& vt = {points_ptr[3 * k + 0],
                                     points_ptr[3 * k + 1],
                                     points_ptr[3 * k + 2]};
        const Eigen::Vector3d& nt = {normals_ptr[3 * k + 0],
                                     normals_ptr[3 * k + 1],
                                     normals_ptr[3 * k + 2]};
        double it = (colors_ptr[3 * k + 0] + colors_ptr[3 * k + 1] +
                     colors_ptr[3 * k + 2]) /
                    3.0;

        // approximate image gradient of vt's tangential plane
        // size_t nn = point_idx.size();
        Eigen::MatrixXd A(max_knn, 3);
        Eigen::MatrixXd b(max_knn, 1);
        A.setZero();
        b.setZero();

        int i = 1;
        for (i = 1; i < max_knn; i++) {
            int neighbour_idx = neighbour_indices_ptr[max_knn * k + i];

            if (neighbour_idx == -1) {
                break;
            }

            Eigen::Vector3d vt_adj = {points_ptr[3 * neighbour_idx + 0],
                                      points_ptr[3 * neighbour_idx + 1],
                                      points_ptr[3 * neighbour_idx + 2]};

            // projection (p') of a point p on a plane defined by normal n,
            // where o is the closest point to p on the plane, is given by:
            // p' = p - [(p - o).dot(n)] * n
            Eigen::Vector3d vt_proj = vt_adj - (vt_adj - vt).dot(nt) * nt;
            double it_adj = (colors_ptr[3 * neighbour_idx + 0] +
                             colors_ptr[3 * neighbour_idx + 1] +
                             colors_ptr[3 * neighbour_idx + 2]) /
                            3.0;
            A(i - 1, 0) = (vt_proj(0) - vt(0));
            A(i - 1, 1) = (vt_proj(1) - vt(1));
            A(i - 1, 2) = (vt_proj(2) - vt(2));
            b(i - 1, 0) = (it_adj - it);
        }

        // adds orthogonal constraint
        A(i, 0) = (i)*nt(0);
        A(i, 1) = (i)*nt(1);
        A(i, 2) = (i)*nt(2);
        b(i, 0) = 0;

        A.resize(i + 1, 3);
        b.resize(i + 1, 1);

        // solving linear equation
        bool is_success = false;
        Eigen::MatrixXd x;
        std::tie(is_success, x) = utility::SolveLinearSystemPSD(
                A.transpose() * A, A.transpose() * b);

        if (is_success) {
            color_gradients_ptr[3 * k] = x(0, 0);
            color_gradients_ptr[3 * k + 1] = x(1, 0);
            color_gradients_ptr[3 * k + 2] = x(2, 0);
        }
    }
}

// core::kernel::CPULauncher::LaunchGeneralKernel(
//         n, [&](int64_t workload_idx) {
//             float color_t = (colors_ptr[3 * workload_idx + 0] +
//                              colors_ptr[3 * workload_idx + 1] +
//                              colors_ptr[3 * workload_idx + 2]) /
//                             3.0;

//             int neighbours_count = 0;

//             // TODO: try binary search to find -1.
//             // TODO: Tensor::Find(scalar), to get indices for first
//             // occurances.
//             for (int i = 0; i < max_knn; i++) {
//                 if (neighbour_indices_ptr[max_knn * workload_idx + i] ==
//                     -1) {
//                     break;
//                 }
//                 neighbours_count++;
//             }

//             float A[3 * neighbours_count] = {0};
//             float b[neighbours_count] = {0};
//             // float AtA[9] = {0};
//             // float Atb[3] = {0};

//             // projection (p') of a point p on a plane defined by normal n,
//             // where o is the closest point to p on the plane (point on
//             // plane where the line from the point p perpendicual to plane,
//             // meets the plane) is defined by the equation:
//             // p' = p - [(p - o).dot(n)] * n
//             // => p' = p - [p.dot(n) + d], where d = - o.dot(n)
//             float d = -(points_ptr[3 * workload_idx + 0] *
//                                 normals_ptr[3 * workload_idx + 0] +
//                         points_ptr[3 * workload_idx + 1] *
//                                 normals_ptr[3 * workload_idx + 1] +
//                         points_ptr[3 * workload_idx + 2] *
//                                 normals_ptr[3 * workload_idx + 2]);

//             for (int i = 1; i < neighbours_count; i++) {
//                 // float AtA_local[9] = {0};

//                 int neighbour_idx =
//                         neighbour_indices_ptr[max_knn * workload_idx + i];
//                 float vt_adj[3] = {points_ptr[3 * neighbour_idx + 0],
//                                    points_ptr[3 * neighbour_idx + 1],
//                                    points_ptr[3 * neighbour_idx + 2]};

//                 // r = vt_adj.dot(n) + d
//                 float r = (vt_adj[0] * normals_ptr[3 * workload_idx + 0] +
//                            vt_adj[1] * normals_ptr[3 * workload_idx + 1] +
//                            vt_adj[2] * normals_ptr[3 * workload_idx + 2]) +
//                           d;

//                 //  vt_proj = vt_adj - r * n
//                 float vt_proj[3] = {
//                         vt_adj[0] - r * normals_ptr[3 * workload_idx + 0],
//                         vt_adj[1] - r * normals_ptr[3 * workload_idx + 1],
//                         vt_adj[2] - r * normals_ptr[3 * workload_idx + 2]};

//                 float it_adj = (colors_ptr[3 * neighbour_idx + 0] +
//                                 colors_ptr[3 * neighbour_idx + 1] +
//                                 colors_ptr[3 * neighbour_idx + 2]) /
//                                3.0;

//                 A[3 * (i - 1) + 0] =
//                         vt_proj[0] - points_ptr[3 * workload_idx + 0];
//                 A[3 * (i - 1) + 1] =
//                         vt_proj[1] - points_ptr[3 * workload_idx + 1];
//                 A[3 * (i - 1) + 2] =
//                         vt_proj[2] - points_ptr[3 * workload_idx + 2];
//                 b[i - 1] = it_adj - color_t;
//             }
//             A[3 * (neighbours_count - 1) + 0] =
//                     (neighbours_count - 1) *
//                     normals_ptr[3 * workload_idx + 0];
//             A[3 * (neighbours_count - 1) + 1] =
//                     (neighbours_count - 1) *
//                     normals_ptr[3 * workload_idx + 1];
//             A[3 * (neighbours_count - 1) + 2] =
//                     (neighbours_count - 1) *
//                     normals_ptr[3 * workload_idx + 2];
//             b[neighbours_count - 1] = 0;
//         });

}  // namespace pointcloud
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
