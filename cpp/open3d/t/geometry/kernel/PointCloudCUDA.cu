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
#include "open3d/core/kernel/CUDALauncher.cuh"
#include "open3d/t/geometry/kernel/GeometryIndexer.h"
#include "open3d/t/geometry/kernel/GeometryMacros.h"
#include "open3d/t/geometry/kernel/PointCloud.h"
#include "open3d/t/geometry/kernel/PointCloudImpl.h"
#include "open3d/t/pipelines/kernel/SVD3x3CUDA.cuh"
#include "open3d/utility/Console.h"

namespace open3d {
namespace t {
namespace geometry {
namespace kernel {
namespace pointcloud {

void ProjectCUDA(
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

    // Pass 1: depth map
    core::kernel::CUDALauncher::LaunchGeneralKernel(
            n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
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
                float d_old = atomicExch(depth_ptr, d);
                if (d_old > 0) {
                    atomicMinf(depth_ptr, d_old);
                }
            });

    // Pass 2: color map
    if (!has_colors) return;

    NDArrayIndexer color_indexer(image_colors.value().get(), 2);
    float precision_bound = depth_scale * 1e-4;
    core::kernel::CUDALauncher::LaunchGeneralKernel(
            n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
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

                float dmap = *depth_indexer.GetDataPtr<float>(
                        static_cast<int64_t>(u), static_cast<int64_t>(v));
                float d = zc * depth_scale;
                if (d < dmap + precision_bound) {
                    uint8_t* color_ptr = color_indexer.GetDataPtr<uint8_t>(
                            static_cast<int64_t>(u), static_cast<int64_t>(v));
                    color_ptr[0] = static_cast<uint8_t>(
                            point_colors_ptr[3 * workload_idx + 0] * 255.0);
                    color_ptr[1] = static_cast<uint8_t>(
                            point_colors_ptr[3 * workload_idx + 1] * 255.0);
                    color_ptr[2] = static_cast<uint8_t>(
                            point_colors_ptr[3 * workload_idx + 2] * 255.0);
                }
            });
}

// template <typename T>
__global__ void EstimatePointWiseColorGradientCUDAKernel(
        const float* points_ptr,
        const float* normals_ptr,
        const float* colors_ptr,
        const int64_t* neighbour_indices_ptr,
        const int64_t* neighbour_row_splits_ptr,
        float* color_gradients_ptr,
        const int64_t max_nn,
        const int64_t n) {
    const int k = threadIdx.x + blockIdx.x * blockDim.x;
    if (k >= n) return;

    int64_t neighbour_start_idx = neighbour_row_splits_ptr[k];
    int64_t neighbour_count =
            neighbour_row_splits_ptr[k + 1] - neighbour_start_idx;
    neighbour_count = neighbour_count < max_nn ? neighbour_count : max_nn;

    if (neighbour_count >= 4) {
        float vt[3] = {points_ptr[3 * k + 0], points_ptr[3 * k + 1],
                       points_ptr[3 * k + 2]};

        float nt[3] = {normals_ptr[3 * k + 0], normals_ptr[3 * k + 1],
                       normals_ptr[3 * k + 2]};
        float it = (colors_ptr[3 * k + 0] + colors_ptr[3 * k + 1] +
                    colors_ptr[3 * k + 2]) /
                   3.0;

        float AtA_local[9] = {0};
        float Atb_local[3] = {0};

        // approximate image gradient of vt's tangential plane
        // projection (p') of a point p on a plane defined by normal n,
        // where o is the closest point to p on the plane, is given by:
        // p' = p - [(p - o).dot(n)] * n
        // p' = p - [(p.dot(n) - s)] * n [where s = o.dot(n)]
        // Computing the scalar s.
        float s = vt[0] * nt[0] + vt[1] * nt[1] + vt[2] * nt[2];

        int i = 1;
        for (i = 1; i < neighbour_count; i++) {
            int64_t neighbour_idx =
                    neighbour_indices_ptr[neighbour_start_idx + i];

            if (neighbour_idx == -1) {
                break;
            }

            float vt_adj[3] = {points_ptr[3 * neighbour_idx + 0],
                               points_ptr[3 * neighbour_idx + 1],
                               points_ptr[3 * neighbour_idx + 2]};

            // p' = p - d * n [where d = p.dot(n) - s]
            // Computing the scalar d.
            float d = vt_adj[0] * nt[0] + vt_adj[1] * nt[1] +
                      vt_adj[2] * nt[2] - s;

            // Computing the p' (projection of the point).
            float vt_proj[3] = {vt_adj[0] - d * nt[0], vt_adj[1] - d * nt[1],
                                vt_adj[2] - d * nt[2]};

            float it_adj = (colors_ptr[3 * neighbour_idx + 0] +
                            colors_ptr[3 * neighbour_idx + 1] +
                            colors_ptr[3 * neighbour_idx + 2]) /
                           3.0;

            float A[3] = {vt_proj[0] - vt[0], vt_proj[1] - vt[1],
                          vt_proj[2] - vt[2]};
            float b = it_adj - it;

            AtA_local[0] += A[0] * A[0];
            AtA_local[1] += A[1] * A[0];
            AtA_local[2] += A[2] * A[0];
            AtA_local[4] += A[1] * A[1];
            AtA_local[5] += A[2] * A[1];
            AtA_local[8] += A[2] * A[2];

            Atb_local[0] += A[0] * b;
            Atb_local[1] += A[1] * b;
            Atb_local[2] += A[2] * b;
        }

        // Orthogonal constraint.
        float A[3] = {(i - 1) * nt[0], (i - 1) * nt[1], (i - 1) * nt[2]};

        AtA_local[0] += A[0] * A[0];
        AtA_local[1] += A[0] * A[1];
        AtA_local[2] += A[0] * A[2];
        AtA_local[4] += A[1] * A[1];
        AtA_local[5] += A[1] * A[2];
        AtA_local[8] += A[2] * A[2];

        // Symmetry.
        AtA_local[3] = AtA_local[1];
        AtA_local[6] = AtA_local[2];
        AtA_local[7] = AtA_local[5];

        solve_svd3x3(AtA_local[0], AtA_local[1], AtA_local[2], AtA_local[3],
                     AtA_local[4], AtA_local[5], AtA_local[6], AtA_local[7],
                     AtA_local[8], Atb_local[0], Atb_local[1], Atb_local[2],
                     color_gradients_ptr[3 * k + 0],
                     color_gradients_ptr[3 * k + 1],
                     color_gradients_ptr[3 * k + 2]);

        // DEBUG KERNEL.
        // printf("\n AtA: \n %f, %f, %f \n %f, %f, %f \n %f, %f, %f",
        //        AtA_local[0], AtA_local[1], AtA_local[2], AtA_local[3],
        //        AtA_local[4], AtA_local[5], AtA_local[6], AtA_local[7],
        //        AtA_local[8]);
        // printf("\n Atb: \n %f, %f, %f", Atb_local[0], Atb_local[1],
        //        Atb_local[2]);
        // printf("\n X: \n %f, %f, %f", color_gradients_ptr[3 * k + 0],
        //        color_gradients_ptr[3 * k + 1],
        // color_gradients_ptr[3 * k + 2]);

    } else {
        color_gradients_ptr[3 * k] = 0;
        color_gradients_ptr[3 * k + 1] = 0;
        color_gradients_ptr[3 * k + 2] = 0;
    }
}

void EstimatePointWiseColorGradientCUDA(
        const core::Tensor& points,
        const core::Tensor& normals,
        const core::Tensor& colors,
        const core::Tensor& neighbour_indices,
        const core::Tensor& neighbour_row_splits,
        core::Tensor& color_gradients,
        const int64_t& max_nn) {
    // core::Dtype dtype = points.GetDtype();

    const int64_t n = points.GetLength();
    const dim3 blocks((n + 1024 - 1) / 1024);
    const dim3 threads(1024);

    EstimatePointWiseColorGradientCUDAKernel<<<blocks, threads>>>(
            points.GetDataPtr<float>(), normals.GetDataPtr<float>(),
            colors.GetDataPtr<float>(), neighbour_indices.GetDataPtr<int64_t>(),
            neighbour_row_splits.GetDataPtr<int64_t>(),
            color_gradients.GetDataPtr<float>(), max_nn, n);

    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
}

}  // namespace pointcloud
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
