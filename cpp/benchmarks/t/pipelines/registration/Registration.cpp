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

#include "open3d/t/pipelines/registration/Registration.h"

#include <benchmark/benchmark.h>

#include "open3d/core/nns/NearestNeighborSearch.h"
#include "open3d/t/io/PointCloudIO.h"
#include "open3d/t/pipelines/registration/TransformationEstimation.h"

// Testing parameters:
// Filename for pointcloud registration data.
static const std::string source_pointcloud_filename =
        std::string(TEST_DATA_DIR) + "/ICP/cloud_bin_0.pcd";
static const std::string target_pointcloud_filename =
        std::string(TEST_DATA_DIR) + "/ICP/cloud_bin_1.pcd";

static const std::string source_colored_pcd_filename =
        std::string(TEST_DATA_DIR) + "/ColoredICP/frag_115.ply";
static const std::string target_colored_pcd_filename =
        std::string(TEST_DATA_DIR) + "/ColoredICP/frag_116.ply";

static const double voxel_downsampling_factor = 0.05;

// ICP ConvergenceCriteria.
static const double relative_fitness = 1e-6;
static const double relative_rmse = 1e-6;
static const int max_iterations = 5;

// NNS parameter.
static const double max_correspondence_distance = 0.075;

// Initial transformation guess for registation.
static const std::vector<float> initial_transform_flat{
        0.862, 0.011, -0.507, 0.5,  -0.139, 0.967, -0.215, 0.7,
        0.487, 0.255, 0.835,  -1.4, 0.0,    0.0,   0.0,    1.0};

namespace open3d {
namespace t {
namespace pipelines {
namespace registration {

static std::tuple<geometry::PointCloud, geometry::PointCloud>
LoadTensorPointCloudFromFile(const std::string& source_pointcloud_filename,
                             const std::string& target_pointcloud_filename,
                             const double voxel_downsample_factor,
                             const core::Dtype& dtype,
                             const core::Device& device) {
    geometry::PointCloud source, target;

    io::ReadPointCloud(source_pointcloud_filename, source,
                       {"auto", false, false, true});
    io::ReadPointCloud(target_pointcloud_filename, target,
                       {"auto", false, false, true});

    // Eliminates the case of impractical values (including negative).
    if (voxel_downsample_factor > 0.001) {
        source = source.VoxelDownSample(voxel_downsample_factor);
        target = target.VoxelDownSample(voxel_downsample_factor);
    }

    geometry::PointCloud source_device(device), target_device(device);

    source_device.SetPoints(source.GetPoints().To(device, dtype));
    if (source.HasPointColors()) {
        if (source.GetPointColors().GetDtype() == core::Dtype::UInt8) {
            source_device.SetPointColors(
                    source.GetPointColors().To(device, dtype).Div(255.0));
        } else if (source.GetPointColors().GetDtype() == core::Dtype::Float32 ||
                   source.GetPointColors().GetDtype() == core::Dtype::Float64) {
            source_device.SetPointColors(
                    source.GetPointColors().To(device, dtype));
        } else {
            utility::LogError(
                    " Only UInt8, Float32, Float64 type colors supported.");
        }
    }

    target_device.SetPoints(target.GetPoints().To(device, dtype));
    target_device.SetPointNormals(target.GetPointNormals().To(device, dtype));
    if (target.HasPointColors()) {
        if (target.GetPointColors().GetDtype() == core::Dtype::UInt8) {
            target_device.SetPointColors(
                    target.GetPointColors().To(device, dtype).Div(255.0));
        } else if (target.GetPointColors().GetDtype() == core::Dtype::Float32 ||
                   target.GetPointColors().GetDtype() == core::Dtype::Float64) {
            target_device.SetPointColors(
                    target.GetPointColors().To(device, dtype));
        } else {
            utility::LogError(
                    " Only UInt8, Float32, Float64 type colors supported.");
        }
    }

    return std::make_tuple(source_device, target_device);
}

static void BenchmarkRegistrationICP(benchmark::State& state,
                                     const core::Device& device,
                                     const core::Dtype& dtype,
                                     const TransformationEstimationType& type) {
    std::string source_pcd_filename;
    std::string target_pcd_filename;

    core::Tensor init_trans;

    std::shared_ptr<TransformationEstimation> estimation;
    if (type == TransformationEstimationType::PointToPlane) {
        estimation = std::make_shared<TransformationEstimationPointToPlane>();
        source_pcd_filename = source_pointcloud_filename;
        target_pcd_filename = target_pointcloud_filename;

        init_trans = core::Tensor(initial_transform_flat, {4, 4},
                                  core::Dtype::Float32, device)
                             .To(dtype);

    } else if (type == TransformationEstimationType::PointToPoint) {
        estimation = std::make_shared<TransformationEstimationPointToPoint>();
        source_pcd_filename = source_colored_pcd_filename;
        target_pcd_filename = target_colored_pcd_filename;
        init_trans = core::Tensor::Eye(4, core::Dtype::Float64,
                                       core::Device("CPU:0"));
        // init_trans = core::Tensor(initial_transform_flat, {4, 4},
        //                           core::Dtype::Float32, device)
        //                      .To(dtype);
    } else if (type == TransformationEstimationType::ColoredICP) {
        estimation = std::make_shared<TransformationEstimationColoredICP>();
        source_pcd_filename = source_colored_pcd_filename;
        target_pcd_filename = target_colored_pcd_filename;

        init_trans = core::Tensor::Eye(4, core::Dtype::Float64,
                                       core::Device("CPU:0"));
    }

    geometry::PointCloud source(device), target(device);

    std::tie(source, target) = LoadTensorPointCloudFromFile(
            source_pcd_filename, target_pcd_filename, voxel_downsampling_factor,
            dtype, device);

    RegistrationResult reg_result(init_trans);

    // Warm up.
    reg_result = RegistrationICP(
            source, target, max_correspondence_distance, init_trans,
            *estimation,
            ICPConvergenceCriteria(relative_fitness, relative_rmse,
                                   max_iterations));
    for (auto _ : state) {
        reg_result = RegistrationICP(
                source, target, max_correspondence_distance, init_trans,
                *estimation,
                ICPConvergenceCriteria(relative_fitness, relative_rmse,
                                       max_iterations));
    }

    utility::LogDebug(" PointCloud Size: Source: {}  Target: {}",
                      source.GetPoints().GetShape().ToString(),
                      target.GetPoints().GetShape().ToString());
    utility::LogDebug(" Max iterations: {}, Max_correspondence_distance : {}",
                      max_iterations, max_correspondence_distance);
}

BENCHMARK_CAPTURE(BenchmarkRegistrationICP,
                  PointToPlane / CPU32,
                  core::Device("CPU:0"),
                  core::Dtype::Float32,
                  TransformationEstimationType::PointToPlane)
        ->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(BenchmarkRegistrationICP,
                  PointToPlane / CUDA32,
                  core::Device("CUDA:0"),
                  core::Dtype::Float32,
                  TransformationEstimationType::PointToPlane)
        ->Unit(benchmark::kMillisecond);
#endif

BENCHMARK_CAPTURE(BenchmarkRegistrationICP,
                  PointToPoint / CPU32,
                  core::Device("CPU:0"),
                  core::Dtype::Float32,
                  TransformationEstimationType::PointToPoint)
        ->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(BenchmarkRegistrationICP,
                  PointToPoint / CUDA32,
                  core::Device("CUDA:0"),
                  core::Dtype::Float32,
                  TransformationEstimationType::PointToPoint)
        ->Unit(benchmark::kMillisecond);
#endif

BENCHMARK_CAPTURE(BenchmarkRegistrationICP,
                  ColoredICP / CPU32,
                  core::Device("CPU:0"),
                  core::Dtype::Float32,
                  TransformationEstimationType::ColoredICP)
        ->Unit(benchmark::kMillisecond);

// #ifdef BUILD_CUDA_MODULE
// BENCHMARK_CAPTURE(BenchmarkRegistrationICP,
//                   ColoredICP / CUDA32,
//                   core::Device("CUDA:0"),
//                   core::Dtype::Float32,
//                   TransformationEstimationType::ColoredICP)
//         ->Unit(benchmark::kMillisecond);
// #endif

BENCHMARK_CAPTURE(BenchmarkRegistrationICP,
                  PointToPlane / CPU64,
                  core::Device("CPU:0"),
                  core::Dtype::Float64,
                  TransformationEstimationType::PointToPlane)
        ->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(BenchmarkRegistrationICP,
                  PointToPlane / CUDA64,
                  core::Device("CUDA:0"),
                  core::Dtype::Float64,
                  TransformationEstimationType::PointToPlane)
        ->Unit(benchmark::kMillisecond);
#endif

BENCHMARK_CAPTURE(BenchmarkRegistrationICP,
                  PointToPoint / CPU64,
                  core::Device("CPU:0"),
                  core::Dtype::Float64,
                  TransformationEstimationType::PointToPoint)
        ->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(BenchmarkRegistrationICP,
                  PointToPoint / CUDA64,
                  core::Device("CUDA:0"),
                  core::Dtype::Float64,
                  TransformationEstimationType::PointToPoint)
        ->Unit(benchmark::kMillisecond);
#endif

}  // namespace registration
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
