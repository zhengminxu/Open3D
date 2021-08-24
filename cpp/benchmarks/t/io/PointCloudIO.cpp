// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include "open3d/io/PointCloudIO.h"

#include <benchmark/benchmark.h>

#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/io/PointCloudIO.h"

namespace open3d {
namespace t {
namespace geometry {

static const std::string path_pcd = "/home/rey/dataset/pointcloud/bedroom.ply";

static const std::string output_tpcd = "/home/rey/test_tpcd.pcd";

// std::string(TEST_DATA_DIR) + "/fragment.pcd";
static const std::string path_ply =
        std::string(TEST_DATA_DIR) + "/fragment.ply";

void ReadTensorPointCloud(benchmark::State& state,
                          const std::string& file_path) {
    t::geometry::PointCloud pcd;
    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);
    t::io::ReadPointCloud(file_path, pcd, {"auto", false, false, false});
    utility::SetVerbosityLevel(utility::VerbosityLevel::Info);

    for (auto _ : state) {
        t::io::ReadPointCloud(file_path, pcd, {"auto", false, false, false});
    }
}

void ReadLegacyPointCloud(benchmark::State& state,
                          const std::string& file_path) {
    open3d::geometry::PointCloud pcd;
    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);
    open3d::io::ReadPointCloud(file_path, pcd, {"auto", false, false, false});
    utility::SetVerbosityLevel(utility::VerbosityLevel::Info);

    for (auto _ : state) {
        open3d::io::ReadPointCloud(file_path, pcd,
                                   {"auto", false, false, false});
    }
}

void WriteTensorPointCloud(benchmark::State& state,
                           const std::string& file_path) {
    t::geometry::PointCloud pcd;
    t::io::ReadPointCloud(file_path, pcd, {"auto", false, false, false});

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);
    t::io::WritePointCloud(
            output_tpcd, pcd,
            open3d::io::WritePointCloudOption(false, true, false, {}));

    t::io::ReadPointCloud(output_tpcd, pcd, {"auto", false, false, false});    
    utility::LogInfo("points : \n{}", pcd.GetPointPositions().ToString());

    for (auto _ : state) {
        std::string filename_loop = "t_pcd.pcd";
        t::io::WritePointCloud(
                filename_loop, pcd,
                open3d::io::WritePointCloudOption(false, true, false, {}));
    }
}

void WriteLegacyPointCloud(benchmark::State& state,
                           const std::string& file_path) {
    open3d::geometry::PointCloud pcd;
    open3d::io::ReadPointCloud(file_path, pcd, {"auto", false, false, false});

    open3d::io::WritePointCloud(
            "l_pcd_test.pcd", pcd,
            open3d::io::WritePointCloudOption(false, true, false, {}));

    for (auto _ : state) {
        std::string filename_loop = "l_pcd.pcd";
        open3d::io::WritePointCloud(
                filename_loop, pcd,
                open3d::io::WritePointCloudOption(false, true, false, {}));
    }
}

BENCHMARK_CAPTURE(ReadTensorPointCloud, PLY, path_ply)
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(ReadLegacyPointCloud, PLY, path_ply)
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(ReadTensorPointCloud, PCD, path_pcd)
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(ReadLegacyPointCloud, PCD, path_pcd)
        ->Unit(benchmark::kMillisecond);

// BENCHMARK_CAPTURE(WriteTensorPointCloud, PLY, path_ply)
//         ->Unit(benchmark::kMillisecond);

// BENCHMARK_CAPTURE(WriteLegacyPointCloud, PLY, path_ply)
//         ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(WriteLegacyPointCloud, PCD, path_pcd)
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(WriteTensorPointCloud, PCD, path_pcd)
        ->Unit(benchmark::kMillisecond);

}  // namespace geometry
}  // namespace t
}  // namespace open3d
