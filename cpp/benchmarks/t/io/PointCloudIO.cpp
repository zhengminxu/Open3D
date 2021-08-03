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

static const std::string file_path = "/home/rey/bedroom.ply";

static const std::string output_path = "/home/rey/";

void WriteTensorPointCloud(benchmark::State& state,
                           const std::string& file_extension) {
    std::string save_filename =
            output_path + "test_io_tpointcloud." + file_extension;

    // Creating pointcloud from a base pointcloud and saving it with required
    // extension.
    t::geometry::PointCloud base_pcd;
    t::io::ReadPointCloud(file_path, base_pcd, {"auto", false, false, false});
    t::io::WritePointCloud(save_filename, base_pcd);

    // Reading pointcloud with the `file_extension`.
    t::geometry::PointCloud pcd;
    t::io::ReadPointCloud(save_filename, pcd, {"auto", false, false, false});
    // std::remove(save_filename.c_str());

    // Warm up.
    t::io::WritePointCloud(save_filename, pcd);
    // std::remove(save_filename.c_str());

    for (auto _ : state) {
        t::io::WritePointCloud(save_filename, pcd);
        // std::remove(save_filename.c_str());
    }
}

void ReadTensorPointCloud(benchmark::State& state,
                          const std::string& file_extension) {
    std::string save_filename =
            output_path + "test_io_tpointcloud." + file_extension;

    // Creating pointcloud from a base pointcloud and saving it with required
    // extension.
    t::geometry::PointCloud base_pcd;
    t::io::ReadPointCloud(file_path, base_pcd, {"auto", false, false, false});
    t::io::WritePointCloud(save_filename, base_pcd);

    // Reading pointcloud with the `file_extension`.
    t::geometry::PointCloud pcd;
    t::io::ReadPointCloud(save_filename, pcd, {"auto", false, false, false});

    for (auto _ : state) {
        // Reading pointcloud with the `file_extension`.
        t::geometry::PointCloud pcd;
        t::io::ReadPointCloud(save_filename, pcd,
                              {"auto", false, false, false});
    }

    std::remove(save_filename.c_str());
}

void WriteLegacyPointCloud(benchmark::State& state,
                           const std::string& file_extension) {
    std::string save_filename =
            output_path + "test_io_lpointcloud." + file_extension;

    // Creating pointcloud from a base pointcloud and saving it with required
    // extension.
    open3d::geometry::PointCloud base_pcd;
    open3d::io::ReadPointCloud(file_path, base_pcd,
                               {"auto", false, false, false});
    open3d::io::WritePointCloud(save_filename, base_pcd);

    // Reading pointcloud with the `file_extension`.
    open3d::geometry::PointCloud pcd;
    open3d::io::ReadPointCloud(save_filename, pcd,
                               {"auto", false, false, false});
    std::remove(save_filename.c_str());

    // Warm up.
    open3d::io::WritePointCloud(save_filename, pcd);
    std::remove(save_filename.c_str());

    for (auto _ : state) {
        open3d::io::WritePointCloud(save_filename, pcd);
        std::remove(save_filename.c_str());
    }
}

void ReadLegacyPointCloud(benchmark::State& state,
                          const std::string& file_extension) {
    std::string save_filename =
            output_path + "test_io_lpointcloud." + file_extension;

    // Creating pointcloud from a base pointcloud and saving it with required
    // extension.
    open3d::geometry::PointCloud base_pcd;
    open3d::io::ReadPointCloud(file_path, base_pcd,
                               {"auto", false, false, false});
    open3d::io::WritePointCloud(save_filename, base_pcd);

    // Reading pointcloud with the `file_extension`.
    open3d::geometry::PointCloud pcd;
    open3d::io::ReadPointCloud(save_filename, pcd,
                               {"auto", false, false, false});

    for (auto _ : state) {
        // Reading pointcloud with the `file_extension`.
        open3d::geometry::PointCloud pcd;
        open3d::io::ReadPointCloud(save_filename, pcd,
                                   {"auto", false, false, false});
    }

    std::remove(save_filename.c_str());
}

// BENCHMARK_CAPTURE(ReadLegacyPointCloud, PLY, "ply")
//         ->Unit(benchmark::kMillisecond);

// BENCHMARK_CAPTURE(ReadTensorPointCloud, PLY, path_ply)
//         ->Unit(benchmark::kMillisecond);

// BENCHMARK_CAPTURE(ReadLegacyPointCloud, PCD, path_pcd)
//         ->Unit(benchmark::kMillisecond);

// BENCHMARK_CAPTURE(ReadTensorPointCloud, PCD, path_pcd)
//         ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(WriteLegacyPointCloud, PLY, "ply")
        ->Unit(benchmark::kMillisecond);

// BENCHMARK_CAPTURE(WriteTensorPointCloud, PLY, "ply")
//         ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(WriteLegacyPointCloud, NPZ, "npz")
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(WriteTensorPointCloud, NPZ, "npz")
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(WriteLegacyPointCloud, PCD, "pcd")
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(WriteTensorPointCloud, PCD, "pcd")
        ->Unit(benchmark::kMillisecond);

// BENCHMARK_CAPTURE(WriteLegacyPointCloud, XYZI, path_ply)
//         ->Unit(benchmark::kMillisecond);

// BENCHMARK_CAPTURE(WriteTensorPointCloud, XYZI, "xyzi")
//         ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(ReadLegacyPointCloud, PLY, "ply")
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(ReadTensorPointCloud, PLY, "ply")
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(ReadLegacyPointCloud, NPZ, "npz")
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(ReadTensorPointCloud, NPZ, "npz")
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(ReadLegacyPointCloud, PCD, "pcd")
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(ReadTensorPointCloud, PCD, "pcd")
        ->Unit(benchmark::kMillisecond);

// BENCHMARK_CAPTURE(WriteLegacyPointCloud, XYZI, path_ply)
//         ->Unit(benchmark::kMillisecond);

// BENCHMARK_CAPTURE(ReadTensorPointCloud, XYZI, "xyzi")
//         ->Unit(benchmark::kMillisecond);

}  // namespace geometry
}  // namespace t
}  // namespace open3d
