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

#include "open3d/t/io/file_format/FileXYZIRGB.h"

#include <cstdio>

#include "open3d/core/Dtype.h"
#include "open3d/core/Tensor.h"
#include "open3d/utility/Helper.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace t {
namespace io {

bool ReadXYZIRGB(geometry::PointCloud &pointcloud,
                 const int64_t &num_points,
                 utility::filesystem::CFile &file,
                 const char *line_buffer,
                 utility::CountingProgressReporter &reporter,
                 const core::Dtype &color_dtype) {
    pointcloud.SetPoints(core::Tensor({num_points, 3}, core::Float64));
    auto points_ptr = pointcloud.GetPoints().GetDataPtr<double>();
    pointcloud.SetPointAttr("intensities",
                            core::Tensor({num_points, 1}, core::Float64));
    auto intensities_ptr =
            pointcloud.GetPointAttr("intensities").GetDataPtr<double>();
    pointcloud.SetPointColors(core::Tensor({num_points, 3}, color_dtype));

    if (color_dtype == core::UInt8) {
        auto colors_ptr = pointcloud.GetPointColors().GetDataPtr<uint8_t>();

        int64_t idx = 0;
        while (idx < num_points && (line_buffer = file.ReadLine())) {
            double x, y, z, i;
            int r, g, b;
            // X Y Z I R G B.
            if (sscanf(line_buffer, "%lf %lf %lf %lf %d %d %d", &x, &y, &z, &i,
                       &r, &g, &b) == 7) {
                points_ptr[3 * idx + 0] = x;
                points_ptr[3 * idx + 1] = y;
                points_ptr[3 * idx + 2] = z;
                intensities_ptr[idx] = i;
                colors_ptr[3 * idx + 0] = r;
                colors_ptr[3 * idx + 1] = g;
                colors_ptr[3 * idx + 2] = b;
            } else {
                utility::LogWarning("Read failed at line: {}", line_buffer);
                return false;
            }
            idx++;
            if (idx % 1000 == 0) {
                reporter.Update(idx);
            }
        }
        return true;
    } else if (color_dtype == core::Float64) {
        auto colors_ptr = pointcloud.GetPointColors().GetDataPtr<double>();

        int64_t idx = 0;
        while (idx < num_points && (line_buffer = file.ReadLine())) {
            double x, y, z, i, r, g, b;
            // X Y Z I R G B.
            if (sscanf(line_buffer, "%lf %lf %lf %lf %lf %lf %lf", &x, &y, &z,
                       &i, &r, &g, &b) == 7) {
                points_ptr[3 * idx + 0] = x;
                points_ptr[3 * idx + 1] = y;
                points_ptr[3 * idx + 2] = z;
                intensities_ptr[idx] = i;
                colors_ptr[3 * idx + 0] = r;
                colors_ptr[3 * idx + 1] = g;
                colors_ptr[3 * idx + 2] = b;
            } else {
                utility::LogWarning("Read failed at line: {}", line_buffer);
                return false;
            }
            idx++;
            if (idx % 1000 == 0) {
                reporter.Update(idx);
            }
        }
        return true;
    } else {
        utility::LogError(
                "Read failed. Only Float64 and UInt8 type color attribute "
                "supported.");
        return false;
    }
    return false;
}

bool ReadXYZRGB(geometry::PointCloud &pointcloud,
                const int64_t &num_points,
                utility::filesystem::CFile &file,
                const char *line_buffer,
                utility::CountingProgressReporter &reporter,
                const core::Dtype &color_dtype) {
    pointcloud.SetPoints(core::Tensor({num_points, 3}, core::Float64));
    auto points_ptr = pointcloud.GetPoints().GetDataPtr<double>();
    pointcloud.SetPointColors(core::Tensor({num_points, 3}, color_dtype));

    if (color_dtype == core::UInt8) {
        auto colors_ptr = pointcloud.GetPointColors().GetDataPtr<uint8_t>();

        int64_t idx = 0;
        while (idx < num_points && (line_buffer = file.ReadLine())) {
            double x, y, z;
            int r, g, b;
            // X Y Z R G B.
            if (sscanf(line_buffer, "%lf %lf %lf %d %d %d", &x, &y, &z, &r, &g,
                       &b) == 6) {
                points_ptr[3 * idx + 0] = x;
                points_ptr[3 * idx + 1] = y;
                points_ptr[3 * idx + 2] = z;
                colors_ptr[3 * idx + 0] = r;
                colors_ptr[3 * idx + 1] = g;
                colors_ptr[3 * idx + 2] = b;
            } else {
                utility::LogWarning("Read failed at line: {}", line_buffer);
                return false;
            }
            idx++;
            if (idx % 1000 == 0) {
                reporter.Update(idx);
            }
        }
        return true;
    } else if (color_dtype == core::Float64) {
        auto colors_ptr = pointcloud.GetPointColors().GetDataPtr<double>();

        int64_t idx = 0;
        while (idx < num_points && (line_buffer = file.ReadLine())) {
            double x, y, z, r, g, b;
            // X Y Z R G B.
            if (sscanf(line_buffer, "%lf %lf %lf %lf %lf %lf", &x, &y, &z, &r,
                       &g, &b) == 6) {
                points_ptr[3 * idx + 0] = x;
                points_ptr[3 * idx + 1] = y;
                points_ptr[3 * idx + 2] = z;
                colors_ptr[3 * idx + 0] = r;
                colors_ptr[3 * idx + 1] = g;
                colors_ptr[3 * idx + 2] = b;
            } else {
                utility::LogWarning("Read failed at line: {}", line_buffer);
                return false;
            }
            idx++;
            if (idx % 1000 == 0) {
                reporter.Update(idx);
            }
        }
        return true;
    } else {
        utility::LogError(
                "Read failed. Only Float64 and UInt8 type color attribute "
                "supported.");
        return false;
    }
    return false;
}

bool ReadXYZI(geometry::PointCloud &pointcloud,
              const int64_t &num_points,
              utility::filesystem::CFile &file,
              const char *line_buffer,
              utility::CountingProgressReporter &reporter) {
    pointcloud.SetPoints(core::Tensor({num_points, 3}, core::Float64));
    auto points_ptr = pointcloud.GetPoints().GetDataPtr<double>();
    pointcloud.SetPointAttr("intensities",
                            core::Tensor({num_points, 1}, core::Float64));
    auto intensities_ptr =
            pointcloud.GetPointAttr("intensities").GetDataPtr<double>();

    int64_t idx = 0;
    while (idx < num_points && (line_buffer = file.ReadLine())) {
        double x, y, z, i;
        // X Y Z I.
        if (sscanf(line_buffer, "%lf %lf %lf %lf", &x, &y, &z, &i) == 4) {
            points_ptr[3 * idx + 0] = x;
            points_ptr[3 * idx + 1] = y;
            points_ptr[3 * idx + 2] = z;
            intensities_ptr[idx] = i;
        } else {
            utility::LogWarning("Read failed at line: {}", line_buffer);
            return false;
        }
        idx++;
        if (idx % 1000 == 0) {
            reporter.Update(idx);
        }
    }
    return true;
}

bool ReadXYZN(geometry::PointCloud &pointcloud,
              const int64_t &num_points,
              utility::filesystem::CFile &file,
              const char *line_buffer,
              utility::CountingProgressReporter &reporter) {
    pointcloud.SetPoints(core::Tensor({num_points, 3}, core::Float64));
    auto points_ptr = pointcloud.GetPoints().GetDataPtr<double>();
    pointcloud.SetPointNormals(core::Tensor({num_points, 3}, core::Float64));
    auto normals_ptr = pointcloud.GetPointNormals().GetDataPtr<double>();

    int64_t idx = 0;
    while (idx < num_points && (line_buffer = file.ReadLine())) {
        double x, y, z, nx, ny, nz;
        // X Y Z NX NY NZ.
        if (sscanf(line_buffer, "%lf %lf %lf %lf %lf %lf", &x, &y, &z, &nx, &ny,
                   &nz) == 6) {
            points_ptr[3 * idx + 0] = x;
            points_ptr[3 * idx + 1] = y;
            points_ptr[3 * idx + 2] = z;
            normals_ptr[3 * idx + 0] = nx;
            normals_ptr[3 * idx + 1] = ny;
            normals_ptr[3 * idx + 2] = nz;
        } else {
            utility::LogWarning("Read failed at line: {}", line_buffer);
            return false;
        }
        idx++;
        if (idx % 1000 == 0) {
            reporter.Update(idx);
        }
    }
    return true;
}

bool ReadXYZ(geometry::PointCloud &pointcloud,
             const int64_t &num_points,
             utility::filesystem::CFile &file,
             const char *line_buffer,
             utility::CountingProgressReporter &reporter) {
    pointcloud.SetPoints(core::Tensor({num_points, 3}, core::Float64));
    auto points_ptr = pointcloud.GetPoints().GetDataPtr<double>();
    pointcloud.SetPointAttr("intensities",
                            core::Tensor({num_points, 1}, core::Float64));

    int64_t idx = 0;
    while (idx < num_points && (line_buffer = file.ReadLine())) {
        double x, y, z;
        // X Y Z.
        if (sscanf(line_buffer, "%lf %lf %lf", &x, &y, &z) == 3) {
            points_ptr[3 * idx + 0] = x;
            points_ptr[3 * idx + 1] = y;
            points_ptr[3 * idx + 2] = z;
        } else {
            utility::LogWarning("Read failed at line: {}", line_buffer);
            return false;
        }
        idx++;
        if (idx % 1000 == 0) {
            reporter.Update(idx);
        }
    }
    return true;
}

open3d::io::FileGeometry ReadFileGeometryTypeXYZI(const std::string &path) {
    return open3d::io::CONTAINS_POINTS;
}

bool ReadPointCloudFromXYZIRGB(const std::string &filename,
                               geometry::PointCloud &pointcloud,
                               const open3d::io::ReadPointCloudOption &params) {
    try {
        utility::filesystem::CFile file;
        if (!file.Open(filename, "r")) {
            utility::LogWarning("Read XYZIRGB failed: unable to open file: {}",
                                filename);
            return false;
        }
        utility::CountingProgressReporter reporter(params.update_progress);
        reporter.SetTotal(file.GetFileSize());
        int64_t num_points = file.GetNumLines();

        pointcloud.Clear();
        const char *line_buffer = nullptr;
        return ReadXYZIRGB(pointcloud, num_points, file, line_buffer, reporter,
                           core::Float64);

    } catch (const std::exception &e) {
        utility::LogWarning("Read XYZ failed with exception: {}", e.what());
        return false;
    }
}

bool ReadPointCloudFromXYZRGB(const std::string &filename,
                              geometry::PointCloud &pointcloud,
                              const open3d::io::ReadPointCloudOption &params) {
    try {
        utility::filesystem::CFile file;
        if (!file.Open(filename, "r")) {
            utility::LogWarning("Read XYZRGB failed: unable to open file: {}",
                                filename);
            return false;
        }
        utility::CountingProgressReporter reporter(params.update_progress);
        reporter.SetTotal(file.GetFileSize());
        int64_t num_points = file.GetNumLines();

        pointcloud.Clear();
        const char *line_buffer = nullptr;
        return ReadXYZRGB(pointcloud, num_points, file, line_buffer, reporter,
                          core::Float64);

    } catch (const std::exception &e) {
        utility::LogWarning("Read XYZ failed with exception: {}", e.what());
        return false;
    }
}

bool ReadPointCloudFromXYZI(const std::string &filename,
                            geometry::PointCloud &pointcloud,
                            const open3d::io::ReadPointCloudOption &params) {
    try {
        utility::filesystem::CFile file;
        if (!file.Open(filename, "r")) {
            utility::LogWarning("Read XYZI failed: unable to open file: {}",
                                filename);
            return false;
        }
        utility::CountingProgressReporter reporter(params.update_progress);
        reporter.SetTotal(file.GetFileSize());
        int64_t num_points = file.GetNumLines();

        pointcloud.Clear();
        const char *line_buffer = nullptr;
        return ReadXYZI(pointcloud, num_points, file, line_buffer, reporter);

    } catch (const std::exception &e) {
        utility::LogWarning("Read XYZ failed with exception: {}", e.what());
        return false;
    }
}

bool ReadPointCloudFromXYZN(const std::string &filename,
                            geometry::PointCloud &pointcloud,
                            const open3d::io::ReadPointCloudOption &params) {
    try {
        utility::filesystem::CFile file;
        if (!file.Open(filename, "r")) {
            utility::LogWarning("Read XYZN failed: unable to open file: {}",
                                filename);
            return false;
        }
        utility::CountingProgressReporter reporter(params.update_progress);
        reporter.SetTotal(file.GetFileSize());
        int64_t num_points = file.GetNumLines();

        pointcloud.Clear();
        const char *line_buffer = nullptr;
        return ReadXYZN(pointcloud, num_points, file, line_buffer, reporter);

    } catch (const std::exception &e) {
        utility::LogWarning("Read XYZ failed with exception: {}", e.what());
        return false;
    }
}

bool ReadPointCloudFromXYZ(const std::string &filename,
                           geometry::PointCloud &pointcloud,
                           const open3d::io::ReadPointCloudOption &params) {
    try {
        utility::filesystem::CFile file;
        if (!file.Open(filename, "r")) {
            utility::LogWarning("Read XYZ failed: unable to open file: {}",
                                filename);
            return false;
        }
        utility::CountingProgressReporter reporter(params.update_progress);
        reporter.SetTotal(file.GetFileSize());
        int64_t num_points = file.GetNumLines();

        pointcloud.Clear();
        const char *line_buffer = nullptr;
        return ReadXYZ(pointcloud, num_points, file, line_buffer, reporter);

    } catch (const std::exception &e) {
        utility::LogWarning("Read XYZ failed with exception: {}", e.what());
        return false;
    }
}

bool WritePointCloudToXYZI(const std::string &filename,
                           const geometry::PointCloud &pointcloud,
                           const open3d::io::WritePointCloudOption &params) {
    if (!pointcloud.HasPointAttr("intensities")) {
        return false;
    }

    try {
        utility::filesystem::CFile file;
        if (!file.Open(filename, "w")) {
            utility::LogWarning("Write XYZI failed: unable to open file: {}",
                                filename);
            return false;
        }
        utility::CountingProgressReporter reporter(params.update_progress);
        const core::Tensor &points = pointcloud.GetPoints();
        if (!points.GetShape().IsCompatible({utility::nullopt, 3})) {
            utility::LogWarning(
                    "Write XYZI failed: Shape of points is {}, but it should "
                    "be Nx3.",
                    points.GetShape());
            return false;
        }
        const core::Tensor &intensities =
                pointcloud.GetPointAttr("intensities");
        if (points.GetShape(0) != intensities.GetShape(0)) {
            utility::LogWarning(
                    "Write XYZI failed: Points ({}) and intensities ({}) have "
                    "different lengths.",
                    points.GetShape(0), intensities.GetShape(0));
            return false;
        }
        reporter.SetTotal(points.GetShape(0));

        for (int i = 0; i < points.GetShape(0); i++) {
            if (fprintf(file.GetFILE(), "%.10f %.10f %.10f %.10f\n",
                        points[i][0].Item<double>(),
                        points[i][1].Item<double>(),
                        points[i][2].Item<double>(),
                        intensities[i][0].Item<double>()) < 0) {
                utility::LogWarning(
                        "Write XYZI failed: unable to write file: {}",
                        filename);
                return false;  // error happened during writing.
            }
            if (i % 1000 == 0) {
                reporter.Update(i);
            }
        }
        reporter.Finish();
        return true;
    } catch (const std::exception &e) {
        utility::LogWarning("Write XYZI failed with exception: {}", e.what());
        return false;
    }
}

}  // namespace io
}  // namespace t
}  // namespace open3d
