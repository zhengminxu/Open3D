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
    pointcloud.SetPointPositions(core::Tensor({num_points, 3}, core::Float64));
    auto positions_ptr = pointcloud.GetPointPositions().GetDataPtr<double>();
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
                positions_ptr[3 * idx + 0] = x;
                positions_ptr[3 * idx + 1] = y;
                positions_ptr[3 * idx + 2] = z;
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
                positions_ptr[3 * idx + 0] = x;
                positions_ptr[3 * idx + 1] = y;
                positions_ptr[3 * idx + 2] = z;
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
    pointcloud.SetPointPositions(core::Tensor({num_points, 3}, core::Float64));
    auto positions_ptr = pointcloud.GetPointPositions().GetDataPtr<double>();
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
                positions_ptr[3 * idx + 0] = x;
                positions_ptr[3 * idx + 1] = y;
                positions_ptr[3 * idx + 2] = z;
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
                positions_ptr[3 * idx + 0] = x;
                positions_ptr[3 * idx + 1] = y;
                positions_ptr[3 * idx + 2] = z;
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
    pointcloud.SetPointPositions(core::Tensor({num_points, 3}, core::Float64));
    auto positions_ptr = pointcloud.GetPointPositions().GetDataPtr<double>();
    pointcloud.SetPointAttr("intensities",
                            core::Tensor({num_points, 1}, core::Float64));
    auto intensities_ptr =
            pointcloud.GetPointAttr("intensities").GetDataPtr<double>();

    int64_t idx = 0;
    while (idx < num_points && (line_buffer = file.ReadLine())) {
        double x, y, z, i;
        // X Y Z I.
        if (sscanf(line_buffer, "%lf %lf %lf %lf", &x, &y, &z, &i) == 4) {
            positions_ptr[3 * idx + 0] = x;
            positions_ptr[3 * idx + 1] = y;
            positions_ptr[3 * idx + 2] = z;
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
    pointcloud.SetPointPositions(core::Tensor({num_points, 3}, core::Float64));
    auto positions_ptr = pointcloud.GetPointPositions().GetDataPtr<double>();
    pointcloud.SetPointNormals(core::Tensor({num_points, 3}, core::Float64));
    auto normals_ptr = pointcloud.GetPointNormals().GetDataPtr<double>();

    int64_t idx = 0;
    while (idx < num_points && (line_buffer = file.ReadLine())) {
        double x, y, z, nx, ny, nz;
        // X Y Z NX NY NZ.
        if (sscanf(line_buffer, "%lf %lf %lf %lf %lf %lf", &x, &y, &z, &nx, &ny,
                   &nz) == 6) {
            positions_ptr[3 * idx + 0] = x;
            positions_ptr[3 * idx + 1] = y;
            positions_ptr[3 * idx + 2] = z;
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
    pointcloud.SetPointPositions(core::Tensor({num_points, 3}, core::Float64));
    auto positions_ptr = pointcloud.GetPointPositions().GetDataPtr<double>();
    pointcloud.SetPointAttr("intensities",
                            core::Tensor({num_points, 1}, core::Float64));

    int64_t idx = 0;
    while (idx < num_points && (line_buffer = file.ReadLine())) {
        double x, y, z;
        // X Y Z.
        if (sscanf(line_buffer, "%lf %lf %lf", &x, &y, &z) == 3) {
            positions_ptr[3 * idx + 0] = x;
            positions_ptr[3 * idx + 1] = y;
            positions_ptr[3 * idx + 2] = z;
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

open3d::io::FileGeometry ReadFileGeometryTypeXYZIRGB(const std::string &path) {
    return open3d::io::CONTAINS_POINTS;
}

open3d::io::FileGeometry ReadFileGeometryTypeXYZRGB(const std::string &path) {
    return open3d::io::CONTAINS_POINTS;
}

open3d::io::FileGeometry ReadFileGeometryTypeXYZI(const std::string &path) {
    return open3d::io::CONTAINS_POINTS;
}

open3d::io::FileGeometry ReadFileGeometryTypeXYZN(const std::string &path) {
    return open3d::io::CONTAINS_POINTS;
}

open3d::io::FileGeometry ReadFileGeometryTypeXYZ(const std::string &path) {
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

static core::Tensor ConvertColorsToDouble(const core::Tensor &colors) {
    double normalization_factor = 1.0;
    const core::Dtype point_color_dtype = colors.GetDtype();

    if (point_color_dtype == core::UInt8) {
        normalization_factor =
                1.0 / static_cast<double>(std::numeric_limits<uint8_t>::max());
    } else if (point_color_dtype == core::UInt16) {
        normalization_factor =
                1.0 / static_cast<double>(std::numeric_limits<uint16_t>::max());
    } else if (point_color_dtype != core::Float32 &&
               point_color_dtype != core::Float64) {
        utility::LogError(
                "Dtype {} of color attribute is not supported for "
                "conversion to Float64 and will be skipped. "
                "Supported dtypes include UInt8, UIn16, Float32, and "
                "Float64",
                point_color_dtype.ToString());
    }

    if (normalization_factor != 1.0) {
        return colors.To(core::Float64) * normalization_factor;

    } else {
        return colors;
    }
}

bool WritePointCloudToXYZIRGB(const std::string &filename,
                              const geometry::PointCloud &pointcloud,
                              const open3d::io::WritePointCloudOption &params) {
    if (!pointcloud.HasPointAttr("intensities") ||
        !pointcloud.HasPointColors()) {
        return false;
    }

    try {
        utility::filesystem::CFile file;
        if (!file.Open(filename, "w")) {
            utility::LogWarning("Write XYZIRGB failed: unable to open file: {}",
                                filename);
            return false;
        }
        utility::CountingProgressReporter reporter(params.update_progress);
        const core::Tensor positions =
                pointcloud.GetPointPositions().To(core::Float64);
        const core::Tensor intensities =
                pointcloud.GetPointAttr("intensities").To(core::Float64);
        const core::Tensor colors =
                ConvertColorsToDouble(pointcloud.GetPointColors());

        if (positions.GetLength() != intensities.GetLength()) {
            utility::LogWarning(
                    "Write XYZIRGB failed: Positions ({}) and intensities ({}) "
                    "have different lengths.",
                    positions.GetLength(), intensities.GetLength());
            return false;
        }
        if (positions.GetLength() != colors.GetLength()) {
            utility::LogWarning(
                    "Write XYZIRGB failed: Positions ({}) and colors ({}) have "
                    "different lengths.",
                    positions.GetLength(), colors.GetLength());
            return false;
        }
        reporter.SetTotal(positions.GetLength());

        auto positions_ptr = positions.GetDataPtr<double>();
        auto intensity_ptr = intensities.GetDataPtr<double>();
        auto colors_ptr = colors.GetDataPtr<double>();

        for (int i = 0; i < positions.GetLength(); i++) {
            if (fprintf(file.GetFILE(),
                        "%.10f %.10f %.10f %.10f %.10f %.10f %.10f\n",
                        positions_ptr[3 * i], positions_ptr[3 * i + 1],
                        positions_ptr[3 * i + 2], intensity_ptr[3 * i],
                        colors_ptr[3 * i], colors_ptr[3 * i + 1],
                        colors_ptr[3 * i + 2]) < 0) {
                utility::LogWarning(
                        "Write XYZIRGB failed: unable to write file: {}",
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
        utility::LogWarning("Write XYZIRGB failed with exception: {}",
                            e.what());
        return false;
    }
}

bool WritePointCloudToXYZRGB(const std::string &filename,
                             const geometry::PointCloud &pointcloud,
                             const open3d::io::WritePointCloudOption &params) {
    if (!pointcloud.HasPointColors()) {
        return false;
    }

    try {
        utility::filesystem::CFile file;
        if (!file.Open(filename, "w")) {
            utility::LogWarning("Write XYZRGB failed: unable to open file: {}",
                                filename);
            return false;
        }
        utility::CountingProgressReporter reporter(params.update_progress);
        const core::Tensor positions =
                pointcloud.GetPointPositions().To(core::Float64);

        const core::Tensor colors =
                ConvertColorsToDouble(pointcloud.GetPointColors());

        if (positions.GetLength() != colors.GetLength()) {
            utility::LogWarning(
                    "Write XYZRGB failed: Positions ({}) and colors ({}) have "
                    "different lengths.",
                    positions.GetLength(), colors.GetLength());
            return false;
        }
        reporter.SetTotal(positions.GetLength());

        auto positions_ptr = positions.GetDataPtr<double>();
        auto colors_ptr = colors.GetDataPtr<double>();

        for (int i = 0; i < positions.GetLength(); i++) {
            if (fprintf(file.GetFILE(), "%.10f %.10f %.10f %.10f %.10f %.10f\n",
                        positions_ptr[3 * i], positions_ptr[3 * i + 1],
                        positions_ptr[3 * i + 2], colors_ptr[3 * i],
                        colors_ptr[3 * i + 1], colors_ptr[3 * i + 2]) < 0) {
                utility::LogWarning(
                        "Write XYZRGB failed: unable to write file: {}",
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
        utility::LogWarning("Write XYZRGB failed with exception: {}", e.what());
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
        const core::Tensor positions =
                pointcloud.GetPointPositions().To(core::Float64);
        const core::Tensor intensities =
                pointcloud.GetPointAttr("intensities").To(core::Float64);

        if (positions.GetLength() != intensities.GetLength()) {
            utility::LogWarning(
                    "Write XYZI failed: Positions ({}) and intensities ({}) "
                    "have different lengths.",
                    positions.GetLength(), intensities.GetLength());
            return false;
        }

        reporter.SetTotal(positions.GetLength());

        auto positions_ptr = positions.GetDataPtr<double>();
        auto intensity_ptr = intensities.GetDataPtr<double>();

        for (int i = 0; i < positions.GetLength(); i++) {
            if (fprintf(file.GetFILE(), "%.10f %.10f %.10f %.10f\n",
                        positions_ptr[3 * i], positions_ptr[3 * i + 1],
                        positions_ptr[3 * i + 2], intensity_ptr[3 * i]) < 0) {
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

bool WritePointCloudToXYZN(const std::string &filename,
                           const geometry::PointCloud &pointcloud,
                           const open3d::io::WritePointCloudOption &params) {
    if (!pointcloud.HasPointNormals()) {
        return false;
    }

    try {
        utility::filesystem::CFile file;
        if (!file.Open(filename, "w")) {
            utility::LogWarning("Write XYZN failed: unable to open file: {}",
                                filename);
            return false;
        }
        utility::CountingProgressReporter reporter(params.update_progress);
        const core::Tensor positions =
                pointcloud.GetPointPositions().To(core::Float64);
        const core::Tensor normals =
                pointcloud.GetPointNormals().To(core::Float64);

        if (positions.GetLength() != normals.GetLength()) {
            utility::LogWarning(
                    "Write XYZN failed: Positions ({}) and normals ({}) "
                    "have different lengths.",
                    positions.GetLength(), normals.GetLength());
            return false;
        }
        reporter.SetTotal(positions.GetLength());

        auto positions_ptr = positions.GetDataPtr<double>();
        auto normals_ptr = normals.GetDataPtr<double>();

        for (int i = 0; i < positions.GetLength(); i++) {
            if (fprintf(file.GetFILE(), "%.10f %.10f %.10f %.10f %.10f %.10f\n",
                        positions_ptr[3 * i], positions_ptr[3 * i + 1],
                        positions_ptr[3 * i + 2], normals_ptr[3 * i],
                        normals_ptr[3 * i + 1], normals_ptr[3 * i + 2]) < 0) {
                utility::LogWarning(
                        "Write XYZN failed: unable to write file: {}",
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
        utility::LogWarning("Write XYZN failed with exception: {}", e.what());
        return false;
    }
}

bool WritePointCloudToXYZ(const std::string &filename,
                          const geometry::PointCloud &pointcloud,
                          const open3d::io::WritePointCloudOption &params) {
    try {
        utility::filesystem::CFile file;
        if (!file.Open(filename, "w")) {
            utility::LogWarning("Write XYZ failed: unable to open file: {}",
                                filename);
            return false;
        }
        utility::CountingProgressReporter reporter(params.update_progress);
        const core::Tensor positions =
                pointcloud.GetPointPositions().To(core::Float64);

        reporter.SetTotal(positions.GetLength());

        auto positions_ptr = positions.GetDataPtr<double>();

        for (int i = 0; i < positions.GetLength(); i++) {
            if (fprintf(file.GetFILE(), "%.10f %.10f %.10f\n",
                        positions_ptr[3 * i], positions_ptr[3 * i + 1],
                        positions_ptr[3 * i + 2]) < 0) {
                utility::LogWarning(
                        "Write XYZ failed: unable to write file: {}", filename);
                return false;  // error happened during writing.
            }
            if (i % 1000 == 0) {
                reporter.Update(i);
            }
        }
        reporter.Finish();
        return true;
    } catch (const std::exception &e) {
        utility::LogWarning("Write XYZ failed with exception: {}", e.what());
        return false;
    }
}

}  // namespace io
}  // namespace t
}  // namespace open3d
