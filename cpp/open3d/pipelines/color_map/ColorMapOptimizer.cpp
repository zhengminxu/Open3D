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

#include "open3d/pipelines/color_map/ColorMapOptimizer.h"

#include "open3d/camera/PinholeCameraTrajectory.h"
#include "open3d/geometry/Image.h"
#include "open3d/geometry/KDTreeFlann.h"
#include "open3d/geometry/RGBDImage.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/io/ClassIO/ImageWarpingFieldIO.h"
#include "open3d/io/ClassIO/PinholeCameraTrajectoryIO.h"
#include "open3d/pipelines/color_map/ColorMapOptimizationJacobian.h"
#include "open3d/pipelines/color_map/ImageWarpingField.h"
#include "open3d/pipelines/color_map/TriangleMeshAndImageUtilities.h"
#include "open3d/utility/Console.h"
#include "open3d/utility/Eigen.h"

namespace open3d {
namespace pipelines {
namespace color_map {

inline std::tuple<float, float, float> Project3DPointAndGetUVDepth(
        const Eigen::Vector3d X,
        const camera::PinholeCameraTrajectory& camera,
        int camid) {
    std::pair<double, double> f =
            camera.parameters_[camid].intrinsic_.GetFocalLength();
    std::pair<double, double> p =
            camera.parameters_[camid].intrinsic_.GetPrincipalPoint();
    Eigen::Vector4d Vt = camera.parameters_[camid].extrinsic_ *
                         Eigen::Vector4d(X(0), X(1), X(2), 1);
    float u = float((Vt(0) * f.first) / Vt(2) + p.first);
    float v = float((Vt(1) * f.second) / Vt(2) + p.second);
    float z = float(Vt(2));
    return std::make_tuple(u, v, z);
}

void ColorMapOptimizer::CreateGradientImages() {
    utility::LogDebug("[ColorMapOptimization] :: CreateGradientImages");
    for (size_t i = 0; i < images_rgbd_.size(); i++) {
        auto gray_image = images_rgbd_[i]->color_.CreateFloatImage();
        auto gray_image_filtered =
                gray_image->Filter(geometry::Image::FilterType::Gaussian3);
        images_gray_.push_back(gray_image_filtered);
        images_dx_.push_back(gray_image_filtered->Filter(
                geometry::Image::FilterType::Sobel3Dx));
        images_dy_.push_back(gray_image_filtered->Filter(
                geometry::Image::FilterType::Sobel3Dy));
        auto color = std::make_shared<geometry::Image>(images_rgbd_[i]->color_);
        auto depth = std::make_shared<geometry::Image>(images_rgbd_[i]->depth_);
        images_color_.push_back(color);
        images_depth_.push_back(depth);
    }
}

std::vector<std::shared_ptr<geometry::Image>>
ColorMapOptimizer::CreateDepthBoundaryMasks(
        const std::vector<std::shared_ptr<geometry::Image>>& images_depth,
        double depth_threshold_for_discontinuity_check,
        double half_dilation_kernel_size_for_discontinuity_map) {
    auto n_images = images_depth.size();
    std::vector<std::shared_ptr<geometry::Image>> masks;
    for (size_t i = 0; i < n_images; i++) {
        utility::LogDebug("[MakeDepthMasks] geometry::Image {:d}/{:d}", i,
                          n_images);
        masks.push_back(images_depth[i]->CreateDepthBoundaryMask(
                depth_threshold_for_discontinuity_check,
                half_dilation_kernel_size_for_discontinuity_map));
    }
    return masks;
}

std::tuple<std::vector<std::vector<int>>, std::vector<std::vector<int>>>
ColorMapOptimizer::CreateVertexAndImageVisibility(
        const geometry::TriangleMesh& mesh,
        const std::vector<std::shared_ptr<geometry::Image>>& images_depth,
        const std::vector<std::shared_ptr<geometry::Image>>& images_mask,
        const camera::PinholeCameraTrajectory& camera_trajectory,
        double maximum_allowable_depth,
        double depth_threshold_for_visibility_check) {
    size_t n_camera = camera_trajectory.parameters_.size();
    size_t n_vertex = mesh.vertices_.size();
    // visibility_image_to_vertex[c]: vertices visible by camera c.
    std::vector<std::vector<int>> visibility_image_to_vertex;
    visibility_image_to_vertex.resize(n_camera);
    // visibility_vertex_to_image[v]: cameras that can see vertex v.
    std::vector<std::vector<int>> visibility_vertex_to_image;
    visibility_vertex_to_image.resize(n_vertex);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int camera_id = 0; camera_id < int(n_camera); camera_id++) {
        for (int vertex_id = 0; vertex_id < int(n_vertex); vertex_id++) {
            Eigen::Vector3d X = mesh.vertices_[vertex_id];
            float u, v, d;
            std::tie(u, v, d) = Project3DPointAndGetUVDepth(
                    X, camera_trajectory, camera_id);
            int u_d = int(round(u)), v_d = int(round(v));
            // Skip if vertex in image boundary.
            if (d < 0.0 ||
                !images_depth[camera_id]->TestImageBoundary(u_d, v_d)) {
                continue;
            }
            // Skip if vertex's depth is too large (e.g. background).
            float d_sensor =
                    *images_depth[camera_id]->PointerAt<float>(u_d, v_d);
            if (d_sensor > maximum_allowable_depth) {
                continue;
            }
            // Check depth boundary mask. If a vertex is located at the boundary
            // of an object, its color will be highly diverse from different
            // viewing angles.
            if (*images_mask[camera_id]->PointerAt<unsigned char>(u_d, v_d) ==
                255) {
                continue;
            }
            // Check depth errors.
            if (std::fabs(d - d_sensor) >=
                depth_threshold_for_visibility_check) {
                continue;
            }
            visibility_image_to_vertex[camera_id].push_back(vertex_id);
#ifdef _OPENMP
#pragma omp critical
#endif
            { visibility_vertex_to_image[vertex_id].push_back(camera_id); }
        }
    }

    for (int camera_id = 0; camera_id < int(n_camera); camera_id++) {
        size_t n_visible_vertex = visibility_image_to_vertex[camera_id].size();
        utility::LogDebug(
                "[cam {:d}]: {:d}/{:d} ({:.5f}%) vertices are visible",
                camera_id, n_visible_vertex, n_vertex,
                double(n_visible_vertex) / n_vertex * 100);
    }

    return std::make_tuple(visibility_vertex_to_image,
                           visibility_image_to_vertex);
}

ColorMapOptimizer::ColorMapOptimizer(
        const geometry::TriangleMesh& mesh,
        const std::vector<std::shared_ptr<geometry::RGBDImage>>& images_rgbd,
        const camera::PinholeCameraTrajectory& camera_trajectory)
    : mesh_(mesh),
      images_rgbd_(images_rgbd),
      camera_trajectory_(camera_trajectory) {
    // Fills images_gray_, images_dx_, images_dy_, images_color_, images_depth_
    CreateGradientImages();
}

void ColorMapOptimizer::RunRigidOptimization(
        int maximum_iteration,
        double maximum_allowable_depth,
        double depth_threshold_for_visibility_check,
        double depth_threshold_for_discontinuity_check,
        double half_dilation_kernel_size_for_discontinuity_map,
        int image_boundary_margin,
        int invisible_vertex_color_knn) {
    utility::LogDebug("[ColorMapOptimization] :: MakingMasks");
    auto images_mask = CreateDepthBoundaryMasks(
            images_depth_, depth_threshold_for_discontinuity_check,
            half_dilation_kernel_size_for_discontinuity_map);

    utility::LogDebug("[ColorMapOptimization] :: VisibilityCheck");
    std::vector<std::vector<int>> visibility_vertex_to_image;
    std::vector<std::vector<int>> visibility_image_to_vertex;
    std::tie(visibility_vertex_to_image, visibility_image_to_vertex) =
            CreateVertexAndImageVisibility(
                    mesh_, images_depth_, images_mask, camera_trajectory_,
                    maximum_allowable_depth,
                    depth_threshold_for_visibility_check);
}

void ColorMapOptimizer::RunNonRigidOptimization(
        int number_of_vertical_anchors,
        double non_rigid_anchor_point_weight,
        int maximum_iteration,
        double maximum_allowable_depth,
        double depth_threshold_for_visibility_check,
        double depth_threshold_for_discontinuity_check,
        double half_dilation_kernel_size_for_discontinuity_map,
        int image_boundary_margin,
        int invisible_vertex_color_knn) {}

}  // namespace color_map
}  // namespace pipelines
}  // namespace open3d
