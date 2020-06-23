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

class ColorMapOptimizer::ColorMapOptimizerImpl {
public:
    ColorMapOptimizerImpl(
            const geometry::TriangleMesh& mesh,
            const std::vector<std::shared_ptr<geometry::RGBDImage>>&
                    images_rgbd,
            const camera::PinholeCameraTrajectory& camera_trajectory)
        : mesh_(mesh),
          images_rgbd_(images_rgbd),
          camera_trajectory_(camera_trajectory) {}

protected:
    geometry::TriangleMesh mesh_;
    std::vector<std::shared_ptr<geometry::RGBDImage>> images_rgbd_;
    camera::PinholeCameraTrajectory camera_trajectory_;
    std::vector<std::shared_ptr<geometry::Image>> images_gray_;
    std::vector<std::shared_ptr<geometry::Image>> images_dx_;
    std::vector<std::shared_ptr<geometry::Image>> images_dy_;
    std::vector<std::shared_ptr<geometry::Image>> images_color_;
    std::vector<std::shared_ptr<geometry::Image>> images_depth_;
};

ColorMapOptimizer::ColorMapOptimizer(
        const geometry::TriangleMesh& mesh,
        const std::vector<std::shared_ptr<geometry::RGBDImage>>& images_rgbd,
        const camera::PinholeCameraTrajectory& camera_trajectory)
    : impl_(new ColorMapOptimizer::ColorMapOptimizerImpl(
              mesh, images_rgbd, camera_trajectory)) {}

static std::vector<ImageWarpingField> CreateWarpingFields(
        const std::vector<std::shared_ptr<geometry::Image>>& images,
        int number_of_vertical_anchors) {
    std::vector<ImageWarpingField> fields;
    for (size_t i = 0; i < images.size(); i++) {
        int width = images[i]->width_;
        int height = images[i]->height_;
        fields.push_back(
                ImageWarpingField(width, height, number_of_vertical_anchors));
    }
    return fields;
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
    std::vector<std::shared_ptr<geometry::Image>> images_mask =
            CreateDepthBoundaryMasks(
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

    utility::LogDebug("[ColorMapOptimization] :: Run Rigid Optimization");
    std::vector<double> proxy_intensity;
    int total_num_ = 0;
    int n_camera = int(camera_trajectory_.parameters_.size());
    SetProxyIntensityForVertex(mesh_, images_gray_, camera_trajectory_,
                               visibility_vertex_to_image, proxy_intensity,
                               image_boundary_margin);
    for (int itr = 0; itr < maximum_iteration; itr++) {
        utility::LogDebug("[Iteration {:04d}] ", itr + 1);
        double residual = 0.0;
        total_num_ = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int c = 0; c < n_camera; c++) {
            Eigen::Matrix4d pose;
            pose = camera_trajectory_.parameters_[c].extrinsic_;

            auto intrinsic = camera_trajectory_.parameters_[c]
                                     .intrinsic_.intrinsic_matrix_;
            auto extrinsic = camera_trajectory_.parameters_[c].extrinsic_;
            ColorMapOptimizationJacobian jac;
            Eigen::Matrix4d intr = Eigen::Matrix4d::Zero();
            intr.block<3, 3>(0, 0) = intrinsic;
            intr(3, 3) = 1.0;

            auto f_lambda = [&](int i, Eigen::Vector6d& J_r, double& r) {
                jac.ComputeJacobianAndResidualRigid(
                        i, J_r, r, mesh_, proxy_intensity, images_gray_[c],
                        images_dx_[c], images_dy_[c], intr, extrinsic,
                        visibility_image_to_vertex[c], image_boundary_margin);
            };
            Eigen::Matrix6d JTJ;
            Eigen::Vector6d JTr;
            double r2;
            std::tie(JTJ, JTr, r2) =
                    utility::ComputeJTJandJTr<Eigen::Matrix6d, Eigen::Vector6d>(
                            f_lambda, int(visibility_image_to_vertex[c].size()),
                            false);

            bool is_success;
            Eigen::Matrix4d delta;
            std::tie(is_success, delta) =
                    utility::SolveJacobianSystemAndObtainExtrinsicMatrix(JTJ,
                                                                         JTr);
            pose = delta * pose;
            camera_trajectory_.parameters_[c].extrinsic_ = pose;
#ifdef _OPENMP
#pragma omp critical
#endif
            {
                residual += r2;
                total_num_ += int(visibility_image_to_vertex[c].size());
            }
        }
        utility::LogDebug("Residual error : {:.6f} (avg : {:.6f})", residual,
                          residual / total_num_);
        SetProxyIntensityForVertex(mesh_, images_gray_, camera_trajectory_,
                                   visibility_vertex_to_image, proxy_intensity,
                                   image_boundary_margin);
    }

    utility::LogDebug("[ColorMapOptimization] :: Set Mesh Color");
    SetGeometryColorAverage(mesh_, images_color_, camera_trajectory_,
                            visibility_vertex_to_image, image_boundary_margin,
                            invisible_vertex_color_knn);
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

    utility::LogDebug("[ColorMapOptimization] :: Run Non-Rigid Optimization");
    auto warping_fields =
            CreateWarpingFields(images_gray_, number_of_vertical_anchors);
    auto warping_fields_init =
            CreateWarpingFields(images_gray_, number_of_vertical_anchors);
    std::vector<double> proxy_intensity;
    auto n_vertex = mesh_.vertices_.size();
    int n_camera = int(camera_trajectory_.parameters_.size());
    SetProxyIntensityForVertex(mesh_, images_gray_, warping_fields,
                               camera_trajectory_, visibility_vertex_to_image,
                               proxy_intensity, image_boundary_margin);
    for (int itr = 0; itr < maximum_iteration; itr++) {
        utility::LogDebug("[Iteration {:04d}] ", itr + 1);
        double residual = 0.0;
        double residual_reg = 0.0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int c = 0; c < n_camera; c++) {
            int nonrigidval = warping_fields[c].anchor_w_ *
                              warping_fields[c].anchor_h_ * 2;
            double rr_reg = 0.0;

            Eigen::Matrix4d pose;
            pose = camera_trajectory_.parameters_[c].extrinsic_;

            auto intrinsic = camera_trajectory_.parameters_[c]
                                     .intrinsic_.intrinsic_matrix_;
            auto extrinsic = camera_trajectory_.parameters_[c].extrinsic_;
            ColorMapOptimizationJacobian jac;
            Eigen::Matrix4d intr = Eigen::Matrix4d::Zero();
            intr.block<3, 3>(0, 0) = intrinsic;
            intr(3, 3) = 1.0;

            auto f_lambda = [&](int i, Eigen::Vector14d& J_r, double& r,
                                Eigen::Vector14i& pattern) {
                jac.ComputeJacobianAndResidualNonRigid(
                        i, J_r, r, pattern, mesh_, proxy_intensity,
                        images_gray_[c], images_dx_[c], images_dy_[c],
                        warping_fields[c], warping_fields_init[c], intr,
                        extrinsic, visibility_image_to_vertex[c],
                        image_boundary_margin);
            };
            Eigen::MatrixXd JTJ;
            Eigen::VectorXd JTr;
            double r2;
            std::tie(JTJ, JTr, r2) =
                    ComputeJTJandJTrNonRigid<Eigen::Vector14d, Eigen::Vector14i,
                                             Eigen::MatrixXd, Eigen::VectorXd>(
                            f_lambda, int(visibility_image_to_vertex[c].size()),
                            nonrigidval, false);

            double weight = non_rigid_anchor_point_weight *
                            visibility_image_to_vertex[c].size() / n_vertex;
            for (int j = 0; j < nonrigidval; j++) {
                double r = weight * (warping_fields[c].flow_(j) -
                                     warping_fields_init[c].flow_(j));
                JTJ(6 + j, 6 + j) += weight * weight;
                JTr(6 + j) += weight * r;
                rr_reg += r * r;
            }

            bool success;
            Eigen::VectorXd result;
            std::tie(success, result) = utility::SolveLinearSystemPSD(
                    JTJ, -JTr, /*prefer_sparse=*/false,
                    /*check_symmetric=*/false,
                    /*check_det=*/false, /*check_psd=*/false);
            Eigen::Vector6d result_pose;
            result_pose << result.block(0, 0, 6, 1);
            auto delta = utility::TransformVector6dToMatrix4d(result_pose);
            pose = delta * pose;

            for (int j = 0; j < nonrigidval; j++) {
                warping_fields[c].flow_(j) += result(6 + j);
            }
            camera_trajectory_.parameters_[c].extrinsic_ = pose;

#ifdef _OPENMP
#pragma omp critical
#endif
            {
                residual += r2;
                residual_reg += rr_reg;
            }
        }
        utility::LogDebug("Residual error : {:.6f}, reg : {:.6f}", residual,
                          residual_reg);
        SetProxyIntensityForVertex(mesh_, images_gray_, warping_fields,
                                   camera_trajectory_,
                                   visibility_vertex_to_image, proxy_intensity,
                                   image_boundary_margin);
    }

    utility::LogDebug("[ColorMapOptimization] :: Set Mesh Color");
    SetGeometryColorAverage(mesh_, images_color_, warping_fields,
                            camera_trajectory_, visibility_vertex_to_image,
                            image_boundary_margin, invisible_vertex_color_knn);
}

}  // namespace color_map
}  // namespace pipelines
}  // namespace open3d
