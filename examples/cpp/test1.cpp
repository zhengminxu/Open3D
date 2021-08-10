// Test for Colored-ICP ComputeRMSE

#include "open3d/Open3D.h"

using namespace open3d;

// double ComputeRMSE(const t::geometry::PointCloud &source,
//                    const t::geometry::PointCloud &target,
//                    const core::Tensor &correspondences,
//                    const double lambda_geometric_) {
//     if (!target.HasPointColors() || !source.HasPointColors()) {
//         utility::LogError(
//                 "Source and/or Target pointcloud missing colors attribute.");
//     }
//     if (!target.HasPointNormals()) {
//         utility::LogError("Target pointcloud missing normals attribute.");
//     }
//     if (!target.HasPointAttr("color_gradients")) {
//         utility::LogError(
//                 "Target pointcloud missing color_gradients attribute.");
//     }

//     const core::Device device = source.GetPointPositions().GetDevice();
//     target.GetPointPositions().AssertDevice(device);
//     correspondences.AssertDevice(device);

//     const core::Dtype dtype = source.GetPointPositions().GetDtype();
//     source.GetPointColors().AssertDtype(dtype);
//     target.GetPointPositions().AssertDtype(dtype);
//     target.GetPointNormals().AssertDtype(dtype);
//     target.GetPointColors().AssertDtype(dtype);
//     target.GetPointAttr("color_gradients").AssertDtype(dtype);
//     correspondences.AssertDtype(core::Dtype::Int64);

//     double sqrt_lambda_geometric = sqrt(lambda_geometric_);
//     double lambda_photometric = 1.0 - lambda_geometric_;
//     double sqrt_lambda_photometric = sqrt(lambda_photometric);

//     core::Tensor valid = correspondences.Ne(-1).Reshape({-1});
//     core::Tensor neighbour_indices =
//             correspondences.IndexGet({valid}).Reshape({-1});

//     // vs - source points (or vertices)
//     // vt - target points
//     // nt - target normals
//     // cs - source colors
//     // ct - target colors
//     // dit - target color gradients
//     // is - source intensity
//     // it - target intensity
//     // vs_proj - source points projection
//     // is_proj - source intensity projection

//     core::Tensor vs = source.GetPointPositions().IndexGet({valid});
//     core::Tensor cs = source.GetPointColors().IndexGet({valid});

//     core::Tensor vt = target.GetPointPositions().IndexGet({neighbour_indices});
//     core::Tensor nt = target.GetPointNormals().IndexGet({neighbour_indices});
//     core::Tensor ct = target.GetPointColors().IndexGet({neighbour_indices});
//     core::Tensor dit = target.GetPointAttr("color_gradients")
//                                .IndexGet({neighbour_indices});

//     // vs_proj = vs - (vs - vt).dot(nt) * nt
//     // d = (vs - vt).dot(nt)
//     const core::Tensor d = (vs - vt).Mul(nt).Sum({1});
//     core::Tensor vs_proj = vs - d.Mul(nt);

//     core::Tensor is = cs.Mean({1});
//     core::Tensor it = ct.Mean({1});

//     // is_proj = (dit.dot(vs_proj - vt)) + it
//     core::Tensor is_proj = (dit.Mul(vs_proj - vt)).Sum({1}).Add(it);

//     core::Tensor residual_geometric = d.Mul(sqrt_lambda_geometric).Sum({1});
//     core::Tensor sq_residual_geometric =
//             residual_geometric.Mul(residual_geometric);
//     core::Tensor residual_photometric =
//             (is - is_proj).Mul(sqrt_lambda_photometric).Sum({1});
//     core::Tensor sq_residual_photometric =
//             residual_photometric.Mul(residual_photometric);

//     double residual = sq_residual_geometric.Add_(sq_residual_photometric)
//                               .Sum({0})
//                               .To(core::Float64)
//                               .Item<double>();

//     return residual;
// }

core::Tensor Dot(core::Tensor A, core::Tensor B) {
    auto C = A.Mul(B).Sum({1});
    return C;
}

int main(int argc, char *argv[]) {
    
    core::Tensor A = core::Tensor::Full({5, 3}, 2.0, core::Float64);
    core::Tensor B = core::Tensor::Full({5, 3}, 3.0, core::Float64);

    std::cout << " A.Dot(B) = " << Dot(A, B).ToString() << std::endl;

    return 0;
}
