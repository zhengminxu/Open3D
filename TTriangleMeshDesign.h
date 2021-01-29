#pragma once

#include <Eigen/Core>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace open3d {

namespace geometry {
class TriangleMesh : public Geometry {
public:
    // Processing.
    TriangleMesh& Scale(...) override;
    TriangleMesh& Rotate(...) override;
    TriangleMesh& ComputeVertexNormals();
    TriangleMesh SubdivideMidpoint(...) const;
    TriangleMesh SimplifyVertexClustering(...) const;

    // Conversion.
    static TriangleMesh CreateFromPointCloudPoisson(const PointCloud&, ...);
    static TriangleMesh CreateFromPointCloudBallPivot(const PointCloud&, ...);

    // Creation.
    static TriangleMesh CreateSphere(...);
    static TriangleMesh CreateBox(...);

protected:
    unordered_map<string, Tensor> vertex_attr_;
    unordered_map<string, Tensor> triangle_attr_;
};
}  // namespace geometry

}  // namespace open3d
