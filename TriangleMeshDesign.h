#pragma once

#include <Eigen/Core>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// clang-format off

namespace open3d {

namespace geometry {
class TriangleMesh : public Geometry {
public:
    // Processing.
    TriangleMesh& Scale(...) override;
    TriangleMesh& Rotate(...) override;
    TriangleMesh& ComputeVertexNormals();
    std::shared_ptr<TriangleMesh> SubdivideMidpoint(...) const;
    std::shared_ptr<TriangleMesh> SimplifyVertexClustering(...) const;

    // Conversion.
    static std::shared_ptr<TriangleMesh> CreateFromPointCloudPoisson(const PointCloud&, ...);
    static std::shared_ptr<TriangleMesh> CreateFromPointCloudBallPivot(const PointCloud&, ...);

    // Creation.
    static std::shared_ptr<TriangleMesh> CreateSphere(...);
    static std::shared_ptr<TriangleMesh> CreateBox(...);

public:
    std::vector<Eigen::Vector3d> vertices_;
    std::vector<Eigen::Vector3d> vertex_normals_;
    std::vector<Eigen::Vector3d> vertex_colors_;
    std::vector<Eigen::Vector3i> triangles_;
    std::vector<Eigen::Vector3d> triangle_normals_;
};
}

namespace io {
bool ReadTriangleMesh(const std::string&, geometry::TriangleMesh&, ...);
bool WriteTriangleMesh(const std::string&, const geometry::TriangleMesh&, ...);
}

namespace visualization {
void Draw(const std::vector<std::shared_ptr<Geometry>>& geometries, ...);
}

}

// clang-format on
