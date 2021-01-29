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
    shared_ptr<TriangleMesh> SubdivideMidpoint(...) const;
    shared_ptr<TriangleMesh> SimplifyVertexClustering(...) const;

    // Conversion.
    static shared_ptr<TriangleMesh> CreateFromPointCloudPoisson(const PointCloud&, ...);
    static shared_ptr<TriangleMesh> CreateFromPointCloudBallPivot(const PointCloud&, ...);

    // Creation.
    static shared_ptr<TriangleMesh> CreateSphere(...);
    static shared_ptr<TriangleMesh> CreateBox(...);

public:
    vector<Eigen::Vector3d> vertices_;
    vector<Eigen::Vector3d> vertex_normals_;
    vector<Eigen::Vector3d> vertex_colors_;
    vector<Eigen::Vector3i> triangles_;
    vector<Eigen::Vector3d> triangle_normals_;
};
}

namespace io {
bool ReadTriangleMesh(const string&, geometry::TriangleMesh&, ...);
bool WriteTriangleMesh(const string&, const geometry::TriangleMesh&, ...);
}

namespace visualization {
void Draw(const vector<shared_ptr<Geometry>>& geometries, ...);
}

}

// clang-format on
