// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 www.open3d.org
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

#include "open3d/tgeometry/TriangleMesh.h"

#include <string>
#include <unordered_map>

#include "pybind/tgeometry/geometry.h"

namespace open3d {
namespace tgeometry {

void pybind_trianglemesh(py::module& m) {
    py::class_<TriangleMesh, PyGeometry<TriangleMesh>,
               std::unique_ptr<TriangleMesh>, Geometry>
            trianglemesh(m, "TriangleMesh",
                         "A trianglemesh contains a set of 3D points.");

    // Constructors.
    // trianglemesh
    //         .def(py::init<core::Dtype, const core::Device&>(), "dtype"_a,
    //              "device"_a)
    //         .def(py::init<const core::TensorList&>(), "points"_a)
    //         .def(py::init<const std::unordered_map<std::string,
    //                                                core::TensorList>&>(),
    //              "map_keys_to_tensorlists"_a);

    // // Point's attributes: points, colors, normals, etc.
    // // def_property_readonly is sufficient, since the returned TensorListMap
    // can
    // // be editable in Python. We don't want the TensorListMp to be replaced
    // // by another TensorListMap in Python.
    // trianglemesh.def_property_readonly("point",
    //                                    &TriangleMesh::GetPointAttrPybind);

    // // Pointcloud specific functions.
    // // TOOD: convert o3d.pybind.core.Tensor (C++ binded Python) to
    // //       o3d.core.Tensor (pure Python wrapper).
    // trianglemesh.def("get_min_bound", &TriangleMesh::GetMinBound,
    //                  "Returns the min bound for point coordinates.");
    // trianglemesh.def("get_max_bound", &TriangleMesh::GetMaxBound,
    //                  "Returns the max bound for point coordinates.");
    // trianglemesh.def("get_center", &TriangleMesh::GetCenter,
    //                  "Returns the center for point coordinates.");
    // trianglemesh.def("transform", &TriangleMesh::Transform,
    // "transformation"_a,
    //                  "Transforms the points and normals (if exist).");
    // trianglemesh.def("translate", &TriangleMesh::Translate, "translation"_a,
    //                  "relative"_a = true, "Translates points.");
    // trianglemesh.def("scale", &TriangleMesh::Scale, "scale"_a, "center"_a,
    //                  "Scale points.");
    // trianglemesh.def("rotate", &TriangleMesh::Rotate, "R"_a, "center"_a,
    //                  "Rotate points and normals (if exist).");
    // trianglemesh.def_static(
    //         "from_legacy_pointcloud", &TriangleMesh::FromLegacyPointCloud,
    //         "pcd_legacy"_a, "dtype"_a = core::Dtype::Float32,
    //         "device"_a = core::Device("CPU:0"),
    //         "Create a TriangleMesh from a legacy Open3D TriangleMesh.");
    // trianglemesh.def("to_legacy_pointcloud",
    // &TriangleMesh::ToLegacyPointCloud,
    //                  "Convert to a legacy Open3D TriangleMesh.");
}

}  // namespace tgeometry
}  // namespace open3d
