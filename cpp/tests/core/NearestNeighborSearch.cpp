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

// #include "open3d/core/nns/NearestNeighborSearch.h"

#include <cmath>
#include <limits>

#include "nanoflann.hpp"
#include "open3d/core/Dtype.h"
#include "open3d/core/SizeVector.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/utility/Helper.h"
#include "tests/UnitTest.h"

// Forward declarations.
namespace nanoflann {

template <class T, class DataSource, typename _DistanceType>
struct L2_Adaptor;

template <class T, class DataSource, typename _DistanceType>
struct L1_Adaptor;

template <typename Distance, class DatasetAdaptor, int DIM, typename IndexType>
class KDTreeSingleIndexAdaptor;

struct SearchParams;

};  // namespace nanoflann

namespace open3d {
namespace tests {

enum Metric { L1 = 0, L2 = 1, Linf = 2 };

struct NanoFlannIndexHolderBase {};

/// NanoFlann Index Holder
template <int METRIC, class T>
struct NanoFlannIndexHolder : NanoFlannIndexHolderBase {
    /// This class is the Adaptor for connecting Open3D Tensor and NanoFlann.
    struct DataAdaptor {
        DataAdaptor(size_t dataset_size, int dimension, const T *const data_ptr)
            : dataset_size_(dataset_size),
              dimension_(dimension),
              data_ptr_(data_ptr) {}

        inline size_t kdtree_get_point_count() const { return dataset_size_; }

        inline T kdtree_get_pt(const size_t idx, const size_t dim) const {
            return data_ptr_[idx * dimension_ + dim];
        }

        template <class BBOX>
        bool kdtree_get_bbox(BBOX &) const {
            return false;
        }

        size_t dataset_size_ = 0;
        int dimension_ = 0;
        const T *const data_ptr_;
    };

    /// Adaptor Selector
    template <int M, typename fake = void>
    struct SelectNanoflannAdaptor {};

    template <typename fake>
    struct SelectNanoflannAdaptor<L2, fake> {
        typedef nanoflann::L2_Adaptor<T, DataAdaptor, T> adaptor_t;
    };

    template <typename fake>
    struct SelectNanoflannAdaptor<L1, fake> {
        typedef nanoflann::L1_Adaptor<T, DataAdaptor, T> adaptor_t;
    };

    /// typedef for KDtree
    typedef nanoflann::KDTreeSingleIndexAdaptor<
            typename SelectNanoflannAdaptor<METRIC>::adaptor_t,
            DataAdaptor,
            -1,
            size_t>
            KDTree_t;

    NanoFlannIndexHolder(size_t dataset_size,
                         int dimension,
                         const T *data_ptr) {
        adaptor_.reset(new DataAdaptor(dataset_size, dimension, data_ptr));
        index_.reset(new KDTree_t(dimension, *adaptor_.get()));
        index_->buildIndex();
    }

    std::unique_ptr<KDTree_t> index_;
    std::unique_ptr<DataAdaptor> adaptor_;
};

void RunTest() {
    std::unique_ptr<NanoFlannIndexHolderBase> holder_;
    std::vector<double> data(300);
    holder_.reset(new NanoFlannIndexHolder<L2, double>(100, 3, data.data()));
}

TEST(NearestNeighborSearch, KnnSearch) {
    while (true) {
        RunTest();
    }
}

}  // namespace tests
}  // namespace open3d
