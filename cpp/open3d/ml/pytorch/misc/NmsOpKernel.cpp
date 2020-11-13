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

#include "open3d/ml/pytorch/misc/NmsOpKernel.h"

#include "open3d/ml/impl/misc/Nms.h"
#include "open3d/ml/pytorch/TorchHelper.h"
#include "torch/script.h"

// [inputs]
// boxes             : (N, 5) float32
// scores            : (N,) float32
// nms_overlap_thresh: double
//
// [return]
// keep_indices      : (M,) int64, the selected box indices
torch::Tensor NmsCPU(torch::Tensor boxes,
                     torch::Tensor scores,
                     double nms_overlap_thresh) {
    std::vector<int64_t> keep_indices = open3d::ml::impl::NmsCPUKernel(
            boxes.data_ptr<float>(), scores.data_ptr<float>(), boxes.size(0),
            nms_overlap_thresh);

    // torch::from_blob does not copy memory, usually a deleter is required. We
    // copy values here for simplicity.
    torch::Tensor keep_tensor =
            torch::from_blob(keep_indices.data(),
                             {static_cast<int64_t>(keep_indices.size())},
                             torch::TensorOptions().dtype(torch::kLong))
                    .clone();
    return keep_tensor;
}
