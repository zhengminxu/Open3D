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
// ----------------------------------------------------------------------------q

#include <vector>

#include "open3d/ml/pytorch/TorchHelper.h"
#include "open3d/ml/pytorch/misc/NmsOpKernel.h"
#include "torch/script.h"

int64_t Nms(torch::Tensor boxes,
            torch::Tensor keep,
            double nms_overlap_thresh) {
    boxes = boxes.contiguous();
    CHECK_TYPE(boxes, kFloat);
    CHECK_TYPE(keep, kInt64);

    if (boxes.is_cuda()) {
#ifdef BUILD_CUDA_MODULE
        return NmsCUDA(boxes, keep, nms_overlap_thresh);
#else
        TORCH_CHECK(false, "Nms was not compiled with CUDA support")
#endif
    } else {
        return NmsCPU(boxes, keep, nms_overlap_thresh);
    }
}

torch::Tensor NmsWithScore(torch::Tensor boxes,
                           torch::Tensor scores,
                           double nms_overlap_thresh) {
    boxes = boxes.contiguous();
    CHECK_TYPE(boxes, kFloat);
    CHECK_TYPE(scores, kFloat);

    if (boxes.is_cuda()) {
#ifdef BUILD_CUDA_MODULE
        return NmsWithScoreCUDA(boxes, scores, nms_overlap_thresh);
#else
        TORCH_CHECK(false, "NmsWithScore was not compiled with CUDA support")

#endif
    } else {
        return NmsWithScoreCPU(boxes, scores, nms_overlap_thresh);
    }
}

static auto registry = torch::RegisterOperators(
        "open3d::nms(Tensor boxes, Tensor keep, float nms_overlap_thresh) -> "
        "int keep_num",
        &Nms);

static auto registry2 = torch::RegisterOperators(
        "open3d::nms_with_score(Tensor boxes, Tensor scores, float "
        "nms_overlap_thresh) -> "
        "Tensor keep_indices",
        &NmsWithScore);
