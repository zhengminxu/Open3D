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

#include "open3d/ml/impl/misc/Nms.h"
#include "open3d/ml/pytorch/TorchHelper.h"
#include "open3d/ml/pytorch/misc/NmsOpKernel.h"
#include "torch/script.h"

#define CHECK_ERROR(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code,
                      const char *file,
                      int line,
                      bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        if (abort) exit(code);
    }
}

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

// [inputs]
// boxes             : (N, 5) float32
// scores            : (N,) float32
// nms_overlap_thresh: double
//
// [return]
// keep_indices      : (M,) int64, the selected box indices
torch::Tensor NmsWithScoreCUDA(torch::Tensor boxes,
                               torch::Tensor scores,
                               double nms_overlap_thresh) {
    // std::vector<int64_t> keep_indices{1, 2, 3};

    // // torch::from_blob does not copy memory, usually a deleter is required.
    // We
    // // copy values here for simplicity.
    // torch::Tensor keep_tensor =
    //         torch::from_blob(keep_indices.data(),
    //                          {static_cast<int64_t>(keep_indices.size())},
    //                          torch::TensorOptions()
    //                                  .dtype(torch::kLong)
    //                                  .device(boxes.device()))
    //                 .clone();
    return scores;
}

int64_t NmsCUDA(torch::Tensor boxes,
                torch::Tensor keep,
                double nms_overlap_thresh) {
    CHECK_CUDA(boxes);
    CHECK_CONTIGUOUS(boxes);
    CHECK_CONTIGUOUS(keep);

    const int num_boxes = boxes.size(0);
    const int num_block_cols =
            DIVUP(num_boxes, open3d::ml::impl::NMS_BLOCK_SIZE);

    // Allocate masks on device.
    uint64_t *mask_ptr = nullptr;
    CHECK_ERROR(cudaMalloc((void **)&mask_ptr,
                           num_boxes * num_block_cols * sizeof(uint64_t)));

    // Call kernel. Results will be saved in masks.
    const float *boxes_ptr = boxes.data_ptr<float>();
    open3d::ml::impl::NmsCUDAKernel(boxes_ptr, mask_ptr, num_boxes,
                                    nms_overlap_thresh);

    // Copy cuda masks to cpu.
    std::vector<uint64_t> mask_cpu(num_boxes * num_block_cols);
    CHECK_ERROR(cudaMemcpy(mask_cpu.data(), mask_ptr,
                           num_boxes * num_block_cols * sizeof(uint64_t),
                           cudaMemcpyDeviceToHost));
    cudaFree(mask_ptr);

    // Write to keep.
    // remv_cpu has num_boxes bits in total. If the bit is 1, the corresponding
    // box will be removed.
    std::vector<uint64_t> remv_cpu(num_block_cols, 0);
    int64_t *keep_ptr = keep.data_ptr<int64_t>();
    int num_to_keep = 0;
    for (int i = 0; i < num_boxes; i++) {
        int block_col_idx = i / open3d::ml::impl::NMS_BLOCK_SIZE;
        int inner_block_col_idx =
                i % open3d::ml::impl::NMS_BLOCK_SIZE;  // threadIdx.x

        // Querying the i-th bit in remv_cpu, counted from the right.
        // - remv_cpu[block_col_idx]: the block bitmap containing the query
        // - 1ULL << inner_block_col_idx: the one-hot bitmap to extract i
        if (!(remv_cpu[block_col_idx] & (1ULL << inner_block_col_idx))) {
            // Keep the i-th box.
            keep_ptr[num_to_keep++] = i;

            // Any box that overlaps with the i-th box will be removed.
            uint64_t *p = mask_cpu.data() + i * num_block_cols;
            for (int j = block_col_idx; j < num_block_cols; j++) {
                remv_cpu[j] |= p[j];
            }
        }
    }

    if (cudaSuccess != cudaGetLastError()) {
        printf("Error!\n");
    }

    return num_to_keep;
}
