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

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

static void NmsCPUKernel(const float *boxes,
                         uint64_t *mask,
                         int num_boxes,
                         float nms_overlap_thresh) {
    // boxes: (N, 5)
    // mask:  (N, N/BS)
    int NMS_BLOCK_SIZE = open3d::ml::impl::NMS_BLOCK_SIZE;

    const int num_block_cols = DIVUP(num_boxes, NMS_BLOCK_SIZE);
    const int num_block_rows = DIVUP(num_boxes, NMS_BLOCK_SIZE);

    // We need the concept of "block" since the mask is a uint64_t binary bit
    // map, and the block size is exactly 64x64. This is also consistent with
    // the CUDA implementation.
    for (int block_col_idx = 0; block_col_idx < num_block_cols;
         block_col_idx++) {
        for (int block_row_idx = 0; block_row_idx < num_block_rows;
             ++block_row_idx) {
            // Local block row size.
            const int row_size = fminf(
                    num_boxes - block_row_idx * NMS_BLOCK_SIZE, NMS_BLOCK_SIZE);
            // Local block col size.
            const int col_size = fminf(
                    num_boxes - block_col_idx * NMS_BLOCK_SIZE, NMS_BLOCK_SIZE);

            // Comparing src and dst. In one block, the following src and dst
            // indices are compared:
            // - src: BS * block_row_idx : BS * block_row_idx + row_size
            // - dst: BS * block_col_idx : BS * block_col_idx + col_size
            //
            // With all blocks, all src and dst indices are compared.
            //
            // Result:
            // mask[i, j] is a 64-bit integer where mask[i, j][k] (k counted
            // from right) is 1 iff box[i] overlaps with box[BS*j+k].
            for (int src_idx = NMS_BLOCK_SIZE * block_row_idx;
                 src_idx < NMS_BLOCK_SIZE * block_row_idx + row_size;
                 src_idx++) {
                uint64_t t = 0;
                for (int dst_idx = NMS_BLOCK_SIZE * block_col_idx;
                     dst_idx < NMS_BLOCK_SIZE * block_col_idx + col_size;
                     dst_idx++) {
                    // Unlike the CUDA impl, both src_idx and dst_idx here are
                    // indexes to the global memory. Thus we need to compute the
                    // local index for dst_idx.
                    if (open3d::ml::impl::iou_bev(boxes + src_idx * 5,
                                                  boxes + dst_idx * 5) >
                        nms_overlap_thresh) {
                        t |= 1ULL << (dst_idx - NMS_BLOCK_SIZE * block_col_idx);
                    }
                }
                mask[src_idx * num_block_cols + block_col_idx] = t;
            }
        }
    }
}

// [inputs]
// boxes             : (N, 5) float32
// scores            : (N,) float32
// nms_overlap_thresh: double
//
// [return]
// selected_indices  : (M,) int64, the selected box indices
torch::Tensor NmsWithScoreCPU(torch::Tensor boxes,
                              torch::Tensor scores,
                              double nms_overlap_thresh) {
    torch::Tensor order =
            std::get<1>(torch::sort(scores, 0, /*descending=*/true));
    torch::Tensor boxes_sorted =
            torch::index_select(boxes, 0, order).contiguous();
    torch::Tensor keep = torch::zeros(
            {boxes.size(0)}, torch::TensorOptions().dtype(torch::kLong));

    CHECK_CONTIGUOUS(boxes_sorted);
    CHECK_CONTIGUOUS(keep);

    const int num_boxes = boxes_sorted.size(0);
    const int num_block_cols =
            DIVUP(num_boxes, open3d::ml::impl::NMS_BLOCK_SIZE);

    // Call kernel. Results will be saved in masks.
    std::vector<uint64_t> mask(num_boxes * num_block_cols);
    NmsCPUKernel(boxes_sorted.data_ptr<float>(), mask.data(), num_boxes,
                 nms_overlap_thresh);

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
            uint64_t *p = mask.data() + i * num_block_cols;
            for (int j = block_col_idx; j < num_block_cols; j++) {
                remv_cpu[j] |= p[j];
            }
        }
    }

    torch::Tensor selected_keep = torch::slice(keep, 0, 0, num_to_keep);
    return torch::index_select(order, 0, selected_keep);
}

int64_t NmsCPU(torch::Tensor boxes,
               torch::Tensor keep,
               double nms_overlap_thresh) {
    CHECK_CONTIGUOUS(boxes);
    CHECK_CONTIGUOUS(keep);

    const int num_boxes = boxes.size(0);
    const int num_block_cols =
            DIVUP(num_boxes, open3d::ml::impl::NMS_BLOCK_SIZE);

    // Call kernel. Results will be saved in masks.
    std::vector<uint64_t> mask(num_boxes * num_block_cols);
    open3d::ml::impl::NmsCPUKernel(boxes.data_ptr<float>(), mask.data(),
                                   num_boxes, nms_overlap_thresh);

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
            uint64_t *p = mask.data() + i * num_block_cols;
            for (int j = block_col_idx; j < num_block_cols; j++) {
                remv_cpu[j] |= p[j];
            }
        }
    }

    return num_to_keep;
}
