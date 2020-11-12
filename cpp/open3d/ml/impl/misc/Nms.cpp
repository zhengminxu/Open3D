#include "open3d/ml/impl/misc/Nms.h"

#include <iostream>

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

namespace open3d {
namespace ml {
namespace impl {

void NmsCPUKernel(const float *boxes,
                  uint64_t *mask,
                  int num_boxes,
                  float nms_overlap_thresh) {
    // boxes: (N, 5)
    // mask:  (N, N/BS)

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
                    if (iou_bev(boxes + src_idx * 5, boxes + dst_idx * 5) >
                        nms_overlap_thresh) {
                        t |= 1ULL << (dst_idx - NMS_BLOCK_SIZE * block_col_idx);
                    }
                }
                mask[src_idx * num_block_cols + block_col_idx] = t;
            }
        }
    }
}

}  // namespace impl
}  // namespace ml
}  // namespace open3d
