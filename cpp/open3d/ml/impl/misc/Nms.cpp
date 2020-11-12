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

    for (int block_col_idx = 0; block_col_idx < num_block_cols;
         block_col_idx++) {
        for (int block_row_idx = 0; block_row_idx < num_block_rows;
             ++block_row_idx) {
            // // Local block row size.
            // const int row_size = fminf(
            //         num_boxes - block_row_idx * NMS_BLOCK_SIZE,
            //         NMS_BLOCK_SIZE);
            // // Local block col size.
            // const int col_size = fminf(
            //         num_boxes - block_col_idx * NMS_BLOCK_SIZE,
            //         NMS_BLOCK_SIZE);
        }
    }

    std::cout << "NumsCPUKernel not implemented" << std::endl;
    exit(1);
}

}  // namespace impl
}  // namespace ml
}  // namespace open3d
