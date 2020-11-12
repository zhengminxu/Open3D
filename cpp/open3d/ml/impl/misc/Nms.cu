/*
3D IoU Calculation and Rotated NMS(modified from 2D NMS written by others)
Written by Shaoshuai Shi
All Rights Reserved 2018.
*/

#include <stdio.h>

#include "open3d/ml/impl/misc/Nms.h"

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

namespace open3d {
namespace ml {
namespace impl {

__global__ void nms_kernel(const int num_boxes,
                           const float nms_overlap_thresh,
                           const float *boxes,
                           uint64_t *mask) {
    // boxes: (N, 5)
    // mask:  (N, N/BS)
    //
    // Kernel launch
    // blocks : (N/BS, N/BS)
    // threads: BS

    // Row-wise block index.
    const int block_row_idx = blockIdx.y;
    // Column-wise block index.
    const int block_col_idx = blockIdx.x;

    // Local block row size.
    const int row_size =
            fminf(num_boxes - block_row_idx * NMS_BLOCK_SIZE, NMS_BLOCK_SIZE);
    // Local block col size.
    const int col_size =
            fminf(num_boxes - block_col_idx * NMS_BLOCK_SIZE, NMS_BLOCK_SIZE);

    // Cololum-wise number of blocks.
    const int num_block_cols = DIVUP(num_boxes, NMS_BLOCK_SIZE);

    // Fill local block_boxes by fetching the global box memory.
    // block_boxes = boxes[NBS*block_col_idx : NBS*block_col_idx+col_size, :].
    __shared__ float block_boxes[NMS_BLOCK_SIZE * 5];
    if (threadIdx.x < col_size) {
        float *dst = block_boxes + threadIdx.x * 5;
        const float *src =
                boxes + (NMS_BLOCK_SIZE * block_col_idx + threadIdx.x) * 5;
        dst[0] = src[0];
        dst[1] = src[1];
        dst[2] = src[2];
        dst[3] = src[3];
        dst[4] = src[4];
    }
    __syncthreads();

    // Comparing src and dst. In one block, the following src and dst indices
    // are compared:
    // - src: BS * block_row_idx : BS * block_row_idx + row_size
    // - dst: BS * block_col_idx : BS * block_col_idx + col_size
    //
    // With all blocks, all src and dst indices are compared.
    //
    // Result:
    // mask[i, j] is a 64-bit integer where mask[i, j][k] (k counted from right)
    // is 1 iff box[i] overlaps with box[BS*j+k].
    if (threadIdx.x < row_size) {
        // src_idx indices the global memory.
        const int src_idx = NMS_BLOCK_SIZE * block_row_idx + threadIdx.x;
        // dst_idx indices the shared memory.
        int dst_idx = block_row_idx == block_col_idx ? threadIdx.x + 1 : 0;

        uint64_t t = 0;
        while (dst_idx < col_size) {
            if (iou_bev(boxes + src_idx * 5, block_boxes + dst_idx * 5) >
                nms_overlap_thresh) {
                t |= 1ULL << dst_idx;
            }
            dst_idx++;
        }
        mask[src_idx * num_block_cols + block_col_idx] = t;
    }
}

void NmsCUDAKernel(const float *boxes,
                   uint64_t *mask,
                   int num_boxes,
                   float nms_overlap_thresh) {
    dim3 blocks(DIVUP(num_boxes, NMS_BLOCK_SIZE),
                DIVUP(num_boxes, NMS_BLOCK_SIZE));
    dim3 threads(NMS_BLOCK_SIZE);
    nms_kernel<<<blocks, threads>>>(num_boxes, nms_overlap_thresh, boxes, mask);
}

}  // namespace impl
}  // namespace ml
}  // namespace open3d
