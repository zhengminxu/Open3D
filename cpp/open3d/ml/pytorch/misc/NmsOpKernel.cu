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

#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

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

static void SortIndices(float *values, int64_t *sort_indices, int64_t N) {
    // Cast to thrust device pointer.
    thrust::device_ptr<float> values_dptr = thrust::device_pointer_cast(values);
    thrust::device_ptr<int64_t> sort_indices_dptr =
            thrust::device_pointer_cast(sort_indices);

    // Fill sort_indices with 0, 1, ..., N-1.
    thrust::sequence(sort_indices_dptr, sort_indices_dptr + N, 0);

    // Sort values and sort_indices together.
    thrust::stable_sort_by_key(values_dptr, values_dptr + N, sort_indices_dptr);
}

__global__ void nms_kernel(const float *boxes,
                           uint64_t *mask,
                           const int N,
                           const float nms_overlap_thresh) {
    // boxes: (N, 5)
    // mask:  (N, N/BS)
    //
    // Kernel launch
    // blocks : (N/BS, N/BS)
    // threads: BS

    const int NMS_BLOCK_SIZE = open3d::ml::impl::NMS_BLOCK_SIZE;

    // Row-wise block index.
    const int block_row_idx = blockIdx.y;
    // Column-wise block index.
    const int block_col_idx = blockIdx.x;

    // Local block row size.
    const int row_size =
            fminf(N - block_row_idx * NMS_BLOCK_SIZE, NMS_BLOCK_SIZE);
    // Local block col size.
    const int col_size =
            fminf(N - block_col_idx * NMS_BLOCK_SIZE, NMS_BLOCK_SIZE);

    // Cololum-wise number of blocks.
    const int num_block_cols = DIVUP(N, NMS_BLOCK_SIZE);

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
            if (open3d::ml::impl::iou_bev(boxes + src_idx * 5,
                                          block_boxes + dst_idx * 5) >
                nms_overlap_thresh) {
                t |= 1ULL << dst_idx;
            }
            dst_idx++;
        }
        mask[src_idx * num_block_cols + block_col_idx] = t;
    }
}

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
    const int N = boxes.size(0);
    const int num_block_cols = DIVUP(N, open3d::ml::impl::NMS_BLOCK_SIZE);

    // Compute sort indices.
    int64_t *sort_indices = nullptr;
    CHECK_ERROR(cudaMalloc((void **)&sort_indices, N * sizeof(int64_t)));
    torch::Tensor scores_copy = scores.clone();
    SortIndices(scores.clone().data_ptr<float>(), sort_indices, N);

    // Allocate masks on device.
    uint64_t *mask_ptr = nullptr;
    CHECK_ERROR(cudaMalloc((void **)&mask_ptr,
                           N * num_block_cols * sizeof(uint64_t)));

    std::vector<int64_t> keep_indices{1, 2, 3};
    torch::Tensor keep_tensor =
            torch::from_blob(keep_indices.data(),
                             {static_cast<int64_t>(keep_indices.size())},
                             torch::TensorOptions().dtype(torch::kLong))
                    .to(boxes.device());

    CHECK_ERROR(cudaFree(sort_indices));
    return keep_tensor;
}

int64_t NmsCUDA(torch::Tensor boxes,
                torch::Tensor keep,
                double nms_overlap_thresh) {
    CHECK_CUDA(boxes);
    CHECK_CONTIGUOUS(boxes);
    CHECK_CONTIGUOUS(keep);

    const int N = boxes.size(0);
    const int num_block_cols = DIVUP(N, open3d::ml::impl::NMS_BLOCK_SIZE);

    // Allocate masks on device.
    uint64_t *mask_ptr = nullptr;
    CHECK_ERROR(cudaMalloc((void **)&mask_ptr,
                           N * num_block_cols * sizeof(uint64_t)));

    // Call kernel. Results will be saved in masks.
    open3d::ml::impl::NmsCUDAKernel(boxes.data_ptr<float>(), mask_ptr, N,
                                    nms_overlap_thresh);

    // Copy cuda masks to cpu.
    std::vector<uint64_t> mask_cpu(N * num_block_cols);
    CHECK_ERROR(cudaMemcpy(mask_cpu.data(), mask_ptr,
                           N * num_block_cols * sizeof(uint64_t),
                           cudaMemcpyDeviceToHost));
    cudaFree(mask_ptr);

    // Write to keep.
    // remv_cpu has N bits in total. If the bit is 1, the corresponding
    // box will be removed.
    std::vector<uint64_t> remv_cpu(num_block_cols, 0);
    int64_t *keep_ptr = keep.data_ptr<int64_t>();
    int num_to_keep = 0;
    for (int i = 0; i < N; i++) {
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
