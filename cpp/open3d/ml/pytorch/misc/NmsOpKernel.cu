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

#include <iostream>

#include "open3d/ml/impl/misc/Nms.h"
#include "open3d/ml/pytorch/misc/NmsOpKernel.h"
#include "torch/script.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

const int THREADS_PER_BLOCK_NMS = sizeof(unsigned long long) * 8;

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

int64_t NmsCUDA(torch::Tensor boxes,
                torch::Tensor keep,
                double nms_overlap_thresh) {
    // params boxes: (N, 5) [x1, y1, x2, y2, ry]
    // params keep: (N)
    printf("01!\n");

    CHECK_INPUT(boxes);
    CHECK_CONTIGUOUS(keep);

    int boxes_num = boxes.size(0);
    const float *boxes_data = boxes.data_ptr<float>();
    int64_t *keep_data = keep.data_ptr<int64_t>();

    const int col_blocks = DIVUP(boxes_num, THREADS_PER_BLOCK_NMS);

    printf("02!\n");

    unsigned long long *mask_data = NULL;
    CHECK_ERROR(
            cudaMalloc((void **)&mask_data,
                       boxes_num * col_blocks * sizeof(unsigned long long)));
    open3d::ml::impl::NmsCUDAKernel(boxes_data, mask_data, boxes_num,
                                    nms_overlap_thresh);

    std::vector<unsigned long long> mask_cpu(boxes_num * col_blocks);

    CHECK_ERROR(cudaMemcpy(&mask_cpu[0], mask_data,
                           boxes_num * col_blocks * sizeof(unsigned long long),
                           cudaMemcpyDeviceToHost));
    printf("03!\n");

    cudaFree(mask_data);

    unsigned long long remv_cpu[col_blocks];
    memset(remv_cpu, 0, col_blocks * sizeof(unsigned long long));

    int num_to_keep = 0;
    std::cout << "boxes_num: " << boxes_num << std::endl;

    for (int i = 0; i < boxes_num; i++) {
        std::cout << "i: " << i << std::endl;
        int nblock = i / THREADS_PER_BLOCK_NMS;
        int inblock = i % THREADS_PER_BLOCK_NMS;

        std::cout << "nblock: " << nblock << std::endl;
        std::cout << "inblock: " << inblock << std::endl;

        if (!(remv_cpu[nblock] & (1ULL << inblock))) {
            std::cout << "inside: 01" << std::endl;
            keep_data[num_to_keep++] = i;
            std::cout << "inside: 02" << std::endl;
            unsigned long long *p = &mask_cpu[0] + i * col_blocks;
            std::cout << "inside: 03" << std::endl;
            for (int j = nblock; j < col_blocks; j++) {
                remv_cpu[j] |= p[j];
            }
        }
    }
    printf("04!\n");

    if (cudaSuccess != cudaGetLastError()) {
        printf("Error!\n");
    }
    printf("05!\n");

    return num_to_keep;
}
