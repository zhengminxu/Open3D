template <typename scalar_t>
static OPEN3D_HOST_DEVICE void CUDAAddElementKernel(const void* lhs,
                                                    const void* rhs,
                                                    void* dst) {
    (scalar_t*)(dst) = *(scalar_t*)(lhs) + *(scalar_t*)(rhs);
}

void Add(const Tensor& lhs, const Tensor& rhs, Tensor& dst) {
    Indexer indexer({lhs, rhs}, dst, DtypePolicy::ALL_SAME);
    DISPATCH_DTYPE(lhs.GetDtype(), [&]() {
        LaunchBinaryEWKernel(
                indexer,
                [] OPEN3D_HOST_DEVICE(const void* lhs, void* rhs, void* dst) {
                    CUDAAddElementKernel<scalar_t>(lhs, rhs, dst);
                });
    }
}

// Pseudo code
void AddKernel(T* lhs, T* rhs, T* dst) { *dst = *lhs + *rhs; }

void Add(const Tensor& lhs, const Tensor& rhs, Tensor& dst) {
    Indexer indexer({lhs, rhs}, dst);
    LaunchElementWiseCUDAKernel(indexer, AddKernel);
}
