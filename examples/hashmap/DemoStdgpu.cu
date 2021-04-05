#include <stdgpu/iterator.h>  // device_begin, device_end
#include <stdgpu/memory.h>    // createDeviceArray, destroyDeviceArray
#include <stdgpu/platform.h>  // STDGPU_HOST_DEVICE
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>

#include <iostream>
#include <sstream>
#include <stdgpu/unordered_map.cuh>  // stdgpu::unordered_map
#include <string>

struct GetFirst {
    STDGPU_HOST_DEVICE int operator()(const thrust::pair<int, int>& x) const {
        return x.first;
    }
};

struct GetSecond {
    STDGPU_HOST_DEVICE int operator()(const thrust::pair<int, int>& x) const {
        return x.second;
    }
};

__global__ void InsertKeyValuePair(
        const int* d_keys,
        const int* d_values,
        int num_keys,
        stdgpu::unordered_map<int, int> key_to_value) {
    // stdgpu::unordered_map pass-by-value does not actually copy.
    int workload_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (workload_idx >= num_keys) {
        return;
    }
    key_to_value.emplace(d_keys[workload_idx], d_values[workload_idx]);
}

template <typename T>
std::string Join(const T& vals, const std::string& delimeter = ", ") {
    std::ostringstream ss;
    for (const auto& val : vals) {
        if (&val != &vals[0]) {
            ss << delimeter;
        }
        ss << val;
    }
    return ss.str();
}

int main() {
    stdgpu::index_t n = 10;

    // Host keys and values
    std::vector<int> h_keys{0, 1, 2, 2, 2, 3, 4, 4, 5};
    std::vector<int> h_values{0, 10, 20, 20, 20, 30, 40, 40, 50};

    // Allocate device
    int* d_keys = createDeviceArray<int>(n);
    int* d_values = createDeviceArray<int>(n);

    // Copy keys and values to device
    thrust::copy(h_keys.begin(), h_keys.end(), stdgpu::device_begin(d_keys));
    thrust::copy(h_values.begin(), h_values.end(),
                 stdgpu::device_begin(d_values));

    // Create map
    stdgpu::unordered_map<int, int> key_to_value =
            stdgpu::unordered_map<int, int>::createDeviceObject(n);

    // Insert to map
    unsigned int num_keys = static_cast<unsigned int>(h_keys.size());
    unsigned int threads = 32;
    unsigned int blocks = (num_keys + threads - 1) / threads;
    InsertKeyValuePair<<<blocks, threads>>>(
            d_keys, d_values, static_cast<int>(num_keys), key_to_value);
    cudaDeviceSynchronize();
    stdgpu::index_t map_size = key_to_value.size();
    std::cout << "num_keys: " << num_keys << ", map_size: " << map_size
              << std::endl;

    // Get all (unique) keys
    auto key_to_value_range_map = key_to_value.device_range();
    int* d_unique_keys = createDeviceArray<int>(map_size);
    thrust::transform(key_to_value_range_map.begin(),
                      key_to_value_range_map.end(),
                      stdgpu::device_begin(d_unique_keys), GetFirst());
    std::vector<int> h_unique_keys(map_size);
    thrust::device_ptr<int> d_unique_keys_ptr(d_unique_keys);
    thrust::copy(d_unique_keys_ptr, d_unique_keys_ptr + map_size,
                 h_unique_keys.begin());
    std::cout << "h_unique_keys: " << Join(h_unique_keys) << std::endl;

    // Get all values (corresponding to all unique keys)
    int* d_unique_values = createDeviceArray<int>(map_size);
    thrust::transform(key_to_value_range_map.begin(),
                      key_to_value_range_map.end(),
                      stdgpu::device_begin(d_unique_values), GetSecond());
    std::vector<int> h_unique_values(map_size);
    thrust::device_ptr<int> d_unique_values_ptr(d_unique_values);
    thrust::copy(d_unique_values_ptr, d_unique_values_ptr + map_size,
                 h_unique_values.begin());
    std::cout << "h_unique_values: " << Join(h_unique_values) << std::endl;

    // Clean up
    destroyDeviceArray<int>(d_keys);
    destroyDeviceArray<int>(d_values);
    destroyDeviceArray<int>(d_unique_keys);
    destroyDeviceArray<int>(d_unique_values);

    return 0;
}
