#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cout << "nie wykryto" << std::endl;
        return 0;
    }

    std::cout << "wykryto " << deviceCount  << std::endl;

    return 0;
}