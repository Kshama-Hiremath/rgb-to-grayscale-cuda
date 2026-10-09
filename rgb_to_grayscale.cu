// RGB to Grayscale using CUDA C++
// Input: synthetic RGB pixels (no external image library required)
// Compile: nvcc -O2 rgb_to_grayscale.cu -o rgb_to_grayscale
// Run:     ./rgb_to_grayscale
//
// Grayscale formula: Y = 0.299R + 0.587G + 0.114B
// One CUDA thread processes one pixel.

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cstdlib>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << "\n"; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

struct RGB {
    unsigned char r, g, b;
};

__global__ void rgbToGrayscale(const RGB* input, unsigned char* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float gray = 0.299f * input[i].r
                   + 0.587f * input[i].g
                   + 0.114f * input[i].b;
        output[i] = static_cast<unsigned char>(gray + 0.5f);
    }
}

void cpuGrayscale(const std::vector<RGB>& input,
                  std::vector<unsigned char>& output) {
    for (size_t i = 0; i < input.size(); ++i) {
        float gray = 0.299f * input[i].r
                   + 0.587f * input[i].g
                   + 0.114f * input[i].b;
        output[i] = static_cast<unsigned char>(gray + 0.5f);
    }
}

int main() {
    const int n = 1'000'000; // one million pixels
    const int threadsPerBlock = 256;
    const int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    std::vector<RGB> input(n);
    std::vector<unsigned char> cpuOutput(n), gpuOutput(n);

    // Reproducible synthetic RGB input.
    for (int i = 0; i < n; ++i) {
        input[i].r = static_cast<unsigned char>((i * 17) % 256);
        input[i].g = static_cast<unsigned char>((i * 31 + 40) % 256);
        input[i].b = static_cast<unsigned char>((i * 13 + 90) % 256);
    }

    auto cpuStart = std::chrono::high_resolution_clock::now();
    cpuGrayscale(input, cpuOutput);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(
        cpuEnd - cpuStart).count();

    RGB* dInput = nullptr;
    unsigned char* dOutput = nullptr;
    CUDA_CHECK(cudaMalloc(&dInput, n * sizeof(RGB)));
    CUDA_CHECK(cudaMalloc(&dOutput, n * sizeof(unsigned char)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Include transfer + kernel + transfer in the end-to-end GPU time.
    auto gpuStart = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(dInput, input.data(), n * sizeof(RGB),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(start));
    rgbToGrayscale<<<blocks, threadsPerBlock>>>(dInput, dOutput, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernelMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernelMs, start, stop));
    CUDA_CHECK(cudaMemcpy(gpuOutput.data(), dOutput, n * sizeof(unsigned char),
                          cudaMemcpyDeviceToHost));
    auto gpuEnd = std::chrono::high_resolution_clock::now();
    double gpuTotalMs = std::chrono::duration<double, std::milli>(
        gpuEnd - gpuStart).count();

    int mismatches = 0;
    for (int i = 0; i < n; ++i) {
        if (std::abs((int)cpuOutput[i] - (int)gpuOutput[i]) > 1)
            ++mismatches;
    }

    std::cout << "CUDA RGB-to-Grayscale Case Study\n";
    std::cout << "GPU: ";
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << prop.name << "\n";
    std::cout << "Pixels: " << n << "\n";
    std::cout << "Threads per block: " << threadsPerBlock << "\n";
    std::cout << "Blocks: " << blocks << "\n";
    std::cout << "CPU time (ms): " << cpuMs << "\n";
    std::cout << "GPU kernel time (ms): " << kernelMs << "\n";
    std::cout << "GPU end-to-end time (ms): " << gpuTotalMs << "\n";
    std::cout << "CPU/GPU end-to-end speedup: " << cpuMs / gpuTotalMs << "x\n";
    std::cout << "Mismatched pixels (tolerance 1): " << mismatches << "\n";
    std::cout << "First 10 grayscale values: ";
    for (int i = 0; i < 10; ++i) std::cout << (int)gpuOutput[i] << " ";
    std::cout << "\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(dInput));
    CUDA_CHECK(cudaFree(dOutput));
    CUDA_CHECK(cudaDeviceReset());

    return mismatches == 0 ? 0 : 2;
}
