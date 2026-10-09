# Image Processing: RGB-to-Grayscale Conversion Using CUDA

## Project Overview

This project implements **RGB-to-Grayscale image processing using CUDA** to demonstrate the effectiveness of GPU-based parallel computing compared with sequential CPU execution.

A color RGB image contains three components: Red, Green, and Blue. Grayscale conversion combines these components into a single intensity value representing the brightness of each pixel.

Instead of processing pixels sequentially on the CPU, the CUDA implementation assigns one GPU thread to each pixel, enabling multiple pixels to be processed in parallel.

## Objectives

- Understand CUDA and GPU-based parallel computing.
- Implement RGB-to-grayscale conversion using CPU and CUDA.
- Compare CPU and GPU execution times for different input sizes.
- Verify output correctness by comparing CPU and GPU results.
- Evaluate performance using speedup calculations.

## Technologies Used

- **Programming Language:** C/C++
- **Parallel Computing:** NVIDIA CUDA
- **Development Environment:** Google Colab
- **GPU:** NVIDIA Tesla T4
- **CUDA Compiler:** NVIDIA CUDA Compiler 13.0

## Grayscale Conversion Formula

The grayscale intensity is calculated using the weighted RGB formula:

`Gray = 0.299R + 0.587G + 0.114B`

Where:
- `R` represents the red intensity.
- `G` represents the green intensity.
- `B` represents the blue intensity.

The calculated intensity is converted to an integer grayscale value.

## Implementation Approach

### CPU Implementation
The CPU processes the pixels sequentially, calculating the grayscale intensity for each pixel.

### CUDA Implementation
The CUDA kernel assigns one GPU thread to each pixel. Threads are organized into blocks, with **256 threads per block** in this experiment.

The number of blocks is calculated using:

`Number of blocks = (N + 255) / 256`

A boundary check ensures that each thread processes only valid pixel indices.

### Execution Workflow

1. Allocate host memory for input and output data.
2. Allocate device memory on the GPU.
3. Copy input data from host memory to device memory.
4. Launch the CUDA kernel to process pixels in parallel.
5. Copy the GPU output back to host memory.
6. Compare CPU and GPU outputs for correctness.
7. Calculate execution time and speedup.

## Experimental Setup

The experiment was conducted in Google Colab using an NVIDIA Tesla T4 GPU and CUDA Compiler 13.0.

Four input sizes were tested: 10,000, 100,000, 1,000,000, and 10,000,000 pixels.

The GPU execution time includes host-to-device transfer, kernel execution, synchronization, and device-to-host transfer. CPU timing measures the sequential conversion.

## Benchmark Results

| Input Size (Pixels) | CPU Time (ms) | GPU Total Time (ms) | Speedup |
|---:|---:|---:|---:|
| 10,000 | 0.0569 | 0.2060 | 0.28x |
| 100,000 | 0.3690 | 0.2869 | 1.29x |
| 1,000,000 | 2.7814 | 1.6536 | 1.68x |
| 10,000,000 | 25.2972 | 13.5599 | 1.87x |

**Correctness:** All tested input sizes passed the CPU-GPU comparison with zero mismatches.

## Performance Analysis

The results show that the GPU was slower for the smallest input because memory transfers and GPU execution overhead outweighed the benefit of parallel processing.

As the input size increased, the GPU achieved better performance relative to the sequential CPU implementation. For 10 million pixels, the measured speedup was approximately **1.87x**.

Speedup is calculated as:

`Speedup = CPU Execution Time / GPU Total Execution Time`

These results demonstrate that GPU acceleration is more beneficial when processing sufficiently large workloads.

## Applications

RGB-to-grayscale conversion is useful in:

- Computer vision and image analysis.
- Image enhancement and preprocessing.
- Edge detection and image segmentation workflows.
- Document processing and OCR preprocessing.
- Machine-learning image preprocessing.

## Conclusion

This project demonstrates the application of CUDA parallel computing to RGB-to-grayscale conversion. The GPU implementation processes independent pixels concurrently and achieved a measured speedup of approximately 1.87x for the largest tested workload.

The experiment also highlights the impact of memory-transfer overhead on small workloads and demonstrates the importance of correctness verification and performance benchmarking.

## How to Run

1. Open the `.ipynb` notebook in Google Colab.
2. Select a GPU runtime if available.
3. Execute the notebook cells in order.
4. Review the CPU and GPU execution times, speedup values, and correctness results.

**Note:** Benchmark results may vary depending on the GPU, runtime environment, and system load.

## Author

**Kshama Shrikantayya Hiremath**

B.E. – Information Science and Engineering  
BMS Institute of Technology and Management  
Academic Year: 2026–2027
