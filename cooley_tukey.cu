#include "common.hu"
#include <cooperative_groups.h>

#define CUDART_PI_F 3.141592654f

__device__ __forceinline__ DATA_TYPE mul(DATA_TYPE a, DATA_TYPE b)
{
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ __forceinline__ DATA_TYPE add(DATA_TYPE a, DATA_TYPE b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}

__device__ __forceinline__ DATA_TYPE twiddle(DATA_TYPE a, size_t n, size_t block, size_t row)
{
    // todo: static
    float f = (-2 * CUDART_PI_F * block * row) / (n * 4);
    return mul(a, make_float2(cosf(f), sinf(f)));
}

__global__ void fft_kernel(DATA_TYPE *output, DATA_TYPE *data, size_t n)
{
    __shared__ DATA_TYPE sm[512];
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;

    const float f_real[] = { 
        1.f,  1.f,  1.f,  1.f,
        1.f,  0.f, -1.f,  0.f,
        1.f, -1.f,  1.f, -1.f,
        1.f,  0.f, -1.f,  0.f
    };

    const float f_imag[] = {
        0.f,  0.f, 0.f,  0.f,
        0.f, -1.f, 0.f,  1.f,
        0.f,  0.f, 0.f,  0.f,
        0.f,  1.f, 0.f, -1.f
    };

    size_t gblock = id / n;
    size_t block  = gblock % 4;
    size_t offset = (gblock - block) * n;
    size_t pos    = id - offset;
    size_t row    = pos / 4;
    size_t col    = id % 4;

    size_t idxl4 = col * 4;
    DATA_TYPE f_0 = make_float2(f_real[idxl4 + 0], f_imag[idxl4 + 0]);
    DATA_TYPE f_1 = make_float2(f_real[idxl4 + 1], f_imag[idxl4 + 1]);
    DATA_TYPE f_2 = make_float2(f_real[idxl4 + 2], f_imag[idxl4 + 2]);
    DATA_TYPE f_3 = make_float2(f_real[idxl4 + 3], f_imag[idxl4 + 3]);

    // transpose, multiply by twiddle factor (W^(col, row))
    DATA_TYPE x = (n >= 4) ? twiddle(data[offset + col * n + row], n, row, col) : data[id];

    sm[threadIdx.x] = x;
    __syncwarp();

    // gemm
    output[offset + col * n + row] =
        add(add(mul(sm[threadIdx.x - col + 0], f_0), mul(sm[threadIdx.x - col + 1], f_1)),
            add(mul(sm[threadIdx.x - col + 2], f_2), mul(sm[threadIdx.x - col + 3], f_3)));
}

__global__ void nested_transpose_kernel(DATA_TYPE *output, DATA_TYPE *data, ushort *tra_map)
{
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    output[id] = data[tra_map[id]];
}

void fft(DATA_TYPE *output, DATA_TYPE *data, ushort *tra_map)
{
    nested_transpose_kernel<<<DATA_SIZE / 128, 128>>>(output, data, tra_map);
    for (size_t n = 1; n <= DATA_SIZE / 16; n *= 16) {
        fft_kernel<<<DATA_SIZE / 128, 128>>>(data, output, n);
        fft_kernel<<<DATA_SIZE / 128, 128>>>(output, data, n * 4);
    }
}

std::vector<float> benchmark(DATA_TYPE *output,
                             DATA_TYPE *data,
                             cudaEvent_t start, cudaEvent_t stop)
{
    DATA_TYPE *dev_output, *dev_middle, *dev_data, *middle;
    ushort *dev_tra_map, *tra_map_r, *tra_map;
    std::vector<float> time(2);

    /*
      Setup
    */
    cudaCheckReturn(cudaMallocHost(&middle, DATA_SIZE * sizeof(DATA_TYPE)));
    cudaCheckReturn(cudaMallocHost(&tra_map_r, DATA_SIZE * sizeof(ushort)));
    cudaCheckReturn(cudaMallocHost(&tra_map, DATA_SIZE * sizeof(ushort)));

    cudaCheckReturn(cudaMalloc(&dev_data,      DATA_SIZE * sizeof(DATA_TYPE)));
    cudaCheckReturn(cudaMalloc(&dev_middle,    DATA_SIZE * sizeof(DATA_TYPE)));
    cudaCheckReturn(cudaMalloc(&dev_output,    DATA_SIZE * sizeof(DATA_TYPE)));

    for (size_t i = 0; i < DATA_SIZE; i++)
        tra_map[i] = i;

    for (size_t n = DATA_SIZE; n >= 16; n /= 4) {
        size_t blocknum = DATA_SIZE / n;
        for (size_t i = 0; i < blocknum; i++) {
            size_t offset = n * i;
            for (size_t j = 0; j < n; j++) {
                size_t row = j / 4;
                size_t col = j % 4;
                tra_map_r[offset + col * (n / 4) + row] = tra_map[offset + j];
            }
            for (size_t j = 0; j < n; j++) {
                tra_map[offset + j] = tra_map_r[offset + j];
            }
        }
    }

    for (size_t i = 0; i < DATA_SIZE; i++) {
        tra_map[tra_map_r[i]] = i;
    }

    cudaCheckReturn(cudaMalloc(&dev_tra_map,   DATA_SIZE * sizeof(ushort)));
    cudaCheckReturn(cudaMemcpy(dev_tra_map, tra_map, DATA_SIZE * sizeof(ushort),
                               cudaMemcpyHostToDevice));

    cudaCheckReturn(cudaMemcpy(dev_data, data, DATA_SIZE * sizeof(DATA_TYPE),
                               cudaMemcpyHostToDevice));

    cufftHandle plan;
    cufftCheckReturn(cufftCreate(&plan));
    long long len = DATA_SIZE;
    size_t ws = 0;

    cufftCheckReturn(
        cufftXtMakePlanMany(
            plan, 1,  &len,
            NULL, 1, 1, CUDA_C_32F,
            NULL, 1, 1, CUDA_C_32F,
            1, &ws, CUDA_C_32F));

    /*
      FFT
    */
    cudaCheckReturn(cudaDeviceSynchronize());
    cudaCheckReturn(cudaEventRecord(start));

    // cufftCheckReturn(cufftXtExec(plan, dev_data, dev_middle, CUFFT_FORWARD));
    fft(dev_middle, dev_data, dev_tra_map);

    cudaCheckReturn(cudaEventRecord(stop));
    cudaCheckReturn(cudaEventSynchronize(stop));
    cudaCheckKernel();

    cudaCheckReturn(cudaEventElapsedTime(&time[0], start, stop));

    /*
      Scaling
    */
    cudaCheckReturn(cudaMemcpy(middle, dev_middle, DATA_SIZE * sizeof(DATA_TYPE),
                               cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < DATA_SIZE; i++) {
        float2 m = middle[i];
        m.x /= DATA_SIZE;
        m.y /= DATA_SIZE;
        middle[i] = m;
    }    

    cudaCheckReturn(cudaMemcpy(dev_middle, middle, DATA_SIZE * sizeof(DATA_TYPE),
                               cudaMemcpyHostToDevice));

    /*
      IFFT
    */
    cudaCheckReturn(cudaDeviceSynchronize());
    cudaCheckReturn(cudaEventRecord(start));

    cufftCheckReturn(cufftXtExec(plan, dev_middle, dev_output, CUFFT_INVERSE));

    cudaCheckReturn(cudaEventRecord(stop));
    cudaCheckReturn(cudaEventSynchronize(stop));
    cudaCheckKernel();

    cudaCheckReturn(cudaEventElapsedTime(&time[1], start, stop));

    /*
      Close
    */
    cufftCheckReturn(cufftDestroy(plan));

    cudaCheckReturn(cudaMemcpy(output, dev_output, DATA_SIZE * sizeof(DATA_TYPE),
                               cudaMemcpyDeviceToHost));

    cudaCheckReturn(cudaFreeHost(middle));
    cudaCheckReturn(cudaFreeHost(tra_map_r));
    cudaCheckReturn(cudaFreeHost(tra_map));

    cudaCheckReturn(cudaFree(dev_output));
    cudaCheckReturn(cudaFree(dev_middle));
    cudaCheckReturn(cudaFree(dev_data));
    cudaCheckReturn(cudaFree(dev_tra_map));

    return time;
}
