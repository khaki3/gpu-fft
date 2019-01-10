#include "common.hu"
#include <cooperative_groups.h>

#define CUDART_PI_F 3.141592654f

__device__ __forceinline__ DATA_TYPE mul(DATA_TYPE a, DATA_TYPE b)
{
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ __forceinline__ DATA_TYPE twiddle(DATA_TYPE a, size_t n, size_t block, size_t row)
{
    // todo: static
    float f = (-2 * CUDART_PI_F * block * row) / n;
    return mul(a, make_float2(cosf(f), sinf(f)));
}

__device__ __forceinline__ void sFFT(DATA_TYPE out[], DATA_TYPE a[], size_t n)
{
    for (int i = 0; i < n; i++) {
        float real = 0.0;
        float imag = 0.0;
        float pow = 2 * CUDART_PI_F * i / (float)n;

        for (int j = 0; j < n; j++) {
            float powh = fmodf(j * pow, 2 * CUDART_PI_F);
            DATA_TYPE d = a[j];

            real +=   d.x * cosf(powh) + d.y * sinf(powh);
            imag += - d.x * sinf(powh) + d.y * cosf(powh);
        }

        out[i] = make_float2(real, imag);
    }
}

__global__ void kernel1_2(DATA_TYPE *output, DATA_TYPE *data, int kernel_id)
{
    DATA_TYPE sample[8];
    DATA_TYPE out[8];
    __shared__ DATA_TYPE block[512];

    for (int i = 0; i < 8; i++)
        sample[i] = data[(blockIdx.x + threadIdx.y * 128 + i * 128 * 8) * 8 + threadIdx.x];

    // 1. 8-point fft
    sFFT(out, sample, 8);

    // 2. transpose through shared memory
    {
        for (int i = 0; i < 8; i++)
            block[(threadIdx.y * blockDim.x + i) * 8 + threadIdx.x] = out[i];

        __syncthreads();

        for (int i = 0; i < 8; i++)
            sample[i] = block[i * blockDim.x * 8 + threadIdx.y * 8 + threadIdx.x];
    }

    // 3. twiddle
    for (int i = 0; i < 8; i++)
        sample[i] = twiddle(sample[i], 64, i, threadIdx.y);

    // 4. 8-point fft
    sFFT(out, sample, 8);

    if (kernel_id == 1) {
        for (int i = 0; i < 8; i++) {
            size_t id = (blockIdx.x + threadIdx.y * 128 + i * 128 * 8);
            size_t row = id / 2;
            size_t rem = id % 2;
            output[(((row % 64) * 64 + row / 64) * 2 + rem) * 8 + threadIdx.x] =
                twiddle(out[i], 64 * 64, id % 64, (id / 64) % 64);
        }
    }

    else {
        for (int i = 0; i < 8; i++) {
            size_t id = (blockIdx.x + threadIdx.y * 128 + i * 128 * 8);
            size_t row = id / 2;
            size_t rem = id % 2;
            output[id * 8 + threadIdx.x] =
                twiddle(out[i], 64 * 64 * 16, row, rem * 8 + threadIdx.x);
        }
    }
}

__global__ void kernel3(DATA_TYPE *output, DATA_TYPE *data)
{
    size_t local_id = threadIdx.y * blockDim.x + threadIdx.x;
    size_t local_pos = local_id * 16;
    size_t global_id = blockIdx.x * blockDim.x * blockDim.y + local_id;
    size_t global_pos = global_id * 16;
    size_t row = global_id;

    DATA_TYPE sample[16];
    DATA_TYPE out[16];

    for (int i = 0; i < 16; i++)
        sample[i] = data[global_pos + i];

    // 1. 16-point fft
    sFFT(out, sample, 16);

    for (int i = 0; i < 16; i++) {
        output[global_pos + i] = out[i];
    }
}

void fft(DATA_TYPE *output, DATA_TYPE *data)
{
    dim3 blockDim1(8, 8, 1);
    dim3 blockDim3(32, 1, 1);
    dim3 gridDim(128);
    kernel1_2<<<gridDim, blockDim1>>>(output, data, 1);
    kernel1_2<<<gridDim, blockDim1>>>(data, output, 2);
    kernel3<<<gridDim, blockDim3>>>(output, data);
}

std::vector<float> benchmark(DATA_TYPE *output,
                             DATA_TYPE *data,
                             cudaEvent_t start, cudaEvent_t stop)
{
    DATA_TYPE *dev_output, *dev_middle, *dev_data, *middle, *middle2;
    std::vector<float> time(2);

    /*
      Setup
    */
    cudaCheckReturn(cudaMallocHost(&middle, DATA_SIZE * sizeof(DATA_TYPE)));
    cudaCheckReturn(cudaMallocHost(&middle2, DATA_SIZE * sizeof(DATA_TYPE)));

    cudaCheckReturn(cudaMalloc(&dev_data,      DATA_SIZE * sizeof(DATA_TYPE)));
    cudaCheckReturn(cudaMalloc(&dev_middle,    DATA_SIZE * sizeof(DATA_TYPE)));
    cudaCheckReturn(cudaMalloc(&dev_output,    DATA_SIZE * sizeof(DATA_TYPE)));

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
    fft(dev_middle, dev_data);

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
        middle2[i] = m;
    }

    for (size_t i = 0; i < 16; i++) {
        for (size_t j = 0; j < 4096; j++) {
            middle[j * 16 + i] = middle2[i * 4096 + j];
        }
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

    cudaCheckReturn(cudaFree(dev_output));
    cudaCheckReturn(cudaFree(dev_middle));
    cudaCheckReturn(cudaFree(dev_data));

    return time;
}
