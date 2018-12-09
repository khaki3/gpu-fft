#include "common.hu"
#include <cooperative_groups.h>

#define CUDART_PI_F 3.141592654f

__device__ __forceinline__ DATA_TYPE mul(DATA_TYPE a, DATA_TYPE b)
{
    // (a_r b_r - a_i b_i), (a_r b_i + a_i b_r)
    DATA_TYPE m  = __floats2half2_rn(-1.0, 1.0);

    // (a_r b_r, a_i b_r)
    DATA_TYPE c = __hmul2(a, __low2half2(b));
    // (- a_i b_i, a_r b_i)
    DATA_TYPE s = __hmul2(m, __hmul2(__lowhigh2highlow(a), __high2half2(b)));

    return __hadd2(c, s);
}

__device__ __forceinline__ DATA_TYPE twiddle(DATA_TYPE a, size_t n, size_t block, size_t row)
{
    // todo: static
    float f = 2 * CUDART_PI_F * block * row / n;
    return mul(a, __floats2half2_rn(cosf(f), - sinf(f)));
}

__global__ void fft(DATA_TYPE *data)
{
    cooperative_groups::grid_group g = cooperative_groups::this_grid();

    __shared__ DATA_TYPE sm[1024];
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;

    const float f_real[] = { 
        1.f,  1.f,  1.f,  1.f,
        1.f,  0.f, -1.f,  0.f,
        1.f, -1.f,  1.f, -1.f,
        1.f,  0.f, -1.f,  0.f
    };

    const float f_imag[] = {
        0.f,  0.f, 0.f,  0.f,
        0.f,  1.f, 0.f, -1.f,
        0.f,  0.f, 0.f,  0.f,
        0.f, -1.f, 0.f,  1.f
    };

    size_t idxl4 = threadIdx.x & 3 * 4;
    DATA_TYPE f_0 = __floats2half2_rn(f_real[idxl4 + 0], f_imag[idxl4 + 0]);
    DATA_TYPE f_1 = __floats2half2_rn(f_real[idxl4 + 1], f_imag[idxl4 + 1]);
    DATA_TYPE f_2 = __floats2half2_rn(f_real[idxl4 + 2], f_imag[idxl4 + 2]);
    DATA_TYPE f_3 = __floats2half2_rn(f_real[idxl4 + 3], f_imag[idxl4 + 3]);

    for (size_t n  = 4; n <= DATA_SIZE / 4; n *= 4) {
        if (n != 4) {
            g.sync();
        }

        size_t gblock = id / n;
        size_t offset = gblock * n;
        size_t block = gblock - (gblock / 4) * 4;
        size_t pos   = id - offset;
        size_t row   = pos / 4;
        size_t col   = pos - row * 4;

        // transpose, multiply by twiddle factor (W^(block, row))
        DATA_TYPE x = twiddle(data[offset + col * n + row], n, block, row);

        sm[threadIdx.x] = x;
        __syncwarp();

        // gemm
        data[id] = __hadd2(__hadd2(mul(sm[threadIdx.x - col + 0], f_0),
                                   mul(sm[threadIdx.x - col + 1], f_1)),
                           __hadd2(mul(sm[threadIdx.x - col + 2], f_2),
                                   mul(sm[threadIdx.x - col + 3], f_3)));
    }
}

std::vector<float> benchmark(DATA_TYPE *output,
                             DATA_TYPE *data,
                             cudaEvent_t start, cudaEvent_t stop)
{
    DATA_TYPE *dev_output, *dev_middle, *dev_data, *middle;
    std::vector<float> time(2);

    /*
      Setup
    */
    cudaCheckReturn(cudaMallocHost(&middle, DATA_SIZE * sizeof(DATA_TYPE)));

    cudaCheckReturn(cudaMalloc(&dev_data,      DATA_SIZE * sizeof(DATA_TYPE)));
    cudaCheckReturn(cudaMalloc(&dev_middle,    DATA_SIZE * sizeof(DATA_TYPE)));
    cudaCheckReturn(cudaMalloc(&dev_output,    DATA_SIZE * sizeof(DATA_TYPE)));

    cudaCheckReturn(cudaMemcpy(dev_middle, data, DATA_SIZE * sizeof(DATA_TYPE),
                               cudaMemcpyHostToDevice));

    cufftHandle plan;
    cufftCheckReturn(cufftCreate(&plan));
    long long len = DATA_SIZE;
    size_t ws = 0;

    cufftCheckReturn(
        cufftXtMakePlanMany(
            plan, 1,  &len,
            NULL, 1, 1, CUDA_C_16F,
            NULL, 1, 1, CUDA_C_16F,
            1, &ws, CUDA_C_16F));

    /*
      FFT
    */
    cudaCheckReturn(cudaDeviceSynchronize());
    cudaCheckReturn(cudaEventRecord(start));

//    cufftCheckReturn(cufftXtExec(plan, dev_data, dev_middle, CUFFT_FORWARD));
    void *args[2];
    args[0] = dev_middle;
    args[1] = NULL;
    cudaLaunchCooperativeKernel((void*)fft, dim3(DATA_SIZE / 1024, 1, 1), dim3(1024, 1, 1), args);

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
        float2 m = __half22float2(middle[i]);
        middle[i] = __floats2half2_rn(m.x / DATA_SIZE, m.y / DATA_SIZE);
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
