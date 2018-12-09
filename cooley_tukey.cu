#include "common.hu"

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

__device__ void _fft(DATA_TYPE *data, DATA_TYPE *fm, DATA_TYPE *workspace, size_t n, size_t id, cublasHandle_t handle)
{
    if (n > 4) {
        _fft(data, fm, workspace, n / 4, id, handle);
        __syncthreads();
    }

    size_t gblock = id / n;
    size_t offset = gblock * n;
    size_t block = gblock - (gblock / 4) * 4;
    size_t pos   = id - offset;
    size_t row   = pos / 4;
    size_t col   = pos - row;

    // transpose, multiply by twiddle factor (W^(block, row))
    DATA_TYPE x = twiddle(data[offset + col * n + row], n, block, row);

    // gemm
    half *a_real = (half*)workspace + DATA_SIZE * 0;
    half *a_imag = (half*)workspace + DATA_SIZE * 1;
    half *c_real = (half*)workspace + DATA_SIZE * 2;
    half *c_imag = (half*)workspace + DATA_SIZE * 3;

    half *fm_real = (half*)fm;
    half *fm_imag = (half*)fm + 16;

    a_real[id] = __high2half(x);
    a_imag[id] = __low2half(x);

    /* real part */
    half alpha = 1;
    half beta = 0;

    cublasCheckReturn(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  DATA_SIZE / 4, 4, 4,
                                  &alpha,
                                  a_real, DATA_SIZE,
                                  fm_real, 16,
                                  &beta,
                                  c_real, DATA_SIZE));

    alpha = -1;
    beta = 1;

    cublasCheckReturn(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  DATA_SIZE / 4, 4, 4,
                                  &alpha,
                                  a_imag, DATA_SIZE,
                                  fm_imag, 16,
                                  &beta,
                                  c_real, DATA_SIZE));

    /* imaginal part */
    alpha = 1;
    beta = 0;

    cublasCheckReturn(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  DATA_SIZE / 4, 4, 4,
                                  &alpha,
                                  a_real, DATA_SIZE,
                                  fm_imag, 16,
                                  &beta,
                                  c_real, DATA_SIZE));

    alpha = 1;
    beta = 1;

    cublasCheckReturn(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  DATA_SIZE / 4, 4, 4,
                                  &alpha,
                                  a_imag, DATA_SIZE,
                                  fm_real, 16,
                                  &beta,
                                  c_real, DATA_SIZE));

    /* finish */
    data[id] = __halves2half2(c_real[id], c_imag[id]);
}

__global__ void fft(DATA_TYPE *data, DATA_TYPE *fm, DATA_TYPE *workspace)
{
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;

    cublasHandle_t handle;
    cublasCheckReturn(cublasCreate(&handle));
    //cublasCheckReturn(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    _fft(data, fm, workspace, DATA_SIZE / 4, id, handle);

    cublasCheckReturn(cublasDestroy(handle));
}

std::vector<float> benchmark(DATA_TYPE *output,
                             DATA_TYPE *data,
                             cudaEvent_t start, cudaEvent_t stop)
{
    DATA_TYPE *dev_output, *dev_middle, *dev_data, *middle, *dev_workspace, *dev_fm, *fm;
    std::vector<float> time(2);

    /*
      Setup
    */
    cudaCheckReturn(cudaMallocHost(&middle, DATA_SIZE * sizeof(DATA_TYPE)));
    cudaCheckReturn(cudaMallocHost(&fm, 4 * 4 * sizeof(DATA_TYPE)));

    cudaCheckReturn(cudaMalloc(&dev_data,      DATA_SIZE * sizeof(DATA_TYPE)));
    cudaCheckReturn(cudaMalloc(&dev_middle,    DATA_SIZE * sizeof(DATA_TYPE)));
    cudaCheckReturn(cudaMalloc(&dev_output,    DATA_SIZE * sizeof(DATA_TYPE)));
    cudaCheckReturn(cudaMalloc(&dev_workspace, 2 * DATA_SIZE * sizeof(DATA_TYPE)));
    cudaCheckReturn(cudaMalloc(&dev_fm,        DATA_SIZE * sizeof(DATA_TYPE)));

    cudaCheckReturn(cudaMemcpy(dev_middle, data, DATA_SIZE * sizeof(DATA_TYPE),
                               cudaMemcpyHostToDevice));

    {
        const float ff[] = { 
            // real part
            1.f,  1.f,  1.f,  1.f,
            1.f,  0.f, -1.f,  0.f,
            1.f, -1.f,  1.f, -1.f,
            1.f,  0.f, -1.f,  0.f,

            // imaginal part
            0.f,  0.f, 0.f,  0.f,
            0.f,  1.f, 0.f, -1.f,
            0.f,  0.f, 0.f,  0.f,
            0.f, -1.f, 0.f,  1.f
        };

        for (size_t i = 0; i < 4 * 4; i++) {
            fm[i] = __floats2half2_rn(ff[i*2], ff[i*2+1]);
        }

        cudaCheckReturn(cudaMemcpy(dev_fm, fm, 4 * 4 * sizeof(DATA_TYPE),
                                   cudaMemcpyHostToDevice));
    }

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
    fft<<<DATA_SIZE / 256, 256>>>(dev_middle, dev_fm, dev_workspace);

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
