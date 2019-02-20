#include "common2.hu"
#include <cmath>

#define PI 3.141592654

__global__ void fft(size_t M, DATA_TYPE *output, DATA_TYPE *data)
{
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;

    double real = 0.0;
    double imag = 0.0;

    if (id < M) {
        double pow = 2 * PI * id / (double)M;

        for (size_t i = 0; i < M; i++) {
            /*
                              r * cos(2pkl/n) + i * sin(2pkl/n)
                                            - r * sin(2pkl/n) + i * con(2pkl/n)
            */
            DATA_TYPE d = data[i];
            double powh = fmod(i * pow, 2 * PI);

            real +=   d.x * cos(powh) + d.y * sin(powh);
            imag += - d.x * sin(powh) + d.y * cos(powh);
        }

        output[id] = make_float2(real, imag);
    }
}

__global__ void ifft(size_t M, DATA_TYPE *output, DATA_TYPE *data)
{
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;

    double real = 0.0;
    double imag = 0.0;

    if (id < M) {
        double pow = 2 * PI * id / (double)M;

        for (size_t i = 0; i < M; i++) {
            /*
                            r * cos(2pkl/n) - i * sin(2pkl/n)
                                          r * sin(2pkl/n) + i * con(2pkl/n)
            */
            DATA_TYPE d = data[i];
            double powh = fmod(i * pow, 2 * PI);

            real += d.x * cos(powh) - d.y * sin(powh);
            imag += d.x * sin(powh) + d.y * cos(powh);
        }

        output[id] = make_float2(real, imag);
    }
}

__inline__ __host__ __device__ double calc_w(double r, double n)
{
    return (- (2 * PI * r) / n);
}

__global__ void kernel_setup_wn(int N, DATA_TYPE *wn)
{
    size_t r = blockIdx.x * blockDim.x + threadIdx.x;

    if (r <= N - 1) {
        double np = calc_w((1 / (double)2) * r * r, N);
        wn[r].x = cos(np);
        wn[r].y = sin(np);
    }
}

__global__ void kernel0(int N, int M, DATA_TYPE *x, DATA_TYPE *wn, DATA_TYPE *y)
{
    size_t l = blockIdx.x * blockDim.x + threadIdx.x;

    if (l <= N - 1) {
        y[l].x = x[l].x * wn[l].x - x[l].y * wn[l].y;
        y[l].y = x[l].x * wn[l].y + x[l].y * wn[l].x;
    }
    else if (l <= M - 1) {
        y[l].x = 0.f;
        y[l].y = 0.f;
    }
}

__global__ void kernel1(int M, DATA_TYPE *CY, DATA_TYPE *hh, DATA_TYPE *hCZ)
{
    size_t r = blockIdx.x * blockDim.x + threadIdx.x;

    if (r <= M - 1) {
       hCZ[r].x = CY[r].x * hh[r].x - CY[r].y * hh[r].y;
       hCZ[r].y = CY[r].x * hh[r].y + CY[r].y * hh[r].x;
    }
}

__global__ void kernel2(int N, int M, DATA_TYPE *CZ, DATA_TYPE *wn, DATA_TYPE *CX)
{
    size_t r = blockIdx.x * blockDim.x + threadIdx.x;

    if (r <= N - 1) {
      CX[r].x = CZ[r].x * wn[r].x - CZ[r].y * wn[r].y;
      CX[r].y = CZ[r].x * wn[r].y + CZ[r].y * wn[r].x;
    }
}

float bluestein(int size, cudaEvent_t start, cudaEvent_t stop,
                DATA_TYPE *dev_data, DATA_TYPE *dev_middle)
{
    float time;

    size_t N = size;
    size_t M = pow(2.0, ceil(log2((double)(N - 1)) + 1));

    cufftHandle plan;
    cufftCheckReturn(cufftCreate(&plan));
    long long len = M;
    size_t ws = 0;

    cufftCheckReturn(
        cufftXtMakePlanMany(
            plan, 1,  &len,
            NULL, 1, 1, CUDA_C_32F,
            NULL, 1, 1, CUDA_C_32F,
            1, &ws, CUDA_C_32F));

    DATA_TYPE *h, *dev_h, *dev_hh, *dev_wn;

    cudaCheckReturn(cudaMallocHost(&h,  M * sizeof(DATA_TYPE)));
    cudaCheckReturn(cudaMalloc(&dev_h,  M * sizeof(DATA_TYPE)));
    cudaCheckReturn(cudaMalloc(&dev_hh, M * sizeof(DATA_TYPE)));

    cudaCheckReturn(cudaMalloc(&dev_wn, M * sizeof(DATA_TYPE)));

    for (int l = 0; l <= N-1; l++) {
      double p = calc_w((- 1 / (double)2) * (l * l), N);;
      h[l].x = cos(p);
      h[l].y = sin(p);
    }
    for (int l = M - N + 1; l <= M - 1; l++) {
      h[l] = h[M - l];
    }
    for (int l = N; l <= M - N; l++) {
      h[l].x = 0.f;
      h[l].y = 0.f;
    }

    cudaCheckReturn(cudaMemcpy(dev_h, h, M * sizeof(DATA_TYPE),
                               cudaMemcpyHostToDevice));

    cufftCheckReturn(cufftXtExec(plan, dev_h, dev_hh, CUFFT_FORWARD));

    const size_t tbsize = 128;

    kernel_setup_wn<<<((N + tbsize - 1) / tbsize), tbsize>>>(N, dev_wn);

    cudaCheckReturn(cudaDeviceSynchronize());
    cudaCheckReturn(cudaEventRecord(start));

    kernel0<<<((M + tbsize - 1) / tbsize), tbsize>>>(N, M, dev_data, dev_wn, dev_middle);

    //fft<<<((M + tbsize - 1) / tbsize), tbsize>>>(M, dev_data, dev_middle);
    cufftCheckReturn(cufftXtExec(plan, dev_middle, dev_data, CUFFT_FORWARD));

    kernel1<<<((M + tbsize - 1) / tbsize), tbsize>>>(M, dev_data, dev_hh, dev_middle);

    //ifft<<<((M + tbsize - 1) / tbsize), tbsize>>>(M, dev_data, dev_middle);
    cufftCheckReturn(cufftXtExec(plan, dev_middle, dev_data, CUFFT_INVERSE));

    kernel2<<<((N + tbsize - 1) / tbsize), tbsize>>>(N, M, dev_data, dev_wn, dev_middle);

    cudaCheckReturn(cudaEventRecord(stop));
    cudaCheckReturn(cudaEventSynchronize(stop));

    cudaCheckReturn(cudaEventElapsedTime(&time, start, stop));

    cudaCheckReturn(cudaFreeHost(h));
    cudaCheckReturn(cudaFree(dev_h));
    cudaCheckReturn(cudaFree(dev_hh));
    cudaCheckReturn(cudaFree(dev_wn));

    cufftCheckReturn(cufftDestroy(plan));

    return time;
}

std::vector<float> benchmark(int size, DATA_TYPE *output,
                             DATA_TYPE *data,
                             cudaEvent_t start, cudaEvent_t stop)
{
    DATA_TYPE *dev_output, *dev_middle, *dev_data, *middle;
    std::vector<float> time(2);
    
    size_t N = size;
    size_t M = pow(2.0, ceil(log2((double)(N - 1)) + 1));

    /*
      Setup
    */
    cudaCheckReturn(cudaMallocHost(&middle, M * sizeof(DATA_TYPE)));

    cudaCheckReturn(cudaMalloc(&dev_data,   M * sizeof(DATA_TYPE)));
    cudaCheckReturn(cudaMalloc(&dev_middle, M * sizeof(DATA_TYPE)));
    cudaCheckReturn(cudaMalloc(&dev_output, M * sizeof(DATA_TYPE)));

    cudaCheckReturn(cudaMemcpy(dev_data, data, N * sizeof(DATA_TYPE),
                               cudaMemcpyHostToDevice));

    cufftHandle plan;
    cufftCheckReturn(cufftCreate(&plan));
    long long len = N;
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
    time[0] = bluestein(size, start, stop, dev_data, dev_middle);
    cudaCheckKernel();

    /*
      Scaling
    */
    cudaCheckReturn(cudaMemcpy(middle, dev_middle, N * sizeof(DATA_TYPE),
                               cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < size; i++) {
        float2 m = middle[i];
        m.x /= N * M;
        m.y /= N * M;
        middle[i] = m;
    }    

    cudaCheckReturn(cudaMemcpy(dev_middle, middle, N * sizeof(DATA_TYPE),
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

    cudaCheckReturn(cudaMemcpy(output, dev_middle, N * sizeof(DATA_TYPE),
                               cudaMemcpyDeviceToHost));

    cudaCheckReturn(cudaFreeHost(middle));

    cudaCheckReturn(cudaFree(dev_output));
    cudaCheckReturn(cudaFree(dev_middle));
    cudaCheckReturn(cudaFree(dev_data));

    return time;
}
