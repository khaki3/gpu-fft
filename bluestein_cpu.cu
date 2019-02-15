#include "common.hu"
#include <cmath>

#define PI 3.141592654f

float calc_w(float r, float n)
{
  return (- (2 * PI * r) / n);
}

float bluestein(cudaEvent_t start, cudaEvent_t stop,
                DATA_TYPE *dev_data, DATA_TYPE *dev_middle)
{
    float time;

    size_t N = DATA_SIZE;
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

    cudaCheckReturn(cudaDeviceSynchronize());
    cudaCheckReturn(cudaEventRecord(start));

    DATA_TYPE *h, *hh, *x, *y, *CY, *hCZ, *CZ, *CX;

    cudaCheckReturn(cudaMallocHost(&h,     M * sizeof(DATA_TYPE)));
    cudaCheckReturn(cudaMallocHost(&hh,    M * sizeof(DATA_TYPE)));
    cudaCheckReturn(cudaMallocHost(&x,     N * sizeof(DATA_TYPE)));
    cudaCheckReturn(cudaMallocHost(&y,     M * sizeof(DATA_TYPE)));
    cudaCheckReturn(cudaMallocHost(&CY,    M * sizeof(DATA_TYPE)));
    cudaCheckReturn(cudaMallocHost(&hCZ,   M * sizeof(DATA_TYPE)));
    cudaCheckReturn(cudaMallocHost(&CZ,    M * sizeof(DATA_TYPE)));
    cudaCheckReturn(cudaMallocHost(&CX,    N * sizeof(DATA_TYPE)));

    for (int l = 0; l <= N-1; l++) {
      float p = calc_w((- 1 / (float)2) * (l * l), N);;
      h[l].x = cosf(p);
      h[l].y = sinf(p);
    }
    for (int l = M - N + 1; l <= M - 1; l++) {
      h[l] = h[M - l];
    }
    for (int l = N; l <= M - N; l++) {
      h[l].x = 0.f;
      h[l].y = 0.f;
    }

    for (int r = 0; r <= M - 1; r++) {
      hh[r].x = 0.f;
      hh[r].y = 0.f;

      for (int l = 0; l <= M - 1; l++) {
        float p = calc_w(r * l, M);
        hh[r].x += h[l].x * cosf(p) - h[l].y * sinf(p);
        hh[r].y += h[l].x * sinf(p) + h[l].y * cosf(p);
      }
    }

    cudaCheckReturn(cudaMemcpy(x, dev_data, N * sizeof(DATA_TYPE),
                               cudaMemcpyDeviceToHost));

    for (int l = 0; l <= N - 1; l++) {
      float p = calc_w((1 / (float)2) * l * l, N);
      y[l].x = x[l].x * cosf(p) - x[l].y * sinf(p);
      y[l].y = x[l].x * sinf(p) + x[l].y * cosf(p);
    }
    for (int l = N; l <= M - 1; l++) {
      y[l].x = 0.f;
      y[l].y = 0.f;
    }

    for (int r = 0; r <= M - 1; r++) {
      CY[r].x = 0.f;
      CY[r].y = 0.f;

      for (int l = 0; l <= M - 1; l++) {
        float p = calc_w(r * l, M);
        CY[r].x += y[l].x * cosf(p) - y[l].y * sinf(p);
        CY[r].y += y[l].x * sinf(p) + y[l].y * cosf(p);
      }
    }

    for (int r = 0; r <= M - 1; r++) {
      hCZ[r].x = CY[r].x * hh[r].x - CY[r].y * hh[r].y;
      hCZ[r].y = CY[r].x * hh[r].y + CY[r].y * hh[r].x;
    }


    for (int r = 0; r <= M - 1; r++) {
      CZ[r].x = 0.f;
      CZ[r].y = 0.f;

      for (int l = 0; l <= M - 1; l++) {
        float p = calc_w(- r * l, M);
        CZ[r].x += hCZ[l].x * cosf(p) - hCZ[l].y * sinf(p);
        CZ[r].y += hCZ[l].x * sinf(p) + hCZ[l].y * cosf(p);
      }

      CZ[r].x /= M;
      CZ[r].y /= M;
    }

    for (int r = 0; r <= N - 1; r++) {
      float p = calc_w((1 / (float)2) * r * r, N);
      CX[r].x = CZ[r].x * cosf(p) - CZ[r].y * sinf(p);
      CX[r].y = CZ[r].x * sinf(p) + CZ[r].y * cosf(p);
    }

    cudaCheckReturn(cudaMemcpy(dev_middle, CX, N * sizeof(DATA_TYPE),
                               cudaMemcpyHostToDevice));

    // cufftCheckReturn(cufftXtExec(plan, dev_data, dev_middle, CUFFT_FORWARD));

    cudaCheckReturn(cudaEventRecord(stop));
    cudaCheckReturn(cudaEventSynchronize(stop));

    cudaCheckReturn(cudaEventElapsedTime(&time, start, stop));

    return time;
}

std::vector<float> benchmark(DATA_TYPE *output,
                             DATA_TYPE *data,
                             cudaEvent_t start, cudaEvent_t stop)
{
    DATA_TYPE *dev_output, *dev_middle, *dev_data, *middle;
    std::vector<float> time(2);
    
    size_t N = DATA_SIZE;
    size_t M = pow(2.0, ceil(log2((double)(N - 1)) + 1));

    /*
      Setup
    */
    cudaCheckReturn(cudaMallocHost(&middle, M * sizeof(DATA_TYPE)));

    cudaCheckReturn(cudaMalloc(&dev_data,   M * sizeof(DATA_TYPE)));
    cudaCheckReturn(cudaMalloc(&dev_middle, M * sizeof(DATA_TYPE)));
    cudaCheckReturn(cudaMalloc(&dev_output, N * sizeof(DATA_TYPE)));

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
    time[0] = bluestein(start, stop, dev_data, dev_middle);
    cudaCheckKernel();

    /*
      Scaling
    */
    cudaCheckReturn(cudaMemcpy(middle, dev_middle, N * sizeof(DATA_TYPE),
                               cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < DATA_SIZE; i++) {
        float2 m = middle[i];
        m.x /= DATA_SIZE;
        m.y /= DATA_SIZE;
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

    cudaCheckReturn(cudaMemcpy(output, dev_output, N * sizeof(DATA_TYPE),
                               cudaMemcpyDeviceToHost));

    cudaCheckReturn(cudaFreeHost(middle));

    cudaCheckReturn(cudaFree(dev_output));
    cudaCheckReturn(cudaFree(dev_middle));
    cudaCheckReturn(cudaFree(dev_data));

    return time;
}
