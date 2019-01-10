#include "common.hu"

#define CUDART_PI_F 3.141592654f

__global__ void fft(DATA_TYPE *output, DATA_TYPE *data)
{
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;

    float real = 0.0;
    float imag = 0.0;

    if (id < DATA_SIZE) {
        float pow = 2 * CUDART_PI_F * id / (float)DATA_SIZE;

        for (size_t i = 0; i < DATA_SIZE; i++) {
            /*
                r * cos(2pkl/n) + i * sin(2pkl/n)
              - r * sin(2pkl/n) + i * con(2pkl/n)
            */
            DATA_TYPE d = data[i];
            float powh = fmodf(i * pow, 2 * CUDART_PI_F);

            real +=   d.x * cosf(powh) + d.y * sinf(powh);
            imag += - d.x * sinf(powh) + d.y * cosf(powh);
        }

        output[id] = make_float2(real, imag);
    }
}

__global__ void ifft(DATA_TYPE *output, DATA_TYPE *data)
{
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;

    float real = 0.0;
    float imag = 0.0;

    if (id < DATA_SIZE) {
        float pow = 2 * CUDART_PI_F * id / (float)DATA_SIZE;

        for (size_t i = 0; i < DATA_SIZE; i++) {
            /*
              r * cos(2pkl/n) - i * sin(2pkl/n)
              r * sin(2pkl/n) + i * con(2pkl/n)
            */
            DATA_TYPE d = data[i];
            float powh = fmodf(i * pow, 2 * CUDART_PI_F);

            real += d.x * cosf(powh) - d.y * sinf(powh);
            imag += d.x * sinf(powh) + d.y * cosf(powh);
        }

        output[id] = make_float2(real, imag);
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

    cudaCheckReturn(cudaMalloc(&dev_data,   DATA_SIZE * sizeof(DATA_TYPE)));
    cudaCheckReturn(cudaMalloc(&dev_middle, DATA_SIZE * sizeof(DATA_TYPE)));
    cudaCheckReturn(cudaMalloc(&dev_output, DATA_SIZE * sizeof(DATA_TYPE)));

    cudaCheckReturn(cudaMemcpy(dev_data, data, DATA_SIZE * sizeof(DATA_TYPE),
                               cudaMemcpyHostToDevice));

    /*
      FFT
    */
    cudaCheckReturn(cudaDeviceSynchronize());
    cudaCheckReturn(cudaEventRecord(start));

    fft<<<DATA_SIZE / 256, 256>>>(dev_middle, dev_data);

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

    ifft<<<DATA_SIZE / 256, 256>>>(dev_output, dev_middle);

    cudaCheckReturn(cudaEventRecord(stop));
    cudaCheckReturn(cudaEventSynchronize(stop));
    cudaCheckKernel();

    cudaCheckReturn(cudaEventElapsedTime(&time[1], start, stop));

    /*
      Close
    */
    cudaCheckReturn(cudaMemcpy(output, dev_output, DATA_SIZE * sizeof(DATA_TYPE),
                               cudaMemcpyDeviceToHost));

    cudaCheckReturn(cudaFreeHost(middle));

    cudaCheckReturn(cudaFree(dev_output));
    cudaCheckReturn(cudaFree(dev_middle));
    cudaCheckReturn(cudaFree(dev_data));

    return time;
}
