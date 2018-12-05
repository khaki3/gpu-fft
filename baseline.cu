#include "common.hu"

#define CUDART_PI_F 3.141592654f

__global__ void fft(DATA_TYPE *output, DATA_TYPE *data)
{
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;

    float2 tmp = make_float2(0.0, 0.0);

    if (id < DATA_SIZE) {
        float pow = 2 * CUDART_PI_F * id / (float)DATA_SIZE;
        DATA_TYPE m = __floats2half2_rn(1.0, -1.0);

        for (size_t i = 0; i < DATA_SIZE; i++) {
            /*
                r * cos(2pkl/n) + i * sin(2pkl/n)
              - r * sin(2pkl/n) + i * con(2pkl/n)
            */

            DATA_TYPE d = data[i];
            DATA_TYPE powh = __float2half2_rn(fmodf(i * pow, 2 * CUDART_PI_F));

            /* 
               c   = (r * cos,   i * cos)
               smr = (i * sin, - r * sin)
            */
            DATA_TYPE c   = __hmul2(d, h2cos(powh));
            DATA_TYPE smr = __hmul2(__lowhigh2highlow(__hmul2(d, h2sin(powh))), m);

            float2 f = __half22float2(__hadd2(c, smr));
            tmp.x += f.x;
            tmp.y += f.y;
        }

        output[id] = __float22half2_rn(tmp);
    }
}

__global__ void ifft(DATA_TYPE *output, DATA_TYPE *data)
{
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;

    float2 tmp = make_float2(0.0, 0.0);

    if (id < DATA_SIZE) {
        float pow = 2 * CUDART_PI_F * id / (float)DATA_SIZE;
        DATA_TYPE m  = __floats2half2_rn(-1.0, 1.0);

        for (size_t i = 0; i < DATA_SIZE; i++) {
            /*
              r * cos(2pkl/n) - i * sin(2pkl/n)
              r * sin(2pkl/n) + i * con(2pkl/n)
            */

            DATA_TYPE d = data[i];
            DATA_TYPE powh = __float2half2_rn(fmodf(i * pow, 2 * CUDART_PI_F));

            /* 
               c   = (  r * cos, i * cos)
               smr = (- i * sin, r * sin)
            */
            DATA_TYPE c   = __hmul2(d, h2cos(powh));
            DATA_TYPE smr = __hmul2(__lowhigh2highlow(__hmul2(d, h2sin(powh))), m);

            float2 f = __half22float2(__hadd2(c, smr));
            tmp.x += f.x;
            tmp.y += f.y;
        }

        output[id] = __float22half2_rn(tmp);
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
