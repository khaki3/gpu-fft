#include "common.hu"

typedef half2 ftype;
long long sig_size = 1<<23;

int main ()
{
  ftype *h_idata = (ftype *)malloc(sig_size*sizeof(ftype));
  ftype *d_idata;
  ftype *d_odata;
  cudaMalloc(&d_idata, sizeof(ftype)*sig_size);
  cudaMalloc(&d_odata, sizeof(ftype)*sig_size);

  cufftHandle plan;
  cufftCheckReturn(cufftCreate(&plan));
  size_t ws = 0;

  cufftCheckReturn(cufftXtMakePlanMany(plan, 1,  &sig_size, NULL, 1, 1, CUDA_C_16F, NULL, 1, 1, CUDA_C_16F, 1, &ws, CUDA_C_16F));
  cufftCheckReturn(cufftXtExec(plan, d_idata, d_odata, CUFFT_FORWARD)); // warm-up

  cudaEvent_t start, stop;
  cudaEventCreate(&start); cudaEventCreate(&stop);
  cudaEventRecord(start);

  cufftCheckReturn(cufftXtExec(plan, d_idata, d_odata, CUFFT_FORWARD));

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float et;

  cudaEventElapsedTime(&et, start, stop);
  printf("forward FFT time for %ld samples: %fms\n", sig_size, et);

  return 0;
}
