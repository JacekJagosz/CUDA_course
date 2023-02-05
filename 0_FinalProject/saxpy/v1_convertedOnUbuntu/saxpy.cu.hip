// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

int main(void)
{
  // this is the max size
  int N = 20 * (1 << 22);
  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

  hipMalloc(&d_x, N*sizeof(float)); 
  hipMalloc(&d_y, N*sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);

  hipMemcpy(d_x, x, N*sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_y, y, N*sizeof(float), hipMemcpyHostToDevice);

  hipEventRecord(start);

  // Perform SAXPY on 1M elements
  saxpy<<<(N+511)/512, 512>>>(N, 2.0f, d_x, d_y);

  hipEventRecord(stop);

  hipMemcpy(y, d_y, N*sizeof(float), hipMemcpyDeviceToHost);

  hipEventSynchronize(stop);
  float milliseconds = 0;
  hipEventElapsedTime(&milliseconds, start, stop);

  float maxError = 0.0f;
  for (int i = 0; i < N; i++) {
    maxError = max(maxError, abs(y[i]-4.0f));
  }

  printf("Max error: %f\n", maxError);
  printf("Effective Bandwidth (GB/s): %f\n", N*4*3/milliseconds/1e6);
}

