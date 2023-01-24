## Writing Application Code for the GPU
W pliku `.cu`:
```C
void CPUFunction()
{
  printf("This function is defined to run on the CPU.\n");
}

__global__ void GPUFunction()
{
  printf("This function is defined to run on the GPU.\n");
}

int main()
{
  CPUFunction();

  GPUFunction<<<1, 1>>>(); //bloki, wątki w bloku
  cudaDeviceSynchronize(); //ważne!
}
```
`__global__` musi być `void`


`threadIdx.x`, `blockIdx.x` oraz `blockDim.x`
`int i = blockIdx.x * blockDim.x + threadIdx.x;`

## Allocating Memory to be accessed on the GPU and the CPU

By pamięć była dostępna na CPU i GPU zamienić `malloc` oraz `free` na `cudaMallocManaged` i `cudaFree`
```C
// CPU-only

int N = 2<<20;
size_t size = N * sizeof(int);

int *a;
a = (int *)malloc(size);

// Use `a` in CPU-only program.

free(a);

// Accelerated

int N = 2<<20;
size_t size = N * sizeof(int);

int *a;
// Note the address of `a` is passed as first argument.
cudaMallocManaged(&a, size);

// Use `a` on the CPU and/or on any GPU in the accelerated system.

cudaFree(a);
```

```C
size_t threads_per_block = 256;
size_t number_of_blocks = (N + threads_per_block - 1) / threads_per_block;//Jest to dlatego bo dzielenie w C zaokrągla w dół, dlatego to działa
```

## Handling Block Configuration Mismatches to Number of Needed Threads
Kiedy nie możemy zagwarantować że N będzie podzielne przez 32:
```C
// Assume `N` is known
int N = 100000;

// Assume we have a desire to set `threads_per_block` exactly to `256`
size_t threads_per_block = 256;

// Ensure there are at least `N` threads in the grid, but only 1 block's worth extra
size_t number_of_blocks = (N + threads_per_block - 1) / threads_per_block;

some_kernel<<<number_of_blocks, threads_per_block>>>(N);
```
czyli wątków będzie więcej niż trzeba:
```C
__global__ some_kernel(int N)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < N) // Check to make sure `idx` maps to some value within `N`
  {
    // Only do work if it does
  }
}
```

## Grid stride loops

<div align="center"><iframe src="https://view.officeapps.live.com/op/view.aspx?src=https://developer.download.nvidia.com/training/courses/C-AC-01-V1/embedded/task1/AC_CUDA_C_6.pptx" width="800px" height="500px" frameborder="0"></iframe></div>

CUDA provides a special variable giving the number of blocks in a grid, gridDim.x. Calculating the total number of threads in a grid then is simply the number of blocks in a grid multiplied by the number of threads in each block, gridDim.x * blockDim.x. With this in mind, here is a verbose example of a grid-stride loop within a kernel:
```C
__global__ void kernel(int *a, int N)
{
  int indexWithinTheGrid = threadIdx.x + blockIdx.x * blockDim.x;
  int gridStride = gridDim.x * blockDim.x;

  for (int i = indexWithinTheGrid; i < N; i += gridStride)
  {
    // do work on a[i];
  }
}
```
### Error handling
```C
cudaError_t err;
err = cudaMallocManaged(&a, N)                    // Assume the existence of `a` and `N`.

if (err != cudaSuccess)                           // `cudaSuccess` is provided by CUDA.
{
  printf("Error: %s\n", cudaGetErrorString(err)); // `cudaGetErrorString` is provided by CUDA.
}
```
Launching kernels, which are defined to return void, do not return a value of type `cudaError_t`. To check for errors occurring at the time of a kernel launch, for example if the launch configuration is erroneous, CUDA provides the `cudaGetLastError` function, which does return a value of type `cudaError_t`.
```C
/*
 * This launch should cause an error, but the kernel itself
 * cannot return it.
 */

someKernel<<<1, -1>>>();  // -1 is not a valid number of threads.

cudaError_t err;
err = cudaGetLastError(); // `cudaGetLastError` will return the error from above.
if (err != cudaSuccess)
{
  printf("Error: %s\n", cudaGetErrorString(err));
}
```
Finally, in order to catch errors that occur asynchronously, for example during the execution of an asynchronous kernel, it is essential to check the status returned by a subsequent synchronizing CUDA runtime API call, such as cudaDeviceSynchronize, which will return an error if one of the kernels launched previously should fail.

### CUDA Error Handling Function
```C
#include <stdio.h>
#include <assert.h>

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

int main()
{

/*
 * The macro can be wrapped around any function returning
 * a value of type `cudaError_t`.
 */

  checkCuda( cudaDeviceSynchronize() )
}
```

### Grids and Blocks of 2 and 3 Dimensions
```C
dim3 threads_per_block(16, 16, 1);
dim3 number_of_blocks(16, 16, 1);
someKernel<<<number_of_blocks, threads_per_block>>>();
```
Given the example just above, the variables `gridDim.x`, `gridDim.y`, `blockDim.x`, and `blockDim.y` inside of someKernel, would all be equal to 16.

### The advaced stuff should be easy, but in the last time during class I forgot to change malloc with CudaMallocManaged
