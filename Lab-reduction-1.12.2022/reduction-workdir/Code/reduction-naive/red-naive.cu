// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

// Functions for the cpu time measuring
#include <chrono>

#include <iostream>

using namespace std::chrono_literals;

template <
    class result_t   = std::chrono::milliseconds,
    class clock_t    = std::chrono::steady_clock,
    class duration_t = std::chrono::milliseconds
>
auto since(std::chrono::time_point<clock_t, duration_t> const& start)
{
    return std::chrono::duration_cast<result_t>(clock_t::now() - start);
}
/*
__global__ void reduce-naive(unsigned int* g_odata, unsigned int* g_idata, unsigned int len) {
	extern __shared__ unsigned int sdata[];

	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	sdata[tid] = 0;

	if (i < len)
	{
	  sdata[tid] = g_idata[i];
	}

	__syncthreads();

	// do reduction in shared mem
	// Interleaved addressing, which causes huge thread divergence
	//  because threads are active/inactive according to their thread IDs
	//  being powers of two. The if conditional here is guaranteed to diverge
	//  threads within a warp.
	for (unsigned int s = 1; s < 2048; s <<= 2) {
		if (tid % (2 * s) == 0) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0) 
		g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduce1(unsigned int* g_odata, unsigned int* g_idata, unsigned int len) {
	extern __shared__ unsigned int sdata[];

	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	sdata[tid] = 0;

	if (i < len)
	{
		sdata[tid] = g_idata[i];
	}

	__syncthreads();

	// Interleaved addressing, threads being active/inactive
	// is no longer depends on thread IDs. Consecutive threadIDs are active
	// the warps are no longer divergent - this is a nice improvement, but
        // what is going on with the shared memory
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		unsigned int index = 2 * s * tid;

		if (index < blockDim.x) {
			sdata[index] += sdata[index + s];
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0)
		g_odata[blockIdx.x] = sdata[0];
}
*/
// Implement stuff here...

int main(void)
{
  auto start = std::chrono::steady_clock::now();
  unsigned int len = 1024;
  unsigned int *tabA;
  cudaMallocHost(&tabA, len * sizeof(unsigned int));
  for(int i=0; i < len; i++){
  	tabA[i]=i;
  }
  //printf("Hello, I'm not doing anything yet...\n");
  std::cout << "Data initialisation took(ms)=" << since(start).count() << std::endl;
  start = std::chrono::steady_clock::now();
  for(int i=1; i<len; i++){
  	tabA[0]+=tabA[i];
  }
  std::cout << "The CPU result is: " << tabA[0] << std::endl;
  std::cout << "CPU reduction took(ms)=" << since(start).count() << std::endl;
  std::cout << "Elapsed(us)=" 
        << since<std::chrono::microseconds>(start).count() << std::endl;
  unsigned int *tabAd;
  cudaMallocManaged((void **)&tabAd, size);
  
}
