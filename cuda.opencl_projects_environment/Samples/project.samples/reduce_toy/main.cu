#include <iostream>
#include <ctime>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include "utils.h"

#include <helper_functions.h>
#include <helper_cuda.h>

//#include "reduce.h"

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

__global__ void reduceunroll(unsigned int* outdata, unsigned int* indata, unsigned int len) {
	extern __shared__ unsigned int sdata[];

	// each thread loads one element from global to shared mem
	// Do the first stage of the reduction on the global-to-shared load step
	// This reduces the previous inefficiency of having half of the threads being
	//  inactive on the first for-loop iteration below (previous first step of reduction)
	// Previously, only less than or equal to 512 out of 1024 threads in a block are active.
	// Now, all 512 threads in a block are active from the start
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	sdata[tid] = 0;

	if (i < len)
	{
		sdata[tid] = indata[i] + indata[i + blockDim.x];
	}

	__syncthreads();

	// do reduction in shared mem
	// this loop now starts with s = 512 / 2 = 256
	for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	if (tid < 32)
	{
		sdata[tid] += sdata[tid + 32];
		sdata[tid] += sdata[tid + 16];
		sdata[tid] += sdata[tid + 8];
		sdata[tid] += sdata[tid + 4];
		sdata[tid] += sdata[tid + 2];
		sdata[tid] += sdata[tid + 1];
	}

	// write result for this block to global mem
	if (tid == 0)
		outdata[blockIdx.x] = sdata[0];
}

void generate_input(unsigned int* input, unsigned int input_len)
{
	for (unsigned int i = 0; i < input_len; ++i)
	{
		input[i] = i;
	}
}

unsigned int cpu_simple_sum(unsigned int* h_in, unsigned int h_in_len)
{
	unsigned int total_sum = 0;

	for (unsigned int i = 0; i < h_in_len; ++i)
	{
		total_sum = total_sum + h_in[i];
	}

	return total_sum;
}

int main()
{
	auto start = std::chrono::steady_clock::now();
  unsigned int len = 102400;
  unsigned int *tabInCPU, *tabOutCPU;
  cudaMallocHost(&tabInCPU, len * sizeof(unsigned int));
  generate_input(tabInCPU, len);
  /*for(int i=0; i < len; i++){
  	tabA[i]=i;
  }*/
  //printf("Hello, I'm not doing anything yet...\n");
  std::cout << "Data initialisation took = " << since(start).count() << " ms"<< std::endl;
  start = std::chrono::steady_clock::now();
  for(int i=1; i<len; i++){
  	tabOutCPU[0]+=tabInCPU[i];
  }
  std::cout << "The CPU result is: " << tabOutCPU[0] << std::endl;
  std::cout << "CPU reduction took = " << since(start).count() << " ms or "\
  << since<std::chrono::microseconds>(start).count() <<  " us" << std::endl;

  unsigned int *tabInGPU, *tabOutGPU;
  cudaMalloc(&tabInGPU, len);
  cudaMalloc(&tabOutGPU, len);
  
  cudaStream_t stream;
  // Allocate CUDA events that we'll use for timing
  cudaEvent_t startGPU, stopGPU;
  checkCudaErrors(cudaEventCreate(&startGPU));
  checkCudaErrors(cudaEventCreate(&stopGPU));

  checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  //checkCudaErrors(
      cudaMemcpyAsync(tabInGPU, tabInCPU, len * sizeof(unsigned int), cudaMemcpyHostToDevice, stream);
  //reduceunroll(unsigned int* outdata, unsigned int* indata, unsigned int len)
  
  // Record the stop event
  checkCudaErrors(cudaEventRecord(stopGPU, stream));

  // Wait for the stop event to complete
  checkCudaErrors(cudaEventSynchronize(stopGPU));

  float msecTotal = 0.0f;
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, startGPU, stopGPU));
}
