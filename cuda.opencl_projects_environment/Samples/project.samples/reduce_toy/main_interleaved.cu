#include <iostream>
#include <ctime>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include "utils.h"

//#include "reduce.h"

__global__ void reduceinter(unsigned int* outdata, unsigned int* indata, unsigned int len) {
	extern __shared__ unsigned int sdata[];

	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	sdata[tid] = 0;

	if (i < len)
	{
		sdata[tid] = indata[i];
	}

	__syncthreads();

	// Interleaved addressing causes significant thread divergence
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		if (tid % (2 * s) == 0) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
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
	std::cout<< "hello" <<std::endl;
	return;
}
