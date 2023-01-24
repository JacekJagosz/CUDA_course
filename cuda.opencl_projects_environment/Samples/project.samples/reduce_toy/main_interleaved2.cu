#include <iostream>
#include <ctime>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include "utils.h"

//#include "reduce.h"

__global__ void reduceinter2(unsigned int* outdata, unsigned int* indata, unsigned int len) {
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

	// do reduction in shared mem
	// Interleaved addressing, but threads being active/inactive
	//  is no longer based on thread IDs being powers of two. Consecutive
	//  threadIDs now run, and thus solves the thread diverging issue within
	//  a warp
	// However, this introduces shared memory bank conflicts, as threads start 
	//  out addressing with a stride of two 32-bit words (unsigned ints),
	//  and further increase the stride as the current power of two grows larger
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		unsigned int index = 2 * s * tid;

		if (index < blockDim.x) {
			sdata[index] += sdata[index + s];
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
