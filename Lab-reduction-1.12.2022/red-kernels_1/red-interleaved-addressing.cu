// remove divergences

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
