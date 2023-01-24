// naive implementation of a general reduce kernel
// can you tell what is wrong with it?

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
