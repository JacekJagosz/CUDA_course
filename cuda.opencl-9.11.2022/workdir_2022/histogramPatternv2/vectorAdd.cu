/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <helper_cuda.h>
/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
 
 #define NUMBINS (26/4+1)
 
 //My own kernel
__global__ void histo_kernel(char *buffer, long size, unsigned int *histo) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  int stride = blockDim.x * gridDim.x;
  while (i < size) {
  	int alphabet_position = buffer[i] - 'a';
  	if (alphabet_position >= 0 && alphabet_position < 26)
  		atomicAdd(&(histo[alphabet_position/4]), 1);
  	i += stride;
  }
}

//Example given bt the teacher
__global__ void histogram_kernel(char *input, unsigned int *bins,
	                                 unsigned int num_elements, 
                                         unsigned int num_bins) {
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Privatized bins
	extern __shared__ unsigned int bins_s[NUMBINS];
	
	for (unsigned int binIdx = threadIdx.x; binIdx < num_bins;
	     binIdx += blockDim.x) { 
             bins_s[binIdx] = 0;
	}
	__syncthreads();

	// Histogram
	for (unsigned int i = tid; i < num_elements;
	i += blockDim.x * gridDim.x) {
	atomicAdd(&(bins_s[(unsigned int)input[i]]), 1);
	}
	__syncthreads();

	// Commit to global memory
	for (unsigned int binIdx = threadIdx.x; binIdx < num_bins;
	binIdx += blockDim.x) {
	atomicAdd(&(bins[binIdx]), bins_s[binIdx]);
	}
        }

/**
 * Host main routine
 */
int main(void) {
  // Error code to check return values for CUDA calls
  cudaError_t err = cudaSuccess;

  // Print the vector length to be used, and compute its size
  //char str[] = "This is a test string";
  std::string str = "Sed ut perspiciatis, unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam eaque ipsa, quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt, explicabo. Nemo enim ipsam voluptatem, quia voluptas sit, aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos, qui ratione voluptatem sequi nesciunt, neque porro quisquam est, qui dolorem ipsum, quia dolor sit, amet, consectetur, adipisci velit, sed quia non numquam eius modi tempora incidunt, ut labore et dolore magnam aliquam quaerat voluptatem. Ut enim ad minima veniam, quis nostrum exercitationem ullam corporis suscipit laboriosam, nisi ut aliquid ex ea commodi consequatur? Quis autem vel eum iure reprehenderit, qui in ea voluptate velit esse, quam nihil molestiae consequatur, vel illum, qui dolorem eum fugiat, quo voluptas nulla pariatur? At vero eos et accusamus et iusto odio dignissimos ducimus, qui blanditiis praesentium voluptatum deleniti atque corrupti, quos dolores et quas molestias excepturi sint, obcaecati cupiditate non provident, similique sunt in culpa, qui officia deserunt mollitia animi, id est laborum et dolorum fuga. Et harum quidem rerum facilis est et expedita distinctio. Nam libero tempore, cum soluta nobis est eligendi optio, cumque nihil impedit, quo minus id, quod maxime placeat, facere possimus, omnis voluptas assumenda est, omnis dolor repellendus. Temporibus autem quibusdam et aut officiis debitis aut rerum necessitatibus saepe eveniet, ut et voluptates repudiandae sint et molestiae non recusandae. Itaque earum rerum hic tenetur a sapiente delectus, ut aut reiciendis voluptatibus maiores alias consequatur aut perferendis doloribus asperiores repellat.";
  //int numElements = 50000;
  //int numElements = strlen(str);
  int numElements = str.length();
  //size_t size = numElements * sizeof(unsigned char);
  unsigned int size = numElements * sizeof(unsigned char);
  unsigned int numBins = NUMBINS;
  //size_t size = sizeof str * sizeof(unsigned char);
  //printf("[Vector addition of %d elements]\n", numElements);

  // Allocate the device input vector A
  char *d_A = NULL;
  err = cudaMallocManaged((void **)&d_A, size);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  //d_A = "This is a test string";
  strcpy(d_A, str.c_str());
  //strcpy(d_A, reinterpret_cast<unsigned char*>(const_cast<char*>(str.c_str())));
  
  // Allocate the device input vector B
  unsigned int *d_B = NULL;
  err = cudaMallocManaged((void **)&d_B, numBins * sizeof(unsigned int));

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Launch the Vector Add CUDA Kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
         threadsPerBlock);
  //histo_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, numElements, d_B);
  histogram_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, numElements, numBins);
  err = cudaGetLastError();

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  cudaDeviceSynchronize();
  printf("Synchronised\n");
  for (int i = 0; i < numBins; ++i) {
    printf("%d : %d\n", i, d_B[i]);
  }

  //printf("Test PASSED\n");
  
  // Free device global memory
  err = cudaFree(d_A);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device vector A (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaFree(d_B);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device vector B (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  printf("Done\n");
  return 0;
}
