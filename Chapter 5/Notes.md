## Warp divergence
If the threads in the same warm are executing different code, respective branches will be executed serially.

## Loop unrolling
```C++
#pragma unroll
for( ; ; ;)
```
## Coalescing of memory
This approach originates from the hardware DRAM burst.
Make sure the data for threads next to each other is next to each other. If there are gaps, then the data takes longer to get, and data from the gaps is not used.
One way to achieve it is tiling.

## Cuda atomic
e.g. `int atomicAdd(int* address, int val);`

## Privatization
