First APU results, 37% performance of the RTX 4000, with only 27.4% of the theoretical TFLOPs, and about 10x smaller power consumption (0,448*2,180*2/7.119)
matrixMul:
```
MatrixA(320,320), MatrixB(640,320)
Computing result using CUDA Kernel...
done
Performance= 216.03 GFlop/s, Time= 0.607 msec, Size= 131072000 Ops, WorkgroupSize= 1024 threads/block
Checking computed result for correctness: Result = PASS
```
deviceQuery:
```
Detected 1 CUDA Capable device(s)

Device 0: ""
  CUDA Driver Version / Runtime Version          50120.3 / 50120.3
  CUDA Capability Major/Minor version number:    9.0
  Total amount of global memory:                 4096 MBytes (4294967296 bytes)
MapSMtoCores for SM 9.0 is undefined.  Default to use 128 Cores/SM
MapSMtoCores for SM 9.0 is undefined.  Default to use 128 Cores/SM
  (007) Multiprocessors, (128) CUDA Cores/MP:    896 CUDADetected 1 CUDA Capable device(s)

Device 0: ""
  CUDA Driver Version / Runtime Version          50120.3 / 50120.3
  CUDA Capability Major/Minor version number:    9.0
  Total amount of global memory:                 4096 MBytes (4294967296 bytes)
MapSMtoCores for SM 9.0 is undefined.  Default to use 128 Cores/SM
MapSMtoCores for SM 9.0 is undefined.  Default to use 128 Cores/SM
  (007) Multiprocessors, (128) CUDA Cores/MP:    896 CUDA Cores
  GPU Max Clock rate:                            1900 MHz (1.90 GHz)
  Memory Clock rate:                             1933 Mhz
  Memory Bus Width:                              128-bit
  L2 Cache Size:                                 1048576 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(16384), 2D=(16384, 16384), 3D=(16384, 16384, 8192)
  Total amount of constant memory:               4294967296 bytes
  Total amount of shared memory per block:       65536 bytes
  Total number of registers available per block: 65536
  Warp size:                                     64
  Maximum number of threads per multiprocessor:  2560
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 1024)
  Max dimension size of a grid size    (x,y,z): (2147483647, 2147483647, 2147483647)
  Maximum memory pitch:                          4294967296 bytes
  Texture alignment:                             256 bytes
  Run time limit on kernels:                     No
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Device has ECC support:                        Disabled
  Device supports Managed Memory:                No
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 4 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 50120.3, CUDA Runtime Version = 50120.3, NumDevs = 1
Result = PASS Cores
  GPU Max Clock rate:                            1900 MHz (1.90 GHz)
  Memory Clock rate:                             1933 Mhz
  Memory Bus Width:                              128-bit
  L2 Cache Size:                                 1048576 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(16384), 2D=(16384, 16384), 3D=(16384, 16384, 8192)
  Total amount of constant memory:               4294967296 bytes
  Total amount of shared memory per block:       65536 bytes
  Total number of registers available per block: 65536
  Warp size:                                     64
  Maximum number of threads per multiprocessor:  2560
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 1024)
  Max dimension size of a grid size    (x,y,z): (2147483647, 2147483647, 2147483647)
  Maximum memory pitch:                          4294967296 bytes
  Texture alignment:                             256 bytes
  Run time limit on kernels:                     No
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Device has ECC support:                        Disabled
  Device supports Managed Memory:                No
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 4 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 50120.3, CUDA Runtime Version = 50120.3, NumDevs = 1
Result = PASS
```
