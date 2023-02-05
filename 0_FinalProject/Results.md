
matrixMul:
Vega7
```
MatrixA(320,320), MatrixB(640,320)
Computing result using CUDA Kernel...
done
Performance= 216.03 GFlop/s, Time= 0.607 msec, Size= 131072000 Ops, WorkgroupSize= 1024 threads/block
Checking computed result for correctness: Result = PASS
```
First APU results, 37% performance of the RTX 4000, with only 27.4% of the theoretical TFLOPs, and about 10x smaller power consumption (0,448*2,180*2/7.119)

RX580
```
MatrixA(320,320), MatrixB(640,320)
Computing result using CUDA Kernel...
done
Performance= 1029.97 GFlop/s, Time= 0.127 msec, Size= 131072000 Ops, WorkgroupSize= 1024 threads/block
Checking computed result for correctness: Result = PASS
```

Quadro 4000
```
GPU Device 0: "Turing" with compute capability 7.5

MatrixA(320,320), MatrixB(640,320)
Computing result using CUDA Kernel...
done
Performance= 591.94 GFlop/s, Time= 0.221 msec, Size= 131072000 Ops, WorkgroupSize= 1024 threads/block
Checking computed result for correctness: Result = PASS
```

vectorAddv2:
Vega7


RX580
```
[Vector addition of 50000 elements]
Copy input data from the host memory to the CUDA device
CUDA kernel launch with 196 blocks of 256 threads
Time= 0.354 msec
Copy output data from the CUDA device to the host memory
Test PASSED
```

Quadro 4000
```
[Vector addition of 50000 elements]
Copy input data from the host memory to the CUDA device
CUDA kernel launch with 196 blocks of 256 threads
Time= 0.006 msec
Copy output data from the CUDA device to the host memory
Test PASSED
Done

```

vectorAddv3:
Vega7


RX580
```
[Vector addition of 5000000 elements]
Copy input data from the host memory to the CUDA device
CUDA kernel launch with 19532 blocks of 256 threads
Time= 2.559 msec
Copy output data from the CUDA device to the host memory
Test PASSED
Done
```

Quadro 4000
```
[Vector addition of 5000000 elements]
Copy input data from the host memory to the CUDA device
CUDA kernel launch with 19532 blocks of 256 threads
Time= 0.163 msec
Copy output data from the CUDA device to the host memory
Test PASSED
Done
```

saxpy:
Vega7


RX580
```
Max error: 0.000000
Effective Bandwidth (GB/s): 174.323280
```

Quadro 4000
```
Max error: 0.000000
Effective Bandwidth (GB/s): 380.884864
```

deviceQuery:
Vega7
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
RX580
```
Detected 1 CUDA Capable device(s)

Device 0: "AMD Radeon RX 580 Series"
  CUDA Driver Version / Runtime Version          50120.3 / 50120.3
  CUDA Capability Major/Minor version number:    8.0
  Total amount of global memory:                 4096 MBytes (4294967296 bytes)
  (036) Multiprocessors, (064) CUDA Cores/MP:    2304 CUDA Cores
  GPU Max Clock rate:                            1340 MHz (1.34 GHz)
  Memory Clock rate:                             1750 Mhz
  Memory Bus Width:                              256-bit
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
  Supports Cooperative Kernel Launch:            No
  Supports MultiDevice Co-op Kernel Launch:      No
  Device PCI Domain ID / Bus ID / location ID:   0 / 38 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 50120.3, CUDA Runtime Version = 50120.3, NumDevs = 1
Result = PASS
```

Quadro 4000
```
Detected 1 CUDA Capable device(s)

Device 0: "Quadro RTX 4000"
  CUDA Driver Version / Runtime Version          11.7 / 11.7
  CUDA Capability Major/Minor version number:    7.5
  Total amount of global memory:                 7982 MBytes (8370192384 bytes)
  (036) Multiprocessors, (064) CUDA Cores/MP:    2304 CUDA Cores
  GPU Max Clock rate:                            1545 MHz (1.54 GHz)
  Memory Clock rate:                             6501 Mhz
  Memory Bus Width:                              256-bit
  L2 Cache Size:                                 4194304 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total shared memory per multiprocessor:        65536 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  1024
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 3 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Managed Memory:                Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 11.7, CUDA Runtime Version = 11.7, NumDevs = 1
Result = PASS
```
