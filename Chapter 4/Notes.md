### Revision - programming vs execution model
Programming model (abstractions):
    - thread hierarchy
    - memory
Execution model:
    - what it actually takes to run my threads, delays, synchronisation

## Closer look at warps
    - warps are aoms of execution of the SMs
    - each block is divided into warps with 32 threads each
    - once a block of threads is assigned to one SM it stays there forever
    - if for a given block threads are not an even multiply of the magic 32, some of the hardware threads will be inactive

    - by dividing data into tiles, so the memory will be closer together, we gain memory locality. With small matrixes naive approach should be faster, with big one tiles should be faster
## Warp scheduling
    - Max 8 blocks per SM (resource permin)
    - up to 1536 threads/sm

    - Each SM implements *zero overhead warp scheduling*
    - For each warp we have just one instruction to be executed
    - At each clock-cycle h/w keeps checking the status of operands for instructions that will be executed next warp
    - When operands are ready for a givn warp, they become **eligible warps**

    - selected, stalled and elligible warps

# LAB PC Setup
    - `ssh -XY 9jagosz@taurus.fis.agh.edu.pl`
    - `ssh -XY 9jagosz@172.20.204.1` or `.16`
    - `cd` into `cuda.opencl-xxxx` folder and do `source setcuda`
    - the makefiles have been altered to work on this machine
    - `make SMS="75"`
## cudaMalloc
- `cudaMalloc` and `cudaMallocManaged` are not the same thing
