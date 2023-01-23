
```bash
hipify-clang vectorAdd.cu -I ~/Downloads/cuda-samples-11.6/Common/
```

```bash
~/Downloads/HIPIFY-rocm-5.4.2/dist/hipify-clang helper_cuda.h -o hip/helper_cuda.h -I ~/Downloads/cuda-samples-11.6/Common/ --experimental
```

```bash
hipcc vectorAdd.cu.hip -o vectorAddRX580 --rocm-device-lib-path=/usr/lib64/amdgcn/bitcode/ -DROCM_PATH=/usr -I ~/Pobrane/cuda-samples-11.6/Common/
```

```bash
hipcc vectorAdd.cu.hip -o vectorAddRX580v2 --rocm-device-lib-path=/usr/lib64/amdgcn/bitcode/ -DROCM_PATH=/usr -I ../../CommonConvertedToHIP_5.4.2/
```

```bash
hipcc commonKernels.cu.hip helperFunctions.cpp.hip matrixMultiplyPerf.cu.hip --rocm-device-lib-path=/usr/lib64/amdgcn/bitcode/ -DROCM_PATH=/usr -I ../CommonConvertedToHIP_5.4.2/
```

## Dead end: Installing necessary cuda libraries for compilation with LLVM:
Download .run file: https://developer.nvidia.com/cuda-11-7-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local

Then extract the files as following. On my maching `/tmp` directory was too small, so I had to create a new directory just for this using: `sudo mkdir /newTmp` and `sudo chmod 777 /newTmp/`.
```bash
sh cuda_11.7.1_515.65.01_linux.run --extract=~/Pobrane/cuda-11.7.1_from_run/ --tmpdir=/newTmp/
```

Inside the extracted folder copy the libs from all the packages, but first create a folder for it, like `mkdir lib64`
`cp -r */lib64/* lib64/`
Move the `lib64` wherever you want and then

# Incompatibilities when converting using HIPify
 - You can't specify which c++ version it is using for converting, like you can when compiling CUDA, but you can when compiling.
 This meant I had to replace c++ 11's sprintf_s with something supported in previous version, even when HIPify should be using C++ 11 by default
 ```
 /tmp/deviceQuery.cpp-040229.hip:308:3: error: use of undeclared identifier 'sprintf_s'
  sprintf_s(cTemp, 10, "%d.%d", driverVersion / 1000,
  ```
 - some device properties and `checkCudaErrors` is not supported
  ```
  deviceQuery.cpp.hip:180:20: error: no member named 'maxTexture1DLayered' in 'hipDeviceProp_t'; did you mean 'maxTexture1DLinear'?
        deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
                   ^~~~~~~~~~~~~~~~~~~
                   maxTexture1DLinear
  /usr/include/hip/hip_runtime_api.h:122:9: note: 'maxTexture1DLinear' declared here
      int maxTexture1DLinear;    ///< Maximum size for 1D textures bound to linear memory
          ^
  deviceQuery.cpp.hip:180:39: error: subscripted value is not an array, pointer, or vector
          deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
          ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^~
  deviceQuery.cpp.hip:180:55: error: no member named 'maxTexture1DLayered' in 'hipDeviceProp_t'; did you mean 'maxTexture1DLinear'?
          deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
                                                        ^~~~~~~~~~~~~~~~~~~
                                                        maxTexture1DLinear
  /usr/include/hip/hip_runtime_api.h:122:9: note: 'maxTexture1DLinear' declared here
      int maxTexture1DLinear;    ///< Maximum size for 1D textures bound to linear memory
          ^
  deviceQuery.cpp.hip:180:74: error: subscripted value is not an array, pointer, or vector
          deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
                                            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^~
  deviceQuery.cpp.hip:184:20: error: no member named 'maxTexture2DLayered' in 'hipDeviceProp_t'; did you mean 'maxTexture1DLinear'?
          deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
                    ^~~~~~~~~~~~~~~~~~~
                    maxTexture1DLinear
  /usr/include/hip/hip_runtime_api.h:122:9: note: 'maxTexture1DLinear' declared here
      int maxTexture1DLinear;    ///< Maximum size for 1D textures bound to linear memory
          ^
  deviceQuery.cpp.hip:184:39: error: subscripted value is not an array, pointer, or vector
          deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
          ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^~
  deviceQuery.cpp.hip:184:55: error: no member named 'maxTexture2DLayered' in 'hipDeviceProp_t'; did you mean 'maxTexture1DLinear'?
          deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
                                                        ^~~~~~~~~~~~~~~~~~~
                                                        maxTexture1DLinear
  /usr/include/hip/hip_runtime_api.h:122:9: note: 'maxTexture1DLinear' declared here
      int maxTexture1DLinear;    ///< Maximum size for 1D textures bound to linear memory
          ^
  deviceQuery.cpp.hip:184:74: error: subscripted value is not an array, pointer, or vector
          deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
                                            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^~
  deviceQuery.cpp.hip:185:20: error: no member named 'maxTexture2DLayered' in 'hipDeviceProp_t'; did you mean 'maxTexture1DLinear'?
          deviceProp.maxTexture2DLayered[2]);
                    ^~~~~~~~~~~~~~~~~~~
                    maxTexture1DLinear
  /usr/include/hip/hip_runtime_api.h:122:9: note: 'maxTexture1DLinear' declared here
      int maxTexture1DLinear;    ///< Maximum size for 1D textures bound to linear memory
          ^
  deviceQuery.cpp.hip:185:39: error: subscripted value is not an array, pointer, or vector
          deviceProp.maxTexture2DLayered[2]);
          ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^~
  deviceQuery.cpp.hip:192:23: error: no member named 'sharedMemPerMultiprocessor' in 'hipDeviceProp_t'; did you mean 'maxSharedMemoryPerMultiProcessor'?
            deviceProp.sharedMemPerMultiprocessor);
                        ^~~~~~~~~~~~~~~~~~~~~~~~~~
                        maxSharedMemoryPerMultiProcessor
  /usr/include/hip/hip_runtime_api.h:114:12: note: 'maxSharedMemoryPerMultiProcessor' declared here
      size_t maxSharedMemoryPerMultiProcessor;  ///< Maximum Shared Memory Per Multiprocessor.
            ^
  deviceQuery.cpp.hip:214:21: error: no member named 'deviceOverlap' in 'hipDeviceProp_t'
          (deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
          ~~~~~~~~~~ ^
  deviceQuery.cpp.hip:214:63: error: no member named 'asyncEngineCount' in 'hipDeviceProp_t'
          (deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
                                                    ~~~~~~~~~~ ^
  deviceQuery.cpp.hip:222:23: error: no member named 'surfaceAlignment' in 'hipDeviceProp_t'
            deviceProp.surfaceAlignment ? "Yes" : "No");
            ~~~~~~~~~~ ^
  deviceQuery.cpp.hip:231:23: error: no member named 'unifiedAddressing' in 'hipDeviceProp_t'
            deviceProp.unifiedAddressing ? "Yes" : "No");
            ~~~~~~~~~~ ^
  deviceQuery.cpp.hip:235:23: error: no member named 'computePreemptionSupported' in 'hipDeviceProp_t'
            deviceProp.computePreemptionSupported ? "Yes" : "No");
            ~~~~~~~~~~ ^
  deviceQuery.cpp.hip:264:7: error: use of undeclared identifier 'checkCudaErrors'
        checkCudaErrors(hipGetDeviceProperties(&prop[i], i));
        ^
  deviceQuery.cpp.hip:288:11: error: use of undeclared identifier 'checkCudaErrors'
            checkCudaErrors(
            ^
  ```
 - Converting helper_cuda.h was the hardest as HIPify can't handle `cuBLAS`, `cuRAND`, `cuFFT`, `cuPARSE`, even though those APIs should be supported: https://github.com/ROCm-Developer-Tools/HIPIFY#-supported-cuda-apis
 - It ignores preprocesor statements, so for example when a file contains `#ifdef`s it will throw errors about e.g. redefinition
 - Oddities, like `__DRIVER_TYPES_H__` never being defined by HIP, nor HIPify converts it to `HIP_INCLUDE_HIP_DRIVER_TYPES_H`. This down the line makes compilation not work
 - No easy way to convert makefiles.

## Problems when compiling
 - Even odder things
 ```
  matrixMultiplyPerf.cu.hip:318:23: error: no matching function for call to 'hipHostGetDevicePointer'
        checkCudaErrors(hipHostGetDevicePointer(&dptrA, hptrA, 0));
 ```

## HIPify is having a crazy high development pace, ass seen by number of commits
 - `/tmp/helper_cuda.h-978032.hip:63:3: warning: 'cuGetErrorName' is experimental in 'HIP'; to hipify it, use the '--experimental' option.`


## Ideas for improvements:
 - HIPify only changes function names, HIP is reimplementing CUDA functions anyways. It shouldn't need installation of CUDA SDK, it necesitates propietary software, limits OS compatibility and makes the whole process more complicated for no good reason. Al least only the headers should be necessary, not relying on LLVM and its CUDA detection
 - ROCm needs to be steadily improved to help with packaging on all distros, not just by Debian and Arch maintainers, but AMD devs need to help too (more directly)
 - hardware compatibility is a lot worse than with CUDA, especially support duration. It is a massive issue for desktop use when you want to have only one version for all users, not a lot of fragmentation. Separate kernels for all familis don't help either.

Sources:
 - https://www.llvm.org/docs/CompileCudaWithLLVM.html
 - https://github.com/NVIDIA/cuda-samples/archive/refs/tags/v11.6.tar.gz

Random:
installing `cuda`
```
The following additional packages will be installed:
  binutils binutils-common binutils-x86-64-linux-gnu build-essential ca-certificates-java cuda-11-7 cuda-cccl-11-7 cuda-command-line-tools-11-7 cuda-compiler-11-7 cuda-cudart-11-7 cuda-cudart-dev-11-7 cuda-cuobjdump-11-7 cuda-cupti-11-7 cuda-cupti-dev-11-7
  cuda-cuxxfilt-11-7 cuda-demo-suite-11-7 cuda-documentation-11-7 cuda-driver-dev-11-7 cuda-drivers cuda-drivers-515 cuda-gdb-11-7 cuda-libraries-11-7 cuda-libraries-dev-11-7 cuda-memcheck-11-7 cuda-nsight-11-7 cuda-nsight-compute-11-7 cuda-nsight-systems-11-7
  cuda-nvcc-11-7 cuda-nvdisasm-11-7 cuda-nvml-dev-11-7 cuda-nvprof-11-7 cuda-nvprune-11-7 cuda-nvrtc-11-7 cuda-nvrtc-dev-11-7 cuda-nvtx-11-7 cuda-nvvp-11-7 cuda-runtime-11-7 cuda-sanitizer-11-7 cuda-toolkit-11-7 cuda-toolkit-11-7-config-common
  cuda-toolkit-11-config-common cuda-toolkit-config-common cuda-tools-11-7 cuda-visual-tools-11-7 dctrl-tools default-jre default-jre-headless dkms dpkg-dev fakeroot fonts-dejavu-extra g++ g++-9 gcc gcc-10-base:i386 gcc-9 gds-tools-11-7 java-common libalgorithm-diff-perl
  libalgorithm-diff-xs-perl libalgorithm-merge-perl libasan5 libatk-wrapper-java libatk-wrapper-java-jni libatomic1:i386 libbinutils libbsd0:i386 libc-dev-bin libc6:i386 libc6-dev libcrypt-dev libcrypt1:i386 libctf-nobfd0 libctf0 libcublas-11-7 libcublas-dev-11-7
  libcufft-11-7 libcufft-dev-11-7 libcufile-11-7 libcufile-dev-11-7 libcurand-11-7 libcurand-dev-11-7 libcusolver-11-7 libcusolver-dev-11-7 libcusparse-11-7 libcusparse-dev-11-7 libdrm-amdgpu1:i386 libdrm-intel1:i386 libdrm-nouveau2:i386 libdrm-radeon1:i386 libdrm2:i386
  libedit2:i386 libegl-mesa0:i386 libegl1:i386 libelf1:i386 libexpat1:i386 libfakeroot libffi7:i386 libgbm1:i386 libgcc-9-dev libgcc-s1:i386 libgl1:i386 libgl1-mesa-dri:i386 libglapi-mesa:i386 libgles2:i386 libglvnd0:i386 libglx-mesa0:i386 libglx0:i386 libidn2-0:i386
  libitm1 libllvm12:i386 liblsan0 libnpp-11-7 libnpp-dev-11-7 libnvidia-cfg1-515 libnvidia-common-515 libnvidia-compute-515 libnvidia-compute-515:i386 libnvidia-decode-515 libnvidia-decode-515:i386 libnvidia-encode-515 libnvidia-encode-515:i386 libnvidia-extra-515
  libnvidia-fbc1-515 libnvidia-fbc1-515:i386 libnvidia-gl-515 libnvidia-gl-515:i386 libnvjpeg-11-7 libnvjpeg-dev-11-7 libopengl0:i386 libpciaccess0:i386 libquadmath0 libsensors5:i386 libstdc++-9-dev libstdc++6:i386 libtinfo5 libtinfo6:i386 libtsan0 libubsan1
  libunistring2:i386 libvulkan1:i386 libwayland-client0:i386 libwayland-server0:i386 libx11-6:i386 libx11-xcb1:i386 libxau6:i386 libxcb-dri2-0:i386 libxcb-dri3-0:i386 libxcb-glx0:i386 libxcb-present0:i386 libxcb-randr0:i386 libxcb-shm0:i386 libxcb-sync1:i386
  libxcb-xfixes0:i386 libxcb-xinerama0 libxcb1:i386 libxdmcp6:i386 libxext6:i386 libxfixes3:i386 libxnvctrl0 libxshmfence1:i386 libxxf86vm1:i386 libzstd1:i386 linux-libc-dev make manpages-dev mesa-vulkan-drivers:i386 nsight-compute-2022.2.1 nsight-systems-2022.1.3
  nvidia-compute-utils-515 nvidia-dkms-515 nvidia-driver-515 nvidia-kernel-common-515 nvidia-kernel-source-515 nvidia-modprobe nvidia-prime nvidia-settings nvidia-utils-515 openjdk-11-jre openjdk-11-jre-headless screen-resolution-extra xserver-xorg-video-nvidia-515
  zlib1g:i386
Suggested packages:
  binutils-doc debtags menu debian-keyring g++-multilib g++-9-multilib gcc-9-doc gcc-multilib autoconf automake libtool flex bison gcc-doc gcc-9-multilib gcc-9-locales glibc-doc:i386 locales:i386 glibc-doc lm-sensors:i386 libstdc++-9-doc make-doc fonts-ipafont-gothic
  fonts-ipafont-mincho fonts-wqy-microhei | fonts-wqy-zenhei
```
installing `nvidia-cuda-toolkit`
```
The following additional packages will be installed:
  cpp-8 g++-8 gcc-8 gcc-8-base javascript-common libaccinj64-10.1 libcublas10
  libcublaslt10 libcudart10.1 libcufft10 libcufftw10 libcuinj64-10.1
  libcupti-dev libcupti-doc libcupti10.1 libcurand10 libcusolver10
  libcusolvermg10 libcusparse10 libegl-dev libgcc-8-dev libgl-dev
  libgl1-mesa-dev libgles-dev libgles1 libglvnd-dev libglx-dev libjs-jquery
  libmpx2 libncurses5 libnppc10 libnppial10 libnppicc10 libnppicom10
  libnppidei10 libnppif10 libnppig10 libnppim10 libnppist10 libnppisu10
  libnppitc10 libnpps10 libnvblas10 libnvgraph10 libnvidia-ml-dev libnvjpeg10
  libnvrtc10.1 libnvtoolsext1 libnvvm3 libopengl-dev libpthread-stubs0-dev
  libstdc++-8-dev libthrust-dev libvdpau-dev libx11-dev libxau-dev libxcb1-dev
  libxdmcp-dev node-html5shiv nvidia-cuda-dev nvidia-cuda-doc nvidia-cuda-gdb
  nvidia-opencl-dev nvidia-profiler nvidia-visual-profiler ocl-icd-opencl-dev
  opencl-c-headers openjdk-8-jre openjdk-8-jre-headless x11proto-core-dev
  x11proto-dev xorg-sgml-doctools xtrans-dev
Suggested packages:
  gcc-8-locales g++-8-multilib gcc-8-doc gcc-8-multilib apache2 | lighttpd
  | httpd libstdc++-8-doc libvdpau-doc libx11-doc libxcb-doc nodejs
  nvidia-driver | nvidia-tesla-440-driver | nvidia-tesla-418-driver
  libpoclu-dev fonts-ipafont-gothic fonts-ipafont-mincho fonts-wqy-microhei
  fonts-wqy-zenhei
```
