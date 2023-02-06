# Viability and performance characteristics of using HIPify for CUDA projects
You can read it [here](https://jacekjagosz.github.io/HIPifyViabilityReport.html)

I ran all the HIP samples on [Solus](https://getsol.us/). You can install HIP from the repository by doing
```
sudo eopkg it rocm-hip-devel
```

Once you have HIP installed you can compile the samples like this:
```bash
hipcc vectorAdd.cu.hip -o vectorAddRX580v2 --rocm-device-lib-path=/usr/lib64/amdgcn/bitcode/ -DROCM_PATH=/usr -I ../../CommonConvertedToHIP_5.4.2/
```

If you want to convert CUDA code yourself you need an OS with CUDA SDK intalled. I used Ubuntu 20.04 in a VM, installed CUDA from [here](https://developer.nvidia.com/cuda-11-7-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local).

Then I installed LLVM 14 (same version as on Solus) from [here](https://apt.llvm.org/), and then I compiled [HIPify from source](https://github.com/ROCm-Developer-Tools/HIPIFY/). But you could also simply [install whole ROCm stack straight from AMD](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation_new.html), if you are already on supported a OS.
