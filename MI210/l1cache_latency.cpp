#include <hip/hip_fp16.h>
#include <stdio.h>

#include <cstdint>
#include <cstdio>
#include <ctime>
#include <iostream>

#include "hip/hip_runtime.h"

const int WARMUP = 100;
const int THREAD = 4;
// number of LDG instructions to be timed
const int ROUND = 50;

template <int ROUND>
__global__ __launch_bounds__(THREAD, 1) void l1_latency_kernel(void **ptr,
                                                               void **ret,
                                                               int32_t *clk) {
  void **ldg_ptr = ptr + threadIdx.x;

  // warm up to populate l1 cache
  for (int i = 0; i < ROUND; ++i) {
    int32_t val;
    asm volatile(
        "flat_load_dword %0, %1;\n"
        "s_waitcnt lgkmcnt(0),vmcnt(0);"
        : "=v"(val)
        : "v"(ldg_ptr)
        : "memory");
  }

  int32_t start;
  int32_t stop;

  asm volatile(
      "s_barrier;\n"  // Wait for data to be returned
      "s_getreg_b32 %0, hwreg(HW_REG_SHADER_CYCLES)\n"
      : "=s"(start)
      :
      : "memory");

  //     for (int i = 0; i < ROUND; ++i) {
  //         asm volatile(
  //         "s_load_b64 %0, %0;\n"  // Load scale data from dram
  //         "s_waitcnt lgkmcnt(0),vmcnt(0);"
  //         : "=s"(ldg_ptr));
  //     }

  //     asm volatile(
  //     "s_barrier;\n"  // Wait for data to be returned
  //     "s_getreg_b32 %0, hwreg(HW_REG_SHADER_CYCLES)\n"
  //     : "=s"(stop)
  //     :
  //     : "memory");

  //     clk[threadIdx.x] = stop - start;

  //     // dummy write back
  //     if (ldg_ptr == nullptr) {
  //         *ret = ldg_ptr;
  //     }
}

int main() {
  void **d_ptr;
  void **d_ret;
  int32_t *d_clock;
  hipMalloc(&d_ptr, THREAD * sizeof(void *));
  hipMalloc(&d_ret, sizeof(void *));
  hipMalloc(&d_clock, THREAD * sizeof(int32_t));

  void **h_ptr;  // =  (void**)malloc(THREAD * sizeof(void *));
  hipMallocHost((void **)&h_ptr, THREAD * sizeof(void *));

  for (int i = 0; i < THREAD; ++i) {
    h_ptr[i] = d_ptr + i;
  }

  hipMemcpy(d_ptr, h_ptr, THREAD * sizeof(void *), hipMemcpyHostToDevice);

  // populate instruction cache
  for (int i = 0; i < WARMUP; ++i) {
    l1_latency_kernel<ROUND><<<1, THREAD>>>(d_ptr, d_ret, d_clock);
  }

  // l1 cache latency benchmark
  // l1_latency_kernel<ROUND><<<1, THREAD>>>(d_ptr, d_ret, d_clock);

  // int32_t h_clk[THREAD];
  // hipMemcpy(h_clk, d_clock, THREAD * sizeof(int32_t), hipMemcpyDeviceToHost);
  // printf("l1 cache latency %u cycles\n", h_clk[0] / ROUND);

  // hipFree(d_ptr);
  // hipFree(d_ret);
  // hipFree(d_clock);
  // free(h_ptr);

  return 0;
}

// template <int ROUND>
// __global__ __launch_bounds__(THREAD, 1)
// void l1_latency_kernel(void **ptr, void **ret, uint32_t *clk) {
//     void **ldg_ptr = ptr + threadIdx.x;

//     // warm up to populate l1 cache
//     for (int i = 0; i < ROUND; ++i) {
//         asm volatile (
//             "ld.global.nc.b64 %0, [%0];\n"
//             : "+l"(ldg_ptr) : : "memory"
//         );
//     }

//     uint32_t start;
//     uint32_t stop;

//     asm volatile (
//         "bar.sync 0;\n"
//         "mov.u32 %0, %%clock;\n"
//         : "=r"(start) : : "memory"
//     );

//     for (int i = 0; i < ROUND; ++i) {
//         /*
//          * dependent LDG instructions to make sure that
//          * LDG latency can not be hidden by parallel LDG.
//          */
//         asm volatile (
//             "ld.global.nc.b64 %0, [%0];\n"
//             : "+l"(ldg_ptr) : : "memory"
//         );
//     }

//     asm volatile (
//         "bar.sync 0;\n"
//         "mov.u32 %0, %%clock;\n"
//         : "=r"(stop) : : "memory"
//     );

//     clk[threadIdx.x] = stop - start;

//     // dummy write back
//     if (ldg_ptr == nullptr) {
//         *ret = ldg_ptr;
//     }
// }
