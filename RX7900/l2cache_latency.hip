#include <hip/hip_fp16.h>
#include <stdio.h>

#include <cstdint>
#include <cstdio>
#include <ctime>
#include <iostream>

#include "hip/hip_runtime.h"

const int WARMUP = 100;
const int ROUND = 10;
const int STRIDE = 128;

template <int ROUND>
__global__ void l2_latency_kernel(uint32_t *stride, uint32_t *ret,
                                  int32_t *clk) {
  const char *ldg_ptr = reinterpret_cast<const char *>(stride + threadIdx.x);
  uint32_t val;
  asm volatile(
      "flat_load_b32 %0, %1;\n"  // Load scale data from dram
      "s_waitcnt lgkmcnt(0),vmcnt(0);"
      : "=v"(val)
      : "v"(ldg_ptr)
      : "memory");

  ldg_ptr += val;

  int32_t start = 0;
  int32_t stop = 0;

  asm volatile(
      "s_barrier;\n"  // Wait for data to be returned
      "s_getreg_b32 %0, hwreg(HW_REG_SHADER_CYCLES)\n"
      : "=s"(start)
      :
      : "memory");

  // #pragma unroll
  for (int i = 0; i < ROUND; ++i) {
    asm volatile(
        "flat_load_b32 %0, %1;\n"
        "s_waitcnt lgkmcnt(0),vmcnt(0);"
        : "=v"(val)
        : "v"(ldg_ptr)
        : "memory"  // Load scale data from dram
    );
    ldg_ptr += val;
  }

  asm volatile(
      "s_barrier;\n"  // Wait for data to be returned
      "s_getreg_b32 %0, hwreg(HW_REG_SHADER_CYCLES)\n"
      : "=s"(stop)
      :
      : "memory");

  clk[threadIdx.x] = (int32_t)(stop - start);

  if (val == 1) {  // To prevent compiler optimizate the loop
    *ret = val;
  }
}

int main() {
  const uint32_t STRIDE_MEM_SIZE = (ROUND + 1) * STRIDE;

  uint32_t *h_stride = (uint32_t *)malloc(STRIDE_MEM_SIZE);

  for (int i = 0; i < STRIDE_MEM_SIZE / sizeof(uint32_t); ++i) {
    h_stride[i] = STRIDE;
  }

  uint32_t *d_stride, *d_ret;
  hipMalloc(&d_stride, STRIDE_MEM_SIZE);
  hipMalloc(&d_ret, sizeof(uint32_t));
  hipMemcpy(d_stride, h_stride, STRIDE_MEM_SIZE, hipMemcpyHostToDevice);

  int32_t *d_clk;
  hipMalloc(&d_clk, 32 * sizeof(int32_t));

  // pupulate l0/l1 i-cache and l2 cache
  for (int i = 0; i < WARMUP; ++i) {
    l2_latency_kernel<ROUND><<<1, 32>>>(d_stride, d_ret, d_clk);
  }
  hipDeviceSynchronize();

  l2_latency_kernel<ROUND><<<1, 32>>>(d_stride, d_ret, d_clk);
  hipDeviceSynchronize();
  // l1 cache latency benchmark
  hipError_t status = hipGetLastError();
  if (status != hipSuccess)
    std::cerr << "Error: HIP reports " << hipGetErrorString(status)
              << std::endl;

  int32_t h_clk[32];
  hipMemcpy(h_clk, d_clk, 32 * sizeof(int32_t), hipMemcpyDeviceToHost);
  printf("l2 cache latency %d cycles\n", h_clk[0] / ROUND);

  hipFree(d_stride);
  hipFree(d_ret);
  hipFree(d_clk);
  free(h_stride);

  return 0;
}
