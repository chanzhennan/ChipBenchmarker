#include <hip/hip_fp16.h>
#include <stdio.h>

#include <cstdint>
#include <cstdio>
#include <ctime>
#include <iostream>

#include "hip/hip_runtime.h"
#include "ldg_stg_inst.h"

const int WARMUP = 100;
const int ROUND = 10;
const int STRIDE = 128;
const int WARPSIZE = 64;

template <int ROUND>
__global__ void l2_latency_kernel(uint32_t *stride, uint32_t *ret,
                                  int32_t *clk) {
  const char *ldg_ptr = reinterpret_cast<const char *>(stride + threadIdx.x);
  uint32_t val = ldg_cg_uint32((void*)ldg_ptr);
  ldg_ptr += val;

  uint64_t start = timestamp();

  // #pragma unroll
  for (int i = 0; i < ROUND; ++i) {
    val = ldg_cg_uint32((void*)ldg_ptr);
    ldg_ptr += val; 
  }

  uint64_t stop = timestamp();

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
  hipMalloc(&d_clk, WARPSIZE * sizeof(int32_t));

  // Pupulate l0/l1 i-cache and l2 cache
  for (int i = 0; i < WARMUP; ++i) {
    l2_latency_kernel<ROUND><<<1, WARPSIZE>>>(d_stride, d_ret, d_clk);
  }
  // No hipDeviceSynchronize() here! because upload addr of flat_load_dword to l2 cache

  l2_latency_kernel<ROUND><<<1, WARPSIZE>>>(d_stride, d_ret, d_clk);
  hipDeviceSynchronize();
  // l1 cache latency benchmark
  hipError_t status = hipGetLastError();
  if (status != hipSuccess)
    std::cerr << "Error: HIP reports " << hipGetErrorString(status)
              << std::endl;

  int32_t h_clk[WARPSIZE];
  hipMemcpy(h_clk, d_clk, WARPSIZE * sizeof(int32_t), hipMemcpyDeviceToHost);
  printf("l2 cache latency %d cycles\n", h_clk[0] / ROUND);

  hipFree(d_stride);
  hipFree(d_ret);
  hipFree(d_clk);
  free(h_stride);

  return 0;
}
