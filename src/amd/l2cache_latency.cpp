
#include <stdio.h>

#include <cstdint>
#include <cstdio>
#include <ctime>
#include <iostream>

#include "ldg_stg_inst.h"

const int WARMUP = 20;
const int ROUND = 10;
const int STRIDE = 128;
const int WARPSIZE = 64;

template <int ROUND>
__global__ void l2_latency_kernel(uint32_t *stride, uint32_t *ret,
                                  uint64_t *clk) {
  const char *ldg_ptr = reinterpret_cast<const char *>(stride + threadIdx.x);
  uint32_t val = ldg_cg_32_char(ldg_ptr);
  ldg_ptr += val;

  // get realtime each threads
  uint64_t start = realtime();

// ldg from L2 since warmup before
#pragma unroll
  for (int i = 0; i < ROUND; ++i) {
    val = ldg_cg_32_char(ldg_ptr);
    ldg_ptr += val;
  }

  // get realtime each threads
  uint64_t stop = realtime();

  // To prevent compiler optimizate the loop
  if (val == 1) *ret = val;

  // store cost time each threads
  clk[(int)threadIdx.x] = stop - start;
}

int main() {
  const uint32_t STRIDE_MEM_SIZE = (ROUND + 1) * STRIDE;
  uint32_t *h_stride = (uint32_t *)malloc(STRIDE_MEM_SIZE);

  for (int i = 0; i < STRIDE_MEM_SIZE / sizeof(uint32_t); ++i) {
    h_stride[i] = i;
  }

  uint32_t *d_stride, *d_ret;
  hipMalloc(&d_stride, STRIDE_MEM_SIZE);
  hipMalloc(&d_ret, sizeof(uint32_t));
  hipMemcpy(d_stride, h_stride, STRIDE_MEM_SIZE, hipMemcpyHostToDevice);

  uint64_t *d_clock;
  hipMalloc(&d_clock, WARPSIZE * sizeof(uint64_t));

  // pupulate l0/l1 i-cache and l2 cache
  for (int i = 0; i < WARMUP; ++i) {
    l2_latency_kernel<ROUND><<<1, WARPSIZE>>>(d_stride, d_ret, d_clock);
    hipDeviceSynchronize();
  }

  l2_latency_kernel<ROUND><<<1, WARPSIZE>>>(d_stride, d_ret, d_clock);
  hipDeviceSynchronize();

  uint64_t h_clk[WARPSIZE];
  hipMemcpy(h_clk, d_clock, WARPSIZE * sizeof(uint64_t), hipMemcpyDeviceToHost);
  printf("l2 cache latency %lu cycles\n", h_clk[0]);

  hipFree(d_stride);
  hipFree(d_ret);
  hipFree(d_clock);
  free(h_stride);

  return 0;
}
