
#include <stdio.h>

#include <cstdint>
#include <cstdio>
#include <ctime>
#include <iostream>

#include "ldg_stg_inst.h"

const int WARMUP = 100;
const int ROUND = 20;
const int STRIDE = 128;
const int WARPSIZE = 64;

template <int ROUND>
__global__ __launch_bounds__(WARPSIZE,
                             1) void l2_latency_kernel(uint32_t *stride,
                                                       uint32_t *ret,
                                                       uint64_t *clk) {
  char *ldg_ptr = reinterpret_cast<char *>(stride + threadIdx.x);
  // uint32_t *ldg_ptr = reinterpret_cast<uint32_t *>(stride + threadIdx.x);
  uint32_t val = ldg_cg_uint32((uint32_t *)ldg_ptr);

  ldg_ptr += val;

  // get realtime each threads
  uint64_t start;
  uint64_t stop;

  asm volatile(
      "s_barrier;\n"         // Wait for data to be returned
      "s_memrealtime %0;\n"  // Message type 0x83 for REALTIME
      "s_waitcnt lgkmcnt(0);"
      : "=s"(start)
      :
      : "memory");

  // ldg from L2 since warmup before
#pragma unroll
  for (int i = 0; i < ROUND; ++i) {
    uint32_t val = ldg_cg_uint32((uint32_t *)ldg_ptr);
    ldg_ptr += val;
  }

  asm volatile(
      "s_barrier;\n"         // Wait for data to be returned
      "s_memrealtime %0;\n"  // Message type 0x83 for REALTIME
      "s_waitcnt lgkmcnt(0);"
      : "=s"(stop)
      :
      : "memory");

  // store cost time each threads
  clk[(int)threadIdx.x] = stop - start;

  // To prevent compiler optimizate the loop
  if (val == 1) *ret = val;
}

int main() {
  static_assert(
      STRIDE >= 32 * sizeof(uint32_t) && STRIDE % sizeof(uint32_t) == 0,
      "invalid 'STRIDE'");

  const uint32_t STRIDE_MEM_SIZE = (ROUND + 1) * STRIDE;

  uint32_t *d_stride, *d_ret;
  uint64_t *d_clock;

  hipMallocManaged(&d_stride, STRIDE_MEM_SIZE);
  hipMallocManaged(&d_ret, sizeof(uint32_t));
  hipMallocManaged(&d_clock, WARPSIZE * sizeof(uint64_t));

  for (int i = 0; i < STRIDE_MEM_SIZE / sizeof(uint32_t); ++i) {
    d_stride[i] = STRIDE;
  }

  // pupulate l0/l1 i-cache and l2 cache
  // for (int i = 0; i < WARMUP; ++i) {
  //   l2_latency_kernel<ROUND><<<1, WARPSIZE>>>(d_stride, d_ret, d_clock);
  // }
  // hipDeviceSynchronize();

  l2_latency_kernel<ROUND><<<1, WARPSIZE>>>(d_stride, d_ret, d_clock);
  hipDeviceSynchronize();

  printf("l2 cache latency %lu cycles\n", d_clock[0] / ROUND);

  hipFree(d_stride);
  hipFree(d_ret);
  hipFree(d_clock);

  return 0;
}
