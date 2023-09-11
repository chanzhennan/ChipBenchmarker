#include <hip/hip_fp16.h>
#include <stdio.h>

#include <cstdio>
#include <iostream>

#include "hip/hip_runtime.h"

// accessed data size in byte, should be smaller than l2 cache size
const int DATA_SIZE_IN_BYTE = (1u << 22) * 16;
// number of LDG instructions
const int N_LDG = (1lu << 20) * 1024;

const int WARMUP_ITER = 200;
const int BENCH_ITER = 200;

__device__ __forceinline__ int ldg_cg(const int *ptr) {
  int ret;
  asm volatile(
      "flat_load_b32 %0, %1;\n"
      "s_waitcnt lgkmcnt(0);"
      : "=v"(ret)
      : "v"(ptr));

  return ret;
}

template <int BLOCK, int UNROLL, int N_DATA>
__global__ void kernel(const int *x, int *y) {
  int offset = (BLOCK * UNROLL * blockIdx.x + threadIdx.x) % N_DATA;
  const int *ldg_ptr = x + offset;
  int reg[UNROLL];

  // #pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    reg[i] = ldg_cg(ldg_ptr + BLOCK * i);
  }

  int sum = 0;
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    sum += reg[i];
  }

  if (sum != 0) {
    *y = sum;
  }
}

int main() {
  const int N_DATA = DATA_SIZE_IN_BYTE / sizeof(int);

  const int UNROLL = 16;
  const int BLOCK = 128;

  static_assert(N_DATA >= UNROLL * BLOCK && N_DATA % (UNROLL * BLOCK) == 0,
                "UNROLL or BLOCK is invalid");

  int *x, *y;
  hipMalloc(&x, N_DATA * sizeof(int));
  hipMalloc(&y, N_DATA * sizeof(int));
  hipMemset(x, 0, N_DATA * sizeof(int));

  int grid = N_LDG / UNROLL / BLOCK;

  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);

  // warm up to cache data into L2
  for (int i = 0; i < WARMUP_ITER; ++i) {
    kernel<BLOCK, UNROLL, N_DATA><<<grid, BLOCK>>>(x, y);
  }
  hipDeviceSynchronize();

  hipEventRecord(start);
  for (int i = 0; i < BENCH_ITER; ++i) {
    kernel<BLOCK, UNROLL, N_DATA><<<grid, BLOCK>>>(x, y);
  }
  hipEventRecord(stop);

  float time_ms = 0.f;
  hipEventSynchronize(stop);
  hipEventElapsedTime(&time_ms, start, stop);
  double gbps = ((double)(N_LDG * sizeof(int)) / 1e9) /
                ((double)time_ms / BENCH_ITER / 1e3);
  printf("L2 cache bandwidth: %fGB/s\n", gbps);

  hipEventDestroy(start);
  hipEventDestroy(stop);

  hipFree(x);
  hipFree(y);

  return 0;
}
