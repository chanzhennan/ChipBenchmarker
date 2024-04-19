#include <hip/hip_fp16.h>
#include <stdio.h>

#include <cstdio>
#include <iostream>

#include "hip/hip_runtime.h"
#include "ldg_stg_inst.h"
#include "timer.h"

// accessed data size in byte, should be smaller than l2 cache size
const int DATA_SIZE_IN_BYTE = (1u << 22) * 16;
// number of LDG instructions
const int N_LDG = (1lu << 20) * 1024;

const int N_DATA = DATA_SIZE_IN_BYTE / sizeof(int);

const int UNROLL = 16;
const int BLOCK = 128;

const int WARMUP_ITER = 200;
const int BENCH_ITER = 200;

const int GRID = N_LDG / UNROLL / BLOCK;

template <int BLOCK, int UNROLL, int N_DATA>
__global__ void kernel(const int *load, int *store) {
  int offset = (BLOCK * UNROLL * blockIdx.x + threadIdx.x) % N_DATA;
  const int *ldg_ptr = load + offset;
  int reg[UNROLL];

  // #pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    reg[i] = ldg_cg_32(ldg_ptr + BLOCK * i);
  }

  int sum = 0;
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    sum += reg[i];
  }

  if (sum != 0) {
    *store = sum;
  }
}

void warmup(const int *load, int *store) {
  // warm up to cache data into L2
  for (int i = 0; i < WARMUP_ITER; ++i) {
    kernel<BLOCK, UNROLL, N_DATA><<<GRID, BLOCK>>>(load, store);
  }
  hipDeviceSynchronize();
}

void run(const int *load, int *store) {
  for (int i = 0; i < BENCH_ITER; ++i) {
    kernel<BLOCK, UNROLL, N_DATA><<<GRID, BLOCK>>>(load, store);
  }
  hipDeviceSynchronize();
}

int main() {
  static_assert(N_DATA >= UNROLL * BLOCK && N_DATA % (UNROLL * BLOCK) == 0,
                "UNROLL or BLOCK is invalid");

  int *load, *store;
  hipMalloc(&load, N_DATA * sizeof(int));
  hipMalloc(&store, N_DATA * sizeof(int));
  hipMemset(load, 0, N_DATA * sizeof(int));

  // HBM -> L2
  auto time_ms = measure<>::execution(warmup, load, store);

  // test L2
  time_ms = measure<>::execution(run, load, store);

  double gbps = ((double)(N_LDG * sizeof(int)) / 1e9) /
                ((double)time_ms / BENCH_ITER / 1e3);
  printf("L2 cache bandwidth: %fGB/s\n", gbps);

  hipFree(load);
  hipFree(store);

  return 0;
}
