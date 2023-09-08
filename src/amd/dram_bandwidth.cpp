#include <hip/hip_fp16.h>
#include <stdio.h>

#include <cstdint>
#include <cstdio>
#include <iostream>

#include "hip/hip_runtime.h"

const int MEMORY_OFFSET = (1u << 20) * 16;
const int BENCH_ITER = 2;

const int BLOCK = 128;
const int LDG_UNROLL = 1;

__device__ __forceinline__  // ptr -> ret
    uint4
    ldg_cs(const void *ptr) {
  uint4 ret;
  asm volatile("FLAT_LOAD_B128 %0, %1;\n" : "=v"(ret) : "v"(ptr));

  return ret;
}

__device__ __forceinline__  // reg -> ptr
    void
    stg_cs(uint4 &reg, void *ptr) {
  asm volatile("FLAT_STORE_B128 %1, %0;\n" : "=v"(reg) : "v"(ptr));
}

template <int BLOCK, int VEC_UNROLL>
__global__ void read_kernel(const void *x, void *y) {
  uint32_t idx = blockIdx.x * BLOCK * VEC_UNROLL + threadIdx.x;
  const uint4 *ldg_ptr = (const uint4 *)x + idx;
  uint4 reg[VEC_UNROLL];

  // #pragma unroll
  for (int i = 0; i < VEC_UNROLL; ++i) {
    reg[i] = ldg_cs(ldg_ptr + i * BLOCK);
  }

  for (int i = 0; i < VEC_UNROLL; ++i) {
    if (reg[i].x == 0) {
      printf("wip %d\n", (int)blockIdx.x);
      // stg_cs(reg[i], (uint4 *)y + i);
    }
  }
}

template <int BLOCK, int VEC_UNROLL>
__global__ void write_kernel(void *y) {
  uint32_t idx = blockIdx.x * BLOCK * VEC_UNROLL + threadIdx.x;

  uint4 *stg_ptr = (uint4 *)y + idx;

#pragma unroll
  for (int i = 0; i < VEC_UNROLL; ++i) {
    uint4 reg = make_uint4(0, 0, 0, 0);
    stg_cs(reg, stg_ptr + i * BLOCK);
  }
}

template <int BLOCK, int VEC_UNROLL>
__global__ void copy_kernel(const void *x, void *y) {
  uint32_t idx = blockIdx.x * BLOCK * VEC_UNROLL + threadIdx.x;

  const uint4 *ldg_ptr = (const uint4 *)x + idx;
  uint4 *stg_ptr = (uint4 *)y + idx;
  uint4 reg[VEC_UNROLL];

#pragma unroll
  for (int i = 0; i < VEC_UNROLL; ++i) {
    reg[i] = ldg_cs(ldg_ptr + i * BLOCK);
  }

#pragma unroll
  for (int i = 0; i < VEC_UNROLL; ++i) {
    stg_cs(reg[i], stg_ptr + i * BLOCK);
  }
}

void benchmark(size_t size_in_byte) {
  printf("%luMB (r+w)\n", size_in_byte / (1 << 20));

  double size_gb = (double)size_in_byte / (1 << 30);

  size_t n = size_in_byte / sizeof(uint4);
  size_t grid = n / (BLOCK * LDG_UNROLL);

  static_assert(MEMORY_OFFSET % sizeof(uint4) == 0, "invalid MEMORY_OFFSET");

  char *ws;
  hipMalloc(&ws, size_in_byte + MEMORY_OFFSET * BENCH_ITER);

  // set all zero for read-only kernel
  hipMemset(ws, 0, size_in_byte + MEMORY_OFFSET * BENCH_ITER);

  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);
  float time_ms = 0.f;

  // warmup
  read_kernel<BLOCK, LDG_UNROLL><<<grid, BLOCK>>>(ws, nullptr);
  write_kernel<BLOCK, LDG_UNROLL><<<grid, BLOCK>>>(ws);
  copy_kernel<BLOCK, LDG_UNROLL>
      <<<grid / 2, BLOCK>>>(ws, ws + size_in_byte / 2);

  // read

  hipEventRecord(start);
  for (int i = BENCH_ITER - 1; i >= 0; --i) {
    read_kernel<BLOCK, LDG_UNROLL>
        <<<grid, BLOCK>>>(ws + i * MEMORY_OFFSET, nullptr);
  }

  hipEventRecord(stop);
  hipEventSynchronize(stop);
  hipEventElapsedTime(&time_ms, start, stop);
  printf("read %fGB/s\n", size_gb * BENCH_ITER / ((double)time_ms / 1000));

  // write
  hipEventRecord(start);
  for (int i = BENCH_ITER - 1; i >= 0; --i) {
    write_kernel<BLOCK, LDG_UNROLL><<<grid, BLOCK>>>(ws + i * MEMORY_OFFSET);
  }
  hipEventRecord(stop);

  hipEventSynchronize(stop);
  hipEventElapsedTime(&time_ms, start, stop);
  printf("write %fGB/s\n", size_gb * BENCH_ITER / ((double)time_ms / 1000));

  // copy
  hipEventRecord(start);
  for (int i = BENCH_ITER - 1; i >= 0; --i) {
    copy_kernel<BLOCK, LDG_UNROLL><<<grid / 2, BLOCK>>>(
        ws + i * MEMORY_OFFSET, ws + i * MEMORY_OFFSET + size_in_byte / 2);
  }
  hipEventRecord(stop);
  hipEventSynchronize(stop);

  hipEventElapsedTime(&time_ms, start, stop);
  printf("copy %fGB/s\n", size_gb * BENCH_ITER / ((double)time_ms / 1000));

  printf("---------------------------\n");

  hipEventDestroy(start);
  hipEventDestroy(stop);

  hipFree(ws);
}

int main() {
  size_t size = (1lu << 20) * 4;

  // 4MB~1GB
  while (size <= (1lu << 30)) {
    benchmark(size);
    size *= 2;
  }

  return 0;
}
