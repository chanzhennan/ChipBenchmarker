#include <hip/hip_fp16.h>
#include <stdio.h>

#include <cstdint>
#include <cstdio>
#include <iostream>

#include "hip/hip_runtime.h"
#include "timer.h"
#include "ldg_stg_inst.h"

const int MEMORY_OFFSET = (1u << 20) * 16;
const int BENCH_ITER = 2;

const int BLOCK = 128;
const int LDG_UNROLL = 1;

template <int BLOCK, int VEC_UNROLL>
__global__ void read_kernel(const void *x, void *y) {
  uint32_t idx = blockIdx.x * BLOCK * VEC_UNROLL + threadIdx.x;
  const uint4 *ldg_ptr = (const uint4 *)x + idx;
  uint4 reg[VEC_UNROLL];

  // #pragma unroll
  for (int i = 0; i < VEC_UNROLL; ++i) {
    reg[i] = ldg_cs_128(ldg_ptr + i * BLOCK);
  }

  for (int i = 0; i < VEC_UNROLL; ++i) {
    if (reg[i].x == -1) {
      stg_cs_128(reg[i], (uint4 *)y + i);
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
    stg_cs_128(reg, stg_ptr + i * BLOCK);
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
    reg[i] = ldg_cs_128(ldg_ptr + i * BLOCK);
  }

#pragma unroll
  for (int i = 0; i < VEC_UNROLL; ++i) {
    stg_cs_128(reg[i], stg_ptr + i * BLOCK);
  }
}

void warmup(char *ws, size_t size_in_byte, size_t grid) {
  // warmup
  read_kernel<BLOCK, LDG_UNROLL><<<grid, BLOCK>>>(ws, nullptr);
  write_kernel<BLOCK, LDG_UNROLL><<<grid, BLOCK>>>(ws);
  copy_kernel<BLOCK, LDG_UNROLL>
      <<<grid / 2, BLOCK>>>(ws, ws + size_in_byte / 2);
}

// read
void read(char *ws, size_t size_in_byte, size_t grid) {

  for (int i = BENCH_ITER - 1; i >= 0; --i) {
    read_kernel<BLOCK, LDG_UNROLL>
        <<<grid, BLOCK>>>(ws + i * MEMORY_OFFSET, nullptr);
  }
}

// write
void write(char *ws, size_t size_in_byte, size_t grid) {
  // write
  for (int i = BENCH_ITER - 1; i >= 0; --i) {
    write_kernel<BLOCK, LDG_UNROLL><<<grid, BLOCK>>>(ws + i * MEMORY_OFFSET);
  }
}

// copy
void copy(char *ws, size_t size_in_byte, size_t grid) {
  // copy
  for (int i = BENCH_ITER - 1; i >= 0; --i) {
    copy_kernel<BLOCK, LDG_UNROLL><<<grid / 2, BLOCK>>>(
        ws + i * MEMORY_OFFSET, ws + i * MEMORY_OFFSET + size_in_byte / 2);
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

  // warmup
  auto time_ms = hip_timer::execution(warmup, ws, size_in_byte, grid);

  // read
  time_ms = hip_timer::execution(read, ws, size_in_byte, grid);
  std::cout << "Read : " << size_gb * BENCH_ITER / ((double)time_ms / 1000)
            << " GB/s\n";

  time_ms = hip_timer::execution(write, ws, size_in_byte, grid);
  std::cout << "Write : " << size_gb * BENCH_ITER / ((double)time_ms / 1000)
            << " GB/s\n";

  time_ms = hip_timer::execution(copy, ws, size_in_byte, grid);
  std::cout << "Copy : " << size_gb * BENCH_ITER / ((double)time_ms / 1000)
            << " GB/s\n";
  
  printf("---------------------------\n");

  hipFree(ws);
}

int main() {
  size_t size = (1lu << 20) * 4;

  // 4MB~2GB
  while (size <= (1lu << 31)) {
    benchmark(size);
    size *= 2;
  }

  return 0;
}
