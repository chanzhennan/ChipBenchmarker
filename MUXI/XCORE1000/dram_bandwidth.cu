#include <mc_common.h>
#include <mc_runtime.h>

#include <cstdint>
#include <cstdio>

#include "__clang_maca_device_functions.h"
#include "common/maca_fp16.h"

const int MEMORY_OFFSET = (1u << 20) * 16;
const int BENCH_ITER = 100;

const int BLOCK = 128;
const int LDG_UNROLL = 1;

template <int BLOCK, int VEC_UNROLL>
__global__ void read_kernel(const void *x, void *y) {
  uint32_t idx = blockIdx.x * BLOCK * VEC_UNROLL + threadIdx.x;

  const uint4 *ldg_ptr = (const uint4 *)x + idx;
  uint4 reg[VEC_UNROLL];

#pragma unroll
  for (int i = 0; i < VEC_UNROLL; ++i) {
    reg[i] = __ldcg(ldg_ptr + i * BLOCK);
  }

// dummy STG
#pragma unroll
  for (int i = 0; i < VEC_UNROLL; ++i) {
    if (reg[i].x != 0) {
      __stcg((uint4 *)y + i, reg[i]);
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
    __stcg(stg_ptr + i * BLOCK, reg);
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
    reg[i] = __ldcg(ldg_ptr + i * BLOCK);
  }

#pragma unroll
  for (int i = 0; i < VEC_UNROLL; ++i) {
    __stcg(stg_ptr + i * BLOCK, reg[i]);
  }
}

void benchmark(size_t size_in_byte) {
  printf("%luMB (r+w)\n", size_in_byte / (1 << 20));

  double size_gb = (double)size_in_byte / (1 << 30);

  size_t n = size_in_byte / sizeof(uint4);
  size_t grid = n / (BLOCK * LDG_UNROLL);

  static_assert(MEMORY_OFFSET % sizeof(uint4) == 0, "invalid MEMORY_OFFSET");

  char *ws;
  mcMalloc(&ws, size_in_byte + MEMORY_OFFSET * BENCH_ITER);

  // set all zero for read-only kernel
  mcMemset(ws, 0, size_in_byte + MEMORY_OFFSET * BENCH_ITER);

  mcEvent_t start, stop;
  mcEventCreate(&start);
  mcEventCreate(&stop);
  float time_ms = 0.f;

  // warmup
  read_kernel<BLOCK, LDG_UNROLL><<<grid, BLOCK>>>(ws, nullptr);
  write_kernel<BLOCK, LDG_UNROLL><<<grid, BLOCK>>>(ws);
  copy_kernel<BLOCK, LDG_UNROLL>
      <<<grid / 2, BLOCK>>>(ws, ws + size_in_byte / 2);

  // read
  mcEventRecord(start);
  for (int i = BENCH_ITER - 1; i >= 0; --i) {
    read_kernel<BLOCK, LDG_UNROLL>
        <<<grid, BLOCK>>>(ws + i * MEMORY_OFFSET, nullptr);
  }
  mcEventRecord(stop);

  mcEventSynchronize(stop);
  mcEventElapsedTime(&time_ms, start, stop);
  printf("read %fGB/s\n", size_gb * BENCH_ITER / ((double)time_ms / 1000));

  // write
  mcEventRecord(start);
  for (int i = BENCH_ITER - 1; i >= 0; --i) {
    write_kernel<BLOCK, LDG_UNROLL><<<grid, BLOCK>>>(ws + i * MEMORY_OFFSET);
  }
  mcEventRecord(stop);

  mcEventSynchronize(stop);
  mcEventElapsedTime(&time_ms, start, stop);
  printf("write %fGB/s\n", size_gb * BENCH_ITER / ((double)time_ms / 1000));

  // copy
  mcEventRecord(start);
  for (int i = BENCH_ITER - 1; i >= 0; --i) {
    copy_kernel<BLOCK, LDG_UNROLL><<<grid / 2, BLOCK>>>(
        ws + i * MEMORY_OFFSET, ws + i * MEMORY_OFFSET + size_in_byte / 2);
  }
  mcEventRecord(stop);
  mcEventSynchronize(stop);

  mcEventElapsedTime(&time_ms, start, stop);
  printf("copy %fGB/s\n", size_gb * BENCH_ITER / ((double)time_ms / 1000));

  printf("---------------------------\n");

  mcEventDestroy(start);
  mcEventDestroy(stop);

  mcFree(ws);
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
