#include <hip/hip_fp16.h>
#include <stdio.h>

#include <cstdint>
#include <cstdio>
#include <iostream>

#include "hip/hip_runtime.h"
#include "ldg_stg_inst.h"
#include "timer.h"

const int MEMORY_OFFSET = (1u << 20) * 16;  // 16M
const int BENCH_ITER = 1;

const int BLOCK = 128;
const int UNROLL = 1;

template <int BLOCK, int UNROLL>
__global__ void read_kernel(const void *x, void *y) {
  uint32_t idx = blockIdx.x * blockDim.x * UNROLL + threadIdx.x;
  const uint4 *ldg_ptr = (const uint4 *)x + idx;
  uint4 reg[UNROLL];

  // #pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    reg[i] = ldg_cs_128(ldg_ptr + i * BLOCK);
  }

  for (int i = 0; i < UNROLL; ++i) {
    if (reg[i].x == -1) {
      stg_cs_128(reg[i], (uint4 *)y + i);
    }
  }
}

template <int BLOCK, int UNROLL>
__global__ void write_kernel(void *y) {
  uint32_t idx = blockIdx.x * blockDim.x * UNROLL + threadIdx.x;

  uint4 *stg_ptr = (uint4 *)y + idx;

#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    uint4 reg = make_uint4(0, 1, 2, 3);
    stg_cs_128(reg, stg_ptr + i * BLOCK);
  }
}

template <int BLOCK, int UNROLL>
__global__ void copy_kernel(const void *x, void *y) {
  uint32_t idx = blockIdx.x * blockDim.x * UNROLL + threadIdx.x;

  const uint4 *ldg_ptr = (const uint4 *)x + idx;
  uint4 *stg_ptr = (uint4 *)y + idx;
  uint4 reg[UNROLL];

#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    reg[i] = ldg_cs_128(ldg_ptr + i * BLOCK);
  }

#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    stg_cs_128(reg[i], stg_ptr + i * BLOCK);
  }
}

void warmup(uint *ws, size_t size_in_byte, size_t grid) {
  // warmup
  char *ws_base = (char *)ws;

  read_kernel<BLOCK, UNROLL><<<grid, BLOCK>>>(ws_base, nullptr);
  write_kernel<BLOCK, UNROLL><<<grid, BLOCK>>>(ws_base);
  // copy_kernel<BLOCK, UNROLL>
  //     <<<grid, BLOCK>>>(ws_base, ws_base + size_in_byte / 2);
}

// read
void read(uint *ws, size_t size_in_byte, size_t grid) {
  char *ws_base = (char *)ws;

  for (int i = BENCH_ITER - 1; i >= 0; --i) {
    read_kernel<BLOCK, UNROLL>
        <<<grid, BLOCK>>>(ws_base + i * MEMORY_OFFSET, nullptr);
  }
}

// write
void write(uint *ws, size_t size_in_byte, size_t grid) {
  // write
  char *ws_base = (char *)ws;
  for (int i = BENCH_ITER - 1; i >= 0; --i) {
    write_kernel<BLOCK, UNROLL><<<grid, BLOCK>>>(ws_base + i * MEMORY_OFFSET);
  }
}

// copy
void copy(uint *ws, size_t size_in_byte, size_t grid) {
  // copy
  char *ws_base = (char *)ws;
  for (int i = BENCH_ITER - 1; i >= 0; --i) {
    copy_kernel<BLOCK, UNROLL>
        <<<grid, BLOCK>>>(ws_base + i * MEMORY_OFFSET,
                          ws_base + i * MEMORY_OFFSET + size_in_byte / 2);
  }
}

void benchmark(size_t size_in_byte) {
  printf("%luMB (r+w)\n", size_in_byte / (1 << 20));
  double size_gb = (double)size_in_byte / (1 << 30);

  size_t n = size_in_byte / sizeof(uint4);
  size_t grid = n / (BLOCK * UNROLL);

  // std::cout << "n " << n << std::endl;
  // std::cout << "grid " << grid << std::endl;
  // std::cout << "size_in_byte " << size_in_byte << std::endl;

  static_assert(MEMORY_OFFSET % sizeof(uint4) == 0, "invalid MEMORY_OFFSET");

  uint datasize = size_in_byte + MEMORY_OFFSET * BENCH_ITER;

  // std::cout  << "datasize" << datasize<< std::endl;

  uint *wordsize;
  hipMallocManaged(&wordsize, datasize);  // 4M + 16M * 2

  // set all zero for read-only kernel
  int num = datasize / sizeof(uint);
  // std::cout << "num " << num << std::endl;

  hipMemset(&wordsize, 0, num);  // 4M + 16M * 2

  // for (int i = 0; i < num; i++) {
  //   wordsize[i] = 0;
  // }
  // std::cout <<  "value  " << wordsize[0] << std::endl;

  // warmup
  auto time_ms = hip_timer::execution(warmup, wordsize, size_in_byte, grid);

  // read
  time_ms = hip_timer::execution(read, wordsize, size_in_byte, grid);
  std::cout << "Read : " << size_gb * BENCH_ITER / ((double)time_ms / 1000)
            << " GB/s\n";

  time_ms = hip_timer::execution(write, wordsize, size_in_byte, grid);
  std::cout << "Write : " << size_gb * BENCH_ITER / ((double)time_ms / 1000)
            << " GB/s\n";

  // time_ms = hip_timer::execution(copy, wordsize, size_in_byte, grid);
  // std::cout << "Copy : " << size_gb * BENCH_ITER / ((double)time_ms / 1000)
  //           << " GB/s\n";

  printf("---------------------------\n");

  hipFree(wordsize);
}

int main() {
  size_t size = (1lu << 20) * 4;  // 4M

  // 1GB~4GB
  while (size <= (1lu << 31)) {
    benchmark(size);
    size *= 2;
  }

  return 0;
}
