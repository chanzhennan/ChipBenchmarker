#include <hip/hip_fp16.h>
#include <stdio.h>

#include <cstdint>
#include <cstdio>
#include <ctime>
#include <iostream>

#include "hip/hip_runtime.h"
#include "ldg_stg_inst.h"

const int WARMUP = 100;
// number of LDS instructions to be timed
const int ROUND = 50;

#define warpsize 64

__global__ __launch_bounds__(warpsize,
                             1) void smem_latency_kernel(const uint32_t *addr,
                                                         uint32_t *ret,
                                                         uint32_t *clk) {
  __shared__ uint32_t smem[warpsize];
  smem[threadIdx.x] = addr[threadIdx.x];

  // convert shared addr -> base addr
  uint32_t smem_addr = (uint32_t)(uintptr_t)&(smem[threadIdx.x]);
  uint32_t val = 0;

  uint32_t start = realtime();

#pragma unroll
  for (int i = 0; i < ROUND; ++i) {
      asm volatile(
      "ds_load_b32 %0 %1;\n"
      "s_waitcnt lgkmcnt(0),vmcnt(0);\n"
      : "=v"(val)  // 使用 "=s" 修饰符，表示将结果保存到 u32addr 变量中
      : "v"(smem_addr)  // 使用 "v" 修饰符，表示输入操作数是一个通用寄存器（v 寄存器）
    );
  }

  // [debug]
  // printf("[%d] %d\n", (int)threadIdx.x, val);

  uint32_t stop = realtime();

  clk[threadIdx.x] = stop - start;

  // dummy write back
  if (smem_addr == ~0x0) {
    *ret = smem_addr;
  }
}

int main() {
  uint32_t *h_addr = (uint32_t *)malloc(warpsize * sizeof(uint32_t));

  for (int i = 0; i < warpsize; ++i) {
    h_addr[i] = i * 4;
  }

  uint32_t *d_addr, *d_ret;
  hipMalloc(&d_addr, warpsize * sizeof(uint32_t));
  hipMalloc(&d_ret, sizeof(uint32_t));
  hipMemcpy(d_addr, h_addr, warpsize * sizeof(uint32_t), hipMemcpyHostToDevice);

  uint32_t *d_clk;
  hipMalloc(&d_clk, warpsize * sizeof(uint32_t));

  // pupulate l0/l1 i-cache
  for (int i = 0; i < WARMUP; ++i) {
    smem_latency_kernel<<<1, warpsize>>>(d_addr, d_ret, d_clk);
  }

  // shared memory latency benchmark
  smem_latency_kernel<<<1, warpsize>>>(d_addr, d_ret, d_clk);

  uint32_t h_clk[warpsize];
  hipMemcpy(h_clk, d_clk, warpsize * sizeof(uint32_t), hipMemcpyDeviceToHost);
  printf("shared memory latency %u cycles\n", h_clk[0] / ROUND);

  hipFree(d_addr);
  hipFree(d_ret);
  hipFree(d_clk);
  hipFreeHost(h_addr);

  return 0;
}
