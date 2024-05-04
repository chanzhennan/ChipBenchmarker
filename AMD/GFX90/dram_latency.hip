#include <hip/hip_fp16.h>
#include <stdio.h>

#include <cstdint>
#include <cstdio>
#include <iostream>

#include "hip/hip_runtime.h"

#include "ldg_stg_inst.h"

// number of LDG instructions to be timed
const int ROUND = 30;
// stride in byte between LDG instructions
// should be greater than L2 cache line size to avoid L2 cache hit
const int STRIDE = 1024;

// workspace size in byte to flush L2 cache
const int L2_FLUSH_SIZE = (1 << 20) * 128;

#define WARP_SIZE 64

template <int BLOCK>
__global__ void flush_l2_kernel(uint32_t *x, uint32_t *y) {
  int warp_id = threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;

  uint32_t *x_ptr = x + blockIdx.x * BLOCK + warp_id * WARP_SIZE;
  uint32_t sum = 0;

  // #pragma unroll
  for (int i = 0; i < WARP_SIZE; ++i) {
    const uint32_t *ldg_ptr = x_ptr + (lane_id ^ i);
    sum += ldg_ptr[0];
  }

  if (sum != 0) {
    *y = sum;
  }
}

void flush_l2() {
  uint32_t *x;
  uint32_t *y;
  hipMalloc(&x, L2_FLUSH_SIZE);
  hipMalloc(&y, sizeof(uint32_t));
  hipMemset(x, 0, L2_FLUSH_SIZE);

  int n = L2_FLUSH_SIZE / sizeof(int);
  flush_l2_kernel<128><<<n / 128, 128>>>(x, y);
  hipDeviceSynchronize();

  hipFree(x);
  hipFree(y);
}

template <int ROUND>
__global__ void dram_latency_kernel(uint32_t *stride, uint32_t *ret,
                                    int32_t *clk) {
  const char *ldg_ptr = reinterpret_cast<const char *>(stride + threadIdx.x);
  uint32_t val;
  asm volatile(
      "flat_load_dword %0, %1;\n"  // Load scale data from dram
      "s_waitcnt lgkmcnt(0),vmcnt(0);"
      : "=v"(val)
      : "v"(ldg_ptr)
      : "memory");

  ldg_ptr += val;

  uint64_t start = timestamp();

  // #pragma unroll
  for (int i = 0; i < ROUND; ++i) {
    asm volatile(
        "flat_load_dword %0, %1;\n"
        "s_waitcnt lgkmcnt(0),vmcnt(0);"
        : "=v"(val)
        : "v"(ldg_ptr)
        : "memory"  // Load scale data from dram
    );
    ldg_ptr += val;

    // if (threadIdx.x == 0){
    //     printf("%d\n", val);
    // }
    
  }

  uint64_t stop = timestamp();


  clk[threadIdx.x] = (int32_t)(stop - start);

  if (val == 1) {  // To prevent compiler optimizate the loop
    *ret = val;
  }
}

int main() {
  static_assert(
      STRIDE >= WARP_SIZE * sizeof(uint32_t) && STRIDE % sizeof(uint32_t) == 0,
      "invalid 'STRIDE'");

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
  hipMalloc(&d_clk, WARP_SIZE * sizeof(int32_t));

  // pupulate l0/l1 i-cache and l2 cache
  dram_latency_kernel<ROUND><<<1, WARP_SIZE>>>(d_stride, d_ret, d_clk);
  hipDeviceSynchronize();

  flush_l2();

  dram_latency_kernel<ROUND><<<1, WARP_SIZE>>>(d_stride, d_ret, d_clk);
  hipDeviceSynchronize();

  //   l1 cache latency benchmark
  hipError_t status = hipGetLastError();
  if (status != hipSuccess)
    std::cerr << "Error: HIP reports " << hipGetErrorString(status)
              << std::endl;

  int32_t h_clk[WARP_SIZE];
  hipMemcpy(h_clk, d_clk, WARP_SIZE * sizeof(int32_t), hipMemcpyDeviceToHost);

  printf("DRAM latency %d cycles\n", h_clk[0] / ROUND);

  hipFree(d_stride);
  hipFree(d_ret);
  hipFree(d_clk);
  free(h_stride);

  return 0;
}
