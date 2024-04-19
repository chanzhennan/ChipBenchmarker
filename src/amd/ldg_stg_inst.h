

#include <hip/hip_fp16.h>

#include "hip/hip_runtime.h"

__device__ __forceinline__ uint4 ldg_cs_128(const void *ptr) {
  uint4 ret;
  asm volatile(
      "flat_load_dwordx4 %0, %1;\n"
      "s_waitcnt lgkmcnt(0);"
      : "=v"(ret)
      : "v"(ptr));

  return ret;
}

__device__ __forceinline__ void stg_cs_128(uint4 &reg, void *ptr) {
  asm volatile(
      "flat_store_dwordx4 %1, %0;\n"
      "s_waitcnt lgkmcnt(0);"
      : "=v"(reg)
      : "v"(ptr));
}

__device__ __forceinline__ int ldg_cg_32(const int *ptr) {
  int ret;
  asm volatile(
      "flat_load_dword %0, %1;\n"
      "s_waitcnt lgkmcnt(0);"
      : "=v"(ret)
      : "v"(ptr));

  return ret;
}

__device__ __forceinline__ void stg_cs_32(uint4 &reg, void *ptr) {
  asm volatile(
      "flat_store_dword %1, %0;\n"
      "s_waitcnt lgkmcnt(0);"
      : "=v"(reg)
      : "v"(ptr));
}
