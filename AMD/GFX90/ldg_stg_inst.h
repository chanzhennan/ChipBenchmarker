

#include <hip/hip_fp16.h>

#include "hip/hip_runtime.h"

__device__ __forceinline__ uint4 ldg_cs_128(const void *ptr) {
  uint4 ret;

  // uint4 tmp = ((uint4*)ptr)[0];
  // printf("xxx %d %d %d %d \n", tmp.x, tmp.y, tmp.z, tmp.w);

  asm volatile(
      "flat_load_dwordx4 %0, %1;\n"
      "s_waitcnt vmcnt(0) lgkmcnt(0);"
      : "=v"(ret)
      : "v"(ptr));

  // printf("kkk %d %d %d %d \n", ret.x, ret.y, ret.z, ret.w);

  return ret;
}

__device__ __forceinline__ void stg_cs_128(uint4 &reg, void *ptr) {
  asm volatile(
      "flat_store_dwordx4 %1, %0;\n"
      "s_waitcnt vmcnt(0) lgkmcnt(0);"
      : "=v"(reg)
      : "v"(ptr));
}

__device__ __forceinline__ int ldg_cg_32(const int *ptr) {
  int ret;
  asm volatile(
      "flat_load_dword %0, %1;\n"
      "s_waitcnt vmcnt(0) lgkmcnt(0);"
      : "=v"(ret)
      : "v"(ptr));

  return ret;
}

__device__ __forceinline__ void stg_cs_32(uint4 &reg, void *ptr) {
  asm volatile(
      "flat_store_dword %1, %0;\n"
      "s_waitcnt vmcnt(0) lgkmcnt(0);"
      : "=v"(reg)
      : "v"(ptr));
}

// char ptr
__device__ __forceinline__ uint32_t ldg_cg_char(char *ptr) {
  uint32_t ret;

  asm volatile(
      "flat_load_dword %0, %1;\n"
      "s_waitcnt vmcnt(0) lgkmcnt(0);"
      : "=v"(ret)
      : "v"(ptr));

  return ret;
}

// uint32 ptr
__device__ __forceinline__ uint32_t ldg_cg_uint32(void *ptr) {
  uint32_t ret;

  asm volatile(
      "flat_load_dword %0, %1;\n"
      "s_waitcnt  vmcnt(0) & lgkmcnt(0);"
      : "=v"(ret)
      : "v"(ptr));

  return ret;
}

__device__ __forceinline__ uint64_t realtime() {
  uint64_t _time;

  // s_memtime        sdst:b64 [Return current 64-bit timestamp.]
  // s_memrealtime    sdst:b64 [Return current 64-bit RTC.]
  // refer to
  // https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/instinct-mi200-cdna2-instruction-set-architecture.pdf

  asm volatile(
      "s_barrier;\n"     // Wait for data to be returned
      "S_MEMTIME %0;\n"  // Message type 0x83 for REALTIME
      "s_waitcnt vmcnt(0) lgkmcnt(0);"
      : "=s"(_time)
      :
      : "memory");
  return _time;
}
