#include <hip/hip_runtime_api.h>

#include "hip/hip_runtime.h"

// !!!! Refer to
// https://github.com/search?q=repo%3Aaditya4d1%2Fgemm-vega64%20Float4&type=code
// !!!
typedef __fp16 Half4 __attribute__((ext_vector_type(4)));
typedef float Float4 __attribute__((ext_vector_type(4)));
typedef __fp16 Half2 __attribute__((ext_vector_type(2)));

__device__ __forceinline__  // ptr -> ret
    uint4
    ldg_cs(const void *ptr) {
  uint4 ret;
  asm volatile(
      "flat_load_b128 %0, %1;\n"
      "s_waitcnt lgkmcnt(0);"
      : "=v"(ret)
      : "v"(ptr));

  return ret;
}

__device__ __forceinline__  // reg -> ptr
    void
    stg_cs(uint4 &reg, void *ptr) {
  asm volatile(
      "flat_store_b128 %1, %0;\n"
      "s_waitcnt lgkmcnt(0);"
      : "=v"(reg)
      : "v"(ptr));
}

__device__ __forceinline__ uint32_t realtime() {
  uint32_t rtn;
  asm volatile(
      "s_barrier;\n"  // Wait for data to be returned
      "s_getreg_b32 %0, hwreg(HW_REG_SHADER_CYCLES)\n"
      : "=s"(rtn)
      :
      : "memory");
  return rtn;
}

inline __device__ void shared_write_b128(Float4 &val, uint32_t &lds) {
  asm volatile("ds_write_b128 %0, %1 \n;" : : "v"(lds), "v"(val));
}
