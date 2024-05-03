#include <hip/hip_fp16.h>
#include <stdio.h>

#include <cstdint>
#include <cstdio>
#include <ctime>
#include <iostream>

#include "hip/hip_runtime.h"
#include "ldg_stg_inst.h"

const int WARMUP = 100;
const int THREAD = 4;
// number of LDG instructions to be timed
const int ROUND = 50;

template <int ROUND>
__global__ __launch_bounds__(THREAD, 1)
void l1_latency_kernel(void **ptr, void **ret, int32_t *clk) {
    void **ldg_ptr = ptr + threadIdx.x;


    uint32_t val;
    // warm up to populate l1 cache
    for(int i = 0; i < ROUND; ++i) {
        asm volatile(
        "flat_store_b32 %1, %0;\n"  
        "s_waitcnt lgkmcnt(0),vmcnt(0);"
        : "=v"(val)
        : "v"(ldg_ptr)
        );
    }
    
    // if (threadIdx.x == 0)
    // {
    //     printf("xx %ld\n", ldg_ptr[0]);
    // }
    
    int32_t start = realtime();


    for(int i = 0; i < ROUND; ++i) {
        asm volatile(
        "flat_store_b32 %1, %0;\n"  
        "s_waitcnt lgkmcnt(0),vmcnt(0);"
        : "=v"(val)
        : "v"(ldg_ptr)
        );
    }

    int32_t stop = realtime();

    clk[threadIdx.x] = stop - start;

    // dummy write back
    if (ldg_ptr == nullptr) {
        *ret = ldg_ptr;
    }
}


int main() {
    void **d_ptr;
    void **d_ret;
    int32_t *d_clk;
    hipMalloc(&d_ptr, THREAD * sizeof(void *));
    hipMalloc(&d_ret, sizeof(void *));
    hipMalloc(&d_clk, THREAD * sizeof(int32_t));

    void **h_ptr =  (void**)malloc(THREAD * sizeof(void *));

    for (int i = 0; i < THREAD; ++i) {
        h_ptr[i] = d_ptr + i;
        // printf("%ld\n", (int64_t)h_ptr[i] );
    }

    hipMemcpy(d_ptr, h_ptr, THREAD * sizeof(void *), hipMemcpyHostToDevice);

    // populate instruction cache
    for (int i = 0; i < WARMUP; ++i) {
        l1_latency_kernel<ROUND><<<1, THREAD>>>(d_ptr, d_ret, d_clk);
    }

    // l1 cache latency benchmark
    l1_latency_kernel<ROUND><<<1, THREAD>>>(d_ptr, d_ret, d_clk);

    int32_t h_clk[THREAD];
    hipMemcpy(h_clk, d_clk, THREAD * sizeof(int32_t), hipMemcpyDeviceToHost);
    printf("l1 cache latency %u cycles\n", h_clk[0] / ROUND);

    hipFree(d_ptr);
    hipFree(d_ret);
    hipFree(d_clk);
    free(h_ptr);

  return 0;
}
