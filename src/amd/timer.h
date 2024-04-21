#include <hip/hip_fp16.h>

#include <chrono>

#include "hip/hip_runtime.h"

// 定义一个模板函数，用于计时
template <typename TimeT = std::chrono::milliseconds>
struct measure {
  template <typename F, typename... Args>
  static typename TimeT::rep execution(F func, Args&&... args) {
    auto start = std::chrono::steady_clock::now();
    func(std::forward<Args>(args)...);
    auto duration = std::chrono::duration_cast<TimeT>(
        std::chrono::steady_clock::now() - start);
    // 计算吞吐量
    return duration.count();
  }
};

struct hip_timer {
  template <typename F, typename... Args>
  static float execution(F func, Args&&... args) {
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float time_ms = 0.f;

    hipEventRecord(start, nullptr);
    func(std::forward<Args>(args)...);
    hipEventRecord(stop, nullptr);
    hipEventSynchronize(stop);

    hipEventElapsedTime(&time_ms, start, stop);

    hipEventDestroy(start);
    hipEventDestroy(stop);

    return time_ms;
  }
};
