cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(benchmark LANGUAGES CXX C HIP)

# Check if at least one backend is enabled
if(NOT BUILD_WITH_CUDA
   AND NOT BUILD_WITH_MI210
   AND NOT BUILD_WITH_MX)
  message(
    FATAL_ERROR
      "At least one backend (BUILD_WITH_CUDA, BUILD_WITH_MI210, or BUILD_WITH_MX) must be ON"
  )
endif()

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -pthread")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -w")

# Add an executable target with HIP source file
function(add_hip_executable targetname filepath)
  get_filename_component(filename ${filepath} NAME_WE) # 获取文件名（不带扩展名）
  add_executable(${targetname} ${filepath}) # 添加可执行文件，使用自定义的目标名称
  target_include_directories(${targetname} PRIVATE ${HIP_INCLUDE_DIRS})
  set_target_properties(${targetname} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
  target_link_libraries(${targetname} PRIVATE hip::host)
endfunction()

# Add an executable target with CUDA source file
function(add_cuda_executable targetname filepath)
  get_filename_component(filename ${filepath} NAME_WE) # Get the filename
                                                       # (without extension)
  add_executable(${targetname} ${filepath}) # Add the executable file with a
                                            # custom target name
  target_include_directories(${targetname} PRIVATE ${CUDA_INCLUDE_DIRS})
  # set_target_properties(${targetname} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
  target_link_libraries(${targetname} PRIVATE cuda::host)
endfunction()

if(BUILD_WITH_CUDA)
  find_package(CUDA REQUIRED PATHS /usr/local/cuda)
  enable_language(CUDA)

  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=${ARCH}")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -std=c++20 -DNDEBUG")

  # Add your CUDA executables here
  add_cuda_executable(dram_bandwidth "CUDA/dram_bandwidth.cuda")
  add_cuda_executable(l2cache_bandwidth "CUDA/dram_latency.cuda")

  add_cuda_executable(l2cache_bandwidth "CUDA/l1cache_latency.cuda")

  add_cuda_executable(l2cache_bandwidth "CUDA/l2cache_bandwidth.cuda")
  add_cuda_executable(l2cache_latency "CUDA/l2cache_latency.cuda")

  add_cuda_executable(l2cache_latency "CUDA/smem_bandwidth.cuda")
  add_cuda_executable(l2cache_latency "CUDA/smem_latency.cuda")

elseif(BUILD_WITH_MI210)

  find_package(HIP REQUIRED PATHS /opt/rocm/hip/cmake)
  enable_language(HIP)

  set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} --offload-arch=gfx90a")
  set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -O3 -std=c++20 -DNDEBUG")

  # Add your HIP executables here
  add_hip_executable(dram_bandwidth "MI210/dram_bandwidth.hip")
  add_hip_executable(l2cache_latency "MI210/l2cache_latency.hip")
  add_hip_executable(l2cache_bandwidth "MI210/l2cache_bandwidth.hip")

elseif(BUILD_WITH_MX)
  # Add your MX executables here
endif()
