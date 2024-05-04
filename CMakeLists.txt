cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(benchmark LANGUAGES CXX C)

# Check if at least one backend is enabled
if(NOT BUILD_WITH_CUDA
   AND NOT BUILD_WITH_GFX90
   AND NOT BUILD_WITH_MX
   AND NOT BUILD_WITH_GFX1100)
  message(
    FATAL_ERROR
      "At least one backend (BUILD_WITH_CUDA, BUILD_WITH_GFX90, or BUILD_WITH_MX) must be ON"
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
  target_link_libraries(${targetname} PRIVATE ${CUDA_LIBRARIES})
endfunction()

if(BUILD_WITH_CUDA)

  if(NOT DEFINED ARCH)
    message(FATAL_ERROR "Please specify the CUDA architecture (ARCH) to build.")
  endif()

  find_package(CUDA REQUIRED)
  enable_language(CUDA)

  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=${ARCH}")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -O3 -DNDEBUG")

  # Add your CUDA executables here
  add_cuda_executable(dram_bandwidth "CUDA/dram_bandwidth.cu")
  add_cuda_executable(dram_latency "CUDA/dram_latency.cu")

  add_cuda_executable(l1cache_latency "CUDA/l1cache_latency.cu")

  add_cuda_executable(l2cache_bandwidth "CUDA/l2cache_bandwidth.cu")
  add_cuda_executable(l2cache_latency "CUDA/l2cache_latency.cu")

  add_cuda_executable(smem_bandwidth "CUDA/smem_bandwidth.cu")
  add_cuda_executable(smem_latency "CUDA/smem_latency.cu")

elseif(BUILD_WITH_GFX90)

  find_package(HIP REQUIRED PATHS /opt/rocm/hip/cmake)
  enable_language(HIP)

  set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} --offload-arch=gfx90a")
  set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -O3 -std=c++20 -DNDEBUG")

  add_hip_executable(dram_bandwidth "AMD/GFX90/dram_bandwidth.hip")
  add_hip_executable(dram_latency "AMD/GFX90/dram_latency.hip")

  add_hip_executable(l1cache_latency "AMD/GFX90/l1cache_latency.hip")

  add_hip_executable(l2cache_latency "AMD/GFX90/l2cache_latency.hip")
  add_hip_executable(l2cache_bandwidth "AMD/GFX90/l2cache_bandwidth.hip")

  add_hip_executable(smem_bandwidth "AMD/GFX90/smem_bandwidth.hip")
  add_hip_executable(smem_latency "AMD/GFX90/smem_latency.hip")

elseif(BUILD_WITH_GFX1100)
  find_package(HIP REQUIRED PATHS /opt/rocm/hip/cmake)
  enable_language(HIP)
  set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} --offload-arch=gfx1100")
  set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -O3 -std=c++20 -DNDEBUG")

  add_hip_executable(dram_bandwidth "AMD/GFX1100/dram_bandwidth.hip")
  add_hip_executable(dram_latency "AMD/GFX1100/dram_latency.hip")

  add_hip_executable(l1cache_latency "AMD/GFX1100/l1cache_latency.hip")

  add_hip_executable(l2cache_latency "AMD/GFX1100/l2cache_latency.hip")
  add_hip_executable(l2cache_bandwidth "AMD/GFX1100/l2cache_bandwidth.hip")

  add_hip_executable(smem_bandwidth "AMD/GFX1100/smem_bandwidth.hip")
  add_hip_executable(smem_latency "AMD/GFX1100/smem_latency.hip")

elseif(BUILD_WITH_MX)
  # Add your MX executables here
endif()
