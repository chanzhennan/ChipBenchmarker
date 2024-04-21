

## Summary
The purpose of this repository is to benchmark the hardware performance of multiple GPU chips across different platforms. It includes testing HBM bandwidth and instruction latency, L2 bandwidth and read latency, L1 read latency, as well as shared memory bandwidth and latency.


## Install

```bash
#fix in build.sh
cmake -DBUILD_WITH_CUDA=ON -DARCH=sm_80 ..

sh build.sh 
```

---

## Benchmark
Devices | Dram BW |  Dram Latency | L1 Latency | L2 BW | L2 Latency | Smem BW | Smem Latency |
|----|----|----|----|----|----|----|----|
|[CUDA SM80](https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html)|✅|✅|✅|✅|✅|✅|✅|
|[AMD GFX1100](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna3-shader-instruction-set-architecture-feb-2023_0.pdf)|✅|✅|✅|✅|✅|🔨|🔨|
|[AMD GFX90](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/instinct-mi200-cdna2-instruction-set-architecture.pdf)|✅|🔨|🔨|✅|✅|🔨|🔨|
|[MACA MXC500](https://www.metax-tech.com/prod.html?cid=2)|🔨|🔨|🔨|🔨|🔨|🔨|🔨|


## Performance

| Devices | Dram BW (Read) | Dram BW (Write) | Dram BW (Copy) | Dram Latency (cycles) | L1 Latency (cycles) | L2 BW | L2 Latency (cycles) | Smem BW (Measured) (byte/cycle) | Smem BW (Theoretical) (byte/cycle) | Smem Latency (cycles) |
|---------|----------------|-----------------|----------------|-----------------------|---------------------|-------|----------------------|---------------------------------|------------------------------------|----------------------|
| [CUDA A100](https://huggingface.co/meta-llama) | 1598.51GB/s | 1732.14GB/s | 1519.82GB/s | 499 | 33 | 🔨 | 332 | 111.734879 | 128 | 23 |
| [AMD RX7900](https://huggingface.co/meta-llama) | 706.31GB/s | 813.87GB/s | 785.77GB/s | 551 | 🔨 | 3253.16GB/s | 340 | 🔨 | 🔨 | 🔨 |

## Device
- [Nvidia A100](https://www.techpowerup.com/gpu-specs/a100-pcie-40-gb.c3623)
- AMD
    - [AMD MI210](https://www.techpowerup.com/gpu-specs/radeon-instinct-mi210.c3857) 
    - [AMD RX7900](https://www.techpowerup.com/gpu-specs/radeon-rx-7900-xt.c3912)
- [MACA MXC500](https://www.metax-tech.com/prod.html?cid=2)

---
