# CUDA-Accelerated LZO1X Compression

An investigation into accelerating the **compression** stage of LZO1X on NVIDIA GPUs. The project explores two complementary kinds of parallelism: processing independent input chunks concurrently (*inter-task parallelism*) and distributing operations of a single compression task across cooperating GPU threads/warps (*intra-task parallelism*).

> **Research status:** Ongoing (Started as Summer 2026 internship that was concluded with a progress report). The benchmark results below refer specifically to the completed **inter-task** implementations documented for the summer practice report. Newer experimental match-finding and encoding variants are described separately and have **no benchmark claim here**. The final benchmarking of all developed kernels will be done on a seperate system, therefore, none of the results presented here are final.

## Motivation

LZO1X is designed for low-overhead compression and particularly fast decompression. Its compressor, however, contains sequential dependencies: a dictionary is updated as input is traversed so matches influence the next input position and encoded output length depends on the discovered match and literal runs. This work asks how much of that computation can use GPU parallelism, and what is lost in exchange for throughput.

## Highlights

- **Up to 2.66× CPU-baseline throughput** in the reported benchmarks (human genome dataset).
- **Up to 804.64 MB/s** measured end-to-end compression throughput using the global-memory inter-task implementation.
- Two inter-task kernels with alternative dictionary placements: **global memory** and **shared memory**.
- Three CUDA streams, asynchronous transfers, and a batched compression pipeline.
- In a later GTX 1660 Ti experiment, warp-parallel match finding delivered **2.03× throughput relative to sequential GPU match finding** with the same sequential encoder (one dataset/configuration).
- Experimental **warp-parallel match finding with separate literal and match encoding warps**.
- Nsight Systems, Nsight Compute, and `nvprof` used to analyze execution behavior where supported.



## Implementations

| Family | Variant | Approach | Status / evidence |
| --- | --- | --- | --- |
| Inter-task | Global-memory dictionary | Each GPU thread compresses an independent input chunk; dictionary state resides in global memory | Implemented and benchmarked in the 2026 report |
| Inter-task | Shared-memory dictionary | Independent chunk compression, with dictionary and selected working data placed in block shared memory | Implemented and benchmarked in the 2026 report |
| Intra-task | Sequential match finding + sequential encoding | A single lane performs compression after cooperative dictionary initialization | Measured and round-trip checked in `lzo_match_compare_1660.cu` |
| Intra-task | Warp-parallel match finding + sequential encoding | One warp checks up to 32 candidate positions per round, preserving dictionary-update ordering; lane 0 retains the encoding role | Measured and round-trip checked in `lzo_match_compare_1660.cu` |
| Intra-task | Warp-parallel match finding + parallel encoding | One match-finding warp communicates match data to cooperating literal and match encoding warps using a block-scope atomic and barriers | Experimental `lzo_intra.cu` kernel; separately benchmarked with its host harness |

### Inter-task compression pipeline

1. Partition an input file into batches and independently compressed thread-sized chunks.
2. Transfer batches to the GPU and launch compression kernels, using three CUDA streams to create opportunities for transfer/compute overlap.
3. Determine per-chunk output sizes, compute offsets using a CUB exclusive prefix scan, and concatenate compressed chunks.
4. Transfer compressed output and associated chunk metadata to host memory.

Independent chunks require boundaries and metadata; the resulting output is **not necessarily a single ordinary LZO1X stream**, nor does it have the same compression ratio as the continuous CPU implementation. Any decompressor must understand the chosen chunk framing and metadata.

### Intra-task experiments

The newer intra-task work examines two decisions separately:

**Match finding: sequential or warp-parallel.** The comparison program contains `compress_sequential` (one active compression lane per chunk) and `compress_warp` (32 candidate positions inspected concurrently). Both use global-memory dictionaries and **sequential encoding**, so their difference isolates the match-finding approach more effectively. Both versions were checked by decompressing every chunk and comparing the reconstructed bytes with the original input.

**Encoding: sequential or parallel.** The newer `compress` kernel in `lzo_intra.cu` assigns warp 0 to match finding, with additional warps splitting literal copying and match information encoding. A block-scoped `cuda::atomic` signals when match information is available; `cuda::barrier` coordinates the encoding warps. Output sizes and offsets must be resolved before independent lanes write their pieces. The available benchmark compares this complete pipeline against CPU, **not against an otherwise identical sequential-encoding GPU implementation**.

The original internship report includes an *intra-task design proposal*, not completed performance measurements for these variants. The implementation has continued evolving since then.

## Newer GTX 1660 Ti experiments (November 2026)

These newer measurements were supplied separately from the internship report. Input: a **1,157,312,766-byte** *Pride and Prejudice* text file. Host machine: **Fedora Linux 44**, Intel **Core i9-10850K**, NVIDIA **GeForce GTX 1660 Ti**, 32 GiB system RAM. GPU throughput is calculated as input size divided by reported elapsed time (decimal MB/s).

| Benchmark / implementation | GPU configuration | Time | Throughput | Relative CPU speedup |
| --- | --- | ---: | ---: | ---: |
| Match-comparison CPU baseline | — | 2,787.70 ms | 415.15 MB/s | 1.00× |
| Sequential GPU match finding, sequential encoding | 32 threads/block; 16 KiB/chunk; 24 MiB batch; 3 streams | 5,015.38 ms | 230.75 MB/s | 0.56× |
| Warp-parallel GPU match finding, sequential encoding | Same configuration | 2,471.93 ms | **468.18 MB/s** | **1.13×** |
| Parallel-encoder run CPU baseline | — | 3,029.52 ms | 382.01 MB/s | 1.00× |
| Warp match finding + parallel encoding | 16 KiB/chunk; 48 MiB batch; 3 streams | 4,521.82 ms | 255.94 MB/s | 0.67× |

In the controlled match-finding comparison, the warp method was **2.03× faster** than sequential GPU match finding. Both output variants passed the program's chunk-wise LZO decompression and full-file byte comparison. Compressed-size ratios differed (0.6410 warp versus 0.7172 sequential), so the current results should **not** be described as identical compressed output.

The parallel-encoding experiment is **not a controlled encoding-only comparison** with the sequential-encoding experiment: the batch sizes differ, the harnesses differ, and CPU timings differ. Thus the 255.94 versus 468.18 MB/s figures do **not establish the cost of parallel encoding on its own**. A matched same-machine A/B test remains future work. These are individual measurements, not multi-run confidence intervals.

## Reported benchmark results

The following measurements are from the **inter-task implementations** in the summer 2026 report. Throughput measures the complete compression pipeline, not isolated kernel execution.

| Dataset | Best global-memory throughput | Best shared-memory throughput |
| --- | ---: | ---: |
| Human genome (GRCh38) | **804.64 MB/s** | 691.44 MB/s |
| Linux kernel source | **776.10 MB/s** | 667.59 MB/s |
| *Moby Dick* text corpus | **516.81 MB/s** | 448.93 MB/s |
| Wikipedia text | **564.50 MB/s** | 510.47 MB/s |

**Best reported speedup:** **2.66×** versus the sequential CPU implementation, using the global-memory kernel on the human genome dataset with a **72 MiB batch**, **32 threads/block**, and **32 KiB/thread**.

### What the measurements showed

- Larger batches generally improved end-to-end throughput by amortizing repeated transfer, launch, scan, and concatenation overhead.
- Increasing threads per block did **not** monotonically improve performance; moderate sizes often worked best.
- Larger per-thread input chunks generally improved the compressed-size ratio, but could reduce throughput.
- Independent GPU chunks produced greater compression overhead than the continuous CPU baseline.
- The shared-memory kernel showed shorter kernel times in profiling captures, yet global-memory configurations attained greater **end-to-end** throughput in the reported benchmarks: kernel time alone did not determine overall performance.

Results are hardware-, dataset-, and configuration-specific. They should not be interpreted as a universal GPU-versus-CPU speedup.

## Benchmark environment

- **GPU:** NVIDIA GeForce GTX 1080 Ti (compute capability 6.1); laboratory workstation equipped with two such GPUs.
- **CPU:** Intel Xeon Silver 4114 @ 2.20 GHz.
- **Operating system:** Ubuntu 20.04.6 LTS.
- **CUDA environment:** NVIDIA driver 470.256.02; CUDA toolkit/compiler versions described in the report (including `nvcc` 10.1.243).
- **Tools:** CUDA C/C++, CMake, CUB, Nsight Systems, Nsight Compute, `nvprof`.
- **Parameters investigated:** batch sizes of 24, 48, and 72 MiB; per-thread inputs of 16, 32, and 48 KiB; several block sizes.

Hardware and toolchain information records the **reported test environment**; it is not a verified build requirement for the latest sources.

## Current development and reproducibility

The project remains under active investigation. In particular, newer comparisons between sequential/warp-parallel match finding and sequential/parallel encoding have not been reproduced from the latest sources for this README.

Before interpreting newer variants as faster or functionally equivalent, they should be checked against a trusted LZO1X baseline, including decompression round trips, edge cases at chunk boundaries, and matched benchmark conditions.


## Limitations and future work

- Quantify the isolated benefit and coordination cost of warp-parallel match finding.
- Compare sequential and parallel encoding under identical inputs and kernel configurations.
- Validate decompression and chunk framing for all variants.
- Investigate whether intra-task parallelism compensates for synchronization, branch divergence, and dictionary access overhead.

## Background and attribution

This is a research implementation inspired by the original **LZO** library by Markus F. X. J. Oberhumer.

- [Original LZO project](https://www.oberhumer.com/opensource/lzo/)

