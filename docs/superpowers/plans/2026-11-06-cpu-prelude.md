# CPU (OpenMP) Prelude for the Native Component — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the same `cuda/src` sources as `beamz._cpu`, a host-platform XLA FFI extension that runs every barrier-free kernel through an OpenMP block scheduler, and pass the CUDA parity suite on CPU for the streamed schedule.

**Architecture:** A third branch of the platform preludes (`BEAMZ_PLATFORM_CPU`). `gpu_device.h` turns kernel qualifiers into `inline` functions, provides thread-local `blockIdx`/`threadIdx`/`blockDim`/`gridDim`, a one-lane "warp", OpenMP atomics and a bf16 struct. A new `BEAMZ_LAUNCH(grid, block, shared, stream, kernel...)` macro replaces `<<<>>>`: on CUDA/HIP it expands to the original launch, on CPU to `beamz::cpu::Launch<kernel>` which runs blocks in an OpenMP loop and threads as nested z/y/x loops (x innermost). `gpu_runtime.h` gains synchronous host stubs. Kernels that need dynamic shared memory (all six barrier kernels) are rejected at launch with `cudaErrorNotSupported`; host scheduling never selects them on CPU.

**Tech Stack:** GCC/Clang C++17, OpenMP, CMake `LANGUAGE CXX` for `.cu`, XLA FFI on the `Host` platform, JAX CPU.

**Spec:** conversation of 2026-11-06 ("add the openmp prelude to the plan and do it"); design discussed just before it. Builds on `2026-11-06-hip-port.md` (branch `hip-port`, PR #522).

## Global Constraints

- CUDA and HIP builds unchanged: every new macro expands to the original tokens there.
- CPU kernels use `-ffp-contract=off`, no `-ffast-math`; `__fmaf_rn` → `std::fma`, `__fmul_rn` → plain multiply.
- Only barrier-free kernels run on CPU. Shared-memory kernels compile but return `cudaErrorNotSupported` if launched.
- `backend="auto"` keeps JAX on CPU; explicit `backend="cuda_streamed"`/`"cuda"` uses `beamz._cpu` when JAX has no GPU. `cuda_mstep` is unavailable on CPU (target not registered).
- No CUDA graphs on CPU: `BeamzLaunchGraph` calls the enqueue function directly.

## Review Focus

1. Thread-local launch indices must be per OpenMP thread; a shared global silently races. → Task 1 host scheduler test with a 2-thread grid checksum.
2. Atomic float adds from concurrent blocks (source injection, adjoint sums) must not lose updates. → Task 1 atomic test.
3. `LaunchTemporalYeeProgram` (no CPML/sources/monitors) always uses a shared-memory kernel; CPU must fall back to per-phase stepping, not error. → Task 3 parity on a PEC no-source case.
4. The DFT lane-0 broadcast (`io.cu`) must give every CPU "thread" its own window/phase. → covered by DFT parity tests in Task 4.
5. Mixed processes: a CUDA venv must not load `_cpu` when a GPU is visible. → Task 4 unit test.

---

### Task 1: CPU prelude and block scheduler

**Files:** Modify `cuda/src/gpu_device.h`, `cuda/src/gpu_runtime.h`; Create `cuda/src/cpu_launch.h`; Test `tests/unit/test_cpu_prelude.py` (compiles a small `.cc` with `-DBEAMZ_PLATFORM_CPU -fopenmp` and runs it).

**Interfaces — Produces:**
- `BEAMZ_LAUNCH(grid, block, shared, stream, ...)`; on CPU `beamz::cpu::Launch<__VA_ARGS__>(dim3 grid, dim3 block, size_t shared)(args...)`; non-zero `shared` sets last error `cudaErrorNotSupported` and runs nothing.
- `BEAMZ_DYNAMIC_SHARED(type, name)` replacing `extern __shared__ type name[]` (CPU: `type* name = nullptr`).
- `BEAMZ_WARP_SIZE` (32 GPU, 1 CPU), `BEAMZ_BROADCAST_LEADER` (`threadIdx.x == 0` GPU, `true` CPU).
- CPU runtime: `cudaError_t` enum (`cudaSuccess`, `cudaErrorInvalidValue`, `cudaErrorMemoryAllocation`, `cudaErrorNotSupported`), thread-local last error, `cudaMemsetAsync` → `memset`, attribute/device queries return success with zero values.

- [ ] Step 1: failing test `test_cpu_scheduler_indices_and_atomics`: kernel writes `blockIdx*blockDim+threadIdx` for a 3-D grid of 37×5×3 blocks of (32,4,2) threads into an array and atomically adds 1.0f into one cell; expect every index exactly once and the sum equal to the thread count; a launch with `shared=16` reports `cudaErrorNotSupported`.
- [ ] Step 2: implement; Step 3: test passes; Step 4: commit.

### Task 2: Use the launch and shared-memory macros everywhere

**Files:** all `<<<` sites (68: adjoint.cu, io.cu, sharded.cu, update.cu, tensor.cuh), 6 `extern __shared__` sites, `io.cu` broadcast, `adjoint.cu` `WarpAdd`; extend `tests/unit/test_cuda_portability.py` (no `<<<` and no `extern __shared__` outside preludes).

- [ ] Step 1: extend portability test → FAIL; Step 2: mechanical rewrite; Step 3: CUDA and HIP builds succeed, portability test passes, CUDA parity subset unchanged; Step 4: commit.

### Task 3: CPU host paths

**Files:** `cuda/CMakeLists.txt` (`BEAMZ_GPU_PLATFORM=CPU` → module `_cpu`, `.cu` as CXX, OpenMP, `-O3 -march=native -ffp-contract=off`), `graph.cu` (direct enqueue on CPU), `program.cu` (`FuseCpmlCore` false on CPU; `LaunchTemporalYeeProgram` → per-phase fallback on CPU), `ffi_handler.cc` (stream-less bindings via a `WithoutStream<&Handler>` wrapper; mstep returns an error), `extension.cc` (`__gpu_platform__ = "cpu"`, no mstep registration).

- [ ] Step 1: CPU build compiles; Step 2: `import beamz._cpu` reports `cpu 30`; Step 3: commit.

### Task 4: Python selection on CPU

**Files:** `beamz/simulation/backend.py`, `tests/unit/test_execution_backend.py`.

**Interfaces — Produces:** `_gpu_platform()` returns `"cpu"` when JAX has no GPU devices; `_load_extension()` imports `beamz._cpu` for it; registration platform `"Host"`; status reports the CPU device; `auto` → JAX.

- [ ] Step 1: failing tests `test_cpu_only_process_loads_cpu_extension`, `test_cpu_extension_registers_host_platform`, `test_auto_keeps_jax_on_cpu`; Step 2: implement; Step 3: unit suite unchanged otherwise; Step 4: commit.

### Task 5: Parity and benchmark on CPU

- [ ] Step 1: run `tests/hardware/test_cuda_backends.py` and `test_cuda_autodiff.py` with `JAX_PLATFORMS=cpu`; triage every failure against the CUDA `main` baseline (identical pre-existing set expected; schedules needing shared memory must skip or fall back, not crash).
- [ ] Step 2: benchmark `jax` vs `cuda_streamed` on CPU at 64×96×128 and 128×256×512 (realistic workload); record GCUPS in `cuda/README.md` (estimate to beat: 0.12 JAX, 0.26 Futhark multicore).
- [ ] Step 3: commit; docs section "CPU (OpenMP)".

## Findings during execution (2026-11-06)

- All 68 launches use `BEAMZ_LAUNCH`; CUDA and HIP parity subsets are unchanged after the rewrite.
- Barrier-only schedules on CPU: `FuseCpmlCore` and `BEAMZ_CUDA_CPML_SPATIAL` are ignored natively, `LaunchTemporalYeeProgram` steps phase by phase, and `platform_cuda_flags` drops `CUDA_TEMPORAL_PAIR | CUDA_CPML_PAIR` at compile time, so two-step requests fall back to single steps instead of failing with error 801.
- **Sharding is not supported by the CPU build.** With `--xla_force_host_platform_device_count`, sharded runs through `beamz._cpu` were wrong (relative error ~1.3, identical for every axis). Correct at `-O0`, under UBSan, and when a global mutex serialised launches across XLA device threads; wrong at `-O1`..`-O3` with GCC 16.2 and clang 23. Thread-local launch indices were ruled out with a two-thread `dlopen` test. Buffer logs show concurrent calls from different device threads touching the same addresses, so the likely cause is cross-device buffer reuse in XLA's host runtime that only a fast native call exposes. The CPU build therefore does not register `beamz_cuda_sharded`; the unit harness keeps its contract build. Multi-device CPU is a test configuration only.
- Floating-point modes: XLA's CPU threads run with MXCSR `0x9fe3` (flush-to-zero and denormals-are-zero), fresh OpenMP workers with `0x1f80`. Blocks therefore rounded tiny values differently depending on which thread ran them, which made `test_temporal_pair_dft_windows_and_intervals[*-101]` differ bit-wise between identical runs (only `dft_vec_im`, found by serialising kernel families: `AccumulateDftPacked`). The launcher now copies the caller's MXCSR modes into every worker; `test_cpu_prelude.py` pins it.
- `test_streamed_cuda_realistic_sources_and_cpml_match_jax[1-binary]` fails on CPU by a rounding margin: one cell differs from XLA's CPU reference by 1.1e-6 of the field scale against a 1e-6 tolerance. Deterministic, not a race.
- Performance: the first CPU build ran at 0.10-0.15 GCUPS, with over half the time in `memcpy`: every simulated CUDA thread copied its by-value launch descriptor (~4 KB). `BEAMZ_ARG(T)`/`BEAMZ_GRID_ARG(T)` (58 parameters) bind struct parameters by reference on CPU and keep by-value parameters on GPUs; the launcher keeps one private argument copy per worker. Result: 0.31-0.35 GCUPS on the 7950X (JAX 0.11-0.14), 91% of samples inside the kernels.
