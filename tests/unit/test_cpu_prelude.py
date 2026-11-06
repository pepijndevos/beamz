"""The CPU platform prelude runs CUDA-style kernels through an OpenMP scheduler."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

SOURCE_DIR = Path(__file__).resolve().parents[2] / "cuda" / "src"
COMPILER = shutil.which("g++") or shutil.which("clang++")

PROGRAM = r"""
#include <cstdio>
#include <vector>
#include "gpu_runtime.h"
#include "gpu_device.h"
#include <xmmintrin.h>

__global__ void Indices(int* out, float* total, int nx, int ny, int nz) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= nx || y >= ny || z >= nz) return;
  atomicAdd(&out[(z * ny + y) * nx + x], 1);
  atomicAdd(total, 1.0f);
}

__global__ void FloatModes(unsigned* out) {
  out[blockIdx.x] = _mm_getcsr() & 0xffc0u;
}

__global__ void NeedsShared(float* out) {
  BEAMZ_DYNAMIC_SHARED(float, scratch);
  (void)scratch;
  out[0] = 1.0f;
}

int main() {
  const dim3 threads(32, 4, 2);
  const dim3 blocks(37, 5, 3);
  const int nx = 37 * 32 - 7, ny = 5 * 4 - 1, nz = 3 * 2;
  std::vector<int> out(static_cast<size_t>(nx) * ny * nz, 0);
  float total = 0.0f;
  cudaStream_t stream = nullptr;
  BEAMZ_LAUNCH(blocks, threads, 0, stream, Indices)(out.data(), &total, nx, ny, nz);
  if (cudaGetLastError() != cudaSuccess) return 2;
  for (int v : out) if (v != 1) return 3;
  if (total != static_cast<float>(out.size())) return 4;
  float flag = 0.0f;
  BEAMZ_LAUNCH(dim3(1), dim3(32), 16, stream, NeedsShared)(&flag);
  if (cudaGetLastError() != cudaErrorNotSupported) return 5;
  if (flag != 0.0f) return 6;
  if (cudaGetLastError() != cudaSuccess) return 7;  // reading clears the error
  // XLA's threads flush denormals; workers must round exactly like the caller.
  const unsigned modes = (_mm_getcsr() & 0xffc0u) | 0x8040u;  // FTZ | DAZ
  _mm_setcsr(modes);
  std::vector<unsigned> seen(4096, 0);
  BEAMZ_LAUNCH(dim3(4096), dim3(1), 0, stream, FloatModes)(seen.data());
  for (unsigned value : seen) if (value != modes) return 8;
  std::printf("ok %zu\n", out.size());
  return 0;
}
"""


@pytest.mark.skipif(COMPILER is None, reason="no C++ compiler")
def test_cpu_scheduler_indices_and_atomics(tmp_path):
    source = tmp_path / "scheduler.cc"
    source.write_text(PROGRAM)
    binary = tmp_path / "scheduler"
    subprocess.run(
        [
            COMPILER,
            "-std=c++17",
            "-O2",
            "-fopenmp",
            "-DBEAMZ_PLATFORM_CPU",
            f"-I{SOURCE_DIR}",
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run(
        [str(binary)], env={"OMP_NUM_THREADS": "4"}, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.startswith("ok ")
