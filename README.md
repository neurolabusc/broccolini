# broccolini

GPU-accelerated spatial normalization (image registration) for NIfTI volumes.

Extracted from [BROCCOLI](https://github.com/wanderine/BROCCOLI) (Eklund et al., 2014) with a pluggable backend architecture supporting CUDA, Metal, OpenCL, and WebGPU.

## Features

- Phase-based multi-scale registration (not mutual information)
- 6 DOF (rigid) or 12 DOF (affine) linear registration
- Optional nonlinear registration with displacement fields
- FLIRT-compatible CLI interface
- CUDA backend (NVIDIA GPUs) — fastest on NVIDIA hardware
- Metal backend (macOS Apple Silicon) — fast, default on macOS
- OpenCL backend — cross-platform (macOS, Linux)
- WebGPU backend — cross-platform via wgpu-native (macOS, Linux, Windows)

## Build

Requires `clang`/`clang++` and `zlib`. The Makefile auto-detects the platform.

```bash
make                    # auto-detect backend (Metal on macOS)
make BACKEND=metal      # force Metal backend
make BACKEND=opencl     # force OpenCL backend
make BACKEND=webgpu     # WebGPU backend (requires wgpu-native)
make BACKEND=cuda       # CUDA backend (requires CUDA toolkit)
```

### CUDA backend (Linux, Windows)

Requires the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (provides `nvcc`). The Makefile auto-detects the GPU architecture. To override:

```bash
make BACKEND=cuda CUDA_ARCH=sm_89
```

### Metal backend (macOS)

Requires Xcode command-line tools (provides Metal framework).

### OpenCL backend

Requires OpenCL headers and runtime. On macOS these come with Xcode; on Linux install your GPU vendor's OpenCL ICD.

The OpenCL backend also requires filter files and the `BROCCOLI_DIR` environment variable:

```bash
export BROCCOLI_DIR=$(pwd)/opencl/
```

### WebGPU backend

Requires [wgpu-native](https://github.com/gfx-rs/wgpu-native/releases) headers and library. Set `WGPU_DIR` to the extracted directory (must contain `include/` and `lib/`):

```bash
make BACKEND=webgpu WGPU_DIR=/path/to/wgpu-native
```

On macOS, ensure `DYLD_LIBRARY_PATH` includes the wgpu-native lib directory at runtime, or install the library system-wide:

```bash
DYLD_LIBRARY_PATH=/path/to/wgpu-native/lib ./broccolini ...
```

The WebGPU backend uses WGSL compute shaders (embedded in the binary) and runs on Metal (macOS), Vulkan (Linux/Windows), or DX12 (Windows) via wgpu-native.

## Usage

```
broccolini [options] -in <input> -ref <reference> -out <output>
```

### Required arguments

| Flag | Description |
|------|-------------|
| `-in <file>` | Input volume (.nii or .nii.gz) |
| `-ref <file>` | Reference/template volume |
| `-out <file>` | Output aligned volume |

### Registration options

| Flag | Default | Description |
|------|---------|-------------|
| `-dof <6\|12>` | 12 | Degrees of freedom (6=rigid, 12=affine) |
| `-lineariter <N>` | 10 | Linear iterations per scale |
| `-nonlineariter <N>` | 0 | Nonlinear iterations (0=linear only) |
| `-coarsestscale <N>` | 4 | Coarsest scale (1, 2, 4, or 8) |
| `-zcut <mm>` | 0 | Z-axis crop in mm |
| `-interp <mode>` | trilinear | nearestneighbour, trilinear, or cubic |

### I/O options

| Flag | Description |
|------|-------------|
| `-mask <file>` | Brain mask for reference (default: ref > 0) |
| `-omat <file>` | Save 4×4 affine matrix (text) |
| `-ofield <prefix>` | Save displacement field as 3 NIfTI volumes |
| `-filters <dir>` | Directory with `.bin` filter files |

### Quick start with example data

The `examples/` folder includes skull-stripped brain images for testing:

- `EPI_brain.nii.gz` — Functional (EPI) volume (64x64x33, 3mm)
- `t1_brain.nii.gz` — Structural (T1) volume (128x181x175, 1mm)
- `MNI152_T1_1mm_brain.nii.gz` — MNI template at 1mm resolution (182x218x182)

Pre-computed reference outputs are in `examples/{cuda,metal,opencl,webgpu}/`.

**Run all backend tests:**

```bash
python3 compare_backends.py                    # auto-detect backends, build, test
python3 compare_backends.py --backends metal    # test specific backend(s)
python3 compare_backends.py --skip-build        # use existing binaries
python3 compare_backends.py -o benchmarks.md    # save report to file
```

This detects available backends, builds each, runs three registration tests (EPI→T1 linear, T1→MNI 1mm linear, T1→MNI 1mm nonlinear), and produces a Markdown table with timing, peak memory, NCC, and HF variance.

**Individual examples:**

```bash
# EPI to T1 registration (linear)
./broccolini \
  -in examples/EPI_brain.nii.gz \
  -ref examples/t1_brain.nii.gz \
  -out /tmp/epi_t1_aligned.nii.gz \
  -omat /tmp/epi_t1_params.txt -verbose

# T1 to MNI 1mm with linear registration
./broccolini \
  -in examples/t1_brain.nii.gz \
  -ref examples/MNI152_T1_1mm_brain.nii.gz \
  -out /tmp/t1_mni_1mm_linear.nii.gz \
  -coarsestscale 8 -zcut 30 -verbose

# T1 to MNI 1mm with nonlinear registration
./broccolini \
  -in examples/t1_brain.nii.gz \
  -ref examples/MNI152_T1_1mm_brain.nii.gz \
  -out /tmp/t1_mni_1mm_nonlinear.nii.gz \
  -nonlineariter 5 -coarsestscale 8 -zcut 30 -verbose
```

### Benchmarks

Reproduce with `python3 compare_backends.py`.

#### Apple M4 (macOS 15.4)

| Task | Metal | OpenCL | WebGPU |
|------|-------|--------|--------|
| EPI to T1 (linear) | 0.3s / 232 MB | 0.6s / 158 MB | 1.0s / 207 MB |
| T1 to MNI 1mm (linear) | 0.5s / 445 MB | 0.8s / 227 MB | 1.5s / 355 MB |
| T1 to MNI 1mm (nonlinear) | 3.4s / 798 MB | 4.2s / 298 MB | 8.1s / 411 MB |

| Task | Metal vs OpenCL | Metal vs WebGPU | OpenCL vs WebGPU |
|------|----------------|-----------------|------------------|
| EPI to T1 (linear) | 0.9999 | 1.0000 | 0.9999 |
| T1 to MNI 1mm (linear) | 0.9998 | 1.0000 | 0.9998 |
| T1 to MNI 1mm (nonlinear) | 0.9971 | 1.0000 | 0.9971 |

| Task | Metal HF | OpenCL HF | WebGPU HF |
|------|----------|-----------|-----------|
| EPI to T1 (linear) | 26.3 | 26.3 | 26.3 |
| T1 to MNI 1mm (linear) | 18.5 | 18.5 | 18.5 |
| T1 to MNI 1mm (nonlinear) | 18.3 | 18.3 | 18.3 |

#### NVIDIA GB10 (Linux aarch64, Driver 580.126.09)

| Task | CUDA | OpenCL | WebGPU |
|------|------|--------|--------|
| EPI to T1 (linear) | 0.7s / 256 MB | 1.0s / 214 MB | 1.1s / 452 MB |
| T1 to MNI 1mm (linear) | 0.9s / 368 MB | 1.4s / 293 MB | 1.6s / 600 MB |
| T1 to MNI 1mm (nonlinear) | 4.6s / 478 MB | 5.5s / 403 MB | 9.1s / 666 MB |

| Task | CUDA vs OpenCL | CUDA vs WebGPU | OpenCL vs WebGPU |
|------|----------------|----------------|------------------|
| EPI to T1 (linear) | 0.9999 | 1.0000 | 0.9999 |
| T1 to MNI 1mm (linear) | 0.9998 | 1.0000 | 0.9998 |
| T1 to MNI 1mm (nonlinear) | 0.9971 | 1.0000 | 0.9971 |

| Task | CUDA HF | OpenCL HF | WebGPU HF |
|------|---------|-----------|-----------|
| EPI to T1 (linear) | 26.3 | 26.3 | 26.3 |
| T1 to MNI 1mm (linear) | 18.5 | 18.5 | 18.5 |
| T1 to MNI 1mm (nonlinear) | 18.3 | 18.3 | 18.3 |

## Filter files

Registration requires pre-computed quadrature filter `.bin` files. The default search path is `../filters/` relative to the executable. Override with `-filters <dir>`.

Required files (27 total):
- `filter{1-3}_{real,imag}_linear_registration.bin` — 3 linear filters
- `filter{1-6}_{real,imag}_nonlinear_registration.bin` — 6 nonlinear filters
- `projection_tensor{1-6}.bin` — 6 projection tensors
- `filter_directions_{x,y,z}.bin` — filter direction vectors

## Directory structure

```
main.c                 — CLI parsing, NIfTI I/O, orchestration
registration.h/c       — Backend-agnostic C API, shared utilities
nifti_io.h/c           — Minimal NIfTI-1/2 reader/writer
Makefile               — Build system
compare_backends.py    — Build, test, and benchmark all backends
filters/               — Quadrature filter .bin files (27 files)
examples/              — Test brain images (EPI, T1, MNI) and reference outputs
cuda/                  — CUDA backend (NVIDIA GPUs)
  cuda_backend.h/cpp       — C vtable adapter
  cuda_registration.h/cu   — CUDA GPU implementation
metal/                 — Metal backend (macOS)
  metal_backend.h/mm       — C vtable adapter
  metal_registration.h/mm  — Metal GPU implementation
  shaders/registration.metal — Compute shaders
opencl/                — OpenCL backend (cross-platform)
  opencl_backend.h/cpp     — C vtable adapter
  opencl_registration.h/cpp — OpenCL GPU implementation
  Eigen/                   — Bundled Eigen (header-only)
  kernels/                 — OpenCL kernel source files
webgpu/                — WebGPU backend (cross-platform via wgpu-native)
  webgpu_backend.h/cpp     — C vtable adapter
  webgpu_registration.h/cpp — WebGPU/WGSL GPU implementation
```

## Architecture

The executable uses a backend vtable pattern: `main.c` is pure C and calls a backend-agnostic `register_volumes()` function pointer. Each backend provides a factory function (e.g. `broc_cuda_create_backend()`) that returns the vtable. Adding a new backend requires implementing the vtable, adding `#ifdef HAVE_<BACKEND>` to `registration.h`, and adding build rules to the Makefile.

## License

LGPL 2.1 — see the [BROCCOLI](https://github.com/wanderine/BROCCOLI) parent project.
