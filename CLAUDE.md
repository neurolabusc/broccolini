# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Broccolini is a GPU-accelerated spatial normalization (image registration) tool for NIfTI brain volumes, extracted from the [BROCCOLI](https://github.com/wanderine/BROCCOLI) project. It uses phase-based multi-scale registration with three GPU backends: Metal (macOS), OpenCL (cross-platform), WebGPU (cross-platform via wgpu-native).

## Build

```bash
# Auto-detects platform (Metal on macOS, OpenCL elsewhere)
make

# Explicit backend selection
make BACKEND=metal
make BACKEND=opencl
make BACKEND=webgpu

make clean
```

WebGPU requires `WGPU_DIR` pointing to the wgpu-native installation. OpenCL requires `BROCCOLI_DIR` for kernel files at runtime.

## Test

```bash
python3 compare_backends.py                    # auto-detect, build, and test all backends
python3 compare_backends.py --backends metal    # test specific backend(s)
python3 compare_backends.py --skip-build        # reuse existing build_<backend>/ binaries
python3 compare_backends.py -o benchmarks.md    # save Markdown report to file
```

Runs three tests per backend: EPI→T1 (linear), T1→MNI 1mm (linear), T1→MNI 1mm (nonlinear). Reports timing, peak memory, NCC, and HF variance. Reference outputs per backend are in `examples/{metal,opencl,webgpu}/`. Requires `numpy` and `nibabel`.

## Architecture

**Backend vtable pattern** — All GPU backends implement a common C interface defined in `registration.h`:

```c
struct broc_backend {
    broc_result (*register_volumes)(...);
    void (*destroy)(broc_backend *self);
    const char *name;
    void *priv;
};
```

Each backend provides a factory function (`broc_metal_create_backend()`, etc.) in its `*_backend.h/cpp|mm` file. `main.c` selects the backend at compile time via `#ifdef` and calls through the vtable.

**Key source layout:**
- `main.c` — CLI parsing and orchestration (FLIRT-compatible interface)
- `registration.h/c` — Shared API, filter loading, volume packing/unpacking utilities
- `nifti_io.h/c` — Minimal NIfTI-1/2 reader/writer (replaces niftilib), handles `.nii.gz`
- `metal/` — Metal backend; shaders in `metal/shaders/registration.metal`
- `opencl/` — OpenCL backend; bulk of code in `broccoli_lib.cpp` (~21K LOC from upstream BROCCOLI); kernels in `opencl/kernels/`
- `webgpu/` — WebGPU backend; WGSL shaders embedded as strings in `webgpu_registration.cpp`

**Registration pipeline:** Load NIfTI volumes → load 27 quadrature filter files → pack to BROCCOLI layout → GPU registration → unpack → save NIfTI output + optional affine matrix and displacement fields.

**Data structures** (all in `registration.h`): `broc_dims`, `broc_voxsize`, `broc_filters`, `broc_reg_params`, `broc_result`.

## Build Dependencies

- zlib (required, for gzip NIfTI)
- Metal + Accelerate frameworks (Metal backend, macOS only)
- OpenCL library (OpenCL backend)
- wgpu-native (WebGPU backend, set `WGPU_DIR`)
- Eigen headers bundled in `opencl/Eigen/` (OpenCL backend only)

## Notes

- Metal backend files use `.mm` (Objective-C++) extension
- The OpenCL backend is substantially larger than Metal/WebGPU because it carries the original BROCCOLI implementation
- Adding a new backend requires implementing the vtable in `registration.h` without touching `main.c` or shared utilities
- License: LGPL 2.1
