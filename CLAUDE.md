# CLAUDE.md

## Project

Broccolini: GPU-accelerated phase-based image registration for NIfTI brain volumes. Extracted from [BROCCOLI](https://github.com/wanderine/BROCCOLI). Four backends: CUDA, Metal, OpenCL, WebGPU.

## Build & Test

```bash
make BACKEND=cuda       # or metal, opencl, webgpu
python3 compare_backends.py   # build, test, benchmark all detected backends
python3 compare_backends.py --backends cuda --skip-build
```

OpenCL needs `BROCCOLI_DIR=$(pwd)/opencl/` at runtime. WebGPU needs `WGPU_DIR`. Tests require `numpy` and `nibabel`.

## Architecture

Backend vtable in `registration.h` — each backend implements `register_volumes()` + `destroy()` via a factory function. `main.c` selects at compile time via `#ifdef`. Priority: Metal → CUDA → WebGPU → OpenCL.

Pipeline: NIfTI → pack to BROCCOLI layout → GPU registration (center-of-mass → multi-scale linear → optional nonlinear) → unpack → NIfTI.

## Gotchas

- Filter `.bin` files (27 total in `filters/`) required at runtime
- BROCCOLI layout is `flipud + transpose(2,0,1)` of NIfTI order
- 12-param format: `[tx,ty,tz, R-I]` where R is 3×3 rotation/scale minus identity
- All four backends follow the same structure: `*_backend.cpp` (C vtable adapter) + `*_registration.{cpp,mm,cu,h}` (GPU implementation in a `*_reg` namespace). Type structs (`QuadratureFilters`, `VolumeDims`, etc.) are duplicated per backend — keep them in sync
- OpenCL backend loads kernel source from `$BROCCOLI_DIR/kernels/` at runtime; kernel variant selection depends on device local memory size
- OpenCL backend (inherited from legacy BROCCOLI code) has known robustness gaps:
  - `clCreateKernel` errors are silently overwritten — a failed kernel creation leads to NULL pointer crash at dispatch time
  - `init()` partial failure leaks OpenCL context/queue/programs (no rollback)
  - `cleanup()` is never called (singleton has no destructor) — OpenCL resources leak at exit
  - No NULL checks on `clCreateBuffer`/`clCreateImage3D` — GPU OOM causes crashes
  - No `clEnqueueNDRangeKernel` error checking — dispatch failures silently ignored
- Metal backend uses `setBytes` for small constant data (<4KB) instead of creating temporary `MTLBuffer` objects — do not regress to `newBufferWithBytes` for scalar/struct shader parameters
