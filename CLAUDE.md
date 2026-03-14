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

- `broccoli_lib.cpp` is ~21K LOC legacy upstream — avoid modifying
- Filter `.bin` files (27 total in `filters/`) required at runtime
- BROCCOLI layout is `flipud + transpose(2,0,1)` of NIfTI order
- 12-param format: `[tx,ty,tz, R-I]` where R is 3×3 rotation/scale minus identity
- Backend adapters (`*_backend.cpp`) and type structs (`QuadratureFilters`, `VolumeDims`, etc.) are duplicated across all backends — ripe for unification
- Metal backend: `encodeFill`/`encodeMultiply`/etc. create small temporary `MTLBuffer` objects per call; these accumulate within each iteration's `@autoreleasepool`. Profile on macOS to assess whether tighter scoping or buffer reuse reduces memory pressure
