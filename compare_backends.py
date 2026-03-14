#!/usr/bin/env python3
"""compare_backends.py — Build, test, and compare broccolini across GPU backends.

Detects which backends are available on the current platform, builds each,
runs the standard registration tests, and produces a Markdown table with
timing, peak memory, accuracy (NCC), and HF variance — matching the
Benchmarks section of README.md.

Usage:
    python3 compare_backends.py                # auto-detect backends, run all
    python3 compare_backends.py --backends metal opencl
    python3 compare_backends.py --skip-build   # use existing binaries
    python3 compare_backends.py --output bench.md
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import time

try:
    import numpy as np
except ImportError:
    print("Error: numpy required. Install with: pip install numpy")
    sys.exit(1)

try:
    import nibabel as nib
except ImportError:
    print("Error: nibabel required. Install with: pip install nibabel")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

EXAMPLES_DIR = os.path.join(SCRIPT_DIR, "examples")
T1 = os.path.join(EXAMPLES_DIR, "t1_brain.nii.gz")
EPI = os.path.join(EXAMPLES_DIR, "EPI_brain.nii.gz")
MNI_1MM = os.path.join(EXAMPLES_DIR, "MNI152_T1_1mm_brain.nii.gz")

# Filter directory: try ../filters/ relative to repo, then ../BROCCOLI/filters/
FILTER_DIR = None
for candidate in [
    os.path.join(SCRIPT_DIR, "filters"),
    os.path.join(SCRIPT_DIR, "..", "filters"),
    os.path.join(SCRIPT_DIR, "..", "BROCCOLI", "filters"),
]:
    if os.path.isdir(candidate) and os.path.exists(
            os.path.join(candidate, "filter1_real_linear_registration.bin")):
        FILTER_DIR = os.path.abspath(candidate)
        break

# Each test: (name, cli_args as list)
TESTS = [
    (
        "EPI to T1 (linear)",
        ["-in", EPI, "-ref", T1, "-omat", "{outdir}/epi_t1_params.txt"],
        "epi_t1_aligned.nii.gz",
    ),
    (
        "T1 to MNI 1mm (linear)",
        [
            "-in", T1, "-ref", MNI_1MM,
            "-coarsestscale", "8", "-zcut", "30",
            "-omat", "{outdir}/t1_mni_1mm_linear_params.txt",
        ],
        "t1_mni_1mm_aligned_linear.nii.gz",
    ),
    (
        "T1 to MNI 1mm (nonlinear)",
        [
            "-in", T1, "-ref", MNI_1MM,
            "-nonlineariter", "5", "-coarsestscale", "8", "-zcut", "30",
            "-omat", "{outdir}/t1_mni_1mm_params.txt",
            "-ofield", "{outdir}/t1_mni_1mm_disp",
        ],
        "t1_mni_1mm_aligned_nonlinear.nii.gz",
    ),
]


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def detect_backends():
    """Return list of backends available on this platform."""
    available = []
    system = platform.system()

    if system == "Darwin":
        # Metal is always available on macOS with Xcode CLI tools
        available.append("metal")

    # OpenCL: check for headers/library
    if system == "Darwin":
        # macOS bundles OpenCL in its SDK
        available.append("opencl")
    elif system == "Linux":
        # Check for libOpenCL on various architectures
        import glob as _glob
        if (shutil.which("clinfo") or
                os.path.exists("/usr/lib/x86_64-linux-gnu/libOpenCL.so") or
                os.path.exists("/usr/lib/x86_64-linux-gnu/libOpenCL.so.1") or
                os.path.exists("/usr/lib/aarch64-linux-gnu/libOpenCL.so") or
                os.path.exists("/usr/lib/aarch64-linux-gnu/libOpenCL.so.1") or
                os.path.exists("/usr/lib64/libOpenCL.so") or
                os.path.exists("/usr/lib/libOpenCL.so") or
                _glob.glob("/usr/lib/*/libOpenCL.so*")):
            available.append("opencl")

    # WebGPU: need wgpu-native
    wgpu_dir = os.environ.get("WGPU_DIR",
                              os.path.join(SCRIPT_DIR, "webgpu", "wgpu-native"))
    if os.path.isdir(os.path.join(wgpu_dir, "include")):
        available.append("webgpu")

    # CUDA: check for nvcc compiler
    if shutil.which("nvcc"):
        available.append("cuda")

    return available


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_backend(backend, src_dir):
    """Build broccolini for a specific backend. Returns path to binary."""
    build_dir = os.path.join(src_dir, f"build_{backend}")
    os.makedirs(build_dir, exist_ok=True)

    make_args = ["make", f"BACKEND={backend}", f"BUILDDIR={build_dir}",
                 f"TARGET={build_dir}/broccolini", "-j4"]

    if backend == "webgpu":
        wgpu_dir = os.environ.get("WGPU_DIR",
                                  os.path.join(src_dir, "webgpu", "wgpu-native"))
        make_args.append(f"WGPU_DIR={wgpu_dir}")

    print(f"  Building {backend}...", end=" ", flush=True)
    result = subprocess.run(make_args, cwd=src_dir,
                            capture_output=True, text=True)
    if result.returncode != 0:
        print("FAILED")
        print(result.stderr)
        return None

    binary = os.path.join(build_dir, "broccolini")
    if not os.path.isfile(binary):
        print("FAILED (binary not found)")
        return None

    print("OK")
    return binary


# ---------------------------------------------------------------------------
# Run a single test
# ---------------------------------------------------------------------------

def run_test(binary, backend, test_name, cli_args, out_filename, out_dir):
    """Run one registration test. Returns dict with timing, peak_mb, out_path."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_filename)

    # Substitute {outdir} in cli args
    args = [a.replace("{outdir}", out_dir) for a in cli_args]
    cmd = [binary, *args, "-out", out_path, "-verbose"]
    if FILTER_DIR:
        cmd.extend(["-filters", FILTER_DIR])

    env = os.environ.copy()
    if backend == "opencl":
        env["BROCCOLI_DIR"] = os.path.join(SCRIPT_DIR, "opencl/")
    if backend == "webgpu":
        wgpu_dir = os.environ.get("WGPU_DIR",
                                  os.path.join(SCRIPT_DIR, "webgpu", "wgpu-native"))
        ld_key = ("DYLD_LIBRARY_PATH" if platform.system() == "Darwin"
                  else "LD_LIBRARY_PATH")
        lib_dir = os.path.join(wgpu_dir, "lib")
        env[ld_key] = lib_dir + ":" + env.get(ld_key, "")

    # Measure with /usr/bin/time on macOS or Linux
    system = platform.system()
    if system == "Darwin":
        # GNU time not available; use resource module via wrapper
        peak_mb, elapsed, returncode, stderr = _run_with_resource(cmd, env)
    else:
        # Linux: use /usr/bin/time -v
        peak_mb, elapsed, returncode, stderr = _run_with_gnu_time(cmd, env)

    if returncode != 0:
        print(f"    {test_name}: FAILED (exit {returncode})")
        if stderr:
            for line in stderr.strip().split("\n")[-5:]:
                print(f"      {line}")
        return None

    return {
        "elapsed": elapsed,
        "peak_mb": peak_mb,
        "out_path": out_path,
    }


def _run_with_resource(cmd, env):
    """Run command and measure peak RSS via /usr/bin/time -l on macOS."""
    time_cmd = ["/usr/bin/time", "-l"] + cmd
    start = time.monotonic()
    proc = subprocess.run(time_cmd, env=env, capture_output=True, text=True)
    elapsed = time.monotonic() - start
    peak_mb = 0.0
    for line in proc.stderr.split("\n"):
        # macOS /usr/bin/time -l: "  NNN  peak memory footprint" (bytes)
        # or older: "  NNN  maximum resident set size"
        if "peak memory footprint" in line or "maximum resident set size" in line:
            try:
                peak_mb = int(line.strip().split()[0]) / (1024 * 1024)
            except (ValueError, IndexError):
                pass
            break
    return peak_mb, elapsed, proc.returncode, proc.stderr


def _run_with_gnu_time(cmd, env):
    """Run command with /usr/bin/time -v on Linux."""
    time_cmd = ["/usr/bin/time", "-v"] + cmd
    start = time.monotonic()
    proc = subprocess.run(time_cmd, env=env, capture_output=True, text=True)
    elapsed = time.monotonic() - start
    peak_mb = 0.0
    for line in proc.stderr.split("\n"):
        if "Maximum resident set size" in line:
            # Value in KB
            try:
                peak_mb = int(line.strip().split()[-1]) / 1024
            except ValueError:
                pass
            break
    return peak_mb, elapsed, proc.returncode, proc.stderr


# ---------------------------------------------------------------------------
# Accuracy metrics
# ---------------------------------------------------------------------------

def load_nifti(path):
    """Load NIfTI file as float32 array, or None if missing."""
    if not os.path.exists(path):
        return None
    return nib.load(path).get_fdata().astype(np.float32)


def ncc(a, b):
    """Normalized cross-correlation."""
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    denom = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    if denom == 0:
        return 0.0
    return float(np.sum(a * b) / denom)


def hf_variance(data):
    """High-frequency variance: mean |v[i]-v[i-1]| over consecutive nonzero voxels."""
    flat = data.flatten().astype(np.float64)
    diffs = np.abs(flat[1:] - flat[:-1])
    mask = (flat[:-1] != 0.0) & (flat[1:] != 0.0)
    valid = diffs[mask]
    if len(valid) < 2:
        return 0.0
    return float(valid.mean())


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(results, backends):
    """Generate Markdown benchmark tables from results.

    results: dict[backend][test_name] -> {elapsed, peak_mb, out_path}
    """
    lines = []
    test_names = [t[0] for t in TESTS]

    # --- Timing / memory table ---
    lines.append("### Performance")
    lines.append("")
    header = "| Task |" + " | ".join(f" {b.capitalize()} " for b in backends) + " |"
    sep = "|------|" + " | ".join("-------" for _ in backends) + " |"
    lines.append(header)
    lines.append(sep)
    for tname in test_names:
        cells = []
        for b in backends:
            r = results.get(b, {}).get(tname)
            if r:
                cells.append(f"{r['elapsed']:.1f}s / {r['peak_mb']:.0f} MB")
            else:
                cells.append("—")
        lines.append(f"| {tname} | " + " | ".join(cells) + " |")
    lines.append("")

    # --- NCC table (cross-backend pairs) ---
    if len(backends) >= 2:
        lines.append("### Cross-backend NCC")
        lines.append("")
        pairs = []
        for i in range(len(backends)):
            for j in range(i + 1, len(backends)):
                pairs.append((backends[i], backends[j]))

        header = "| Task |" + " | ".join(
            f" {a.capitalize()} vs {b.capitalize()} " for a, b in pairs) + " |"
        sep = "|------|" + " | ".join("-------" for _ in pairs) + " |"
        lines.append(header)
        lines.append(sep)

        for tname in test_names:
            cells = []
            for a, b in pairs:
                ra = results.get(a, {}).get(tname)
                rb = results.get(b, {}).get(tname)
                if ra and rb:
                    va = load_nifti(ra["out_path"])
                    vb = load_nifti(rb["out_path"])
                    if va is not None and vb is not None and va.shape == vb.shape:
                        cells.append(f"{ncc(va, vb):.4f}")
                    else:
                        cells.append("shape mismatch")
                else:
                    cells.append("—")
            lines.append(f"| {tname} | " + " | ".join(cells) + " |")
        lines.append("")

    # --- HF variance table ---
    lines.append("### HF variance")
    lines.append("")
    header = "| Task |" + " | ".join(f" {b.capitalize()} HF " for b in backends) + " |"
    sep = "|------|" + " | ".join("-------" for _ in backends) + " |"
    lines.append(header)
    lines.append(sep)
    for tname in test_names:
        cells = []
        for b in backends:
            r = results.get(b, {}).get(tname)
            if r:
                vol = load_nifti(r["out_path"])
                if vol is not None:
                    cells.append(f"{hf_variance(vol):.1f}")
                else:
                    cells.append("—")
            else:
                cells.append("—")
        lines.append(f"| {tname} | " + " | ".join(cells) + " |")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build, test, and compare broccolini across GPU backends")
    parser.add_argument("--backends", nargs="+",
                        help="Backends to test (default: auto-detect)")
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip building; use existing build_<backend>/broccolini")
    parser.add_argument("--output", "-o", default=None,
                        help="Write Markdown report to file (default: stdout)")
    args = parser.parse_args()

    backends = args.backends or detect_backends()
    if not backends:
        print("No backends detected. Use --backends to specify manually.")
        sys.exit(1)

    # Verify test data exists
    for path in [T1, EPI, MNI_1MM]:
        if not os.path.exists(path):
            print(f"Missing test data: {path}")
            sys.exit(1)

    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Backends: {', '.join(backends)}")
    print()

    # Build each backend
    binaries = {}
    for backend in backends:
        if args.skip_build:
            binary = os.path.join(SCRIPT_DIR, f"build_{backend}", "broccolini")
            if not os.path.isfile(binary):
                print(f"  {backend}: binary not found at {binary}, skipping")
                continue
            binaries[backend] = binary
        else:
            binary = build_backend(backend, SCRIPT_DIR)
            if binary:
                binaries[backend] = binary

    if not binaries:
        print("No backends built successfully.")
        sys.exit(1)

    # Run tests
    results = {}  # results[backend][test_name] = {...}
    print()
    for backend, binary in binaries.items():
        print(f"=== Testing {backend} ===")
        results[backend] = {}
        out_dir = os.path.join(EXAMPLES_DIR, backend)

        for test_name, cli_args, out_filename in TESTS:
            print(f"  {test_name}...", end=" ", flush=True)
            r = run_test(binary, backend, test_name, cli_args, out_filename, out_dir)
            if r:
                results[backend][test_name] = r
                print(f"{r['elapsed']:.1f}s / {r['peak_mb']:.0f} MB")
            else:
                print("FAILED")
        print()

    # Generate report
    active_backends = [b for b in backends if b in results and results[b]]
    report = generate_report(results, active_backends)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report written to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
