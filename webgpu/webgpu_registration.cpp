// webgpu_registration.cpp — WebGPU compute backend for BROCCOLI image registration
// Uses wgpu-native C API (webgpu.h + wgpu.h)

#include "webgpu_registration.h"

#include <webgpu/webgpu.h>
#include <wgpu.h>

#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cassert>
#include <array>

// ============================================================
//  WGSL Kernel Sources
// ============================================================

static const char* HELPERS = R"wgsl(
fn idx3(x: i32, y: i32, z: i32, W: i32, H: i32) -> i32 {
    return x + y * W + z * W * H;
}

struct Dims {
    W: i32,
    H: i32,
    D: i32,
}
)wgsl";

static const char* KERNEL_FILL_FLOAT = R"wgsl(
struct Params { value: f32 }
@group(0) @binding(0) var<storage, read_write> buf: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&buf)) { return; }
    buf[i] = params.value;
}
)wgsl";

static const char* KERNEL_FILL_VEC2 = R"wgsl(
@group(0) @binding(0) var<storage, read_write> buf: array<vec2<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&buf)) { return; }
    buf[i] = vec2<f32>(0.0, 0.0);
}
)wgsl";

static const char* KERNEL_ADD_VOLUMES = R"wgsl(
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&A)) { return; }
    A[i] = A[i] + B[i];
}
)wgsl";

static const char* KERNEL_MULTIPLY_VOLUME = R"wgsl(
struct Params { factor: f32 }
@group(0) @binding(0) var<storage, read_write> vol: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&vol)) { return; }
    vol[i] = vol[i] * params.factor;
}
)wgsl";

static const char* KERNEL_MULTIPLY_VOLUMES = R"wgsl(
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&A)) { return; }
    A[i] = A[i] * B[i];
}
)wgsl";

static const char* KERNEL_COLUMN_MAXS = R"wgsl(
fn idx3(x: i32, y: i32, z: i32, W: i32, H: i32) -> i32 {
    return x + y * W + z * W * H;
}

struct Dims {
    W: i32,
    H: i32,
    D: i32,
}

@group(0) @binding(0) var<storage, read_write> columnMaxs: array<f32>;
@group(0) @binding(1) var<storage, read> volume: array<f32>;
@group(0) @binding(2) var<uniform> dims: Dims;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let y = i32(gid.x);
    let z = i32(gid.y);
    if (y >= dims.H || z >= dims.D) { return; }
    var mx = volume[idx3(0, y, z, dims.W, dims.H)];
    for (var x = 1; x < dims.W; x++) {
        mx = max(mx, volume[idx3(x, y, z, dims.W, dims.H)]);
    }
    columnMaxs[y + z * dims.H] = mx;
}
)wgsl";

static const char* KERNEL_ROW_MAXS = R"wgsl(
struct Dims { W: i32, H: i32, D: i32 }
@group(0) @binding(0) var<storage, read_write> rowMaxs: array<f32>;
@group(0) @binding(1) var<storage, read> columnMaxs: array<f32>;
@group(0) @binding(2) var<uniform> dims: Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let z = i32(gid.x);
    if (z >= dims.D) { return; }
    var mx = columnMaxs[z * dims.H];
    for (var y = 1; y < dims.H; y++) {
        mx = max(mx, columnMaxs[y + z * dims.H]);
    }
    rowMaxs[z] = mx;
}
)wgsl";

static const char* KERNEL_CONV3D_FULL = R"wgsl(
fn idx3(x: i32, y: i32, z: i32, W: i32, H: i32) -> i32 {
    return x + y * W + z * W * H;
}

struct Dims {
    W: i32,
    H: i32,
    D: i32,
}

@group(0) @binding(0) var<storage, read_write> response1: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> response2: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> response3: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> volume: array<f32>;
@group(0) @binding(4) var<storage, read> f1r: array<f32>;
@group(0) @binding(5) var<storage, read> f1i: array<f32>;
@group(0) @binding(6) var<storage, read> f2r: array<f32>;
@group(0) @binding(7) var<storage, read> f2i: array<f32>;
@group(0) @binding(8) var<storage, read> f3r: array<f32>;
@group(0) @binding(9) var<storage, read> f3i: array<f32>;
@group(0) @binding(10) var<uniform> dims: Dims;

fn safe_vol(x: i32, y: i32, z: i32) -> f32 {
    if (x < 0 || x >= dims.W || y < 0 || y >= dims.H || z < 0 || z >= dims.D) {
        return 0.0;
    }
    return volume[idx3(x, y, z, dims.W, dims.H)];
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);
    let z = i32(gid.z);
    if (x >= dims.W || y >= dims.H || z >= dims.D) { return; }

    var s1r: f32 = 0.0; var s1i: f32 = 0.0;
    var s2r: f32 = 0.0; var s2i: f32 = 0.0;
    var s3r: f32 = 0.0; var s3i: f32 = 0.0;

    for (var fz = 0; fz < 7; fz++) {
        for (var fy = 0; fy < 7; fy++) {
            for (var fx = 0; fx < 7; fx++) {
                let p = safe_vol(x + 3 - fx, y + 3 - fy, z + 3 - fz);
                let fi = fx + fy * 7 + fz * 49;
                s1r += f1r[fi] * p;
                s1i += f1i[fi] * p;
                s2r += f2r[fi] * p;
                s2i += f2i[fi] * p;
                s3r += f3r[fi] * p;
                s3i += f3i[fi] * p;
            }
        }
    }

    let outIdx = idx3(x, y, z, dims.W, dims.H);
    response1[outIdx] = vec2<f32>(s1r, s1i);
    response2[outIdx] = vec2<f32>(s2r, s2i);
    response3[outIdx] = vec2<f32>(s3r, s3i);
}
)wgsl";

static const char* KERNEL_SEPARABLE_CONV_ROWS = R"wgsl(
fn idx3(x: i32, y: i32, z: i32, W: i32, H: i32) -> i32 {
    return x + y * W + z * W * H;
}

struct Dims {
    W: i32,
    H: i32,
    D: i32,
}

@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> filterY: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);
    let z = i32(gid.z);
    if (x >= dims.W || y >= dims.H || z >= dims.D) { return; }

    var sum: f32 = 0.0;
    for (var fy = -4; fy <= 4; fy++) {
        let yy = y + fy;
        var val: f32 = 0.0;
        if (yy >= 0 && yy < dims.H) {
            val = input[idx3(x, yy, z, dims.W, dims.H)];
        }
        sum += val * filterY[4 - fy];
    }
    output[idx3(x, y, z, dims.W, dims.H)] = sum;
}
)wgsl";

static const char* KERNEL_SEPARABLE_CONV_COLS = R"wgsl(
fn idx3(x: i32, y: i32, z: i32, W: i32, H: i32) -> i32 {
    return x + y * W + z * W * H;
}

struct Dims {
    W: i32,
    H: i32,
    D: i32,
}

@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> filterX: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);
    let z = i32(gid.z);
    if (x >= dims.W || y >= dims.H || z >= dims.D) { return; }

    var sum: f32 = 0.0;
    for (var fx = -4; fx <= 4; fx++) {
        let xx = x + fx;
        var val: f32 = 0.0;
        if (xx >= 0 && xx < dims.W) {
            val = input[idx3(xx, y, z, dims.W, dims.H)];
        }
        sum += val * filterX[4 - fx];
    }
    output[idx3(x, y, z, dims.W, dims.H)] = sum;
}
)wgsl";

static const char* KERNEL_SEPARABLE_CONV_RODS = R"wgsl(
fn idx3(x: i32, y: i32, z: i32, W: i32, H: i32) -> i32 {
    return x + y * W + z * W * H;
}

struct Dims {
    W: i32,
    H: i32,
    D: i32,
}

@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> filterZ: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);
    let z = i32(gid.z);
    if (x >= dims.W || y >= dims.H || z >= dims.D) { return; }

    var sum: f32 = 0.0;
    for (var fz = -4; fz <= 4; fz++) {
        let zz = z + fz;
        var val: f32 = 0.0;
        if (zz >= 0 && zz < dims.D) {
            val = input[idx3(x, y, zz, dims.W, dims.H)];
        }
        sum += val * filterZ[4 - fz];
    }
    output[idx3(x, y, z, dims.W, dims.H)] = sum;
}
)wgsl";

static const char* KERNEL_PHASE_DIFF_CERT = R"wgsl(
fn idx3(x: i32, y: i32, z: i32, W: i32, H: i32) -> i32 {
    return x + y * W + z * W * H;
}

struct Dims {
    W: i32,
    H: i32,
    D: i32,
}

@group(0) @binding(0) var<storage, read_write> phaseDiff: array<f32>;
@group(0) @binding(1) var<storage, read_write> certainties: array<f32>;
@group(0) @binding(2) var<storage, read> q1: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> q2: array<vec2<f32>>;
@group(0) @binding(4) var<uniform> dims: Dims;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x); let y = i32(gid.y); let z = i32(gid.z);
    if (x >= dims.W || y >= dims.H || z >= dims.D) { return; }

    let i = idx3(x, y, z, dims.W, dims.H);
    let a = q1[i];
    let c = q2[i];

    let cpReal = a.x * c.x + a.y * c.y;
    let cpImag = a.y * c.x - a.x * c.y;
    var phase = 0.0;
    if (abs(cpReal) > 1e-30 || abs(cpImag) > 1e-30) {
        phase = atan2(cpImag, cpReal);
    }

    let prodReal = a.x * c.x - a.y * c.y;
    let prodImag = a.y * c.x + a.x * c.y;
    let cosHalf = cos(phase * 0.5);

    phaseDiff[i] = phase;
    let mag = prodReal * prodReal + prodImag * prodImag;
    var cert = 0.0;
    if (mag > 0.0) { cert = sqrt(mag) * cosHalf * cosHalf; }
    certainties[i] = cert;
}
)wgsl";

static const char* KERNEL_PHASE_GRAD_X = R"wgsl(
fn idx3(x: i32, y: i32, z: i32, W: i32, H: i32) -> i32 {
    return x + y * W + z * W * H;
}

struct Dims {
    W: i32,
    H: i32,
    D: i32,
}

@group(0) @binding(0) var<storage, read_write> gradients: array<f32>;
@group(0) @binding(1) var<storage, read> q1: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> q2: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x); let y = i32(gid.y); let z = i32(gid.z);
    if (x < 1 || x >= dims.W - 1 || y >= dims.H || z >= dims.D) { return; }

    let i0 = idx3(x, y, z, dims.W, dims.H);
    let im = idx3(x-1, y, z, dims.W, dims.H);
    let ip = idx3(x+1, y, z, dims.W, dims.H);

    var tr: f32 = 0.0; var ti: f32 = 0.0;

    // q1[ip] * conj(q1[i0])
    var a = q1[ip]; var c = q1[i0];
    tr += a.x*c.x + a.y*c.y; ti += a.y*c.x - a.x*c.y;
    a = q1[i0]; c = q1[im];
    tr += a.x*c.x + a.y*c.y; ti += a.y*c.x - a.x*c.y;
    a = q2[ip]; c = q2[i0];
    tr += a.x*c.x + a.y*c.y; ti += a.y*c.x - a.x*c.y;
    a = q2[i0]; c = q2[im];
    tr += a.x*c.x + a.y*c.y; ti += a.y*c.x - a.x*c.y;

    var g = 0.0;
    if (abs(tr) > 1e-30 || abs(ti) > 1e-30) { g = atan2(ti, tr); }
    gradients[i0] = g;
}
)wgsl";

static const char* KERNEL_PHASE_GRAD_Y = R"wgsl(
fn idx3(x: i32, y: i32, z: i32, W: i32, H: i32) -> i32 {
    return x + y * W + z * W * H;
}

struct Dims {
    W: i32,
    H: i32,
    D: i32,
}

@group(0) @binding(0) var<storage, read_write> gradients: array<f32>;
@group(0) @binding(1) var<storage, read> q1: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> q2: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x); let y = i32(gid.y); let z = i32(gid.z);
    if (x >= dims.W || y < 1 || y >= dims.H - 1 || z >= dims.D) { return; }

    let i0 = idx3(x, y, z, dims.W, dims.H);
    let im = idx3(x, y-1, z, dims.W, dims.H);
    let ip = idx3(x, y+1, z, dims.W, dims.H);

    var tr: f32 = 0.0; var ti: f32 = 0.0;

    var a = q1[ip]; var c = q1[i0];
    tr += a.x*c.x + a.y*c.y; ti += a.y*c.x - a.x*c.y;
    a = q1[i0]; c = q1[im];
    tr += a.x*c.x + a.y*c.y; ti += a.y*c.x - a.x*c.y;
    a = q2[ip]; c = q2[i0];
    tr += a.x*c.x + a.y*c.y; ti += a.y*c.x - a.x*c.y;
    a = q2[i0]; c = q2[im];
    tr += a.x*c.x + a.y*c.y; ti += a.y*c.x - a.x*c.y;

    var g = 0.0;
    if (abs(tr) > 1e-30 || abs(ti) > 1e-30) { g = atan2(ti, tr); }
    gradients[i0] = g;
}
)wgsl";

static const char* KERNEL_PHASE_GRAD_Z = R"wgsl(
fn idx3(x: i32, y: i32, z: i32, W: i32, H: i32) -> i32 {
    return x + y * W + z * W * H;
}

struct Dims {
    W: i32,
    H: i32,
    D: i32,
}

@group(0) @binding(0) var<storage, read_write> gradients: array<f32>;
@group(0) @binding(1) var<storage, read> q1: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> q2: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x); let y = i32(gid.y); let z = i32(gid.z);
    if (x >= dims.W || y >= dims.H || z < 1 || z >= dims.D - 1) { return; }

    let i0 = idx3(x, y, z, dims.W, dims.H);
    let im = idx3(x, y, z-1, dims.W, dims.H);
    let ip = idx3(x, y, z+1, dims.W, dims.H);

    var tr: f32 = 0.0; var ti: f32 = 0.0;

    var a = q1[ip]; var c = q1[i0];
    tr += a.x*c.x + a.y*c.y; ti += a.y*c.x - a.x*c.y;
    a = q1[i0]; c = q1[im];
    tr += a.x*c.x + a.y*c.y; ti += a.y*c.x - a.x*c.y;
    a = q2[ip]; c = q2[i0];
    tr += a.x*c.x + a.y*c.y; ti += a.y*c.x - a.x*c.y;
    a = q2[i0]; c = q2[im];
    tr += a.x*c.x + a.y*c.y; ti += a.y*c.x - a.x*c.y;

    var g = 0.0;
    if (abs(tr) > 1e-30 || abs(ti) > 1e-30) { g = atan2(ti, tr); }
    gradients[i0] = g;
}
)wgsl";

static const char* KERNEL_AMATRIX_HVECTOR_2D = R"wgsl(
fn idx3(x: i32, y: i32, z: i32, W: i32, H: i32) -> i32 {
    return x + y * W + z * W * H;
}

struct AMatrixParams {
    W: i32, H: i32, D: i32,
    filterSize: i32,
    directionOffset: i32,
    hVectorOffset: i32,
}

@group(0) @binding(0) var<storage, read_write> A2D: array<f32>;
@group(0) @binding(1) var<storage, read_write> h2D: array<f32>;
@group(0) @binding(2) var<storage, read> phaseDiff: array<f32>;
@group(0) @binding(3) var<storage, read> phaseGrad: array<f32>;
@group(0) @binding(4) var<storage, read> certainty: array<f32>;
@group(0) @binding(5) var<uniform> p: AMatrixParams;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let y = i32(gid.x);
    let z = i32(gid.y);
    let fhalf = (p.filterSize - 1) / 2;

    if (y < fhalf || y >= p.H - fhalf || z < fhalf || z >= p.D - fhalf) { return; }

    let yf = f32(y) - (f32(p.H) - 1.0) * 0.5;
    let zf = f32(z) - (f32(p.D) - 1.0) * 0.5;

    var aval: array<f32, 10>;
    var hval: array<f32, 4>;

    for (var x = fhalf; x < p.W - fhalf; x++) {
        let xf = f32(x) - (f32(p.W) - 1.0) * 0.5;
        let i = idx3(x, y, z, p.W, p.H);
        let pd = phaseDiff[i];
        let pg = phaseGrad[i];
        let cert = certainty[i];
        let cpp = cert * pg * pg;
        let cpd = cert * pg * pd;

        aval[0] += cpp;
        aval[1] += xf * cpp;
        aval[2] += yf * cpp;
        aval[3] += zf * cpp;
        aval[4] += xf * xf * cpp;
        aval[5] += xf * yf * cpp;
        aval[6] += xf * zf * cpp;
        aval[7] += yf * yf * cpp;
        aval[8] += yf * zf * cpp;
        aval[9] += zf * zf * cpp;

        hval[0] += cpd;
        hval[1] += xf * cpd;
        hval[2] += yf * cpd;
        hval[3] += zf * cpd;
    }

    let HD = p.H * p.D;
    let base = y + z * p.H + p.directionOffset * HD;
    for (var k = 0; k < 10; k++) {
        A2D[base + k * HD] = aval[k];
    }

    let hBase = y + z * p.H + p.hVectorOffset * HD;
    h2D[hBase] = hval[0];
    let extraBase = y + z * p.H + (3 + p.hVectorOffset * 3) * HD;
    h2D[extraBase + 0 * HD] = hval[1];
    h2D[extraBase + 1 * HD] = hval[2];
    h2D[extraBase + 2 * HD] = hval[3];
}
)wgsl";

static const char* KERNEL_AMATRIX_1D = R"wgsl(
@group(0) @binding(0) var<storage, read_write> A1D: array<f32>;
@group(0) @binding(1) var<storage, read> A2D: array<f32>;
@group(0) @binding(2) var<uniform> params: vec4<i32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let z = i32(gid.x);
    let element = i32(gid.y);
    let H = params.x;
    let D = params.y;
    let filterSize = params.z;
    let fhalf = (filterSize - 1) / 2;
    if (z < fhalf || z >= D - fhalf || element >= 30) { return; }

    var sum: f32 = 0.0;
    let base = z * H + element * H * D;
    for (var y = fhalf; y < H - fhalf; y++) {
        sum += A2D[base + y];
    }
    A1D[z + element * D] = sum;
}
)wgsl";

static const char* KERNEL_AMATRIX_FINAL = R"wgsl(
const parameterIndices = array<vec2<i32>, 30>(
    vec2<i32>(0,0), vec2<i32>(3,0), vec2<i32>(4,0), vec2<i32>(5,0),
    vec2<i32>(3,3), vec2<i32>(4,3), vec2<i32>(5,3), vec2<i32>(4,4),
    vec2<i32>(5,4), vec2<i32>(5,5),
    vec2<i32>(1,1), vec2<i32>(6,1), vec2<i32>(7,1), vec2<i32>(8,1),
    vec2<i32>(6,6), vec2<i32>(7,6), vec2<i32>(8,6), vec2<i32>(7,7),
    vec2<i32>(8,7), vec2<i32>(8,8),
    vec2<i32>(2,2), vec2<i32>(9,2), vec2<i32>(10,2), vec2<i32>(11,2),
    vec2<i32>(9,9), vec2<i32>(10,9), vec2<i32>(11,9), vec2<i32>(10,10),
    vec2<i32>(11,10), vec2<i32>(11,11)
);

@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read> A1D: array<f32>;
@group(0) @binding(2) var<uniform> params: vec4<i32>;

@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let element = i32(gid.x);
    if (element >= 30) { return; }
    let D = params.x;
    let filterSize = params.y;
    let fhalf = (filterSize - 1) / 2;

    var sum: f32 = 0.0;
    let base = element * D;
    for (var z = fhalf; z < D - fhalf; z++) {
        sum += A1D[base + z];
    }

    let ij = parameterIndices[element];
    A[ij.x + ij.y * 12] = sum;
}
)wgsl";

static const char* KERNEL_HVECTOR_1D = R"wgsl(
@group(0) @binding(0) var<storage, read_write> h1D: array<f32>;
@group(0) @binding(1) var<storage, read> h2D: array<f32>;
@group(0) @binding(2) var<uniform> params: vec4<i32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let z = i32(gid.x);
    let element = i32(gid.y);
    let H = params.x;
    let D = params.y;
    let filterSize = params.z;
    let fhalf = (filterSize - 1) / 2;
    if (z < fhalf || z >= D - fhalf || element >= 12) { return; }

    var sum: f32 = 0.0;
    let base = z * H + element * H * D;
    for (var y = fhalf; y < H - fhalf; y++) {
        sum += h2D[base + y];
    }
    h1D[z + element * D] = sum;
}
)wgsl";

static const char* KERNEL_HVECTOR_FINAL = R"wgsl(
@group(0) @binding(0) var<storage, read_write> h: array<f32>;
@group(0) @binding(1) var<storage, read> h1D: array<f32>;
@group(0) @binding(2) var<uniform> params: vec4<i32>;

@compute @workgroup_size(16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let element = i32(gid.x);
    if (element >= 12) { return; }
    let D = params.x;
    let filterSize = params.y;
    let fhalf = (filterSize - 1) / 2;

    var sum: f32 = 0.0;
    let base = element * D;
    for (var z = fhalf; z < D - fhalf; z++) {
        sum += h1D[base + z];
    }
    h[element] = sum;
}
)wgsl";

static const char* KERNEL_TENSOR_COMPONENTS = R"wgsl(
fn idx3(x: i32, y: i32, z: i32, W: i32, H: i32) -> i32 {
    return x + y * W + z * W * H;
}

struct Dims {
    W: i32,
    H: i32,
    D: i32,
}

struct TensorParams {
    m11: f32, m12: f32, m13: f32,
    m22: f32, m23: f32, m33: f32,
}

@group(0) @binding(0) var<storage, read_write> t11: array<f32>;
@group(0) @binding(1) var<storage, read_write> t12: array<f32>;
@group(0) @binding(2) var<storage, read_write> t13: array<f32>;
@group(0) @binding(3) var<storage, read_write> t22: array<f32>;
@group(0) @binding(4) var<storage, read_write> t23: array<f32>;
@group(0) @binding(5) var<storage, read_write> t33: array<f32>;
@group(0) @binding(6) var<storage, read> q2: array<vec2<f32>>;
@group(0) @binding(7) var<uniform> dims: Dims;
@group(0) @binding(8) var<uniform> tp: TensorParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x); let y = i32(gid.y); let z = i32(gid.z);
    if (x >= dims.W || y >= dims.H || z >= dims.D) { return; }

    let i = idx3(x, y, z, dims.W, dims.H);
    let q2v = q2[i];
    let mag = sqrt(q2v.x * q2v.x + q2v.y * q2v.y);

    t11[i] += mag * tp.m11;
    t12[i] += mag * tp.m12;
    t13[i] += mag * tp.m13;
    t22[i] += mag * tp.m22;
    t23[i] += mag * tp.m23;
    t33[i] += mag * tp.m33;
}
)wgsl";

static const char* KERNEL_TENSOR_NORMS = R"wgsl(
fn idx3(x: i32, y: i32, z: i32, W: i32, H: i32) -> i32 {
    return x + y * W + z * W * H;
}

struct Dims {
    W: i32,
    H: i32,
    D: i32,
}

@group(0) @binding(0) var<storage, read_write> norms: array<f32>;
@group(0) @binding(1) var<storage, read> t11: array<f32>;
@group(0) @binding(2) var<storage, read> t12: array<f32>;
@group(0) @binding(3) var<storage, read> t13: array<f32>;
@group(0) @binding(4) var<storage, read> t22: array<f32>;
@group(0) @binding(5) var<storage, read> t23: array<f32>;
@group(0) @binding(6) var<storage, read> t33: array<f32>;
@group(0) @binding(7) var<uniform> dims: Dims;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x); let y = i32(gid.y); let z = i32(gid.z);
    if (x >= dims.W || y >= dims.H || z >= dims.D) { return; }
    let i = idx3(x, y, z, dims.W, dims.H);
    let v11 = t11[i]; let v12 = t12[i]; let v13 = t13[i];
    let v22 = t22[i]; let v23 = t23[i]; let v33 = t33[i];
    norms[i] = sqrt(v11*v11 + 2.0*v12*v12 + 2.0*v13*v13 + v22*v22 + 2.0*v23*v23 + v33*v33);
}
)wgsl";

static const char* KERNEL_AMATRICES_HVECTORS = R"wgsl(
fn idx3(x: i32, y: i32, z: i32, W: i32, H: i32) -> i32 {
    return x + y * W + z * W * H;
}

struct Dims {
    W: i32,
    H: i32,
    D: i32,
}

struct MorphonParams {
    W: i32, H: i32, D: i32, FILTER: i32,
}

@group(0) @binding(0) var<storage, read_write> a11: array<f32>;
@group(0) @binding(1) var<storage, read_write> a12: array<f32>;
@group(0) @binding(2) var<storage, read_write> a13: array<f32>;
@group(0) @binding(3) var<storage, read_write> a22: array<f32>;
@group(0) @binding(4) var<storage, read_write> a23: array<f32>;
@group(0) @binding(5) var<storage, read_write> a33: array<f32>;
@group(0) @binding(6) var<storage, read_write> h1: array<f32>;
@group(0) @binding(7) var<storage, read_write> h2: array<f32>;
@group(0) @binding(8) var<storage, read_write> h3: array<f32>;
@group(0) @binding(9) var<storage, read> q1: array<vec2<f32>>;
@group(0) @binding(10) var<storage, read> q2: array<vec2<f32>>;
@group(0) @binding(11) var<storage, read> t11: array<f32>;
@group(0) @binding(12) var<storage, read> t12: array<f32>;
@group(0) @binding(13) var<storage, read> t13: array<f32>;
@group(0) @binding(14) var<storage, read> t22: array<f32>;
@group(0) @binding(15) var<storage, read> t23: array<f32>;
@group(0) @binding(16) var<storage, read> t33: array<f32>;
@group(0) @binding(17) var<storage, read> filterDirX: array<f32>;
@group(0) @binding(18) var<storage, read> filterDirY: array<f32>;
@group(0) @binding(19) var<storage, read> filterDirZ: array<f32>;
@group(0) @binding(20) var<uniform> p: MorphonParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x); let y = i32(gid.y); let z = i32(gid.z);
    if (x >= p.W || y >= p.H || z >= p.D) { return; }

    let i = idx3(x, y, z, p.W, p.H);
    let q1v = q1[i];
    let q2v = q2[i];

    let qqR = q1v.x * q2v.x + q1v.y * q2v.y;
    let qqI = -q1v.x * q2v.y + q1v.y * q2v.x;
    var pd = 0.0;
    if (abs(qqR) > 1e-30 || abs(qqI) > 1e-30) { pd = atan2(qqI, qqR); }
    let Aqq = qqR * qqR + qqI * qqI;
    let cosH = cos(pd * 0.5);
    var cert = 0.0;
    if (Aqq > 0.0) { cert = sqrt(sqrt(Aqq)) * cosH * cosH; }

    let T11 = t11[i]; let T12 = t12[i]; let T13 = t13[i];
    let T22 = t22[i]; let T23 = t23[i]; let T33 = t33[i];

    let tt11 = T11*T11 + T12*T12 + T13*T13;
    let tt12 = T11*T12 + T12*T22 + T13*T23;
    let tt13 = T11*T13 + T12*T23 + T13*T33;
    let tt22 = T12*T12 + T22*T22 + T23*T23;
    let tt23 = T12*T13 + T22*T23 + T23*T33;
    let tt33 = T13*T13 + T23*T23 + T33*T33;

    let fdx = filterDirX[p.FILTER];
    let fdy = filterDirY[p.FILTER];
    let fdz = filterDirZ[p.FILTER];

    let cpd = cert * pd;
    let hh1 = cpd * (fdx * tt11 + fdy * tt12 + fdz * tt13);
    let hh2 = cpd * (fdx * tt12 + fdy * tt22 + fdz * tt23);
    let hh3 = cpd * (fdx * tt13 + fdy * tt23 + fdz * tt33);

    if (p.FILTER == 0) {
        a11[i] = cert * tt11;
        a12[i] = cert * tt12;
        a13[i] = cert * tt13;
        a22[i] = cert * tt22;
        a23[i] = cert * tt23;
        a33[i] = cert * tt33;
        h1[i] = hh1;
        h2[i] = hh2;
        h3[i] = hh3;
    } else {
        a11[i] += cert * tt11;
        a12[i] += cert * tt12;
        a13[i] += cert * tt13;
        a22[i] += cert * tt22;
        a23[i] += cert * tt23;
        a33[i] += cert * tt33;
        h1[i] += hh1;
        h2[i] += hh2;
        h3[i] += hh3;
    }
}
)wgsl";

static const char* KERNEL_DISPLACEMENT_UPDATE = R"wgsl(
fn idx3(x: i32, y: i32, z: i32, W: i32, H: i32) -> i32 {
    return x + y * W + z * W * H;
}

struct Dims {
    W: i32,
    H: i32,
    D: i32,
}

@group(0) @binding(0) var<storage, read_write> dispX: array<f32>;
@group(0) @binding(1) var<storage, read_write> dispY: array<f32>;
@group(0) @binding(2) var<storage, read_write> dispZ: array<f32>;
@group(0) @binding(3) var<storage, read> a11: array<f32>;
@group(0) @binding(4) var<storage, read> a12: array<f32>;
@group(0) @binding(5) var<storage, read> a13: array<f32>;
@group(0) @binding(6) var<storage, read> a22: array<f32>;
@group(0) @binding(7) var<storage, read> a23: array<f32>;
@group(0) @binding(8) var<storage, read> a33: array<f32>;
@group(0) @binding(9) var<storage, read> rh1: array<f32>;
@group(0) @binding(10) var<storage, read> rh2: array<f32>;
@group(0) @binding(11) var<storage, read> rh3: array<f32>;
@group(0) @binding(12) var<uniform> dims: Dims;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x); let y = i32(gid.y); let z = i32(gid.z);
    if (x >= dims.W || y >= dims.H || z >= dims.D) { return; }

    let i = idx3(x, y, z, dims.W, dims.H);
    let A11 = a11[i]; let A12 = a12[i]; let A13 = a13[i];
    let A22 = a22[i]; let A23 = a23[i]; let A33 = a33[i];
    let H1 = rh1[i]; let H2 = rh2[i]; let H3 = rh3[i];

    let det = A11*A22*A33 - A11*A23*A23 - A12*A12*A33
            + A12*A23*A13 + A13*A12*A23 - A13*A22*A13;

    let trace = A11 + A22 + A33;
    let epsilon = 0.01 * trace * trace * trace / 27.0 + 1e-16;
    let denom = det + epsilon;

    var dx = 0.0; var dy = 0.0; var dz = 0.0;
    if (abs(denom) > 1e-30) {
        let norm = 0.2 / denom;
        dx = norm * (H1*(A22*A33 - A23*A23) - H2*(A12*A33 - A13*A23) + H3*(A12*A23 - A13*A22));
        dy = norm * (H2*(A11*A33 - A13*A13) - H3*(A11*A23 - A13*A12) - H1*(A12*A33 - A23*A13));
        dz = norm * (H3*(A11*A22 - A12*A12) - H2*(A11*A23 - A12*A13) + H1*(A12*A23 - A22*A13));
        // Clamp extreme values
        if (abs(dx) > 1e6) { dx = 0.0; }
        if (abs(dy) > 1e6) { dy = 0.0; }
        if (abs(dz) > 1e6) { dz = 0.0; }
    }

    dispX[i] = dx;
    dispY[i] = dy;
    dispZ[i] = dz;
}
)wgsl";

static const char* KERNEL_INTERPOLATE_LINEAR = R"wgsl(
fn idx3(x: i32, y: i32, z: i32, W: i32, H: i32) -> i32 {
    return x + y * W + z * W * H;
}
struct Dims { W: i32, H: i32, D: i32 }

@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<storage, read> volume: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

fn safe_read(x: i32, y: i32, z: i32, W: i32, H: i32, D: i32) -> f32 {
    if (x < 0 || x >= W || y < 0 || y >= H || z < 0 || z >= D) {
        return 0.0;
    }
    return volume[idx3(x, y, z, W, H)];
}

fn trilinear(px: f32, py: f32, pz: f32, W: i32, H: i32, D: i32) -> f32 {
    let x0 = i32(floor(px));
    let y0 = i32(floor(py));
    let z0 = i32(floor(pz));
    let fx = px - f32(x0);
    let fy = py - f32(y0);
    let fz = pz - f32(z0);
    let c000 = safe_read(x0, y0, z0, W, H, D);
    let c100 = safe_read(x0+1, y0, z0, W, H, D);
    let c010 = safe_read(x0, y0+1, z0, W, H, D);
    let c110 = safe_read(x0+1, y0+1, z0, W, H, D);
    let c001 = safe_read(x0, y0, z0+1, W, H, D);
    let c101 = safe_read(x0+1, y0, z0+1, W, H, D);
    let c011 = safe_read(x0, y0+1, z0+1, W, H, D);
    let c111 = safe_read(x0+1, y0+1, z0+1, W, H, D);
    let c00 = c000*(1.0-fx) + c100*fx;
    let c10 = c010*(1.0-fx) + c110*fx;
    let c01 = c001*(1.0-fx) + c101*fx;
    let c11 = c011*(1.0-fx) + c111*fx;
    let c0 = c00*(1.0-fy) + c10*fy;
    let c1 = c01*(1.0-fy) + c11*fy;
    return c0*(1.0-fz) + c1*fz;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x); let y = i32(gid.y); let z = i32(gid.z);
    if (x >= dims.W || y >= dims.H || z >= dims.D) { return; }

    let xf = f32(x) - (f32(dims.W) - 1.0) * 0.5;
    let yf = f32(y) - (f32(dims.H) - 1.0) * 0.5;
    let zf = f32(z) - (f32(dims.D) - 1.0) * 0.5;

    let px = f32(x) + params[0] + params[3]*xf + params[4]*yf + params[5]*zf;
    let py = f32(y) + params[1] + params[6]*xf + params[7]*yf + params[8]*zf;
    let pz = f32(z) + params[2] + params[9]*xf + params[10]*yf + params[11]*zf;

    output[idx3(x, y, z, dims.W, dims.H)] = trilinear(px, py, pz, dims.W, dims.H, dims.D);
}
)wgsl";

static const char* KERNEL_INTERPOLATE_NONLINEAR = R"wgsl(
fn idx3(x: i32, y: i32, z: i32, W: i32, H: i32) -> i32 {
    return x + y * W + z * W * H;
}
struct Dims { W: i32, H: i32, D: i32 }

@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<storage, read> volume: array<f32>;
@group(0) @binding(2) var<storage, read> dX: array<f32>;
@group(0) @binding(3) var<storage, read> dY: array<f32>;
@group(0) @binding(4) var<storage, read> dZ: array<f32>;
@group(0) @binding(5) var<uniform> dims: Dims;

fn safe_read(x: i32, y: i32, z: i32, W: i32, H: i32, D: i32) -> f32 {
    if (x < 0 || x >= W || y < 0 || y >= H || z < 0 || z >= D) {
        return 0.0;
    }
    return volume[idx3(x, y, z, W, H)];
}

fn trilinear(px: f32, py: f32, pz: f32, W: i32, H: i32, D: i32) -> f32 {
    let x0 = i32(floor(px));
    let y0 = i32(floor(py));
    let z0 = i32(floor(pz));
    let fx = px - f32(x0);
    let fy = py - f32(y0);
    let fz = pz - f32(z0);
    let c000 = safe_read(x0, y0, z0, W, H, D);
    let c100 = safe_read(x0+1, y0, z0, W, H, D);
    let c010 = safe_read(x0, y0+1, z0, W, H, D);
    let c110 = safe_read(x0+1, y0+1, z0, W, H, D);
    let c001 = safe_read(x0, y0, z0+1, W, H, D);
    let c101 = safe_read(x0+1, y0, z0+1, W, H, D);
    let c011 = safe_read(x0, y0+1, z0+1, W, H, D);
    let c111 = safe_read(x0+1, y0+1, z0+1, W, H, D);
    let c00 = c000*(1.0-fx) + c100*fx;
    let c10 = c010*(1.0-fx) + c110*fx;
    let c01 = c001*(1.0-fx) + c101*fx;
    let c11 = c011*(1.0-fx) + c111*fx;
    let c0 = c00*(1.0-fy) + c10*fy;
    let c1 = c01*(1.0-fy) + c11*fy;
    return c0*(1.0-fz) + c1*fz;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x); let y = i32(gid.y); let z = i32(gid.z);
    if (x >= dims.W || y >= dims.H || z >= dims.D) { return; }

    let i = idx3(x, y, z, dims.W, dims.H);
    let px = f32(x) + dX[i];
    let py = f32(y) + dY[i];
    let pz = f32(z) + dZ[i];

    output[i] = trilinear(px, py, pz, dims.W, dims.H, dims.D);
}
)wgsl";

static const char* KERNEL_RESCALE_VOLUME = R"wgsl(
fn idx3(x: i32, y: i32, z: i32, W: i32, H: i32) -> i32 {
    return x + y * W + z * W * H;
}

struct Dims { W: i32, H: i32, D: i32 }
struct Scales { x: f32, y: f32, z: f32 }

@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<storage, read> volume: array<f32>;
@group(0) @binding(2) var<uniform> dims: Dims;
@group(0) @binding(3) var<uniform> scales: Scales;
@group(0) @binding(4) var<uniform> srcDims: Dims;

fn safe_read(x: i32, y: i32, z: i32, W: i32, H: i32, D: i32) -> f32 {
    if (x < 0 || x >= W || y < 0 || y >= H || z < 0 || z >= D) {
        return 0.0;
    }
    return volume[idx3(x, y, z, W, H)];
}

fn trilinear(px: f32, py: f32, pz: f32, W: i32, H: i32, D: i32) -> f32 {
    let x0 = i32(floor(px));
    let y0 = i32(floor(py));
    let z0 = i32(floor(pz));
    let fx = px - f32(x0);
    let fy = py - f32(y0);
    let fz = pz - f32(z0);
    let c000 = safe_read(x0, y0, z0, W, H, D);
    let c100 = safe_read(x0+1, y0, z0, W, H, D);
    let c010 = safe_read(x0, y0+1, z0, W, H, D);
    let c110 = safe_read(x0+1, y0+1, z0, W, H, D);
    let c001 = safe_read(x0, y0, z0+1, W, H, D);
    let c101 = safe_read(x0+1, y0, z0+1, W, H, D);
    let c011 = safe_read(x0, y0+1, z0+1, W, H, D);
    let c111 = safe_read(x0+1, y0+1, z0+1, W, H, D);
    let c00 = c000*(1.0-fx) + c100*fx;
    let c10 = c010*(1.0-fx) + c110*fx;
    let c01 = c001*(1.0-fx) + c101*fx;
    let c11 = c011*(1.0-fx) + c111*fx;
    let c0 = c00*(1.0-fy) + c10*fy;
    let c1 = c01*(1.0-fy) + c11*fy;
    return c0*(1.0-fz) + c1*fz;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x); let y = i32(gid.y); let z = i32(gid.z);
    if (x >= dims.W || y >= dims.H || z >= dims.D) { return; }

    let px = f32(x) * scales.x;
    let py = f32(y) * scales.y;
    let pz = f32(z) * scales.z;

    output[idx3(x, y, z, dims.W, dims.H)] = trilinear(px, py, pz, srcDims.W, srcDims.H, srcDims.D);
}
)wgsl";

static const char* KERNEL_COPY_VOLUME_TO_NEW = R"wgsl(
fn idx3(x: i32, y: i32, z: i32, W: i32, H: i32) -> i32 {
    return x + y * W + z * W * H;
}

struct CopyParams {
    newW: i32, newH: i32, newD: i32,
    srcW: i32, srcH: i32, srcD: i32,
    xDiff: i32, yDiff: i32, zDiff: i32,
    mmZCut: i32,
    voxelSizeZ: f32,
    _pad: i32,
}

@group(0) @binding(0) var<storage, read_write> newVol: array<f32>;
@group(0) @binding(1) var<storage, read> srcVol: array<f32>;
@group(0) @binding(2) var<uniform> p: CopyParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x); let y = i32(gid.y); let z = i32(gid.z);

    var xNew: i32; var xSrc: i32;
    var yNew: i32; var ySrc: i32;
    var zNew: i32; var zSrc: i32;

    if (p.xDiff > 0) {
        xNew = x; xSrc = x + i32(round(f32(p.xDiff) / 2.0));
    } else {
        xNew = x + i32(round(f32(abs(p.xDiff)) / 2.0)); xSrc = x;
    }
    if (p.yDiff > 0) {
        yNew = y; ySrc = y + i32(round(f32(p.yDiff) / 2.0));
    } else {
        yNew = y + i32(round(f32(abs(p.yDiff)) / 2.0)); ySrc = y;
    }
    if (p.zDiff > 0) {
        zNew = z; zSrc = z + i32(round(f32(p.zDiff) / 2.0)) + i32(round(f32(p.mmZCut) / p.voxelSizeZ));
    } else {
        zNew = z + i32(round(f32(abs(p.zDiff)) / 2.0));
        zSrc = z + i32(round(f32(p.mmZCut) / p.voxelSizeZ));
    }

    if (xSrc < 0 || xSrc >= p.srcW || ySrc < 0 || ySrc >= p.srcH || zSrc < 0 || zSrc >= p.srcD) { return; }
    if (xNew < 0 || xNew >= p.newW || yNew < 0 || yNew >= p.newH || zNew < 0 || zNew >= p.newD) { return; }

    newVol[idx3(xNew, yNew, zNew, p.newW, p.newH)] = srcVol[idx3(xSrc, ySrc, zSrc, p.srcW, p.srcH)];
}
)wgsl";

static const char* KERNEL_ADD_LINEAR_NONLINEAR_DISP = R"wgsl(
fn idx3(x: i32, y: i32, z: i32, W: i32, H: i32) -> i32 {
    return x + y * W + z * W * H;
}
struct Dims { W: i32, H: i32, D: i32 }

@group(0) @binding(0) var<storage, read_write> dispX: array<f32>;
@group(0) @binding(1) var<storage, read_write> dispY: array<f32>;
@group(0) @binding(2) var<storage, read_write> dispZ: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<f32>;
@group(0) @binding(4) var<uniform> dims: Dims;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x); let y = i32(gid.y); let z = i32(gid.z);
    if (x >= dims.W || y >= dims.H || z >= dims.D) { return; }

    let xf = f32(x) - (f32(dims.W) - 1.0) * 0.5;
    let yf = f32(y) - (f32(dims.H) - 1.0) * 0.5;
    let zf = f32(z) - (f32(dims.D) - 1.0) * 0.5;

    let i = idx3(x, y, z, dims.W, dims.H);
    dispX[i] += params[0] + params[3]*xf + params[4]*yf + params[5]*zf;
    dispY[i] += params[1] + params[6]*xf + params[7]*yf + params[8]*zf;
    dispZ[i] += params[2] + params[9]*xf + params[10]*yf + params[11]*zf;
}
)wgsl";

// ============================================================
//  Kernel Registry
// ============================================================

static std::unordered_map<std::string, const char*> KERNELS = {
    {"fillFloat",             KERNEL_FILL_FLOAT},
    {"fillVec2",              KERNEL_FILL_VEC2},
    {"addVolumes",            KERNEL_ADD_VOLUMES},
    {"multiplyVolume",        KERNEL_MULTIPLY_VOLUME},
    {"multiplyVolumes",       KERNEL_MULTIPLY_VOLUMES},
    {"calculateColumnMaxs",   KERNEL_COLUMN_MAXS},
    {"calculateRowMaxs",      KERNEL_ROW_MAXS},
    {"conv3D_Full",           KERNEL_CONV3D_FULL},
    {"separableConvRows",     KERNEL_SEPARABLE_CONV_ROWS},
    {"separableConvColumns",  KERNEL_SEPARABLE_CONV_COLS},
    {"separableConvRods",     KERNEL_SEPARABLE_CONV_RODS},
    {"phaseDiffCert",         KERNEL_PHASE_DIFF_CERT},
    {"phaseGradX",            KERNEL_PHASE_GRAD_X},
    {"phaseGradY",            KERNEL_PHASE_GRAD_Y},
    {"phaseGradZ",            KERNEL_PHASE_GRAD_Z},
    {"amatrixHvector2D",      KERNEL_AMATRIX_HVECTOR_2D},
    {"amatrix1D",             KERNEL_AMATRIX_1D},
    {"amatrixFinal",          KERNEL_AMATRIX_FINAL},
    {"hvector1D",             KERNEL_HVECTOR_1D},
    {"hvectorFinal",          KERNEL_HVECTOR_FINAL},
    {"tensorComponents",      KERNEL_TENSOR_COMPONENTS},
    {"tensorNorms",           KERNEL_TENSOR_NORMS},
    {"amatricesHvectors",     KERNEL_AMATRICES_HVECTORS},
    {"displacementUpdate",    KERNEL_DISPLACEMENT_UPDATE},
    {"interpolateLinear",     KERNEL_INTERPOLATE_LINEAR},
    {"interpolateNonLinear",  KERNEL_INTERPOLATE_NONLINEAR},
    {"rescaleVolume",         KERNEL_RESCALE_VOLUME},
    {"copyVolumeToNew",       KERNEL_COPY_VOLUME_TO_NEW},
    {"addLinearNonLinearDisp", KERNEL_ADD_LINEAR_NONLINEAR_DISP},
};

// ============================================================
//  WebGPU Context
// ============================================================

namespace webgpu_reg {
namespace {

struct Dims {
    int W, H, D;
};

// Callback data structures
struct AdapterData {
    WGPUAdapter adapter = nullptr;
    WGPUFuture future = {};
};

struct DeviceData {
    WGPUDevice device = nullptr;
    WGPUFuture future = {};
};

struct MapData {
    bool done = false;
    WGPUFuture future = {};
};

// Callbacks
static void onAdapter(WGPURequestAdapterStatus status, WGPUAdapter adapter,
                       WGPUStringView message, void* ud1, void* ud2) {
    auto* data = (AdapterData*)ud1;
    if (status == WGPURequestAdapterStatus_Success) {
        data->adapter = adapter;
    } else {
        fprintf(stderr, "Adapter request failed: %.*s\n", (int)message.length, message.data);
    }
}

static void onDevice(WGPURequestDeviceStatus status, WGPUDevice device,
                      WGPUStringView message, void* ud1, void* ud2) {
    auto* data = (DeviceData*)ud1;
    if (status == WGPURequestDeviceStatus_Success) {
        data->device = device;
    } else {
        fprintf(stderr, "Device request failed: %.*s\n", (int)message.length, message.data);
    }
}

static void onMapAsync(WGPUMapAsyncStatus status, WGPUStringView message,
                        void* ud1, void* ud2) {
    auto* data = (MapData*)ud1;
    data->done = (status == WGPUMapAsyncStatus_Success);
}

class WebGPUContext {
public:
    WGPUInstance instance = nullptr;
    WGPUAdapter adapter = nullptr;
    WGPUDevice device = nullptr;
    WGPUQueue queue = nullptr;
    std::unordered_map<std::string, WGPUComputePipeline> pipelines;
    std::unordered_map<std::string, WGPUShaderModule> shaderModules;

    void init() {
        instance = wgpuCreateInstance(nullptr);
        assert(instance && "Failed to create WebGPU instance");

        // Request adapter
        WGPURequestAdapterOptions opts = {};
        opts.powerPreference = WGPUPowerPreference_HighPerformance;

        AdapterData adata = {};
        WGPURequestAdapterCallbackInfo adapterCb = {};
        adapterCb.mode = WGPUCallbackMode_AllowProcessEvents;
        adapterCb.callback = onAdapter;
        adapterCb.userdata1 = &adata;

        wgpuInstanceRequestAdapter(instance, &opts, adapterCb);
        while (!adata.adapter) {
            wgpuInstanceProcessEvents(instance);
        }
        adapter = adata.adapter;
        assert(adapter && "Failed to get WebGPU adapter");

        // Request device with adapter limits (avoids zero-default issues)
        WGPULimits requiredLimits = {};
        wgpuAdapterGetLimits(adapter, &requiredLimits);
        // Override specific limits we need
        if (requiredLimits.maxStorageBuffersPerShaderStage < 21)
            requiredLimits.maxStorageBuffersPerShaderStage = 21;
        if (requiredLimits.maxBufferSize < 256u * 1024 * 1024)
            requiredLimits.maxBufferSize = 256u * 1024 * 1024;
        if (requiredLimits.maxStorageBufferBindingSize < 256u * 1024 * 1024)
            requiredLimits.maxStorageBufferBindingSize = 256u * 1024 * 1024;

        WGPUDeviceDescriptor devDesc = {};
        devDesc.requiredLimits = &requiredLimits;

        DeviceData ddata = {};
        WGPURequestDeviceCallbackInfo deviceCb = {};
        deviceCb.mode = WGPUCallbackMode_AllowProcessEvents;
        deviceCb.callback = onDevice;
        deviceCb.userdata1 = &ddata;

        wgpuAdapterRequestDevice(adapter, &devDesc, deviceCb);
        while (!ddata.device) {
            wgpuInstanceProcessEvents(instance);
        }
        device = ddata.device;
        assert(device && "Failed to get WebGPU device");

        queue = wgpuDeviceGetQueue(device);
        assert(queue && "Failed to get device queue");
    }

    WGPUComputePipeline getPipeline(const std::string& name) {
        auto it = pipelines.find(name);
        if (it != pipelines.end()) return it->second;

        auto kit = KERNELS.find(name);
        assert(kit != KERNELS.end() && "Unknown kernel name");

        // Create shader module
        WGPUShaderSourceWGSL wgslSource = {};
        wgslSource.chain.sType = WGPUSType_ShaderSourceWGSL;
        wgslSource.code.data = kit->second;
        wgslSource.code.length = WGPU_STRLEN;

        WGPUShaderModuleDescriptor smd = {};
        smd.nextInChain = &wgslSource.chain;

        WGPUShaderModule module = wgpuDeviceCreateShaderModule(device, &smd);
        assert(module && "Failed to create shader module");
        shaderModules[name] = module;

        // Create compute pipeline
        WGPUComputePipelineDescriptor cpd = {};
        cpd.compute.module = module;
        cpd.compute.entryPoint.data = "main";
        cpd.compute.entryPoint.length = WGPU_STRLEN;

        WGPUComputePipeline pipeline = wgpuDeviceCreateComputePipeline(device, &cpd);
        assert(pipeline && "Failed to create compute pipeline");
        pipelines[name] = pipeline;
        return pipeline;
    }

    WGPUBuffer newBuffer(size_t bytes, const void* data = nullptr) {
        if (bytes < 4) bytes = 4;  // minimum buffer size
        WGPUBufferDescriptor desc = {};
        desc.size = bytes;
        desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst;
        desc.mappedAtCreation = false;
        WGPUBuffer buf = wgpuDeviceCreateBuffer(device, &desc);
        if (data) {
            wgpuQueueWriteBuffer(queue, buf, 0, data, bytes);
        }
        return buf;
    }

    WGPUBuffer newUniform(const void* data, size_t bytes) {
        if (bytes < 4) bytes = 4;
        WGPUBufferDescriptor desc = {};
        desc.size = bytes;
        desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
        desc.mappedAtCreation = false;
        WGPUBuffer buf = wgpuDeviceCreateBuffer(device, &desc);
        wgpuQueueWriteBuffer(queue, buf, 0, data, bytes);
        return buf;
    }

    void writeBuffer(WGPUBuffer buf, const void* data, size_t bytes) {
        wgpuQueueWriteBuffer(queue, buf, 0, data, bytes);
    }

    void readBuffer(WGPUBuffer buf, void* out, size_t bytes) {
        // Create staging buffer
        WGPUBufferDescriptor desc = {};
        desc.size = bytes;
        desc.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
        WGPUBuffer staging = wgpuDeviceCreateBuffer(device, &desc);

        // Copy GPU buffer to staging
        WGPUCommandEncoderDescriptor encDesc = {};
        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, &encDesc);
        wgpuCommandEncoderCopyBufferToBuffer(encoder, buf, 0, staging, 0, bytes);
        WGPUCommandBufferDescriptor cbDesc = {};
        WGPUCommandBuffer cb = wgpuCommandEncoderFinish(encoder, &cbDesc);
        wgpuQueueSubmit(queue, 1, &cb);

        // Map staging buffer
        MapData mdata = {};
        WGPUBufferMapCallbackInfo mapCb = {};
        mapCb.mode = WGPUCallbackMode_AllowProcessEvents;
        mapCb.callback = onMapAsync;
        mapCb.userdata1 = &mdata;

        wgpuBufferMapAsync(staging, WGPUMapMode_Read, 0, bytes, mapCb);
        while (!mdata.done) {
            wgpuDevicePoll(device, true, NULL);
        }

        const void* mapped = wgpuBufferGetConstMappedRange(staging, 0, bytes);
        memcpy(out, mapped, bytes);
        wgpuBufferUnmap(staging);
        wgpuBufferRelease(staging);
        wgpuCommandBufferRelease(cb);
        wgpuCommandEncoderRelease(encoder);
    }

    void copyBuffer(WGPUBuffer src, WGPUBuffer dst, size_t bytes) {
        WGPUCommandEncoderDescriptor encDesc = {};
        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, &encDesc);
        wgpuCommandEncoderCopyBufferToBuffer(encoder, src, 0, dst, 0, bytes);
        WGPUCommandBufferDescriptor cbDesc = {};
        WGPUCommandBuffer cb = wgpuCommandEncoderFinish(encoder, &cbDesc);
        wgpuQueueSubmit(queue, 1, &cb);
        wgpuDevicePoll(device, true, nullptr);
        wgpuCommandBufferRelease(cb);
        wgpuCommandEncoderRelease(encoder);
    }

    void dispatch(const std::string& kernelName,
                  const std::vector<std::pair<uint32_t, WGPUBuffer>>& bindings,
                  const std::vector<size_t>& bufSizes,
                  uint32_t gx, uint32_t gy, uint32_t gz) {
        WGPUComputePipeline pipeline = getPipeline(kernelName);
        WGPUBindGroupLayout layout = wgpuComputePipelineGetBindGroupLayout(pipeline, 0);

        std::vector<WGPUBindGroupEntry> entries(bindings.size());
        for (size_t i = 0; i < bindings.size(); i++) {
            entries[i] = {};
            entries[i].binding = bindings[i].first;
            entries[i].buffer = bindings[i].second;
            entries[i].offset = 0;
            entries[i].size = bufSizes[i];
        }

        WGPUBindGroupDescriptor bgDesc = {};
        bgDesc.layout = layout;
        bgDesc.entryCount = entries.size();
        bgDesc.entries = entries.data();
        WGPUBindGroup bindGroup = wgpuDeviceCreateBindGroup(device, &bgDesc);

        WGPUCommandEncoderDescriptor encDesc = {};
        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, &encDesc);
        WGPUComputePassDescriptor passDesc = {};
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
        wgpuComputePassEncoderSetPipeline(pass, pipeline);
        wgpuComputePassEncoderSetBindGroup(pass, 0, bindGroup, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, gx, gy, gz);
        wgpuComputePassEncoderEnd(pass);

        WGPUCommandBufferDescriptor cbDesc = {};
        WGPUCommandBuffer cb = wgpuCommandEncoderFinish(encoder, &cbDesc);
        wgpuQueueSubmit(queue, 1, &cb);
        wgpuDevicePoll(device, true, nullptr);

        wgpuBindGroupRelease(bindGroup);
        wgpuBindGroupLayoutRelease(layout);
        wgpuComputePassEncoderRelease(pass);
        wgpuCommandBufferRelease(cb);
        wgpuCommandEncoderRelease(encoder);
    }

    size_t bufferSize(WGPUBuffer buf) {
        return wgpuBufferGetSize(buf);
    }

    static WebGPUContext& get() {
        static WebGPUContext ctx;
        static bool inited = false;
        if (!inited) {
            ctx.init();
            inited = true;
        }
        return ctx;
    }
};

// ============================================================
//  Dispatch helpers
// ============================================================

static uint32_t divCeil(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

struct WG { uint32_t x, y, z; };

static WG wg3d(int W, int H, int D) {
    return { divCeil(W, 8), divCeil(H, 8), (uint32_t)D };
}

static WG wg2d(int W, int H) {
    return { divCeil(W, 8), divCeil(H, 8), 1 };
}

static WG wg1d(int N) {
    return { divCeil(N, 256), 1, 1 };
}

// Helper: create Dims uniform buffer
static WGPUBuffer dimsUniform(WebGPUContext& c, int W, int H, int D) {
    Dims dims = {W, H, D};
    return c.newUniform(&dims, sizeof(Dims));
}

// Convenience dispatch wrapper
static void gpuDispatch(WebGPUContext& c, const std::string& kernel,
                        const std::vector<std::pair<uint32_t, WGPUBuffer>>& bindings,
                        WG wg) {
    std::vector<size_t> sizes;
    for (auto& b : bindings) sizes.push_back(c.bufferSize(b.second));
    c.dispatch(kernel, bindings, sizes, wg.x, wg.y, wg.z);
}

// ============================================================
//  GPU Operations
// ============================================================

static void fillBuffer(WebGPUContext& c, WGPUBuffer buf, float value, int count) {
    WGPUBuffer params = c.newUniform(&value, sizeof(float));
    gpuDispatch(c, "fillFloat", {{0, buf}, {1, params}}, wg1d(count));
    wgpuBufferRelease(params);
}

static void fillVec2Buffer(WebGPUContext& c, WGPUBuffer buf, int count) {
    gpuDispatch(c, "fillVec2", {{0, buf}}, wg1d(count));
}

static void addVolumes(WebGPUContext& c, WGPUBuffer A, WGPUBuffer B, int count) {
    gpuDispatch(c, "addVolumes", {{0, A}, {1, B}}, wg1d(count));
}

static void multiplyVolume(WebGPUContext& c, WGPUBuffer vol, float factor, int count) {
    WGPUBuffer params = c.newUniform(&factor, sizeof(float));
    gpuDispatch(c, "multiplyVolume", {{0, vol}, {1, params}}, wg1d(count));
    wgpuBufferRelease(params);
}

static void multiplyVolumes(WebGPUContext& c, WGPUBuffer A, WGPUBuffer B, int count) {
    gpuDispatch(c, "multiplyVolumes", {{0, A}, {1, B}}, wg1d(count));
}

static float calculateMax(WebGPUContext& c, WGPUBuffer volume, int W, int H, int D) {
    WGPUBuffer dims = dimsUniform(c, W, H, D);
    WGPUBuffer colMaxs = c.newBuffer(H * D * sizeof(float));
    WGPUBuffer rowMaxs = c.newBuffer(D * sizeof(float));

    gpuDispatch(c, "calculateColumnMaxs", {{0, colMaxs}, {1, volume}, {2, dims}}, wg2d(H, D));
    gpuDispatch(c, "calculateRowMaxs", {{0, rowMaxs}, {1, colMaxs}, {2, dims}}, wg1d(D));

    std::vector<float> rowData(D);
    c.readBuffer(rowMaxs, rowData.data(), D * sizeof(float));

    float mx = rowData[0];
    for (int i = 1; i < D; i++) mx = std::max(mx, rowData[i]);

    wgpuBufferRelease(dims);
    wgpuBufferRelease(colMaxs);
    wgpuBufferRelease(rowMaxs);
    return mx;
}

// ============================================================
//  3D Nonseparable Convolution
// ============================================================

static void nonseparableConvolution3D(WebGPUContext& c,
    WGPUBuffer resp1, WGPUBuffer resp2, WGPUBuffer resp3,
    WGPUBuffer volume,
    const float* fReal1, const float* fImag1,
    const float* fReal2, const float* fImag2,
    const float* fReal3, const float* fImag3,
    int W, int H, int D)
{
    WGPUBuffer dims = dimsUniform(c, W, H, D);
    WGPUBuffer f1r = c.newBuffer(343 * sizeof(float), fReal1);
    WGPUBuffer f1i = c.newBuffer(343 * sizeof(float), fImag1);
    WGPUBuffer f2r = c.newBuffer(343 * sizeof(float), fReal2);
    WGPUBuffer f2i = c.newBuffer(343 * sizeof(float), fImag2);
    WGPUBuffer f3r = c.newBuffer(343 * sizeof(float), fReal3);
    WGPUBuffer f3i = c.newBuffer(343 * sizeof(float), fImag3);

    gpuDispatch(c, "conv3D_Full", {
        {0, resp1}, {1, resp2}, {2, resp3}, {3, volume},
        {4, f1r}, {5, f1i}, {6, f2r}, {7, f2i}, {8, f3r}, {9, f3i},
        {10, dims},
    }, wg3d(W, H, D));

    wgpuBufferRelease(dims);
    wgpuBufferRelease(f1r); wgpuBufferRelease(f1i);
    wgpuBufferRelease(f2r); wgpuBufferRelease(f2i);
    wgpuBufferRelease(f3r); wgpuBufferRelease(f3i);
}

// ============================================================
//  Smoothing
// ============================================================

static void createSmoothingFilter(float* filter, float sigma) {
    float sum = 0;
    for (int i = 0; i < 9; i++) {
        float x = float(i) - 4.0f;
        filter[i] = expf(-0.5f * x * x / (sigma * sigma));
        sum += filter[i];
    }
    for (int i = 0; i < 9; i++) filter[i] /= sum;
}

static void performSmoothing(WebGPUContext& c, WGPUBuffer output, WGPUBuffer input,
                              int W, int H, int D, const float* smoothFilter) {
    WGPUBuffer dims = dimsUniform(c, W, H, D);
    WGPUBuffer filtBuf = c.newBuffer(9 * sizeof(float), smoothFilter);
    int vol = W * H * D;
    WGPUBuffer temp1 = c.newBuffer(vol * sizeof(float));
    WGPUBuffer temp2 = c.newBuffer(vol * sizeof(float));

    gpuDispatch(c, "separableConvRows",
        {{0, temp1}, {1, input}, {2, filtBuf}, {3, dims}}, wg3d(W, H, D));
    gpuDispatch(c, "separableConvColumns",
        {{0, temp2}, {1, temp1}, {2, filtBuf}, {3, dims}}, wg3d(W, H, D));
    gpuDispatch(c, "separableConvRods",
        {{0, output}, {1, temp2}, {2, filtBuf}, {3, dims}}, wg3d(W, H, D));

    wgpuBufferRelease(dims);
    wgpuBufferRelease(filtBuf);
    wgpuBufferRelease(temp1);
    wgpuBufferRelease(temp2);
}

static void performSmoothingInPlace(WebGPUContext& c, WGPUBuffer volume,
                                     int W, int H, int D, const float* smoothFilter) {
    int vol = W * H * D;
    WGPUBuffer output = c.newBuffer(vol * sizeof(float));
    performSmoothing(c, output, volume, W, H, D, smoothFilter);
    c.copyBuffer(output, volume, vol * sizeof(float));
    wgpuBufferRelease(output);
}

static void batchSmoothInPlace(WebGPUContext& c, const std::vector<WGPUBuffer>& volumes,
                                int W, int H, int D, const float* smoothFilter) {
    for (WGPUBuffer vol : volumes) {
        performSmoothingInPlace(c, vol, W, H, D, smoothFilter);
    }
}

// ============================================================
//  Volume operations
// ============================================================

static WGPUBuffer rescaleVolume(WebGPUContext& c, WGPUBuffer input,
                                 int srcW, int srcH, int srcD,
                                 int dstW, int dstH, int dstD,
                                 float scaleX, float scaleY, float scaleZ) {
    int vol = dstW * dstH * dstD;
    WGPUBuffer output = c.newBuffer(vol * sizeof(float));
    fillBuffer(c, output, 0.0f, vol);
    WGPUBuffer dims = dimsUniform(c, dstW, dstH, dstD);
    float scales[3] = {scaleX, scaleY, scaleZ};
    WGPUBuffer scalesBuf = c.newUniform(scales, sizeof(scales));
    WGPUBuffer srcDims = dimsUniform(c, srcW, srcH, srcD);

    gpuDispatch(c, "rescaleVolume", {
        {0, output}, {1, input}, {2, dims}, {3, scalesBuf}, {4, srcDims},
    }, wg3d(dstW, dstH, dstD));

    wgpuBufferRelease(dims);
    wgpuBufferRelease(scalesBuf);
    wgpuBufferRelease(srcDims);
    return output;
}

static WGPUBuffer copyVolumeToNew(WebGPUContext& c, WGPUBuffer src,
                                    int srcW, int srcH, int srcD,
                                    int dstW, int dstH, int dstD,
                                    int mmZCut, float voxelSizeZ) {
    int vol = dstW * dstH * dstD;
    WGPUBuffer dst = c.newBuffer(vol * sizeof(float));
    fillBuffer(c, dst, 0.0f, vol);

    struct CopyParams {
        int newW, newH, newD;
        int srcW, srcH, srcD;
        int xDiff, yDiff, zDiff;
        int mmZCut;
        float voxelSizeZ;
        int _pad;
    };

    CopyParams cp = {
        dstW, dstH, dstD,
        srcW, srcH, srcD,
        srcW - dstW, srcH - dstH, srcD - dstD,
        mmZCut, voxelSizeZ, 0
    };

    WGPUBuffer paramBuf = c.newUniform(&cp, sizeof(CopyParams));

    int dispW = std::max(srcW, dstW);
    int dispH = std::max(srcH, dstH);
    int dispD = std::max(srcD, dstD);

    gpuDispatch(c, "copyVolumeToNew",
        {{0, dst}, {1, src}, {2, paramBuf}}, wg3d(dispW, dispH, dispD));

    wgpuBufferRelease(paramBuf);
    return dst;
}

static WGPUBuffer changeVolumesResolutionAndSize(WebGPUContext& c, WGPUBuffer input,
    int srcW, int srcH, int srcD, VoxelSize srcVox,
    int dstW, int dstH, int dstD, VoxelSize dstVox, int mmZCut)
{
    float scaleX = srcVox.x / dstVox.x;
    float scaleY = srcVox.y / dstVox.y;
    float scaleZ = srcVox.z / dstVox.z;

    int interpW = (int)roundf(srcW * scaleX);
    int interpH = (int)roundf(srcH * scaleY);
    int interpD = (int)roundf(srcD * scaleZ);

    float voxDiffX = (float)(srcW - 1) / std::max(interpW - 1, 1);
    float voxDiffY = (float)(srcH - 1) / std::max(interpH - 1, 1);
    float voxDiffZ = (float)(srcD - 1) / std::max(interpD - 1, 1);

    WGPUBuffer interpolated = rescaleVolume(c, input, srcW, srcH, srcD,
                                             interpW, interpH, interpD,
                                             voxDiffX, voxDiffY, voxDiffZ);

    WGPUBuffer result = copyVolumeToNew(c, interpolated, interpW, interpH, interpD,
                                         dstW, dstH, dstD, mmZCut, dstVox.z);
    wgpuBufferRelease(interpolated);
    return result;
}

static WGPUBuffer changeVolumeSize(WebGPUContext& c, WGPUBuffer input,
                                     int srcW, int srcH, int srcD,
                                     int dstW, int dstH, int dstD) {
    float scaleX = (float)(srcW - 1) / std::max(dstW - 1, 1);
    float scaleY = (float)(srcH - 1) / std::max(dstH - 1, 1);
    float scaleZ = (float)(srcD - 1) / std::max(dstD - 1, 1);
    return rescaleVolume(c, input, srcW, srcH, srcD, dstW, dstH, dstD,
                         scaleX, scaleY, scaleZ);
}

// ============================================================
//  Interpolation
// ============================================================

static void interpolateLinear(WebGPUContext& c, WGPUBuffer output, WGPUBuffer volume,
                               const float* params, int W, int H, int D) {
    WGPUBuffer dims = dimsUniform(c, W, H, D);
    WGPUBuffer paramsBuf = c.newBuffer(12 * sizeof(float), params);

    // Handle buffer aliasing (output == volume)
    if (output == volume) {
        size_t bytes = (size_t)W * H * D * sizeof(float);
        WGPUBuffer tmp = c.newBuffer(bytes);
        c.copyBuffer(volume, tmp, bytes);
        gpuDispatch(c, "interpolateLinear", {
            {0, output}, {1, tmp}, {2, paramsBuf}, {3, dims},
        }, wg3d(W, H, D));
        wgpuBufferRelease(tmp);
    } else {
        gpuDispatch(c, "interpolateLinear", {
            {0, output}, {1, volume}, {2, paramsBuf}, {3, dims},
        }, wg3d(W, H, D));
    }
    wgpuBufferRelease(dims);
    wgpuBufferRelease(paramsBuf);
}

static void interpolateNonLinear(WebGPUContext& c, WGPUBuffer output, WGPUBuffer volume,
                                  WGPUBuffer dispX, WGPUBuffer dispY, WGPUBuffer dispZ,
                                  int W, int H, int D) {
    WGPUBuffer dims = dimsUniform(c, W, H, D);

    // Handle buffer aliasing
    if (output == volume) {
        size_t bytes = (size_t)W * H * D * sizeof(float);
        WGPUBuffer tmp = c.newBuffer(bytes);
        c.copyBuffer(volume, tmp, bytes);
        gpuDispatch(c, "interpolateNonLinear", {
            {0, output}, {1, tmp}, {2, dispX}, {3, dispY}, {4, dispZ}, {5, dims},
        }, wg3d(W, H, D));
        wgpuBufferRelease(tmp);
    } else {
        gpuDispatch(c, "interpolateNonLinear", {
            {0, output}, {1, volume}, {2, dispX}, {3, dispY}, {4, dispZ}, {5, dims},
        }, wg3d(W, H, D));
    }
    wgpuBufferRelease(dims);
}

static void addLinearNonLinearDisplacement(WebGPUContext& c,
    WGPUBuffer dispX, WGPUBuffer dispY, WGPUBuffer dispZ,
    const float* params, int W, int H, int D)
{
    WGPUBuffer dims = dimsUniform(c, W, H, D);
    WGPUBuffer paramsBuf = c.newBuffer(12 * sizeof(float), params);
    gpuDispatch(c, "addLinearNonLinearDisp", {
        {0, dispX}, {1, dispY}, {2, dispZ}, {3, paramsBuf}, {4, dims},
    }, wg3d(W, H, D));
    wgpuBufferRelease(dims);
    wgpuBufferRelease(paramsBuf);
}

// ============================================================
//  CPU helpers
// ============================================================

static void centerOfMass(const float* vol, int W, int H, int D,
                          float& cx, float& cy, float& cz) {
    double sum = 0, sx = 0, sy = 0, sz = 0;
    for (int z = 0; z < D; z++)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                float v = vol[x + y * W + z * W * H];
                if (v > 0) {
                    sum += v;
                    sx += v * x;
                    sy += v * y;
                    sz += v * z;
                }
            }
    if (sum > 0) {
        cx = (float)(sx / sum);
        cy = (float)(sy / sum);
        cz = (float)(sz / sum);
    } else {
        cx = W * 0.5f;
        cy = H * 0.5f;
        cz = D * 0.5f;
    }
}

static void solveEquationSystem(float* A, float* h, double* params, int n) {
    // Convert to double for precision
    std::vector<double> Ad(n * n), hd(n);
    for (int i = 0; i < n * n; i++) Ad[i] = A[i];
    for (int i = 0; i < n; i++) hd[i] = h[i];

    // Mirror symmetric matrix
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            Ad[i * n + j] = Ad[j * n + i];

    // Augmented matrix [A | h]
    std::vector<std::vector<double>> aug(n, std::vector<double>(n + 1));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            aug[i][j] = Ad[j * n + i];  // column-major to row-major
        aug[i][n] = hd[i];
    }

    // Forward elimination with partial pivoting
    for (int col = 0; col < n; col++) {
        int pivotRow = col;
        double pivotVal = fabs(aug[col][col]);
        for (int row = col + 1; row < n; row++) {
            if (fabs(aug[row][col]) > pivotVal) {
                pivotVal = fabs(aug[row][col]);
                pivotRow = row;
            }
        }
        if (pivotVal < 1e-30) {
            for (int i = 0; i < n; i++) params[i] = 0;
            return;
        }
        if (pivotRow != col) {
            std::swap(aug[col], aug[pivotRow]);
        }
        for (int row = col + 1; row < n; row++) {
            double factor = aug[row][col] / aug[col][col];
            for (int j = col; j <= n; j++)
                aug[row][j] -= factor * aug[col][j];
        }
    }

    // Back substitution
    for (int row = n - 1; row >= 0; row--) {
        double s = aug[row][n];
        for (int j = row + 1; j < n; j++)
            s -= aug[row][j] * params[j];
        params[row] = s / aug[row][row];
    }
}

// ============================================================
//  Affine parameter composition
// ============================================================

static void paramsToMatrix(const float* p, double M[4][4], float translationScale = 1.0f) {
    M[0][0] = p[3] + 1.0; M[0][1] = p[4];       M[0][2] = p[5];       M[0][3] = p[0] * translationScale;
    M[1][0] = p[6];        M[1][1] = p[7] + 1.0; M[1][2] = p[8];       M[1][3] = p[1] * translationScale;
    M[2][0] = p[9];        M[2][1] = p[10];      M[2][2] = p[11] + 1.0; M[2][3] = p[2] * translationScale;
    M[3][0] = 0;           M[3][1] = 0;          M[3][2] = 0;           M[3][3] = 1.0;
}

static void matrixToParams(const double M[4][4], float* p) {
    p[0]  = (float)M[0][3];
    p[1]  = (float)M[1][3];
    p[2]  = (float)M[2][3];
    p[3]  = (float)(M[0][0] - 1.0);
    p[4]  = (float)M[0][1];
    p[5]  = (float)M[0][2];
    p[6]  = (float)M[1][0];
    p[7]  = (float)(M[1][1] - 1.0);
    p[8]  = (float)M[1][2];
    p[9]  = (float)M[2][0];
    p[10] = (float)M[2][1];
    p[11] = (float)(M[2][2] - 1.0);
}

static void matMul4x4(const double A[4][4], const double B[4][4], double C[4][4]) {
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            C[i][j] = 0;
            for (int k = 0; k < 4; k++)
                C[i][j] += A[i][k] * B[k][j];
        }
}

static void composeAffineParams(float* oldParams, const float* newParams,
                                 float translationScale = 1.0f) {
    double O[4][4], N[4][4], T[4][4];
    paramsToMatrix(oldParams, O, translationScale);
    paramsToMatrix(newParams, N, translationScale);
    matMul4x4(N, O, T);
    matrixToParams(T, oldParams);
}

static void composeAffineParamsNextScale(float* oldParams, const float* newParams) {
    double O[4][4], N[4][4], T[4][4];
    paramsToMatrix(oldParams, O, 2.0f);
    paramsToMatrix(newParams, N, 2.0f);
    matMul4x4(N, O, T);
    matrixToParams(T, oldParams);
}

// ============================================================
//  Linear Registration
// ============================================================

static void alignTwoVolumesLinear(WebGPUContext& c,
    WGPUBuffer alignedVolume, WGPUBuffer referenceVolume,
    const QuadratureFilters& filters,
    int W, int H, int D, int filterSize, int numIterations,
    float* registrationParams, bool verbose)
{
    int vol = W * H * D;

    // Save original aligned volume
    WGPUBuffer originalAligned = c.newBuffer(vol * sizeof(float));
    c.copyBuffer(alignedVolume, originalAligned, vol * sizeof(float));

    // Reset params
    for (int i = 0; i < 12; i++) registrationParams[i] = 0.0f;

    // Allocate filter response buffers (complex = vec2)
    WGPUBuffer q11 = c.newBuffer(vol * 8);
    WGPUBuffer q12 = c.newBuffer(vol * 8);
    WGPUBuffer q13 = c.newBuffer(vol * 8);
    WGPUBuffer q21 = c.newBuffer(vol * 8);
    WGPUBuffer q22 = c.newBuffer(vol * 8);
    WGPUBuffer q23 = c.newBuffer(vol * 8);

    // Phase/certainty buffers
    WGPUBuffer phaseDiff = c.newBuffer(vol * sizeof(float));
    WGPUBuffer certainties = c.newBuffer(vol * sizeof(float));
    WGPUBuffer phaseGrad = c.newBuffer(vol * sizeof(float));

    // A-matrix / h-vector buffers
    int HD = H * D;
    WGPUBuffer A2D = c.newBuffer(30 * HD * sizeof(float));
    WGPUBuffer A1D = c.newBuffer(30 * D * sizeof(float));
    WGPUBuffer Amat = c.newBuffer(144 * sizeof(float));
    WGPUBuffer h2D = c.newBuffer(12 * HD * sizeof(float));
    WGPUBuffer h1D = c.newBuffer(12 * D * sizeof(float));

    // Filter reference volume once
    nonseparableConvolution3D(c, q11, q12, q13, referenceVolume,
        filters.linearReal[0].data(), filters.linearImag[0].data(),
        filters.linearReal[1].data(), filters.linearImag[1].data(),
        filters.linearReal[2].data(), filters.linearImag[2].data(),
        W, H, D);

    const char* gradKernels[3] = {"phaseGradX", "phaseGradY", "phaseGradZ"};
    int dirOffsets[3] = {0, 10, 20};
    int hOffsets[3] = {0, 1, 2};
    WGPUBuffer q1Bufs[3] = {q11, q12, q13};
    WGPUBuffer q2Bufs[3] = {q21, q22, q23};

    for (int iter = 0; iter < numIterations; iter++) {
        // Filter aligned volume
        nonseparableConvolution3D(c, q21, q22, q23, alignedVolume,
            filters.linearReal[0].data(), filters.linearImag[0].data(),
            filters.linearReal[1].data(), filters.linearImag[1].data(),
            filters.linearReal[2].data(), filters.linearImag[2].data(),
            W, H, D);

        // Zero intermediate buffers
        fillBuffer(c, A2D, 0.0f, 30 * HD);
        fillBuffer(c, h2D, 0.0f, 12 * HD);

        WGPUBuffer dims = dimsUniform(c, W, H, D);

        // Process each direction
        for (int d = 0; d < 3; d++) {
            // Phase differences + certainties
            gpuDispatch(c, "phaseDiffCert", {
                {0, phaseDiff}, {1, certainties}, {2, q1Bufs[d]}, {3, q2Bufs[d]}, {4, dims},
            }, wg3d(W, H, D));

            // Phase gradients
            gpuDispatch(c, gradKernels[d], {
                {0, phaseGrad}, {1, q1Bufs[d]}, {2, q2Bufs[d]}, {3, dims},
            }, wg3d(W, H, D));

            // A-matrix and h-vector 2D
            struct AParams { int W, H, D, filterSize, dirOff, hOff; };
            AParams ap = {W, H, D, filterSize, dirOffsets[d], hOffsets[d]};
            WGPUBuffer apBuf = c.newUniform(&ap, sizeof(AParams));
            gpuDispatch(c, "amatrixHvector2D", {
                {0, A2D}, {1, h2D}, {2, phaseDiff}, {3, phaseGrad},
                {4, certainties}, {5, apBuf},
            }, wg2d(H, D));
            wgpuBufferRelease(apBuf);
        }
        wgpuBufferRelease(dims);

        // Reduce A-matrix: 2D -> 1D -> Final
        int32_t hdParams[4] = {H, D, filterSize, 0};
        WGPUBuffer hdParamBuf = c.newUniform(hdParams, sizeof(hdParams));
        gpuDispatch(c, "amatrix1D", {{0, A1D}, {1, A2D}, {2, hdParamBuf}}, wg2d(D, 30));
        gpuDispatch(c, "hvector1D", {{0, h1D}, {1, h2D}, {2, hdParamBuf}}, wg2d(D, 12));

        fillBuffer(c, Amat, 0.0f, 144);
        int32_t dfParams[4] = {D, filterSize, 0, 0};
        WGPUBuffer dfParamBuf = c.newUniform(dfParams, sizeof(dfParams));
        gpuDispatch(c, "amatrixFinal", {{0, Amat}, {1, A1D}, {2, dfParamBuf}}, {1, 1, 1});

        // Read back A-matrix
        float hA[144];
        c.readBuffer(Amat, hA, 144 * sizeof(float));

        // h-vector final reduction on CPU (matches Metal backend)
        std::vector<float> h1DData(12 * D);
        c.readBuffer(h1D, h1DData.data(), 12 * D * sizeof(float));
        int fhalf = (filterSize - 1) / 2;
        float hh[12];
        for (int elem = 0; elem < 12; elem++) {
            float sum = 0;
            for (int z = fhalf; z < D - fhalf; z++) {
                sum += h1DData[elem * D + z];
            }
            hh[elem] = sum;
        }

        double paramsDbl[12];
        solveEquationSystem(hA, hh, paramsDbl, 12);

        float deltaParams[12];
        for (int i = 0; i < 12; i++) deltaParams[i] = (float)paramsDbl[i];
        composeAffineParams(registrationParams, deltaParams);

        // Apply affine transform from original volume
        interpolateLinear(c, alignedVolume, originalAligned, registrationParams, W, H, D);

        wgpuBufferRelease(hdParamBuf);
        wgpuBufferRelease(dfParamBuf);
    }

    // Release buffers
    wgpuBufferRelease(originalAligned);
    wgpuBufferRelease(q11); wgpuBufferRelease(q12); wgpuBufferRelease(q13);
    wgpuBufferRelease(q21); wgpuBufferRelease(q22); wgpuBufferRelease(q23);
    wgpuBufferRelease(phaseDiff); wgpuBufferRelease(certainties); wgpuBufferRelease(phaseGrad);
    wgpuBufferRelease(A2D); wgpuBufferRelease(A1D); wgpuBufferRelease(Amat);
    wgpuBufferRelease(h2D); wgpuBufferRelease(h1D);
}

// Multi-scale linear registration
static void alignTwoVolumesLinearSeveralScales(WebGPUContext& c,
    WGPUBuffer& alignedVolume, WGPUBuffer referenceVolume,
    const QuadratureFilters& filters,
    int W, int H, int D, int filterSize, int numIterations,
    int coarsestScale, float* registrationParams, bool verbose)
{
    int vol = W * H * D;

    for (int i = 0; i < 12; i++) registrationParams[i] = 0.0f;

    WGPUBuffer originalAligned = c.newBuffer(vol * sizeof(float));
    c.copyBuffer(alignedVolume, originalAligned, vol * sizeof(float));

    for (int scale = coarsestScale; scale >= 1; scale /= 2) {
        int sW = (int)roundf((float)W / (float)scale);
        int sH = (int)roundf((float)H / (float)scale);
        int sD = (int)roundf((float)D / (float)scale);

        if (sW < 8 || sH < 8 || sD < 8) continue;

        // Downscale from originals
        WGPUBuffer scaledRef = nullptr;
        WGPUBuffer scaledAligned = nullptr;
        if (scale == 1) {
            scaledRef = referenceVolume;
            scaledAligned = alignedVolume;
        } else {
            scaledRef = changeVolumeSize(c, referenceVolume, W, H, D, sW, sH, sD);
            scaledAligned = changeVolumeSize(c, originalAligned, W, H, D, sW, sH, sD);
        }

        // Pre-transform with accumulated params (non-coarsest)
        if (scale < coarsestScale) {
            interpolateLinear(c, scaledAligned, scaledAligned, registrationParams, sW, sH, sD);
        }

        int iters = (scale == 1) ? (int)ceilf((float)numIterations / 5.0f) : numIterations;

        if (verbose) {
            printf("  Linear scale %d: %dx%dx%d, %d iterations\n", scale, sW, sH, sD, iters);
        }

        float tempParams[12] = {0};
        alignTwoVolumesLinear(c, scaledAligned, scaledRef, filters,
                              sW, sH, sD, filterSize, iters, tempParams, verbose);

        // Compose
        if (scale != 1) {
            composeAffineParamsNextScale(registrationParams, tempParams);
            wgpuBufferRelease(scaledRef);
            wgpuBufferRelease(scaledAligned);
        } else {
            composeAffineParams(registrationParams, tempParams);
        }
    }

    // Final transform at full resolution
    interpolateLinear(c, alignedVolume, originalAligned, registrationParams, W, H, D);
    wgpuBufferRelease(originalAligned);
}

// ============================================================
//  Nonlinear Registration
// ============================================================

static void alignTwoVolumesNonLinear(WebGPUContext& c,
    WGPUBuffer alignedVolume, WGPUBuffer referenceVolume,
    const QuadratureFilters& filters,
    int W, int H, int D, int numIterations,
    WGPUBuffer updateDispX, WGPUBuffer updateDispY, WGPUBuffer updateDispZ,
    bool verbose)
{
    int vol = W * H * D;

    // Allocate filter response buffers (6 filters x 2 volumes)
    WGPUBuffer q1[6], q2[6];
    for (int i = 0; i < 6; i++) {
        q1[i] = c.newBuffer(vol * 8);
        q2[i] = c.newBuffer(vol * 8);
    }

    // Tensor components
    WGPUBuffer t11 = c.newBuffer(vol * sizeof(float));
    WGPUBuffer t12 = c.newBuffer(vol * sizeof(float));
    WGPUBuffer t13 = c.newBuffer(vol * sizeof(float));
    WGPUBuffer t22 = c.newBuffer(vol * sizeof(float));
    WGPUBuffer t23 = c.newBuffer(vol * sizeof(float));
    WGPUBuffer t33 = c.newBuffer(vol * sizeof(float));

    // A-matrix and h-vector
    WGPUBuffer a11 = c.newBuffer(vol * sizeof(float));
    WGPUBuffer a12 = c.newBuffer(vol * sizeof(float));
    WGPUBuffer a13 = c.newBuffer(vol * sizeof(float));
    WGPUBuffer a22 = c.newBuffer(vol * sizeof(float));
    WGPUBuffer a23 = c.newBuffer(vol * sizeof(float));
    WGPUBuffer a33 = c.newBuffer(vol * sizeof(float));
    WGPUBuffer h1 = c.newBuffer(vol * sizeof(float));
    WGPUBuffer h2 = c.newBuffer(vol * sizeof(float));
    WGPUBuffer h3 = c.newBuffer(vol * sizeof(float));

    WGPUBuffer tensorNorms = c.newBuffer(vol * sizeof(float));
    WGPUBuffer dux = c.newBuffer(vol * sizeof(float));
    WGPUBuffer duy = c.newBuffer(vol * sizeof(float));
    WGPUBuffer duz = c.newBuffer(vol * sizeof(float));

    WGPUBuffer originalAligned = c.newBuffer(vol * sizeof(float));
    c.copyBuffer(alignedVolume, originalAligned, vol * sizeof(float));

    // Filter direction buffers
    WGPUBuffer fdxBuf = c.newBuffer(6 * sizeof(float), filters.filterDirectionsX);
    WGPUBuffer fdyBuf = c.newBuffer(6 * sizeof(float), filters.filterDirectionsY);
    WGPUBuffer fdzBuf = c.newBuffer(6 * sizeof(float), filters.filterDirectionsZ);

    float smoothTensor[9], smoothEq[9], smoothDisp[9];
    createSmoothingFilter(smoothTensor, 1.0f);
    createSmoothingFilter(smoothEq, 2.0f);
    createSmoothingFilter(smoothDisp, 2.0f);

    // Filter reference volume (once, 2 batches of 3)
    nonseparableConvolution3D(c, q1[0], q1[1], q1[2], referenceVolume,
        filters.nonlinearReal[0].data(), filters.nonlinearImag[0].data(),
        filters.nonlinearReal[1].data(), filters.nonlinearImag[1].data(),
        filters.nonlinearReal[2].data(), filters.nonlinearImag[2].data(),
        W, H, D);
    nonseparableConvolution3D(c, q1[3], q1[4], q1[5], referenceVolume,
        filters.nonlinearReal[3].data(), filters.nonlinearImag[3].data(),
        filters.nonlinearReal[4].data(), filters.nonlinearImag[4].data(),
        filters.nonlinearReal[5].data(), filters.nonlinearImag[5].data(),
        W, H, D);

    for (int iter = 0; iter < numIterations; iter++) {
        if (verbose) printf("    Nonlinear iter %d/%d\n", iter + 1, numIterations);

        // Filter aligned volume
        nonseparableConvolution3D(c, q2[0], q2[1], q2[2], alignedVolume,
            filters.nonlinearReal[0].data(), filters.nonlinearImag[0].data(),
            filters.nonlinearReal[1].data(), filters.nonlinearImag[1].data(),
            filters.nonlinearReal[2].data(), filters.nonlinearImag[2].data(),
            W, H, D);
        nonseparableConvolution3D(c, q2[3], q2[4], q2[5], alignedVolume,
            filters.nonlinearReal[3].data(), filters.nonlinearImag[3].data(),
            filters.nonlinearReal[4].data(), filters.nonlinearImag[4].data(),
            filters.nonlinearReal[5].data(), filters.nonlinearImag[5].data(),
            W, H, D);

        // Zero tensors and displacement update
        for (auto buf : {t11, t12, t13, t22, t23, t33, dux, duy, duz})
            fillBuffer(c, buf, 0.0f, vol);

        // Compute tensor components (6 filters)
        WGPUBuffer dims = dimsUniform(c, W, H, D);
        for (int f = 0; f < 6; f++) {
            const float* pt = filters.projectionTensors[f];
            float tp[6] = {pt[0], pt[1], pt[2], pt[3], pt[4], pt[5]};
            WGPUBuffer tpBuf = c.newUniform(tp, sizeof(tp));
            gpuDispatch(c, "tensorComponents", {
                {0, t11}, {1, t12}, {2, t13}, {3, t22}, {4, t23}, {5, t33},
                {6, q2[f]}, {7, dims}, {8, tpBuf},
            }, wg3d(W, H, D));
            wgpuBufferRelease(tpBuf);
        }

        // Tensor norms (pre-smooth)
        gpuDispatch(c, "tensorNorms", {
            {0, tensorNorms}, {1, t11}, {2, t12}, {3, t13},
            {4, t22}, {5, t23}, {6, t33}, {7, dims},
        }, wg3d(W, H, D));

        // Smooth tensors
        batchSmoothInPlace(c, {t11, t12, t13, t22, t23, t33}, W, H, D, smoothTensor);

        // Tensor norms (post-smooth) + normalize
        WGPUBuffer dims2 = dimsUniform(c, W, H, D);
        gpuDispatch(c, "tensorNorms", {
            {0, tensorNorms}, {1, t11}, {2, t12}, {3, t13},
            {4, t22}, {5, t23}, {6, t33}, {7, dims2},
        }, wg3d(W, H, D));
        wgpuBufferRelease(dims2);

        float maxNorm = calculateMax(c, tensorNorms, W, H, D);
        if (maxNorm > 0) {
            float invMax = 1.0f / maxNorm;
            for (auto buf : {t11, t12, t13, t22, t23, t33})
                multiplyVolume(c, buf, invMax, vol);
        }

        // A-matrices and h-vectors (6 filters)
        for (int f = 0; f < 6; f++) {
            struct MorphonParams { int W, H, D, FILTER; };
            MorphonParams mp = {W, H, D, f};
            WGPUBuffer mpBuf = c.newUniform(&mp, sizeof(MorphonParams));
            gpuDispatch(c, "amatricesHvectors", {
                {0, a11}, {1, a12}, {2, a13}, {3, a22}, {4, a23}, {5, a33},
                {6, h1}, {7, h2}, {8, h3},
                {9, q1[f]}, {10, q2[f]},
                {11, t11}, {12, t12}, {13, t13}, {14, t22}, {15, t23}, {16, t33},
                {17, fdxBuf}, {18, fdyBuf}, {19, fdzBuf},
                {20, mpBuf},
            }, wg3d(W, H, D));
            wgpuBufferRelease(mpBuf);
        }

        // Smooth A-matrix and h-vector
        batchSmoothInPlace(c, {a11, a12, a13, a22, a23, a33, h1, h2, h3}, W, H, D, smoothEq);

        // Displacement update
        WGPUBuffer dims3 = dimsUniform(c, W, H, D);
        gpuDispatch(c, "displacementUpdate", {
            {0, dux}, {1, duy}, {2, duz},
            {3, a11}, {4, a12}, {5, a13}, {6, a22}, {7, a23}, {8, a33},
            {9, h1}, {10, h2}, {11, h3}, {12, dims3},
        }, wg3d(W, H, D));
        wgpuBufferRelease(dims3);

        // Smooth displacement
        batchSmoothInPlace(c, {dux, duy, duz}, W, H, D, smoothDisp);

        // Accumulate displacement
        addVolumes(c, updateDispX, dux, vol);
        addVolumes(c, updateDispY, duy, vol);
        addVolumes(c, updateDispZ, duz, vol);

        // Interpolate from original with accumulated displacement
        interpolateNonLinear(c, alignedVolume, originalAligned,
                             updateDispX, updateDispY, updateDispZ, W, H, D);

        wgpuBufferRelease(dims);
    }

    // Release buffers
    for (int i = 0; i < 6; i++) {
        wgpuBufferRelease(q1[i]);
        wgpuBufferRelease(q2[i]);
    }
    wgpuBufferRelease(t11); wgpuBufferRelease(t12); wgpuBufferRelease(t13);
    wgpuBufferRelease(t22); wgpuBufferRelease(t23); wgpuBufferRelease(t33);
    wgpuBufferRelease(a11); wgpuBufferRelease(a12); wgpuBufferRelease(a13);
    wgpuBufferRelease(a22); wgpuBufferRelease(a23); wgpuBufferRelease(a33);
    wgpuBufferRelease(h1); wgpuBufferRelease(h2); wgpuBufferRelease(h3);
    wgpuBufferRelease(tensorNorms);
    wgpuBufferRelease(dux); wgpuBufferRelease(duy); wgpuBufferRelease(duz);
    wgpuBufferRelease(originalAligned);
    wgpuBufferRelease(fdxBuf); wgpuBufferRelease(fdyBuf); wgpuBufferRelease(fdzBuf);
}

// Multi-scale nonlinear registration
static void alignTwoVolumesNonLinearSeveralScales(WebGPUContext& c,
    WGPUBuffer& alignedVolume, WGPUBuffer referenceVolume,
    const QuadratureFilters& filters,
    int W, int H, int D, int numIterations, int coarsestScale,
    WGPUBuffer& totalDispX, WGPUBuffer& totalDispY, WGPUBuffer& totalDispZ,
    bool verbose)
{
    int vol = W * H * D;

    WGPUBuffer originalAligned = c.newBuffer(vol * sizeof(float));
    c.copyBuffer(alignedVolume, originalAligned, vol * sizeof(float));

    totalDispX = c.newBuffer(vol * sizeof(float));
    totalDispY = c.newBuffer(vol * sizeof(float));
    totalDispZ = c.newBuffer(vol * sizeof(float));
    fillBuffer(c, totalDispX, 0.0f, vol);
    fillBuffer(c, totalDispY, 0.0f, vol);
    fillBuffer(c, totalDispZ, 0.0f, vol);

    for (int scale = coarsestScale; scale >= 1; scale /= 2) {
        int sW = W / scale;
        int sH = H / scale;
        int sD = D / scale;

        if (sW < 8 || sH < 8 || sD < 8) continue;

        if (verbose) printf("  Nonlinear scale %d: %dx%dx%d\n", scale, sW, sH, sD);

        WGPUBuffer scaledRef = nullptr;
        WGPUBuffer scaledAligned = nullptr;
        if (scale == 1) {
            scaledRef = referenceVolume;
            scaledAligned = alignedVolume;
        } else {
            scaledRef = changeVolumeSize(c, referenceVolume, W, H, D, sW, sH, sD);
            scaledAligned = changeVolumeSize(c, alignedVolume, W, H, D, sW, sH, sD);
        }

        int sVol = sW * sH * sD;
        WGPUBuffer updateX = c.newBuffer(sVol * sizeof(float));
        WGPUBuffer updateY = c.newBuffer(sVol * sizeof(float));
        WGPUBuffer updateZ = c.newBuffer(sVol * sizeof(float));
        fillBuffer(c, updateX, 0.0f, sVol);
        fillBuffer(c, updateY, 0.0f, sVol);
        fillBuffer(c, updateZ, 0.0f, sVol);

        alignTwoVolumesNonLinear(c, scaledAligned, scaledRef, filters,
                                  sW, sH, sD, numIterations,
                                  updateX, updateY, updateZ, verbose);

        if (scale > 1) {
            WGPUBuffer rescX = changeVolumeSize(c, updateX, sW, sH, sD, W, H, D);
            WGPUBuffer rescY = changeVolumeSize(c, updateY, sW, sH, sD, W, H, D);
            WGPUBuffer rescZ = changeVolumeSize(c, updateZ, sW, sH, sD, W, H, D);
            multiplyVolume(c, rescX, (float)scale, vol);
            multiplyVolume(c, rescY, (float)scale, vol);
            multiplyVolume(c, rescZ, (float)scale, vol);
            addVolumes(c, totalDispX, rescX, vol);
            addVolumes(c, totalDispY, rescY, vol);
            addVolumes(c, totalDispZ, rescZ, vol);
            wgpuBufferRelease(rescX);
            wgpuBufferRelease(rescY);
            wgpuBufferRelease(rescZ);

            interpolateNonLinear(c, alignedVolume, originalAligned,
                                 totalDispX, totalDispY, totalDispZ, W, H, D);

            wgpuBufferRelease(scaledRef);
            wgpuBufferRelease(scaledAligned);
        } else {
            addVolumes(c, totalDispX, updateX, vol);
            addVolumes(c, totalDispY, updateY, vol);
            addVolumes(c, totalDispZ, updateZ, vol);
        }

        wgpuBufferRelease(updateX);
        wgpuBufferRelease(updateY);
        wgpuBufferRelease(updateZ);
    }

    wgpuBufferRelease(originalAligned);
}

} // anonymous namespace

// ============================================================
//  PUBLIC API
// ============================================================

T1MNIResult registerT1MNI(
    const float* t1Data, VolumeDims t1Dims, VoxelSize t1Vox,
    const float* mniData, VolumeDims mniDims, VoxelSize mniVox,
    const float* mniBrainData,
    const float* mniMaskData,
    const QuadratureFilters& filters,
    int linearIterations, int nonlinearIterations,
    int coarsestScale, int mmZCut, bool verbose)
{
    auto& c = WebGPUContext::get();

    if (verbose) printf("registerT1MNI: T1 %dx%dx%d -> MNI %dx%dx%d\n",
                        t1Dims.W, t1Dims.H, t1Dims.D,
                        mniDims.W, mniDims.H, mniDims.D);

    int mniVol = mniDims.size();

    // Upload volumes
    WGPUBuffer mniBuf = c.newBuffer(mniVol * sizeof(float), mniData);
    WGPUBuffer mniBrainBuf = c.newBuffer(mniVol * sizeof(float), mniBrainData);
    WGPUBuffer mniMaskBuf = c.newBuffer(mniVol * sizeof(float), mniMaskData);
    WGPUBuffer t1Buf = c.newBuffer(t1Dims.size() * sizeof(float), t1Data);

    // Resample T1 to MNI resolution and size
    WGPUBuffer t1InMNI = changeVolumesResolutionAndSize(c, t1Buf,
        t1Dims.W, t1Dims.H, t1Dims.D, t1Vox,
        mniDims.W, mniDims.H, mniDims.D, mniVox, mmZCut);

    // Center-of-mass alignment
    std::vector<float> mniCpu(mniVol), t1Cpu(mniVol);
    c.readBuffer(mniBuf, mniCpu.data(), mniVol * sizeof(float));
    c.readBuffer(t1InMNI, t1Cpu.data(), mniVol * sizeof(float));

    float cx1, cy1, cz1, cx2, cy2, cz2;
    centerOfMass(mniCpu.data(), mniDims.W, mniDims.H, mniDims.D, cx1, cy1, cz1);
    centerOfMass(t1Cpu.data(), mniDims.W, mniDims.H, mniDims.D, cx2, cy2, cz2);

    float initParams[12] = {0};
    // Round to integers to avoid interpolation blur (matches OpenCL's myround)
    initParams[0] = roundf(cx2 - cx1);
    initParams[1] = roundf(cy2 - cy1);
    initParams[2] = roundf(cz2 - cz1);

    interpolateLinear(c, t1InMNI, t1InMNI, initParams, mniDims.W, mniDims.H, mniDims.D);

    // Save interpolated volume after center-of-mass alignment
    std::vector<float> interpResult(mniVol);
    c.readBuffer(t1InMNI, interpResult.data(), mniVol * sizeof(float));

    // Linear registration
    float regParams[12] = {0};

    if (verbose) printf("Running linear registration (%d iterations)...\n", linearIterations);

    alignTwoVolumesLinearSeveralScales(c, t1InMNI, mniBuf, filters,
        mniDims.W, mniDims.H, mniDims.D,
        7, linearIterations, coarsestScale,
        regParams, verbose);

    // Compose COM shift into registration params (matches OpenCL line 8042)
    composeAffineParams(regParams, initParams);

    // Save linear result — re-rescale original T1 and apply combined
    // COM + affine in a single interpolation to avoid compounded blur
    // (matches OpenCL's "do total interpolation in one step" pattern)
    T1MNIResult result;
    {
        WGPUBuffer freshT1 = changeVolumesResolutionAndSize(c, t1Buf,
            t1Dims.W, t1Dims.H, t1Dims.D, t1Vox,
            mniDims.W, mniDims.H, mniDims.D, mniVox, mmZCut);
        interpolateLinear(c, freshT1, freshT1, regParams, mniDims.W, mniDims.H, mniDims.D);
        result.alignedLinear.resize(mniVol);
        c.readBuffer(freshT1, result.alignedLinear.data(), mniVol * sizeof(float));
        wgpuBufferRelease(freshT1);
    }
    result.params.resize(12);
    for (int i = 0; i < 12; i++) result.params[i] = regParams[i];

    // Nonlinear registration
    if (nonlinearIterations > 0) {
        if (verbose) printf("Running nonlinear registration (%d iterations)...\n", nonlinearIterations);

        WGPUBuffer totalDispX = nullptr, totalDispY = nullptr, totalDispZ = nullptr;

        alignTwoVolumesNonLinearSeveralScales(c, t1InMNI, mniBuf, filters,
            mniDims.W, mniDims.H, mniDims.D,
            nonlinearIterations, coarsestScale,
            totalDispX, totalDispY, totalDispZ, verbose);

        // Combine linear + nonlinear displacement into single field
        addLinearNonLinearDisplacement(c, totalDispX, totalDispY, totalDispZ,
                                       regParams, mniDims.W, mniDims.H, mniDims.D);

        // Copy displacement fields
        result.dispX.resize(mniVol);
        result.dispY.resize(mniVol);
        result.dispZ.resize(mniVol);
        c.readBuffer(totalDispX, result.dispX.data(), mniVol * sizeof(float));
        c.readBuffer(totalDispY, result.dispY.data(), mniVol * sizeof(float));
        c.readBuffer(totalDispZ, result.dispZ.data(), mniVol * sizeof(float));

        // Re-rescale original T1 and apply combined displacement in one step
        // (rescale + single interpolation = 2 passes, not 4)
        {
            WGPUBuffer freshT1 = changeVolumesResolutionAndSize(c, t1Buf,
                t1Dims.W, t1Dims.H, t1Dims.D, t1Vox,
                mniDims.W, mniDims.H, mniDims.D, mniVox, mmZCut);
            interpolateNonLinear(c, freshT1, freshT1, totalDispX, totalDispY, totalDispZ,
                                  mniDims.W, mniDims.H, mniDims.D);
            result.alignedNonLinear.resize(mniVol);
            c.readBuffer(freshT1, result.alignedNonLinear.data(), mniVol * sizeof(float));

            // Skullstrip from the single-step result
            multiplyVolumes(c, freshT1, mniMaskBuf, mniVol);
            result.skullstripped.resize(mniVol);
            c.readBuffer(freshT1, result.skullstripped.data(), mniVol * sizeof(float));
            wgpuBufferRelease(freshT1);
        }

        wgpuBufferRelease(totalDispX);
        wgpuBufferRelease(totalDispY);
        wgpuBufferRelease(totalDispZ);
    } else {
        result.alignedNonLinear = result.alignedLinear;
        result.skullstripped.resize(mniVol, 0);
        result.dispX.resize(mniVol, 0);
        result.dispY.resize(mniVol, 0);
        result.dispZ.resize(mniVol, 0);
    }

    result.interpolated = std::move(interpResult);

    // Release upload buffers
    wgpuBufferRelease(mniBuf);
    wgpuBufferRelease(mniBrainBuf);
    wgpuBufferRelease(mniMaskBuf);
    wgpuBufferRelease(t1Buf);
    wgpuBufferRelease(t1InMNI);

    return result;
}

} // namespace webgpu_reg
