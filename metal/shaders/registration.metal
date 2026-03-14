// Metal compute shaders for BROCCOLI image registration
// Ported from OpenCL kernels in code/Kernels/kernelRegistration.cpp
// and code/Kernels/kernelConvolution.cpp

#include <metal_stdlib>
using namespace metal;

// ============================================================
//  Helpers
// ============================================================

inline int idx3(int x, int y, int z, int W, int H) {
    return x + y * W + z * W * H;
}

inline int idx4(int x, int y, int z, int t, int W, int H, int D) {
    return x + y * W + z * W * H + t * W * H * D;
}

// Struct for passing volume dimensions
struct Dims {
    int W;
    int H;
    int D;
};

// ============================================================
//  Utility kernels
// ============================================================

kernel void fillFloat(device float* buf [[buffer(0)]],
                      constant float& value [[buffer(1)]],
                      uint gid [[thread_position_in_grid]]) {
    buf[gid] = value;
}

kernel void fillFloat2(device float2* buf [[buffer(0)]],
                       constant float2& value [[buffer(1)]],
                       uint gid [[thread_position_in_grid]]) {
    buf[gid] = value;
}

kernel void addVolumes(device float* A [[buffer(0)]],
                       device const float* B [[buffer(1)]],
                       uint gid [[thread_position_in_grid]]) {
    A[gid] += B[gid];
}

kernel void multiplyVolume(device float* vol [[buffer(0)]],
                           constant float& factor [[buffer(1)]],
                           uint gid [[thread_position_in_grid]]) {
    vol[gid] *= factor;
}

kernel void multiplyVolumes(device float* A [[buffer(0)]],
                            device const float* B [[buffer(1)]],
                            uint gid [[thread_position_in_grid]]) {
    A[gid] *= B[gid];
}

// ============================================================
//  Reduction: column max, row max
// ============================================================

kernel void calculateColumnMaxs(device float* columnMaxs [[buffer(0)]],
                                device const float* volume [[buffer(1)]],
                                constant Dims& dims [[buffer(2)]],
                                uint2 gid [[thread_position_in_grid]]) {
    int y = gid.x;
    int z = gid.y;
    if (y >= dims.H || z >= dims.D) return;
    float mx = volume[idx3(0, y, z, dims.W, dims.H)];
    for (int x = 1; x < dims.W; x++) {
        mx = max(mx, volume[idx3(x, y, z, dims.W, dims.H)]);
    }
    columnMaxs[y + z * dims.H] = mx;
}

kernel void calculateRowMaxs(device float* rowMaxs [[buffer(0)]],
                              device const float* columnMaxs [[buffer(1)]],
                              constant Dims& dims [[buffer(2)]],
                              uint gid [[thread_position_in_grid]]) {
    int z = gid;
    if ((int)gid >= dims.D) return;
    float mx = columnMaxs[z * dims.H];
    for (int y = 1; y < dims.H; y++) {
        mx = max(mx, columnMaxs[y + z * dims.H]);
    }
    rowMaxs[z] = mx;
}

// ============================================================
//  Nonseparable 3D convolution — three quadrature filters
//  Processes one z-slice of the 7x7x7 filter per dispatch.
//  Accumulates into filter response buffers (must be zero-initialized).
//  Uses threadgroup memory for the 2D image tile.
// ============================================================

#define HALO 3
#define TILE_W 32
#define TILE_H 32

kernel void nonseparableConv3D_ThreeFilters(
    device float2* response1 [[buffer(0)]],
    device float2* response2 [[buffer(1)]],
    device float2* response3 [[buffer(2)]],
    device const float* volume [[buffer(3)]],
    constant float* filter1Real [[buffer(4)]],
    constant float* filter1Imag [[buffer(5)]],
    constant float* filter2Real [[buffer(6)]],
    constant float* filter2Imag [[buffer(7)]],
    constant float* filter3Real [[buffer(8)]],
    constant float* filter3Imag [[buffer(9)]],
    constant int& zOffset [[buffer(10)]],
    constant Dims& dims [[buffer(11)]],
    uint3 groupId [[threadgroup_position_in_grid]],
    uint3 localId [[thread_position_in_threadgroup]],
    uint3 globalId [[thread_position_in_grid]])
{
    int x = int(groupId.x) * (TILE_W - 2 * HALO) + int(localId.x);
    int y = int(groupId.y) * (TILE_H - 2 * HALO) + int(localId.y);
    int z = int(globalId.z);

    // Shared memory tile: (TILE_H + 0) x (TILE_W + 0)
    // The tile includes halos; we launch TILE_W x TILE_H threads per group.
    threadgroup float tile[TILE_H][TILE_W];

    // Load tile: each thread loads one element
    int srcX = x - HALO;
    int srcY = y - HALO;
    int srcZ = z + zOffset;

    float val = 0.0f;
    if (srcX >= 0 && srcX < dims.W && srcY >= 0 && srcY < dims.H &&
        srcZ >= 0 && srcZ < dims.D) {
        val = volume[idx3(srcX, srcY, srcZ, dims.W, dims.H)];
    }
    tile[localId.y][localId.x] = val;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Only interior threads compute valid filter responses
    if (localId.x < HALO || localId.x >= TILE_W - HALO) return;
    if (localId.y < HALO || localId.y >= TILE_H - HALO) return;

    int outX = x - HALO;
    int outY = y - HALO;
    if (outX < 0 || outX >= dims.W || outY < 0 || outY >= dims.H || z >= dims.D) return;

    // 7x7 2D convolution from tile (reverse iteration to match OpenCL convolution)
    float s1r = 0, s1i = 0, s2r = 0, s2i = 0, s3r = 0, s3i = 0;

    for (int fy = 6; fy >= 0; fy--) {
        for (int fx = 6; fx >= 0; fx--) {
            float p = tile[localId.y - fy + 3][localId.x - fx + 3];
            int fi = fx + fy * 7;
            s1r += filter1Real[fi] * p;
            s1i += filter1Imag[fi] * p;
            s2r += filter2Real[fi] * p;
            s2i += filter2Imag[fi] * p;
            s3r += filter3Real[fi] * p;
            s3i += filter3Imag[fi] * p;
        }
    }

    int outIdx = idx3(outX, outY, z, dims.W, dims.H);
    response1[outIdx] += float2(s1r, s1i);
    response2[outIdx] += float2(s2r, s2i);
    response3[outIdx] += float2(s3r, s3i);
}

// ============================================================
//  Full 3D Nonseparable Convolution (texture3D, no z-loop)
//  Processes entire 7x7x7 filter in a single dispatch.
//  Uses hardware texture cache for efficient 3D neighborhood reads.
// ============================================================

kernel void nonseparableConv3D_Full(
    device float2* response1 [[buffer(0)]],
    device float2* response2 [[buffer(1)]],
    device float2* response3 [[buffer(2)]],
    texture3d<float, access::sample> volume [[texture(0)]],
    constant float* filter1Real [[buffer(3)]],   // 343 floats (7x7x7)
    constant float* filter1Imag [[buffer(4)]],
    constant float* filter2Real [[buffer(5)]],
    constant float* filter2Imag [[buffer(6)]],
    constant float* filter3Real [[buffer(7)]],
    constant float* filter3Imag [[buffer(8)]],
    constant Dims& dims [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]])
{
    int x = gid.x, y = gid.y, z = gid.z;
    if (x >= dims.W || y >= dims.H || z >= dims.D) return;

    constexpr sampler s(coord::pixel, address::clamp_to_zero, filter::nearest);

    float s1r = 0, s1i = 0, s2r = 0, s2i = 0, s3r = 0, s3i = 0;

    // Full 7x7x7 convolution — matches the z-slice loop + 2D convolution pattern
    // Filter index: fi = fx + fy*7 + fz*49
    // Source offset: (3-fx, 3-fy, 3-fz) relative to output voxel
    for (int fz = 0; fz < 7; fz++) {
        float srcZ = float(z + 3 - fz) + 0.5f;
        for (int fy = 0; fy < 7; fy++) {
            float srcY = float(y + 3 - fy) + 0.5f;
            for (int fx = 0; fx < 7; fx++) {
                float srcX = float(x + 3 - fx) + 0.5f;
                float p = volume.sample(s, float3(srcX, srcY, srcZ)).x;
                int fi = fx + fy * 7 + fz * 49;
                s1r += filter1Real[fi] * p;
                s1i += filter1Imag[fi] * p;
                s2r += filter2Real[fi] * p;
                s2i += filter2Imag[fi] * p;
                s3r += filter3Real[fi] * p;
                s3i += filter3Imag[fi] * p;
            }
        }
    }

    int outIdx = idx3(x, y, z, dims.W, dims.H);
    response1[outIdx] = float2(s1r, s1i);
    response2[outIdx] = float2(s2r, s2i);
    response3[outIdx] = float2(s3r, s3i);
}

// ============================================================
//  Separable convolution — rows (along y), columns (along x), rods (along z)
//  Filter size = 9 (hardcoded, matching BROCCOLI's smoothing)
// ============================================================

#define SMOOTH_HALF 4
#define SMOOTH_SIZE 9

kernel void separableConvRows(device float* output [[buffer(0)]],
                              device const float* input [[buffer(1)]],
                              constant float* filterY [[buffer(2)]],
                              constant Dims& dims [[buffer(3)]],
                              uint3 gid [[thread_position_in_grid]]) {
    int x = gid.x;
    int y = gid.y;
    int z = gid.z;
    if (x >= dims.W || y >= dims.H || z >= dims.D) return;

    float sum = 0.0f;
    for (int fy = -SMOOTH_HALF; fy <= SMOOTH_HALF; fy++) {
        int yy = y + fy;
        float val = 0.0f;
        if (yy >= 0 && yy < dims.H) {
            val = input[idx3(x, yy, z, dims.W, dims.H)];
        }
        sum += val * filterY[SMOOTH_HALF - fy];
    }
    output[idx3(x, y, z, dims.W, dims.H)] = sum;
}

kernel void separableConvColumns(device float* output [[buffer(0)]],
                                 device const float* input [[buffer(1)]],
                                 constant float* filterX [[buffer(2)]],
                                 constant Dims& dims [[buffer(3)]],
                                 uint3 gid [[thread_position_in_grid]]) {
    int x = gid.x;
    int y = gid.y;
    int z = gid.z;
    if (x >= dims.W || y >= dims.H || z >= dims.D) return;

    float sum = 0.0f;
    for (int fx = -SMOOTH_HALF; fx <= SMOOTH_HALF; fx++) {
        int xx = x + fx;
        float val = 0.0f;
        if (xx >= 0 && xx < dims.W) {
            val = input[idx3(xx, y, z, dims.W, dims.H)];
        }
        sum += val * filterX[SMOOTH_HALF - fx];
    }
    output[idx3(x, y, z, dims.W, dims.H)] = sum;
}

kernel void separableConvRods(device float* output [[buffer(0)]],
                              device const float* input [[buffer(1)]],
                              constant float* filterZ [[buffer(2)]],
                              constant Dims& dims [[buffer(3)]],
                              uint3 gid [[thread_position_in_grid]]) {
    int x = gid.x;
    int y = gid.y;
    int z = gid.z;
    if (x >= dims.W || y >= dims.H || z >= dims.D) return;

    float sum = 0.0f;
    for (int fz = -SMOOTH_HALF; fz <= SMOOTH_HALF; fz++) {
        int zz = z + fz;
        float val = 0.0f;
        if (zz >= 0 && zz < dims.D) {
            val = input[idx3(x, y, zz, dims.W, dims.H)];
        }
        sum += val * filterZ[SMOOTH_HALF - fz];
    }
    output[idx3(x, y, z, dims.W, dims.H)] = sum;
}

// ============================================================
//  Phase differences and certainties (linear registration)
// ============================================================

kernel void calculatePhaseDifferencesAndCertainties(
    device float* phaseDiff [[buffer(0)]],
    device float* certainties [[buffer(1)]],
    device const float2* q1 [[buffer(2)]],
    device const float2* q2 [[buffer(3)]],
    constant Dims& dims [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])
{
    int x = gid.x, y = gid.y, z = gid.z;
    if (x >= dims.W || y >= dims.H || z >= dims.D) return;

    int i = idx3(x, y, z, dims.W, dims.H);
    float2 a = q1[i];
    float2 c = q2[i];

    // phase = arg(q1 * conj(q2))
    float cpReal = a.x * c.x + a.y * c.y;
    float cpImag = a.y * c.x - a.x * c.y;
    float phase = atan2(cpImag, cpReal);

    // |q1 * q2| for certainty
    float prodReal = a.x * c.x - a.y * c.y;
    float prodImag = a.y * c.x + a.x * c.y;
    float cosHalf = cos(phase * 0.5f);

    phaseDiff[i] = isnan(phase) ? 0.0f : phase;
    float cert = sqrt(prodReal * prodReal + prodImag * prodImag) * cosHalf * cosHalf;
    certainties[i] = isnan(cert) ? 0.0f : cert;
}

// ============================================================
//  Phase gradients (finite difference of quadrature filter responses)
// ============================================================

kernel void calculatePhaseGradientsX(
    device float* gradients [[buffer(0)]],
    device const float2* q1 [[buffer(1)]],
    device const float2* q2 [[buffer(2)]],
    constant Dims& dims [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]])
{
    int x = gid.x, y = gid.y, z = gid.z;
    if (x < 1 || x >= dims.W - 1 || y >= dims.H || z >= dims.D) return;

    int i0 = idx3(x, y, z, dims.W, dims.H);
    int im = idx3(x-1, y, z, dims.W, dims.H);
    int ip = idx3(x+1, y, z, dims.W, dims.H);

    float tr = 0, ti = 0;
    float2 a, c;

    // q1[x+1] * conj(q1[x])
    a = q1[ip]; c = q1[i0];
    tr += a.x*c.x + a.y*c.y; ti += a.y*c.x - a.x*c.y;
    // q1[x] * conj(q1[x-1])
    a = q1[i0]; c = q1[im];
    tr += a.x*c.x + a.y*c.y; ti += a.y*c.x - a.x*c.y;
    // q2[x+1] * conj(q2[x])
    a = q2[ip]; c = q2[i0];
    tr += a.x*c.x + a.y*c.y; ti += a.y*c.x - a.x*c.y;
    // q2[x] * conj(q2[x-1])
    a = q2[i0]; c = q2[im];
    tr += a.x*c.x + a.y*c.y; ti += a.y*c.x - a.x*c.y;

    float g = atan2(ti, tr);
    gradients[i0] = isnan(g) ? 0.0f : g;
}

kernel void calculatePhaseGradientsY(
    device float* gradients [[buffer(0)]],
    device const float2* q1 [[buffer(1)]],
    device const float2* q2 [[buffer(2)]],
    constant Dims& dims [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]])
{
    int x = gid.x, y = gid.y, z = gid.z;
    if (x >= dims.W || y < 1 || y >= dims.H - 1 || z >= dims.D) return;

    int i0 = idx3(x, y, z, dims.W, dims.H);
    int im = idx3(x, y-1, z, dims.W, dims.H);
    int ip = idx3(x, y+1, z, dims.W, dims.H);

    float tr = 0, ti = 0;
    float2 a, c;

    a = q1[ip]; c = q1[i0];
    tr += a.x*c.x + a.y*c.y; ti += a.y*c.x - a.x*c.y;
    a = q1[i0]; c = q1[im];
    tr += a.x*c.x + a.y*c.y; ti += a.y*c.x - a.x*c.y;
    a = q2[ip]; c = q2[i0];
    tr += a.x*c.x + a.y*c.y; ti += a.y*c.x - a.x*c.y;
    a = q2[i0]; c = q2[im];
    tr += a.x*c.x + a.y*c.y; ti += a.y*c.x - a.x*c.y;

    float g = atan2(ti, tr);
    gradients[i0] = isnan(g) ? 0.0f : g;
}

kernel void calculatePhaseGradientsZ(
    device float* gradients [[buffer(0)]],
    device const float2* q1 [[buffer(1)]],
    device const float2* q2 [[buffer(2)]],
    constant Dims& dims [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]])
{
    int x = gid.x, y = gid.y, z = gid.z;
    if (x >= dims.W || y >= dims.H || z < 1 || z >= dims.D - 1) return;

    int i0 = idx3(x, y, z, dims.W, dims.H);
    int im = idx3(x, y, z-1, dims.W, dims.H);
    int ip = idx3(x, y, z+1, dims.W, dims.H);

    float tr = 0, ti = 0;
    float2 a, c;

    a = q1[ip]; c = q1[i0];
    tr += a.x*c.x + a.y*c.y; ti += a.y*c.x - a.x*c.y;
    a = q1[i0]; c = q1[im];
    tr += a.x*c.x + a.y*c.y; ti += a.y*c.x - a.x*c.y;
    a = q2[ip]; c = q2[i0];
    tr += a.x*c.x + a.y*c.y; ti += a.y*c.x - a.x*c.y;
    a = q2[i0]; c = q2[im];
    tr += a.x*c.x + a.y*c.y; ti += a.y*c.x - a.x*c.y;

    float g = atan2(ti, tr);
    gradients[i0] = isnan(g) ? 0.0f : g;
}

// ============================================================
//  A-matrix and h-vector 2D reduction (linear registration)
//  One thread per (y, z) pair, sums across x.
//  Direction-specific offsets into the output arrays.
// ============================================================

struct AMatrixParams {
    int W;
    int H;
    int D;
    int filterSize;
    int directionOffset;   // 0 for X, 10 for Y, 20 for Z
    int hVectorOffset;     // 0 for X, 1 for Y, 2 for Z
};

kernel void calculateAMatrixAndHVector2D(
    device float* A2D [[buffer(0)]],
    device float* h2D [[buffer(1)]],
    device const float* phaseDiff [[buffer(2)]],
    device const float* phaseGrad [[buffer(3)]],
    device const float* certainty [[buffer(4)]],
    constant AMatrixParams& p [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    int y = gid.x;
    int z = gid.y;
    int fhalf = (p.filterSize - 1) / 2;

    if (y < fhalf || y >= p.H - fhalf || z < fhalf || z >= p.D - fhalf) return;

    float yf = float(y) - (float(p.H) - 1.0f) * 0.5f;
    float zf = float(z) - (float(p.D) - 1.0f) * 0.5f;

    float aval[10] = {0,0,0,0,0,0,0,0,0,0};
    float hval[4] = {0,0,0,0};

    for (int x = fhalf; x < p.W - fhalf; x++) {
        float xf = float(x) - (float(p.W) - 1.0f) * 0.5f;
        int i = idx3(x, y, z, p.W, p.H);
        float pd = phaseDiff[i];
        float pg = phaseGrad[i];
        float cert = certainty[i];
        float cpp = cert * pg * pg;
        float cpd = cert * pg * pd;

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

    int HD = p.H * p.D;
    int base = y + z * p.H + p.directionOffset * HD;
    for (int k = 0; k < 10; k++)
        A2D[base + k * HD] = aval[k];

    // h-vector storage: directions at offsets 0, 1, 2 for X, Y, Z;
    // then within each direction: element 0 at dirOffset, elements 1-3 at later offsets
    int hBase = y + z * p.H + p.hVectorOffset * HD;
    h2D[hBase] = hval[0];
    // Elements 1,2,3 are at offsets (3 + directionOffset/10*2)*HD from start,
    // but let's use the exact OpenCL layout:
    // X: h[0] at 0, h[1] at 3*HD, h[2] at 4*HD, h[3] at 5*HD
    // Y: h[0] at HD, h[1] at 6*HD, h[2] at 7*HD, h[3] at 8*HD
    // Z: h[0] at 2*HD, h[1] at 9*HD, h[2] at 10*HD, h[3] at 11*HD
    int hStart = (3 + p.hVectorOffset * 2) * HD + y + z * p.H;
    // Actually the OpenCL offsets are:
    // X: h[0] at 0*HD, h[1,2,3] at 3,4,5 * HD
    // Y: h[0] at 1*HD, h[1,2,3] at 6,7,8 * HD
    // Z: h[0] at 2*HD, h[1,2,3] at 9,10,11 * HD
    int extraBase = y + z * p.H + (3 + p.hVectorOffset * 3) * HD;
    h2D[extraBase + 0 * HD] = hval[1];
    h2D[extraBase + 1 * HD] = hval[2];
    h2D[extraBase + 2 * HD] = hval[3];
}

// 1D reduction: sum A-matrix 2D values over y
kernel void calculateAMatrix1D(
    device float* A1D [[buffer(0)]],
    device const float* A2D [[buffer(1)]],
    constant int& H [[buffer(2)]],
    constant int& D [[buffer(3)]],
    constant int& filterSize [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    int z = gid.x;
    int element = gid.y;  // 0..29
    int fhalf = (filterSize - 1) / 2;
    if (z < fhalf || z >= D - fhalf) return;

    float sum = 0.0f;
    int base = z * H + element * H * D;
    for (int y = fhalf; y < H - fhalf; y++) {
        sum += A2D[base + y];
    }
    A1D[z + element * D] = sum;
}

// Final reduction: sum A-matrix 1D values over z -> scalar matrix elements
// GetParameterIndices mapping (same as OpenCL)
constant int2 parameterIndices[30] = {
    {0,0}, {3,0}, {4,0}, {5,0}, {3,3}, {4,3}, {5,3}, {4,4}, {5,4}, {5,5},
    {1,1}, {6,1}, {7,1}, {8,1}, {6,6}, {7,6}, {8,6}, {7,7}, {8,7}, {8,8},
    {2,2}, {9,2}, {10,2}, {11,2}, {9,9}, {10,9}, {11,9}, {10,10}, {11,10}, {11,11}
};

kernel void calculateAMatrixFinal(
    device float* A [[buffer(0)]],
    device const float* A1D [[buffer(1)]],
    constant int& D [[buffer(2)]],
    constant int& filterSize [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    int element = gid;  // 0..29
    if (element >= 30) return;
    int fhalf = (filterSize - 1) / 2;

    float sum = 0.0f;
    int base = element * D;
    for (int z = fhalf; z < D - fhalf; z++) {
        sum += A1D[base + z];
    }

    int2 ij = parameterIndices[element];
    A[ij.x + ij.y * 12] = sum;
}

// 1D and final reduction for h-vector
kernel void calculateHVector1D(
    device float* h1D [[buffer(0)]],
    device const float* h2D [[buffer(1)]],
    constant int& H [[buffer(2)]],
    constant int& D [[buffer(3)]],
    constant int& filterSize [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    int z = gid.x;
    int element = gid.y;  // 0..11
    int fhalf = (filterSize - 1) / 2;
    if (z < fhalf || z >= D - fhalf) return;

    float sum = 0.0f;
    int base = z * H + element * H * D;
    for (int y = fhalf; y < H - fhalf; y++) {
        sum += h2D[base + y];
    }
    h1D[z + element * D] = sum;
}

kernel void calculateHVectorFinal(
    device float* h [[buffer(0)]],
    device const float* h1D [[buffer(1)]],
    constant int& D [[buffer(2)]],
    constant int& filterSize [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    int element = gid;  // 0..11
    if (element >= 12) return;
    int fhalf = (filterSize - 1) / 2;

    float sum = 0.0f;
    int base = element * D;
    for (int z = fhalf; z < D - fhalf; z++) {
        sum += h1D[base + z];
    }
    h[element] = sum;
}

// ============================================================
//  Nonlinear registration: tensor components
// ============================================================

kernel void calculateTensorComponents(
    device float* t11 [[buffer(0)]],
    device float* t12 [[buffer(1)]],
    device float* t13 [[buffer(2)]],
    device float* t22 [[buffer(3)]],
    device float* t23 [[buffer(4)]],
    device float* t33 [[buffer(5)]],
    device const float2* q1 [[buffer(6)]],
    device const float2* q2 [[buffer(7)]],
    constant float& m11 [[buffer(8)]],
    constant float& m12 [[buffer(9)]],
    constant float& m13 [[buffer(10)]],
    constant float& m22 [[buffer(11)]],
    constant float& m23 [[buffer(12)]],
    constant float& m33 [[buffer(13)]],
    constant Dims& dims [[buffer(14)]],
    uint3 gid [[thread_position_in_grid]])
{
    int x = gid.x, y = gid.y, z = gid.z;
    if (x >= dims.W || y >= dims.H || z >= dims.D) return;

    int i = idx3(x, y, z, dims.W, dims.H);
    float2 q2v = q2[i];
    float mag = sqrt(q2v.x * q2v.x + q2v.y * q2v.y);

    t11[i] += mag * m11;
    t12[i] += mag * m12;
    t13[i] += mag * m13;
    t22[i] += mag * m22;
    t23[i] += mag * m23;
    t33[i] += mag * m33;
}

// ============================================================
//  Tensor norms (Frobenius)
// ============================================================

kernel void calculateTensorNorms(
    device float* norms [[buffer(0)]],
    device const float* t11 [[buffer(1)]],
    device const float* t12 [[buffer(2)]],
    device const float* t13 [[buffer(3)]],
    device const float* t22 [[buffer(4)]],
    device const float* t23 [[buffer(5)]],
    device const float* t33 [[buffer(6)]],
    constant Dims& dims [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]])
{
    int x = gid.x, y = gid.y, z = gid.z;
    if (x >= dims.W || y >= dims.H || z >= dims.D) return;

    int i = idx3(x, y, z, dims.W, dims.H);
    float v11 = t11[i], v12 = t12[i], v13 = t13[i];
    float v22 = t22[i], v23 = t23[i], v33 = t33[i];

    norms[i] = sqrt(v11*v11 + 2*v12*v12 + 2*v13*v13 + v22*v22 + 2*v23*v23 + v33*v33);
}

// ============================================================
//  Nonlinear A-matrices and h-vectors
// ============================================================

struct MorphonParams {
    int W;
    int H;
    int D;
    int FILTER;  // 0-5, determines = vs +=
};

kernel void calculateAMatricesAndHVectors(
    device float* a11 [[buffer(0)]],
    device float* a12 [[buffer(1)]],
    device float* a13 [[buffer(2)]],
    device float* a22 [[buffer(3)]],
    device float* a23 [[buffer(4)]],
    device float* a33 [[buffer(5)]],
    device float* h1 [[buffer(6)]],
    device float* h2 [[buffer(7)]],
    device float* h3 [[buffer(8)]],
    device const float2* q1 [[buffer(9)]],
    device const float2* q2 [[buffer(10)]],
    device const float* t11 [[buffer(11)]],
    device const float* t12 [[buffer(12)]],
    device const float* t13 [[buffer(13)]],
    device const float* t22 [[buffer(14)]],
    device const float* t23 [[buffer(15)]],
    device const float* t33 [[buffer(16)]],
    device const float* filterDirX [[buffer(17)]],
    device const float* filterDirY [[buffer(18)]],
    device const float* filterDirZ [[buffer(19)]],
    constant MorphonParams& p [[buffer(20)]],
    uint3 gid [[thread_position_in_grid]])
{
    int x = gid.x, y = gid.y, z = gid.z;
    if (x >= p.W || y >= p.H || z >= p.D) return;

    int i = idx3(x, y, z, p.W, p.H);

    float2 q1v = q1[i];
    float2 q2v = q2[i];

    // q1 * conj(q2)
    float qqR = q1v.x * q2v.x + q1v.y * q2v.y;
    float qqI = -q1v.x * q2v.y + q1v.y * q2v.x;
    float pd = atan2(qqI, qqR);
    float Aqq = sqrt(qqR * qqR + qqI * qqI);
    float cosH = cos(pd * 0.5f);
    float cert = sqrt(Aqq) * cosH * cosH;

    // T^2 components
    float T11 = t11[i], T12 = t12[i], T13 = t13[i];
    float T22 = t22[i], T23 = t23[i], T33 = t33[i];

    float tt11 = T11*T11 + T12*T12 + T13*T13;
    float tt12 = T11*T12 + T12*T22 + T13*T23;
    float tt13 = T11*T13 + T12*T23 + T13*T33;
    float tt22 = T12*T12 + T22*T22 + T23*T23;
    float tt23 = T12*T13 + T22*T23 + T23*T33;
    float tt33 = T13*T13 + T23*T23 + T33*T33;

    float fdx = filterDirX[p.FILTER];
    float fdy = filterDirY[p.FILTER];
    float fdz = filterDirZ[p.FILTER];

    float cpd = cert * pd;
    // Protect against NaN from ill-conditioned phase computation
    if (isnan(cpd)) cpd = 0.0f;
    float hh1 = cpd * (fdx * tt11 + fdy * tt12 + fdz * tt13);
    float hh2 = cpd * (fdx * tt12 + fdy * tt22 + fdz * tt23);
    float hh3 = cpd * (fdx * tt13 + fdy * tt23 + fdz * tt33);

    // FILTER==0: assignment to clear stale data (d_a11 reused for tensor norms)
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

// ============================================================
//  Displacement update (Cramer's rule with regularization)
// ============================================================

kernel void calculateDisplacementUpdate(
    device float* dispX [[buffer(0)]],
    device float* dispY [[buffer(1)]],
    device float* dispZ [[buffer(2)]],
    device const float* a11 [[buffer(3)]],
    device const float* a12 [[buffer(4)]],
    device const float* a13 [[buffer(5)]],
    device const float* a22 [[buffer(6)]],
    device const float* a23 [[buffer(7)]],
    device const float* a33 [[buffer(8)]],
    device const float* h1 [[buffer(9)]],
    device const float* h2 [[buffer(10)]],
    device const float* h3 [[buffer(11)]],
    constant Dims& dims [[buffer(12)]],
    uint3 gid [[thread_position_in_grid]])
{
    int x = gid.x, y = gid.y, z = gid.z;
    if (x >= dims.W || y >= dims.H || z >= dims.D) return;

    int i = idx3(x, y, z, dims.W, dims.H);

    float A11 = a11[i], A12 = a12[i], A13 = a13[i];
    float A22 = a22[i], A23 = a23[i], A33 = a33[i];
    float H1 = h1[i], H2 = h2[i], H3 = h3[i];

    float det = A11*A22*A33 - A11*A23*A23 - A12*A12*A33
              + A12*A23*A13 + A13*A12*A23 - A13*A22*A13;

    // Regularized inversion with step-size damping
    float trace = A11 + A22 + A33;
    float epsilon = 0.01f * trace * trace * trace / 27.0f + 1e-16f;
    float norm = 0.2f / (det + epsilon);

    float dx = norm * (H1*(A22*A33 - A23*A23) - H2*(A12*A33 - A13*A23) + H3*(A12*A23 - A13*A22));
    float dy = norm * (H2*(A11*A33 - A13*A13) - H3*(A11*A23 - A13*A12) - H1*(A12*A33 - A23*A13));
    float dz = norm * (H3*(A11*A22 - A12*A12) - H2*(A11*A23 - A12*A13) + H1*(A12*A23 - A22*A13));

    // Replace NaN/Inf with 0 (ill-conditioned regions produce no displacement)
    dispX[i] = isnan(dx) || isinf(dx) ? 0.0f : dx;
    dispY[i] = isnan(dy) || isinf(dy) ? 0.0f : dy;
    dispZ[i] = isnan(dz) || isinf(dz) ? 0.0f : dz;
}

// ============================================================
//  Interpolation using 3D textures (hardware trilinear)
// ============================================================

constexpr sampler linearSampler(coord::pixel, address::clamp_to_zero, filter::linear);
constexpr sampler nearestSampler(coord::pixel, address::clamp_to_zero, filter::nearest);

// Affine interpolation (linear registration)
kernel void interpolateLinearLinear(
    device float* output [[buffer(0)]],
    texture3d<float, access::sample> volume [[texture(0)]],
    constant float* params [[buffer(1)]],
    constant Dims& dims [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]])
{
    int x = gid.x, y = gid.y, z = gid.z;
    if (x >= dims.W || y >= dims.H || z >= dims.D) return;

    float xf = float(x) - (float(dims.W) - 1.0f) * 0.5f;
    float yf = float(y) - (float(dims.H) - 1.0f) * 0.5f;
    float zf = float(z) - (float(dims.D) - 1.0f) * 0.5f;

    float3 pos;
    pos.x = float(x) + params[0] + params[3]*xf + params[4]*yf  + params[5]*zf  + 0.5f;
    pos.y = float(y) + params[1] + params[6]*xf + params[7]*yf  + params[8]*zf  + 0.5f;
    pos.z = float(z) + params[2] + params[9]*xf + params[10]*yf + params[11]*zf + 0.5f;

    float val = volume.sample(linearSampler, pos).x;
    output[idx3(x, y, z, dims.W, dims.H)] = val;
}

// Nonlinear interpolation (displacement field)
kernel void interpolateLinearNonLinear(
    device float* output [[buffer(0)]],
    texture3d<float, access::sample> volume [[texture(0)]],
    device const float* dispX [[buffer(1)]],
    device const float* dispY [[buffer(2)]],
    device const float* dispZ [[buffer(3)]],
    constant Dims& dims [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])
{
    int x = gid.x, y = gid.y, z = gid.z;
    if (x >= dims.W || y >= dims.H || z >= dims.D) return;

    int i = idx3(x, y, z, dims.W, dims.H);
    float3 pos;
    pos.x = float(x) + dispX[i] + 0.5f;
    pos.y = float(y) + dispY[i] + 0.5f;
    pos.z = float(z) + dispZ[i] + 0.5f;

    float val = volume.sample(linearSampler, pos).x;
    output[i] = val;
}

// ============================================================
//  Rescale volume (voxel size change) using texture
// ============================================================

kernel void rescaleVolumeLinear(
    device float* output [[buffer(0)]],
    texture3d<float, access::sample> volume [[texture(0)]],
    constant float& scaleX [[buffer(1)]],
    constant float& scaleY [[buffer(2)]],
    constant float& scaleZ [[buffer(3)]],
    constant Dims& dims [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])
{
    int x = gid.x, y = gid.y, z = gid.z;
    if (x >= dims.W || y >= dims.H || z >= dims.D) return;

    float3 pos;
    pos.x = float(x) * scaleX + 0.5f;
    pos.y = float(y) * scaleY + 0.5f;
    pos.z = float(z) * scaleZ + 0.5f;

    output[idx3(x, y, z, dims.W, dims.H)] = volume.sample(linearSampler, pos).x;
}

// ============================================================
//  Copy volume to new grid (with dimension offsets)
// ============================================================

struct CopyParams {
    int newW, newH, newD;
    int srcW, srcH, srcD;
    int xDiff, yDiff, zDiff;
    int mmZCut;
    float voxelSizeZ;
};

kernel void copyVolumeToNew(
    device float* newVol [[buffer(0)]],
    device const float* srcVol [[buffer(1)]],
    constant CopyParams& p [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]])
{
    int x = gid.x, y = gid.y, z = gid.z;

    int xNew, xSrc, yNew, ySrc, zNew, zSrc;

    if (p.xDiff > 0) {
        xNew = x; xSrc = x + int(round(float(p.xDiff) / 2.0f));
    } else {
        xNew = x + int(round(float(abs(p.xDiff)) / 2.0f)); xSrc = x;
    }
    if (p.yDiff > 0) {
        yNew = y; ySrc = y + int(round(float(p.yDiff) / 2.0f));
    } else {
        yNew = y + int(round(float(abs(p.yDiff)) / 2.0f)); ySrc = y;
    }
    if (p.zDiff > 0) {
        zNew = z; zSrc = z + int(round(float(p.zDiff) / 2.0f)) + int(round(float(p.mmZCut) / p.voxelSizeZ));
    } else {
        zNew = z + int(round(float(abs(p.zDiff)) / 2.0f));
        zSrc = z + int(round(float(p.mmZCut) / p.voxelSizeZ));
    }

    if (xSrc < 0 || xSrc >= p.srcW || ySrc < 0 || ySrc >= p.srcH || zSrc < 0 || zSrc >= p.srcD) return;
    if (xNew < 0 || xNew >= p.newW || yNew < 0 || yNew >= p.newH || zNew < 0 || zNew >= p.newD) return;

    newVol[idx3(xNew, yNew, zNew, p.newW, p.newH)] = srcVol[idx3(xSrc, ySrc, zSrc, p.srcW, p.srcH)];
}

// ============================================================
//  Add linear and nonlinear displacement fields
// ============================================================

kernel void addLinearAndNonLinearDisplacement(
    device float* dispX [[buffer(0)]],
    device float* dispY [[buffer(1)]],
    device float* dispZ [[buffer(2)]],
    constant float* params [[buffer(3)]],
    constant Dims& dims [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])
{
    int x = gid.x, y = gid.y, z = gid.z;
    if (x >= dims.W || y >= dims.H || z >= dims.D) return;

    float xf = float(x) - (float(dims.W) - 1.0f) * 0.5f;
    float yf = float(y) - (float(dims.H) - 1.0f) * 0.5f;
    float zf = float(z) - (float(dims.D) - 1.0f) * 0.5f;

    int i = idx3(x, y, z, dims.W, dims.H);
    dispX[i] += params[0] + params[3]*xf + params[4]*yf  + params[5]*zf;
    dispY[i] += params[1] + params[6]*xf + params[7]*yf  + params[8]*zf;
    dispZ[i] += params[2] + params[9]*xf + params[10]*yf + params[11]*zf;
}
