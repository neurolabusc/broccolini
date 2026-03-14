// cuda_registration.cu — CUDA compute backend for BROCCOLI image registration
// Ported from the Metal backend with CUDA-native optimizations:
//   - CUDA 3D textures for hardware trilinear interpolation
//   - Constant memory for small filter coefficients
//   - Stream-based dispatch for overlapping compute/transfer

#include "cuda_registration.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cassert>
#include <mutex>

// ============================================================
//  Error checking
// ============================================================

#include <stdexcept>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        char msg[256]; \
        snprintf(msg, sizeof(msg), "CUDA error at %s:%d: %s", \
                 __FILE__, __LINE__, cudaGetErrorString(err)); \
        throw std::runtime_error(msg); \
    } \
} while(0)

// ============================================================
//  Internal types
// ============================================================

namespace cuda_reg {
namespace {

struct Dims { int W, H, D; };

// ============================================================
//  CUDA Context (singleton)
// ============================================================

struct CudaContext {
    bool initialized = false;

    void init() {
        if (initialized) return;
        int deviceCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        assert(deviceCount > 0 && "No CUDA device found");
        CUDA_CHECK(cudaSetDevice(0));
        initialized = true;
    }

    // Allocate device memory
    float* newBuffer(size_t count) {
        float* ptr = nullptr;
        CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(float)));
        CUDA_CHECK(cudaMemset(ptr, 0, count * sizeof(float)));
        return ptr;
    }

    // Allocate and upload
    float* newBuffer(const float* data, size_t count) {
        float* ptr = nullptr;
        CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(ptr, data, count * sizeof(float), cudaMemcpyHostToDevice));
        return ptr;
    }

    // Download to host
    void readBuffer(float* host, const float* device, size_t count) {
        CUDA_CHECK(cudaMemcpy(host, device, count * sizeof(float), cudaMemcpyDeviceToHost));
    }

    // Upload to device
    void writeBuffer(float* device, const float* host, size_t count) {
        CUDA_CHECK(cudaMemcpy(device, host, count * sizeof(float), cudaMemcpyHostToDevice));
    }

    void freeBuffer(float* ptr) {
        if (ptr) cudaFree(ptr);
    }

    // Create a 3D texture object from a device buffer
    // Returns a cudaTextureObject_t for hardware-cached reads with trilinear interpolation
    cudaTextureObject_t createTexture3D(const float* deviceBuf, int W, int H, int D,
                                         cudaArray_t* arrayOut, bool linear = true) {
        // Create 3D array
        cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
        cudaExtent extent = make_cudaExtent(W, H, D);
        CUDA_CHECK(cudaMalloc3DArray(arrayOut, &desc, extent));

        // Copy device buffer → 3D array
        cudaMemcpy3DParms p = {};
        p.srcPtr = make_cudaPitchedPtr((void*)deviceBuf, W * sizeof(float), W, H);
        p.dstArray = *arrayOut;
        p.extent = extent;
        p.kind = cudaMemcpyDeviceToDevice;
        CUDA_CHECK(cudaMemcpy3D(&p));

        // Create texture object
        cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = *arrayOut;

        cudaTextureDesc texDesc = {};
        texDesc.addressMode[0] = cudaAddressModeBorder;
        texDesc.addressMode[1] = cudaAddressModeBorder;
        texDesc.addressMode[2] = cudaAddressModeBorder;
        texDesc.filterMode = linear ? cudaFilterModeLinear : cudaFilterModePoint;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        cudaTextureObject_t tex = 0;
        CUDA_CHECK(cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr));
        return tex;
    }

    void destroyTexture3D(cudaTextureObject_t tex, cudaArray_t array) {
        cudaDestroyTextureObject(tex);
        cudaFreeArray(array);
    }
};

CudaContext& ctx() {
    static CudaContext c;
    if (!c.initialized) c.init();
    return c;
}

// ============================================================
//  Dispatch helpers
// ============================================================

inline dim3 gridFor3D(int W, int H, int D, dim3 block) {
    return dim3((W + block.x - 1) / block.x,
                (H + block.y - 1) / block.y,
                (D + block.z - 1) / block.z);
}

inline dim3 gridFor1D(int count, int blockSize = 256) {
    return dim3((count + blockSize - 1) / blockSize);
}

// Standard block sizes
static const dim3 BLOCK_3D(8, 8, 4);
static const dim3 BLOCK_2D(16, 16);
static const int BLOCK_1D = 256;

// ============================================================
//  CUDA Kernels
// ============================================================

// --- Utility kernels ---

__global__ void k_fill(float* buf, float value, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) buf[i] = value;
}

__global__ void k_fill_float2(float* buf, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // float2 stored as interleaved: buf[2*i], buf[2*i+1]
    if (i < count) { buf[2*i] = 0.0f; buf[2*i+1] = 0.0f; }
}

__global__ void k_add(float* A, const float* B, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) A[i] += B[i];
}

__global__ void k_multiply(float* vol, float factor, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) vol[i] *= factor;
}

__global__ void k_multiply_volumes(float* A, const float* B, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) A[i] *= B[i];
}

// --- Reduction: column max, row max ---

__global__ void k_column_maxs(float* colMaxs, const float* volume,
                               int W, int H, int D) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= H || z >= D) return;
    float mx = volume[0 + y * W + z * W * H];
    for (int x = 1; x < W; x++)
        mx = fmaxf(mx, volume[x + y * W + z * W * H]);
    colMaxs[y + z * H] = mx;
}

__global__ void k_row_maxs(float* rowMaxs, const float* colMaxs, int H, int D) {
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    if (z >= D) return;
    float mx = colMaxs[z * H];
    for (int y = 1; y < H; y++)
        mx = fmaxf(mx, colMaxs[y + z * H]);
    rowMaxs[z] = mx;
}

// --- 3D nonseparable convolution (3 quadrature filters, texture-cached) ---

__global__ void k_conv3d_full(
    float* __restrict__ resp1, float* __restrict__ resp2, float* __restrict__ resp3,
    cudaTextureObject_t volume,
    const float* __restrict__ f1r, const float* __restrict__ f1i,
    const float* __restrict__ f2r, const float* __restrict__ f2i,
    const float* __restrict__ f3r, const float* __restrict__ f3i,
    int W, int H, int D)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= W || y >= H || z >= D) return;

    float s1r = 0, s1i = 0, s2r = 0, s2i = 0, s3r = 0, s3i = 0;

    for (int fz = 0; fz < 7; fz++) {
        float srcZ = (float)(z + 3 - fz) + 0.5f;
        for (int fy = 0; fy < 7; fy++) {
            float srcY = (float)(y + 3 - fy) + 0.5f;
            for (int fx = 0; fx < 7; fx++) {
                float srcX = (float)(x + 3 - fx) + 0.5f;
                float p = tex3D<float>(volume, srcX, srcY, srcZ);
                int fi = fx + fy * 7 + fz * 49;
                s1r += f1r[fi] * p;
                s1i += f1i[fi] * p;
                s2r += f2r[fi] * p;
                s2i += f2i[fi] * p;
                s3r += f3r[fi] * p;
                s3i += f3i[fi] * p;
            }
        }
    }

    int idx = x + y * W + z * W * H;
    resp1[2*idx]   = s1r; resp1[2*idx+1] = s1i;
    resp2[2*idx]   = s2r; resp2[2*idx+1] = s2i;
    resp3[2*idx]   = s3r; resp3[2*idx+1] = s3i;
}

// --- Separable smoothing (9-tap) ---

__global__ void k_smooth_rows(float* output, const float* input,
                               const float* filter, int W, int H, int D) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= W || y >= H || z >= D) return;

    float sum = 0.0f;
    for (int fy = -4; fy <= 4; fy++) {
        int yy = y + fy;
        float val = (yy >= 0 && yy < H) ? input[x + yy * W + z * W * H] : 0.0f;
        sum += val * filter[4 - fy];
    }
    output[x + y * W + z * W * H] = sum;
}

__global__ void k_smooth_cols(float* output, const float* input,
                               const float* filter, int W, int H, int D) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= W || y >= H || z >= D) return;

    float sum = 0.0f;
    for (int fx = -4; fx <= 4; fx++) {
        int xx = x + fx;
        float val = (xx >= 0 && xx < W) ? input[xx + y * W + z * W * H] : 0.0f;
        sum += val * filter[4 - fx];
    }
    output[x + y * W + z * W * H] = sum;
}

__global__ void k_smooth_rods(float* output, const float* input,
                               const float* filter, int W, int H, int D) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= W || y >= H || z >= D) return;

    float sum = 0.0f;
    for (int fz = -4; fz <= 4; fz++) {
        int zz = z + fz;
        float val = (zz >= 0 && zz < D) ? input[x + y * W + zz * W * H] : 0.0f;
        sum += val * filter[4 - fz];
    }
    output[x + y * W + z * W * H] = sum;
}

// --- Phase differences and certainties ---

__global__ void k_phase_diff_cert(float* phaseDiff, float* certainties,
                                   const float* q1, const float* q2,
                                   int W, int H, int D) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= W || y >= H || z >= D) return;

    int i = x + y * W + z * W * H;
    float a_r = q1[2*i], a_i = q1[2*i+1];
    float c_r = q2[2*i], c_i = q2[2*i+1];

    // phase = arg(q1 * conj(q2))
    float cpR = a_r * c_r + a_i * c_i;
    float cpI = a_i * c_r - a_r * c_i;
    float phase = atan2f(cpI, cpR);

    // |q1 * q2| for certainty
    float pR = a_r * c_r - a_i * c_i;
    float pI = a_i * c_r + a_r * c_i;
    float cosH = cosf(phase * 0.5f);
    float cert = sqrtf(pR * pR + pI * pI) * cosH * cosH;

    phaseDiff[i] = isnan(phase) ? 0.0f : phase;
    certainties[i] = isnan(cert) ? 0.0f : cert;
}

// --- Phase gradients ---

__device__ void phaseGradHelper(float* gradients, const float* q1, const float* q2,
                                 int i0, int im, int ip) {
    float tr = 0, ti = 0;
    float ar, ai, cr, ci;

    // q1[ip] * conj(q1[i0])
    ar = q1[2*ip]; ai = q1[2*ip+1]; cr = q1[2*i0]; ci = q1[2*i0+1];
    tr += ar*cr + ai*ci; ti += ai*cr - ar*ci;
    // q1[i0] * conj(q1[im])
    ar = q1[2*i0]; ai = q1[2*i0+1]; cr = q1[2*im]; ci = q1[2*im+1];
    tr += ar*cr + ai*ci; ti += ai*cr - ar*ci;
    // q2[ip] * conj(q2[i0])
    ar = q2[2*ip]; ai = q2[2*ip+1]; cr = q2[2*i0]; ci = q2[2*i0+1];
    tr += ar*cr + ai*ci; ti += ai*cr - ar*ci;
    // q2[i0] * conj(q2[im])
    ar = q2[2*i0]; ai = q2[2*i0+1]; cr = q2[2*im]; ci = q2[2*im+1];
    tr += ar*cr + ai*ci; ti += ai*cr - ar*ci;

    float g = atan2f(ti, tr);
    gradients[i0] = isnan(g) ? 0.0f : g;
}

__global__ void k_phase_grad_x(float* grad, const float* q1, const float* q2,
                                int W, int H, int D) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x < 1 || x >= W-1 || y >= H || z >= D) return;
    int i0 = x + y*W + z*W*H;
    phaseGradHelper(grad, q1, q2, i0, i0-1, i0+1);
}

__global__ void k_phase_grad_y(float* grad, const float* q1, const float* q2,
                                int W, int H, int D) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= W || y < 1 || y >= H-1 || z >= D) return;
    int i0 = x + y*W + z*W*H;
    phaseGradHelper(grad, q1, q2, i0, i0-W, i0+W);
}

__global__ void k_phase_grad_z(float* grad, const float* q1, const float* q2,
                                int W, int H, int D) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= W || y >= H || z < 1 || z >= D-1) return;
    int i0 = x + y*W + z*W*H;
    phaseGradHelper(grad, q1, q2, i0, i0-W*H, i0+W*H);
}

// --- A-matrix and h-vector 2D reduction ---

__global__ void k_amatrix_hvector_2d(
    float* A2D, float* h2D,
    const float* phaseDiff, const float* phaseGrad, const float* certainty,
    int W, int H, int D, int filterSize, int dirOff, int hOff)
{
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.y * blockDim.y + threadIdx.y;
    int fhalf = (filterSize - 1) / 2;
    if (y < fhalf || y >= H - fhalf || z < fhalf || z >= D - fhalf) return;

    float yf = (float)y - ((float)H - 1.0f) * 0.5f;
    float zf = (float)z - ((float)D - 1.0f) * 0.5f;

    float aval[10] = {};
    float hval[4] = {};

    for (int x = fhalf; x < W - fhalf; x++) {
        float xf = (float)x - ((float)W - 1.0f) * 0.5f;
        int i = x + y * W + z * W * H;
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

    int HD = H * D;
    int base = y + z * H + dirOff * HD;
    for (int k = 0; k < 10; k++)
        A2D[base + k * HD] = aval[k];

    int hBase = y + z * H + hOff * HD;
    h2D[hBase] = hval[0];
    int extraBase = y + z * H + (3 + hOff * 3) * HD;
    h2D[extraBase + 0 * HD] = hval[1];
    h2D[extraBase + 1 * HD] = hval[2];
    h2D[extraBase + 2 * HD] = hval[3];
}

// --- A-matrix reduction 1D and final ---

__global__ void k_amatrix_1d(float* A1D, const float* A2D,
                              int H, int D, int filterSize) {
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    int element = blockIdx.y * blockDim.y + threadIdx.y;
    int fhalf = (filterSize - 1) / 2;
    if (z < fhalf || z >= D - fhalf || element >= 30) return;

    float sum = 0.0f;
    int base = z * H + element * H * D;
    for (int y = fhalf; y < H - fhalf; y++)
        sum += A2D[base + y];
    A1D[z + element * D] = sum;
}

__constant__ int d_parameterIndices[60]; // 30 pairs of (row, col)

__global__ void k_amatrix_final(float* A, const float* A1D,
                                 int D, int filterSize) {
    int element = blockIdx.x * blockDim.x + threadIdx.x;
    if (element >= 30) return;
    int fhalf = (filterSize - 1) / 2;

    float sum = 0.0f;
    int base = element * D;
    for (int z = fhalf; z < D - fhalf; z++)
        sum += A1D[base + z];

    int row = d_parameterIndices[2 * element];
    int col = d_parameterIndices[2 * element + 1];
    A[row + col * 12] = sum;
}

// --- h-vector reduction ---

__global__ void k_hvector_1d(float* h1D, const float* h2D,
                              int H, int D, int filterSize) {
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    int element = blockIdx.y * blockDim.y + threadIdx.y;
    int fhalf = (filterSize - 1) / 2;
    if (z < fhalf || z >= D - fhalf || element >= 12) return;

    float sum = 0.0f;
    int base = z * H + element * H * D;
    for (int y = fhalf; y < H - fhalf; y++)
        sum += h2D[base + y];
    h1D[z + element * D] = sum;
}

// --- Tensor components (nonlinear) ---

__global__ void k_tensor_components(
    float* t11, float* t12, float* t13,
    float* t22, float* t23, float* t33,
    const float* q2, // complex, interleaved
    float m11, float m12, float m13, float m22, float m23, float m33,
    int W, int H, int D)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= W || y >= H || z >= D) return;

    int i = x + y * W + z * W * H;
    float qr = q2[2*i], qi = q2[2*i+1];
    float mag = sqrtf(qr * qr + qi * qi);

    t11[i] += mag * m11;
    t12[i] += mag * m12;
    t13[i] += mag * m13;
    t22[i] += mag * m22;
    t23[i] += mag * m23;
    t33[i] += mag * m33;
}

// --- Tensor norms (Frobenius) ---

__global__ void k_tensor_norms(
    float* norms,
    const float* t11, const float* t12, const float* t13,
    const float* t22, const float* t23, const float* t33,
    int W, int H, int D)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= W || y >= H || z >= D) return;

    int i = x + y * W + z * W * H;
    float v11 = t11[i], v12 = t12[i], v13 = t13[i];
    float v22 = t22[i], v23 = t23[i], v33 = t33[i];
    norms[i] = sqrtf(v11*v11 + 2*v12*v12 + 2*v13*v13 + v22*v22 + 2*v23*v23 + v33*v33);
}

// --- Nonlinear A-matrices and h-vectors ---

__global__ void k_amatrices_hvectors(
    float* a11, float* a12, float* a13,
    float* a22, float* a23, float* a33,
    float* h1, float* h2, float* h3,
    const float* q1, const float* q2,
    const float* t11, const float* t12, const float* t13,
    const float* t22, const float* t23, const float* t33,
    const float* fdx, const float* fdy, const float* fdz,
    int W, int H, int D, int FILTER)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= W || y >= H || z >= D) return;

    int i = x + y * W + z * W * H;

    float q1r = q1[2*i], q1i = q1[2*i+1];
    float q2r = q2[2*i], q2i = q2[2*i+1];

    float qqR = q1r * q2r + q1i * q2i;
    float qqI = -q1r * q2i + q1i * q2r;
    float pd = atan2f(qqI, qqR);
    float Aqq = sqrtf(qqR * qqR + qqI * qqI);
    float cosH = cosf(pd * 0.5f);
    float cert = sqrtf(Aqq) * cosH * cosH;

    float T11 = t11[i], T12 = t12[i], T13 = t13[i];
    float T22 = t22[i], T23 = t23[i], T33 = t33[i];

    // T^2
    float tt11 = T11*T11 + T12*T12 + T13*T13;
    float tt12 = T11*T12 + T12*T22 + T13*T23;
    float tt13 = T11*T13 + T12*T23 + T13*T33;
    float tt22 = T12*T12 + T22*T22 + T23*T23;
    float tt23 = T12*T13 + T22*T23 + T23*T33;
    float tt33 = T13*T13 + T23*T23 + T33*T33;

    float fx = fdx[FILTER], fy = fdy[FILTER], fz = fdz[FILTER];

    float cpd = cert * pd;
    if (isnan(cpd)) cpd = 0.0f;
    float hh1 = cpd * (fx * tt11 + fy * tt12 + fz * tt13);
    float hh2 = cpd * (fx * tt12 + fy * tt22 + fz * tt23);
    float hh3 = cpd * (fx * tt13 + fy * tt23 + fz * tt33);

    if (FILTER == 0) {
        a11[i] = cert * tt11; a12[i] = cert * tt12; a13[i] = cert * tt13;
        a22[i] = cert * tt22; a23[i] = cert * tt23; a33[i] = cert * tt33;
        h1[i] = hh1; h2[i] = hh2; h3[i] = hh3;
    } else {
        a11[i] += cert * tt11; a12[i] += cert * tt12; a13[i] += cert * tt13;
        a22[i] += cert * tt22; a23[i] += cert * tt23; a33[i] += cert * tt33;
        h1[i] += hh1; h2[i] += hh2; h3[i] += hh3;
    }
}

// --- Displacement update (Cramer's rule) ---

__global__ void k_displacement_update(
    float* dispX, float* dispY, float* dispZ,
    const float* a11, const float* a12, const float* a13,
    const float* a22, const float* a23, const float* a33,
    const float* h1, const float* h2, const float* h3,
    int W, int H, int D)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= W || y >= H || z >= D) return;

    int i = x + y * W + z * W * H;

    float A11 = a11[i], A12 = a12[i], A13 = a13[i];
    float A22 = a22[i], A23 = a23[i], A33 = a33[i];
    float H1 = h1[i], H2 = h2[i], H3 = h3[i];

    float det = A11*A22*A33 - A11*A23*A23 - A12*A12*A33
              + A12*A23*A13 + A13*A12*A23 - A13*A22*A13;

    float trace = A11 + A22 + A33;
    float epsilon = 0.01f * trace * trace * trace / 27.0f + 1e-16f;
    float norm = 0.2f / (det + epsilon);

    float dx = norm * (H1*(A22*A33-A23*A23) - H2*(A12*A33-A13*A23) + H3*(A12*A23-A13*A22));
    float dy = norm * (H2*(A11*A33-A13*A13) - H3*(A11*A23-A13*A12) - H1*(A12*A33-A23*A13));
    float dz = norm * (H3*(A11*A22-A12*A12) - H2*(A11*A23-A12*A13) + H1*(A12*A23-A22*A13));

    dispX[i] = (isnan(dx) || isinf(dx)) ? 0.0f : dx;
    dispY[i] = (isnan(dy) || isinf(dy)) ? 0.0f : dy;
    dispZ[i] = (isnan(dz) || isinf(dz)) ? 0.0f : dz;
}

// --- Interpolation kernels (texture-based trilinear) ---

__global__ void k_interpolate_linear(
    float* output, cudaTextureObject_t volume,
    const float* params, int W, int H, int D)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= W || y >= H || z >= D) return;

    float xf = (float)x - ((float)W - 1.0f) * 0.5f;
    float yf = (float)y - ((float)H - 1.0f) * 0.5f;
    float zf = (float)z - ((float)D - 1.0f) * 0.5f;

    float px = (float)x + params[0] + params[3]*xf + params[4]*yf  + params[5]*zf  + 0.5f;
    float py = (float)y + params[1] + params[6]*xf + params[7]*yf  + params[8]*zf  + 0.5f;
    float pz = (float)z + params[2] + params[9]*xf + params[10]*yf + params[11]*zf + 0.5f;

    output[x + y*W + z*W*H] = tex3D<float>(volume, px, py, pz);
}

__global__ void k_interpolate_nonlinear(
    float* output, cudaTextureObject_t volume,
    const float* dispX, const float* dispY, const float* dispZ,
    int W, int H, int D)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= W || y >= H || z >= D) return;

    int i = x + y*W + z*W*H;
    float px = (float)x + dispX[i] + 0.5f;
    float py = (float)y + dispY[i] + 0.5f;
    float pz = (float)z + dispZ[i] + 0.5f;

    output[i] = tex3D<float>(volume, px, py, pz);
}

// --- Rescale volume ---

__global__ void k_rescale(float* output, cudaTextureObject_t volume,
                           float scaleX, float scaleY, float scaleZ,
                           int W, int H, int D) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= W || y >= H || z >= D) return;

    float px = (float)x * scaleX + 0.5f;
    float py = (float)y * scaleY + 0.5f;
    float pz = (float)z * scaleZ + 0.5f;
    output[x + y*W + z*W*H] = tex3D<float>(volume, px, py, pz);
}

// --- Copy volume to new grid ---

__global__ void k_copy_volume_to_new(
    float* dst, const float* src,
    int newW, int newH, int newD,
    int srcW, int srcH, int srcD,
    int xDiff, int yDiff, int zDiff,
    int mmZCut, float voxelSizeZ)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int xNew, xSrc, yNew, ySrc, zNew, zSrc;

    if (xDiff > 0) { xNew = x; xSrc = x + (int)roundf((float)xDiff / 2.0f); }
    else { xNew = x + (int)roundf((float)(-xDiff) / 2.0f); xSrc = x; }
    if (yDiff > 0) { yNew = y; ySrc = y + (int)roundf((float)yDiff / 2.0f); }
    else { yNew = y + (int)roundf((float)(-yDiff) / 2.0f); ySrc = y; }
    if (zDiff > 0) {
        zNew = z; zSrc = z + (int)roundf((float)zDiff / 2.0f) + (int)roundf((float)mmZCut / voxelSizeZ);
    } else {
        zNew = z + (int)roundf((float)(-zDiff) / 2.0f);
        zSrc = z + (int)roundf((float)mmZCut / voxelSizeZ);
    }

    if (xSrc < 0 || xSrc >= srcW || ySrc < 0 || ySrc >= srcH || zSrc < 0 || zSrc >= srcD) return;
    if (xNew < 0 || xNew >= newW || yNew < 0 || yNew >= newH || zNew < 0 || zNew >= newD) return;

    dst[xNew + yNew * newW + zNew * newW * newH] = src[xSrc + ySrc * srcW + zSrc * srcW * srcH];
}

// --- Add linear + nonlinear displacement ---

__global__ void k_add_linear_nonlinear_disp(
    float* dispX, float* dispY, float* dispZ,
    const float* params, int W, int H, int D)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= W || y >= H || z >= D) return;

    float xf = (float)x - ((float)W - 1.0f) * 0.5f;
    float yf = (float)y - ((float)H - 1.0f) * 0.5f;
    float zf = (float)z - ((float)D - 1.0f) * 0.5f;

    int i = x + y*W + z*W*H;
    dispX[i] += params[0] + params[3]*xf + params[4]*yf  + params[5]*zf;
    dispY[i] += params[1] + params[6]*xf + params[7]*yf  + params[8]*zf;
    dispZ[i] += params[2] + params[9]*xf + params[10]*yf + params[11]*zf;
}

// ============================================================
//  Host-side helper functions
// ============================================================

void initParameterIndices() {
    static std::once_flag flag;
    std::call_once(flag, []() {
        int indices[60] = {
            0,0, 3,0, 4,0, 5,0, 3,3, 4,3, 5,3, 4,4, 5,4, 5,5,
            1,1, 6,1, 7,1, 8,1, 6,6, 7,6, 8,6, 7,7, 8,7, 8,8,
            2,2, 9,2, 10,2, 11,2, 9,9, 10,9, 11,9, 10,10, 11,10, 11,11
        };
        CUDA_CHECK(cudaMemcpyToSymbol(d_parameterIndices, indices, sizeof(indices)));
    });
}

// Fill buffer on device
void fillBuffer(float* buf, float value, int count) {
    k_fill<<<gridFor1D(count), BLOCK_1D>>>(buf, value, count);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void addVolumes(float* A, const float* B, int count) {
    k_add<<<gridFor1D(count), BLOCK_1D>>>(A, B, count);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void multiplyVolume(float* vol, float factor, int count) {
    k_multiply<<<gridFor1D(count), BLOCK_1D>>>(vol, factor, count);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void multiplyVolumes(float* A, const float* B, int count) {
    k_multiply_volumes<<<gridFor1D(count), BLOCK_1D>>>(A, B, count);
    CUDA_CHECK(cudaDeviceSynchronize());
}

float calculateMax(float* volume, int W, int H, int D) {
    auto& c = ctx();
    float* colMaxs = c.newBuffer(H * D);
    float* rowMaxs = c.newBuffer(D);

    dim3 block2(16, 16);
    dim3 grid2((H + 15) / 16, (D + 15) / 16);
    k_column_maxs<<<grid2, block2>>>(colMaxs, volume, W, H, D);
    k_row_maxs<<<gridFor1D(D), BLOCK_1D>>>(rowMaxs, colMaxs, H, D);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> hostMaxs(D);
    c.readBuffer(hostMaxs.data(), rowMaxs, D);

    float mx = hostMaxs[0];
    for (int i = 1; i < D; i++) mx = std::max(mx, hostMaxs[i]);

    c.freeBuffer(colMaxs);
    c.freeBuffer(rowMaxs);
    return mx;
}

// 3D convolution with 3 quadrature filters
void nonseparableConvolution3D(
    float* resp1, float* resp2, float* resp3,
    float* volume,
    const float* filterReal1, const float* filterImag1,
    const float* filterReal2, const float* filterImag2,
    const float* filterReal3, const float* filterImag3,
    int W, int H, int D)
{
    auto& c = ctx();

    // Upload filters to device
    float* d_f1r = c.newBuffer(filterReal1, 343);
    float* d_f1i = c.newBuffer(filterImag1, 343);
    float* d_f2r = c.newBuffer(filterReal2, 343);
    float* d_f2i = c.newBuffer(filterImag2, 343);
    float* d_f3r = c.newBuffer(filterReal3, 343);
    float* d_f3i = c.newBuffer(filterImag3, 343);

    // Create texture from volume (nearest-neighbor for convolution indexing)
    cudaArray_t array;
    cudaTextureObject_t tex = c.createTexture3D(volume, W, H, D, &array, false);

    dim3 grid = gridFor3D(W, H, D, BLOCK_3D);
    k_conv3d_full<<<grid, BLOCK_3D>>>(
        resp1, resp2, resp3, tex,
        d_f1r, d_f1i, d_f2r, d_f2i, d_f3r, d_f3i,
        W, H, D);
    CUDA_CHECK(cudaDeviceSynchronize());

    c.destroyTexture3D(tex, array);
    c.freeBuffer(d_f1r); c.freeBuffer(d_f1i);
    c.freeBuffer(d_f2r); c.freeBuffer(d_f2i);
    c.freeBuffer(d_f3r); c.freeBuffer(d_f3i);
}

// Smoothing filter creation
void createSmoothingFilter(float* filter, float sigma) {
    float sum = 0;
    for (int i = 0; i < 9; i++) {
        float x = (float)i - 4.0f;
        filter[i] = expf(-0.5f * x * x / (sigma * sigma));
        sum += filter[i];
    }
    for (int i = 0; i < 9; i++) filter[i] /= sum;
}

// 3-pass separable smoothing
void performSmoothing(float* output, float* input, int W, int H, int D,
                      const float* d_filter) {
    auto& c = ctx();
    float* temp1 = c.newBuffer(W * H * D);
    float* temp2 = c.newBuffer(W * H * D);

    dim3 grid = gridFor3D(W, H, D, BLOCK_3D);
    k_smooth_rows<<<grid, BLOCK_3D>>>(temp1, input, d_filter, W, H, D);
    CUDA_CHECK(cudaDeviceSynchronize());
    k_smooth_cols<<<grid, BLOCK_3D>>>(temp2, temp1, d_filter, W, H, D);
    CUDA_CHECK(cudaDeviceSynchronize());
    k_smooth_rods<<<grid, BLOCK_3D>>>(output, temp2, d_filter, W, H, D);
    CUDA_CHECK(cudaDeviceSynchronize());

    c.freeBuffer(temp1);
    c.freeBuffer(temp2);
}

void performSmoothingInPlace(float* volume, int W, int H, int D,
                              const float* d_filter) {
    auto& c = ctx();
    int vol = W * H * D;
    float* output = c.newBuffer(vol);
    performSmoothing(output, volume, W, H, D, d_filter);
    CUDA_CHECK(cudaMemcpy(volume, output, vol * sizeof(float), cudaMemcpyDeviceToDevice));
    c.freeBuffer(output);
}

// Batch smooth multiple volumes in place
void batchSmoothInPlace(std::initializer_list<float*> volumes, int W, int H, int D,
                         const float* d_filter) {
    for (float* vol : volumes)
        performSmoothingInPlace(vol, W, H, D, d_filter);
}

// Rescale volume
float* rescaleVolume(float* input, int srcW, int srcH, int srcD,
                     int dstW, int dstH, int dstD,
                     float scaleX, float scaleY, float scaleZ) {
    auto& c = ctx();

    cudaArray_t array;
    cudaTextureObject_t tex = c.createTexture3D(input, srcW, srcH, srcD, &array, true);

    float* output = c.newBuffer(dstW * dstH * dstD);
    dim3 grid = gridFor3D(dstW, dstH, dstD, BLOCK_3D);
    k_rescale<<<grid, BLOCK_3D>>>(output, tex, scaleX, scaleY, scaleZ, dstW, dstH, dstD);
    CUDA_CHECK(cudaDeviceSynchronize());

    c.destroyTexture3D(tex, array);
    return output;
}

float* copyVolumeToNew(float* src, int srcW, int srcH, int srcD,
                        int dstW, int dstH, int dstD,
                        int mmZCut, float voxelSizeZ) {
    auto& c = ctx();
    float* dst = c.newBuffer(dstW * dstH * dstD);

    int dispW = std::max(srcW, dstW);
    int dispH = std::max(srcH, dstH);
    int dispD = std::max(srcD, dstD);

    dim3 grid = gridFor3D(dispW, dispH, dispD, BLOCK_3D);
    k_copy_volume_to_new<<<grid, BLOCK_3D>>>(
        dst, src, dstW, dstH, dstD, srcW, srcH, srcD,
        srcW - dstW, srcH - dstH, srcD - dstD,
        mmZCut, voxelSizeZ);
    CUDA_CHECK(cudaDeviceSynchronize());

    return dst;
}

float* changeVolumesResolutionAndSize(
    float* input, int srcW, int srcH, int srcD,
    VoxelSize srcVox, int dstW, int dstH, int dstD,
    VoxelSize dstVox, int mmZCut)
{
    float scaleX = srcVox.x / dstVox.x;
    float scaleY = srcVox.y / dstVox.y;
    float scaleZ = srcVox.z / dstVox.z;

    int interpW = (int)roundf(srcW * scaleX);
    int interpH = (int)roundf(srcH * scaleY);
    int interpD = (int)roundf(srcD * scaleZ);

    float voxDiffX = (interpW > 1) ? (float)(srcW - 1) / (float)(interpW - 1) : 0.0f;
    float voxDiffY = (interpH > 1) ? (float)(srcH - 1) / (float)(interpH - 1) : 0.0f;
    float voxDiffZ = (interpD > 1) ? (float)(srcD - 1) / (float)(interpD - 1) : 0.0f;

    float* interpolated = rescaleVolume(input, srcW, srcH, srcD,
                                         interpW, interpH, interpD,
                                         voxDiffX, voxDiffY, voxDiffZ);

    float* result = copyVolumeToNew(interpolated, interpW, interpH, interpD,
                                     dstW, dstH, dstD, mmZCut, dstVox.z);
    ctx().freeBuffer(interpolated);
    return result;
}

float* changeVolumeSize(float* input, int srcW, int srcH, int srcD,
                         int dstW, int dstH, int dstD) {
    float scaleX = (dstW > 1) ? (float)(srcW - 1) / (float)(dstW - 1) : 0.0f;
    float scaleY = (dstH > 1) ? (float)(srcH - 1) / (float)(dstH - 1) : 0.0f;
    float scaleZ = (dstD > 1) ? (float)(srcD - 1) / (float)(dstD - 1) : 0.0f;
    return rescaleVolume(input, srcW, srcH, srcD, dstW, dstH, dstD,
                         scaleX, scaleY, scaleZ);
}

// Center of mass (CPU)
void centerOfMass(const float* vol, int W, int H, int D,
                  float& cx, float& cy, float& cz) {
    double sum = 0, sx = 0, sy = 0, sz = 0;
    for (int z = 0; z < D; z++)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                float v = vol[x + y * W + z * W * H];
                if (v > 0) {
                    sum += v; sx += v * x; sy += v * y; sz += v * z;
                }
            }
    if (sum > 0) { cx = sx/sum; cy = sy/sum; cz = sz/sum; }
    else { cx = W*0.5f; cy = H*0.5f; cz = D*0.5f; }
}

// GPU interpolation with affine params
void interpolateLinear(float* output, float* volume,
                       const float* params, int W, int H, int D) {
    auto& c = ctx();
    cudaArray_t array;
    cudaTextureObject_t tex = c.createTexture3D(volume, W, H, D, &array, true);

    float* d_params = c.newBuffer(params, 12);
    dim3 grid = gridFor3D(W, H, D, BLOCK_3D);
    k_interpolate_linear<<<grid, BLOCK_3D>>>(output, tex, d_params, W, H, D);
    CUDA_CHECK(cudaDeviceSynchronize());

    c.destroyTexture3D(tex, array);
    c.freeBuffer(d_params);
}

void interpolateNonLinear(float* output, float* volume,
                           float* dispX, float* dispY, float* dispZ,
                           int W, int H, int D) {
    auto& c = ctx();
    cudaArray_t array;
    cudaTextureObject_t tex = c.createTexture3D(volume, W, H, D, &array, true);

    dim3 grid = gridFor3D(W, H, D, BLOCK_3D);
    k_interpolate_nonlinear<<<grid, BLOCK_3D>>>(output, tex, dispX, dispY, dispZ, W, H, D);
    CUDA_CHECK(cudaDeviceSynchronize());

    c.destroyTexture3D(tex, array);
}

void addLinearNonLinearDisplacement(float* dispX, float* dispY, float* dispZ,
                                    const float* params, int W, int H, int D) {
    auto& c = ctx();
    float* d_params = c.newBuffer(params, 12);
    dim3 grid = gridFor3D(W, H, D, BLOCK_3D);
    k_add_linear_nonlinear_disp<<<grid, BLOCK_3D>>>(dispX, dispY, dispZ, d_params, W, H, D);
    CUDA_CHECK(cudaDeviceSynchronize());
    c.freeBuffer(d_params);
}

// Solve equation system (CPU, Gaussian elimination with partial pivoting)
void solveEquationSystem(float* A, float* h, double* params, int n) {
    double Ad[144], hd[12];
    for (int i = 0; i < n * n; i++) Ad[i] = A[i];
    for (int i = 0; i < n; i++) hd[i] = h[i];

    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            Ad[i * n + j] = Ad[j * n + i];

    double aug[12][13];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            aug[i][j] = Ad[j * n + i];
        aug[i][n] = hd[i];
    }

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
            for (int j = 0; j <= n; j++)
                std::swap(aug[col][j], aug[pivotRow][j]);
        }
        for (int row = col + 1; row < n; row++) {
            double factor = aug[row][col] / aug[col][col];
            for (int j = col; j <= n; j++)
                aug[row][j] -= factor * aug[col][j];
        }
    }

    for (int row = n - 1; row >= 0; row--) {
        double sum = aug[row][n];
        for (int j = row + 1; j < n; j++)
            sum -= aug[row][j] * params[j];
        params[row] = sum / aug[row][row];
    }
}

// Affine parameter composition
static void paramsToMatrix(const float* p, double M[4][4], float translationScale = 1.0f) {
    M[0][0] = p[3]+1.0; M[0][1] = p[4];     M[0][2] = p[5];     M[0][3] = p[0]*translationScale;
    M[1][0] = p[6];     M[1][1] = p[7]+1.0;  M[1][2] = p[8];     M[1][3] = p[1]*translationScale;
    M[2][0] = p[9];     M[2][1] = p[10];     M[2][2] = p[11]+1.0; M[2][3] = p[2]*translationScale;
    M[3][0] = 0;        M[3][1] = 0;         M[3][2] = 0;         M[3][3] = 1.0;
}

static void matrixToParams(const double M[4][4], float* p) {
    p[0]  = (float)M[0][3]; p[1] = (float)M[1][3]; p[2]  = (float)M[2][3];
    p[3]  = (float)(M[0][0]-1.0); p[4] = (float)M[0][1]; p[5]  = (float)M[0][2];
    p[6]  = (float)M[1][0]; p[7] = (float)(M[1][1]-1.0); p[8]  = (float)M[1][2];
    p[9]  = (float)M[2][0]; p[10] = (float)M[2][1]; p[11] = (float)(M[2][2]-1.0);
}

static void matMul4x4(const double A[4][4], const double B[4][4], double C[4][4]) {
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            C[i][j] = 0;
            for (int k = 0; k < 4; k++)
                C[i][j] += A[i][k] * B[k][j];
        }
}

void composeAffineParams(float* oldParams, const float* newParams) {
    double O[4][4], N[4][4], T[4][4];
    paramsToMatrix(oldParams, O);
    paramsToMatrix(newParams, N);
    matMul4x4(N, O, T);
    matrixToParams(T, oldParams);
}

void composeAffineParamsNextScale(float* oldParams, const float* newParams) {
    double O[4][4], N[4][4], T[4][4];
    paramsToMatrix(oldParams, O, 2.0f);
    paramsToMatrix(newParams, N, 2.0f);
    matMul4x4(N, O, T);
    matrixToParams(T, oldParams);
}

// ============================================================
//  LINEAR REGISTRATION
// ============================================================

void alignTwoVolumesLinear(
    float* alignedVolume,
    float* referenceVolume,
    const QuadratureFilters& filters,
    int W, int H, int D,
    int filterSize,
    int numIterations,
    float* registrationParams,
    bool verbose)
{
    auto& c = ctx();
    int vol = W * H * D;

    initParameterIndices();

    float* originalAligned = c.newBuffer(vol);
    CUDA_CHECK(cudaMemcpy(originalAligned, alignedVolume, vol * sizeof(float), cudaMemcpyDeviceToDevice));

    for (int i = 0; i < 12; i++) registrationParams[i] = 0.0f;

    // Filter response buffers (complex = 2 floats per element)
    float* q11 = c.newBuffer(vol * 2);
    float* q12 = c.newBuffer(vol * 2);
    float* q13 = c.newBuffer(vol * 2);
    float* q21 = c.newBuffer(vol * 2);
    float* q22 = c.newBuffer(vol * 2);
    float* q23 = c.newBuffer(vol * 2);

    float* phaseDiff = c.newBuffer(vol);
    float* certainties = c.newBuffer(vol);
    float* phaseGrad = c.newBuffer(vol);

    int HD = H * D;
    float* A2D = c.newBuffer(30 * HD);
    float* A1D = c.newBuffer(30 * D);
    float* Amat = c.newBuffer(144);
    float* h2D = c.newBuffer(12 * HD);
    float* h1D = c.newBuffer(12 * D);
    float* hvec = c.newBuffer(12);

    // Filter reference once
    nonseparableConvolution3D(q11, q12, q13, referenceVolume,
        filters.linearReal[0].data(), filters.linearImag[0].data(),
        filters.linearReal[1].data(), filters.linearImag[1].data(),
        filters.linearReal[2].data(), filters.linearImag[2].data(),
        W, H, D);

    dim3 grid3 = gridFor3D(W, H, D, BLOCK_3D);

    for (int iter = 0; iter < numIterations; iter++) {
        // Filter aligned volume
        nonseparableConvolution3D(q21, q22, q23, alignedVolume,
            filters.linearReal[0].data(), filters.linearImag[0].data(),
            filters.linearReal[1].data(), filters.linearImag[1].data(),
            filters.linearReal[2].data(), filters.linearImag[2].data(),
            W, H, D);

        // Zero intermediate buffers
        CUDA_CHECK(cudaMemset(A2D, 0, 30 * HD * sizeof(float)));
        CUDA_CHECK(cudaMemset(h2D, 0, 12 * HD * sizeof(float)));

        // Process each direction
        struct { int dirOff; int hOff; float* q1; float* q2; } dirs[3] = {
            {0, 0, q11, q21}, {10, 1, q12, q22}, {20, 2, q13, q23}
        };

        for (int d = 0; d < 3; d++) {
            // Phase differences + certainties
            k_phase_diff_cert<<<grid3, BLOCK_3D>>>(
                phaseDiff, certainties, dirs[d].q1, dirs[d].q2, W, H, D);

            // Phase gradients
            if (d == 0) k_phase_grad_x<<<grid3, BLOCK_3D>>>(phaseGrad, dirs[d].q1, dirs[d].q2, W, H, D);
            else if (d == 1) k_phase_grad_y<<<grid3, BLOCK_3D>>>(phaseGrad, dirs[d].q1, dirs[d].q2, W, H, D);
            else k_phase_grad_z<<<grid3, BLOCK_3D>>>(phaseGrad, dirs[d].q1, dirs[d].q2, W, H, D);
            CUDA_CHECK(cudaDeviceSynchronize());

            // A-matrix and h-vector 2D
            dim3 block2(16, 16);
            dim3 grid2((H + 15)/16, (D + 15)/16);
            k_amatrix_hvector_2d<<<grid2, block2>>>(
                A2D, h2D, phaseDiff, phaseGrad, certainties,
                W, H, D, filterSize, dirs[d].dirOff, dirs[d].hOff);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Reduce A: 2D -> 1D
        {
            dim3 block2(16, 16);
            dim3 gridA((D + 15)/16, (30 + 15)/16);
            dim3 gridH((D + 15)/16, (12 + 15)/16);
            k_amatrix_1d<<<gridA, block2>>>(A1D, A2D, H, D, filterSize);
            k_hvector_1d<<<gridH, block2>>>(h1D, h2D, H, D, filterSize);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // A final
        CUDA_CHECK(cudaMemset(Amat, 0, 144 * sizeof(float)));
        k_amatrix_final<<<gridFor1D(30), BLOCK_1D>>>(Amat, A1D, D, filterSize);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Read back and solve on CPU
        float hA[144], hh[12];
        c.readBuffer(hA, Amat, 144);

        // h-vector final reduction on CPU (matches Metal)
        {
            int fhalf = (filterSize - 1) / 2;
            std::vector<float> h1Dp(12 * D);
            c.readBuffer(h1Dp.data(), h1D, 12 * D);
            for (int elem = 0; elem < 12; elem++) {
                float sum = 0;
                for (int z = fhalf; z < D - fhalf; z++)
                    sum += h1Dp[elem * D + z];
                hh[elem] = sum;
            }
        }

        double paramsDbl[12];
        solveEquationSystem(hA, hh, paramsDbl, 12);

        float deltaParams[12];
        for (int i = 0; i < 12; i++) deltaParams[i] = (float)paramsDbl[i];
        composeAffineParams(registrationParams, deltaParams);

        // Apply from original
        interpolateLinear(alignedVolume, originalAligned, registrationParams, W, H, D);
    }

    c.freeBuffer(originalAligned);
    c.freeBuffer(q11); c.freeBuffer(q12); c.freeBuffer(q13);
    c.freeBuffer(q21); c.freeBuffer(q22); c.freeBuffer(q23);
    c.freeBuffer(phaseDiff); c.freeBuffer(certainties); c.freeBuffer(phaseGrad);
    c.freeBuffer(A2D); c.freeBuffer(A1D); c.freeBuffer(Amat);
    c.freeBuffer(h2D); c.freeBuffer(h1D); c.freeBuffer(hvec);
}

void alignTwoVolumesLinearSeveralScales(
    float*& alignedVolume,
    float* referenceVolume,
    const QuadratureFilters& filters,
    int W, int H, int D,
    int filterSize,
    int numIterations,
    int coarsestScale,
    float* registrationParams,
    bool verbose)
{
    auto& c = ctx();
    int vol = W * H * D;

    for (int i = 0; i < 12; i++) registrationParams[i] = 0.0f;

    float* originalAligned = c.newBuffer(vol);
    CUDA_CHECK(cudaMemcpy(originalAligned, alignedVolume, vol * sizeof(float), cudaMemcpyDeviceToDevice));

    for (int scale = coarsestScale; scale >= 1; scale /= 2) {
        int sW = (int)roundf((float)W / (float)scale);
        int sH = (int)roundf((float)H / (float)scale);
        int sD = (int)roundf((float)D / (float)scale);
        if (sW < 8 || sH < 8 || sD < 8) continue;

        float* scaledRef = (scale == 1) ? referenceVolume :
            changeVolumeSize(referenceVolume, W, H, D, sW, sH, sD);
        float* scaledAligned = (scale == 1) ? alignedVolume :
            changeVolumeSize(originalAligned, W, H, D, sW, sH, sD);

        if (scale < coarsestScale)
            interpolateLinear(scaledAligned, scaledAligned, registrationParams, sW, sH, sD);

        int iters = (scale == 1) ? (int)ceilf((float)numIterations / 5.0f) : numIterations;

        if (verbose) printf("  Linear scale %d: %dx%dx%d, %d iterations\n", scale, sW, sH, sD, iters);

        float tempParams[12] = {0};
        alignTwoVolumesLinear(scaledAligned, scaledRef, filters,
                              sW, sH, sD, filterSize, iters, tempParams, verbose);

        if (scale != 1) {
            composeAffineParamsNextScale(registrationParams, tempParams);
            c.freeBuffer(scaledRef);
            c.freeBuffer(scaledAligned);
        } else {
            composeAffineParams(registrationParams, tempParams);
        }
    }

    interpolateLinear(alignedVolume, originalAligned, registrationParams, W, H, D);
    c.freeBuffer(originalAligned);
}

// ============================================================
//  NONLINEAR REGISTRATION
// ============================================================

void alignTwoVolumesNonLinear(
    float* alignedVolume,
    float* referenceVolume,
    const QuadratureFilters& filters,
    int W, int H, int D,
    int numIterations,
    float* updateDispX, float* updateDispY, float* updateDispZ,
    bool verbose)
{
    auto& c = ctx();
    int vol = W * H * D;

    float* q1[6], *q2[6];
    for (int i = 0; i < 6; i++) {
        q1[i] = c.newBuffer(vol * 2);
        q2[i] = c.newBuffer(vol * 2);
    }

    float* t11 = c.newBuffer(vol); float* t12 = c.newBuffer(vol);
    float* t13 = c.newBuffer(vol); float* t22 = c.newBuffer(vol);
    float* t23 = c.newBuffer(vol); float* t33 = c.newBuffer(vol);

    float* a11 = c.newBuffer(vol); float* a12 = c.newBuffer(vol);
    float* a13 = c.newBuffer(vol); float* a22 = c.newBuffer(vol);
    float* a23 = c.newBuffer(vol); float* a33 = c.newBuffer(vol);
    float* h1 = c.newBuffer(vol); float* h2 = c.newBuffer(vol);
    float* h3 = c.newBuffer(vol);

    float* tensorNorms = c.newBuffer(vol);
    float* dux = c.newBuffer(vol); float* duy = c.newBuffer(vol);
    float* duz = c.newBuffer(vol);

    float* originalAligned = c.newBuffer(vol);
    CUDA_CHECK(cudaMemcpy(originalAligned, alignedVolume, vol * sizeof(float), cudaMemcpyDeviceToDevice));

    // Upload filter directions
    float* d_fdx = c.newBuffer(filters.filterDirectionsX, 6);
    float* d_fdy = c.newBuffer(filters.filterDirectionsY, 6);
    float* d_fdz = c.newBuffer(filters.filterDirectionsZ, 6);

    // Smoothing filters (device)
    float smoothTensor[9], smoothEq[9], smoothDisp[9];
    createSmoothingFilter(smoothTensor, 1.0f);
    createSmoothingFilter(smoothEq, 2.0f);
    createSmoothingFilter(smoothDisp, 2.0f);
    float* d_smoothTensor = c.newBuffer(smoothTensor, 9);
    float* d_smoothEq = c.newBuffer(smoothEq, 9);
    float* d_smoothDisp = c.newBuffer(smoothDisp, 9);

    dim3 grid3 = gridFor3D(W, H, D, BLOCK_3D);

    // Filter reference once
    nonseparableConvolution3D(q1[0], q1[1], q1[2], referenceVolume,
        filters.nonlinearReal[0].data(), filters.nonlinearImag[0].data(),
        filters.nonlinearReal[1].data(), filters.nonlinearImag[1].data(),
        filters.nonlinearReal[2].data(), filters.nonlinearImag[2].data(),
        W, H, D);
    nonseparableConvolution3D(q1[3], q1[4], q1[5], referenceVolume,
        filters.nonlinearReal[3].data(), filters.nonlinearImag[3].data(),
        filters.nonlinearReal[4].data(), filters.nonlinearImag[4].data(),
        filters.nonlinearReal[5].data(), filters.nonlinearImag[5].data(),
        W, H, D);

    for (int iter = 0; iter < numIterations; iter++) {
        if (verbose) printf("    Nonlinear iter %d/%d\n", iter + 1, numIterations);

        // Filter aligned
        nonseparableConvolution3D(q2[0], q2[1], q2[2], alignedVolume,
            filters.nonlinearReal[0].data(), filters.nonlinearImag[0].data(),
            filters.nonlinearReal[1].data(), filters.nonlinearImag[1].data(),
            filters.nonlinearReal[2].data(), filters.nonlinearImag[2].data(),
            W, H, D);
        nonseparableConvolution3D(q2[3], q2[4], q2[5], alignedVolume,
            filters.nonlinearReal[3].data(), filters.nonlinearImag[3].data(),
            filters.nonlinearReal[4].data(), filters.nonlinearImag[4].data(),
            filters.nonlinearReal[5].data(), filters.nonlinearImag[5].data(),
            W, H, D);

        // Zero tensors + displacement update
        for (auto buf : {t11, t12, t13, t22, t23, t33, dux, duy, duz})
            CUDA_CHECK(cudaMemset(buf, 0, vol * sizeof(float)));

        // Tensor components (6 filters)
        for (int f = 0; f < 6; f++) {
            const float* pt = filters.projectionTensors[f];
            k_tensor_components<<<grid3, BLOCK_3D>>>(
                t11, t12, t13, t22, t23, t33,
                q2[f], pt[0], pt[1], pt[2], pt[3], pt[4], pt[5],
                W, H, D);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Tensor norms (pre-smooth)
        k_tensor_norms<<<grid3, BLOCK_3D>>>(tensorNorms, t11, t12, t13, t22, t23, t33, W, H, D);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Smooth tensors
        batchSmoothInPlace({t11, t12, t13, t22, t23, t33}, W, H, D, d_smoothTensor);

        // Tensor norms (post-smooth) + normalize
        k_tensor_norms<<<grid3, BLOCK_3D>>>(tensorNorms, t11, t12, t13, t22, t23, t33, W, H, D);
        CUDA_CHECK(cudaDeviceSynchronize());

        float maxNorm = calculateMax(tensorNorms, W, H, D);
        if (maxNorm > 0) {
            float invMax = 1.0f / maxNorm;
            for (auto buf : {t11, t12, t13, t22, t23, t33})
                multiplyVolume(buf, invMax, vol);
        }

        // A-matrices and h-vectors (6 filters)
        for (int f = 0; f < 6; f++) {
            k_amatrices_hvectors<<<grid3, BLOCK_3D>>>(
                a11, a12, a13, a22, a23, a33, h1, h2, h3,
                q1[f], q2[f], t11, t12, t13, t22, t23, t33,
                d_fdx, d_fdy, d_fdz, W, H, D, f);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Smooth A + h
        batchSmoothInPlace({a11, a12, a13, a22, a23, a33, h1, h2, h3}, W, H, D, d_smoothEq);

        // Displacement update
        k_displacement_update<<<grid3, BLOCK_3D>>>(
            dux, duy, duz, a11, a12, a13, a22, a23, a33, h1, h2, h3, W, H, D);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Smooth displacement
        batchSmoothInPlace({dux, duy, duz}, W, H, D, d_smoothDisp);

        // Accumulate
        addVolumes(updateDispX, dux, vol);
        addVolumes(updateDispY, duy, vol);
        addVolumes(updateDispZ, duz, vol);

        // Re-interpolate
        interpolateNonLinear(alignedVolume, originalAligned,
                             updateDispX, updateDispY, updateDispZ, W, H, D);
    }

    // Cleanup
    for (int i = 0; i < 6; i++) { c.freeBuffer(q1[i]); c.freeBuffer(q2[i]); }
    c.freeBuffer(t11); c.freeBuffer(t12); c.freeBuffer(t13);
    c.freeBuffer(t22); c.freeBuffer(t23); c.freeBuffer(t33);
    c.freeBuffer(a11); c.freeBuffer(a12); c.freeBuffer(a13);
    c.freeBuffer(a22); c.freeBuffer(a23); c.freeBuffer(a33);
    c.freeBuffer(h1); c.freeBuffer(h2); c.freeBuffer(h3);
    c.freeBuffer(tensorNorms);
    c.freeBuffer(dux); c.freeBuffer(duy); c.freeBuffer(duz);
    c.freeBuffer(originalAligned);
    c.freeBuffer(d_fdx); c.freeBuffer(d_fdy); c.freeBuffer(d_fdz);
    c.freeBuffer(d_smoothTensor); c.freeBuffer(d_smoothEq); c.freeBuffer(d_smoothDisp);
}

void alignTwoVolumesNonLinearSeveralScales(
    float*& alignedVolume,
    float* referenceVolume,
    const QuadratureFilters& filters,
    int W, int H, int D,
    int numIterations,
    int coarsestScale,
    float*& totalDispX, float*& totalDispY, float*& totalDispZ,
    bool verbose)
{
    auto& c = ctx();
    int vol = W * H * D;

    float* originalAligned = c.newBuffer(vol);
    CUDA_CHECK(cudaMemcpy(originalAligned, alignedVolume, vol * sizeof(float), cudaMemcpyDeviceToDevice));

    totalDispX = c.newBuffer(vol);
    totalDispY = c.newBuffer(vol);
    totalDispZ = c.newBuffer(vol);

    for (int scale = coarsestScale; scale >= 1; scale /= 2) {
        int sW = W / scale, sH = H / scale, sD = D / scale;
        if (sW < 8 || sH < 8 || sD < 8) continue;

        if (verbose) printf("  Nonlinear scale %d: %dx%dx%d\n", scale, sW, sH, sD);

        float* scaledRef = (scale == 1) ? referenceVolume :
            changeVolumeSize(referenceVolume, W, H, D, sW, sH, sD);
        float* scaledAligned = (scale == 1) ? alignedVolume :
            changeVolumeSize(alignedVolume, W, H, D, sW, sH, sD);

        int sVol = sW * sH * sD;
        float* updateX = c.newBuffer(sVol);
        float* updateY = c.newBuffer(sVol);
        float* updateZ = c.newBuffer(sVol);

        alignTwoVolumesNonLinear(scaledAligned, scaledRef, filters,
                                  sW, sH, sD, numIterations,
                                  updateX, updateY, updateZ, verbose);

        if (scale > 1) {
            float* rescX = changeVolumeSize(updateX, sW, sH, sD, W, H, D);
            float* rescY = changeVolumeSize(updateY, sW, sH, sD, W, H, D);
            float* rescZ = changeVolumeSize(updateZ, sW, sH, sD, W, H, D);
            multiplyVolume(rescX, (float)scale, vol);
            multiplyVolume(rescY, (float)scale, vol);
            multiplyVolume(rescZ, (float)scale, vol);
            addVolumes(totalDispX, rescX, vol);
            addVolumes(totalDispY, rescY, vol);
            addVolumes(totalDispZ, rescZ, vol);
            c.freeBuffer(rescX); c.freeBuffer(rescY); c.freeBuffer(rescZ);

            interpolateNonLinear(alignedVolume, originalAligned,
                                 totalDispX, totalDispY, totalDispZ, W, H, D);

            c.freeBuffer(scaledRef);
            c.freeBuffer(scaledAligned);
        } else {
            addVolumes(totalDispX, updateX, vol);
            addVolumes(totalDispY, updateY, vol);
            addVolumes(totalDispZ, updateZ, vol);
        }
        c.freeBuffer(updateX); c.freeBuffer(updateY); c.freeBuffer(updateZ);
    }

    c.freeBuffer(originalAligned);
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
    int linearIterations, int nonlinearIterations, int coarsestScale,
    int mmZCut, bool verbose)
{
    try {
    auto& c = ctx();
    int mniVol = mniDims.size();

    if (verbose) printf("registerT1MNI: T1 %dx%dx%d -> MNI %dx%dx%d\n",
                        t1Dims.W, t1Dims.H, t1Dims.D,
                        mniDims.W, mniDims.H, mniDims.D);

    // Upload volumes
    float* mniBuf = c.newBuffer(mniData, mniVol);
    float* mniBrainBuf = c.newBuffer(mniBrainData, mniVol);
    float* mniMaskBuf = c.newBuffer(mniMaskData, mniVol);
    float* t1Buf = c.newBuffer(t1Data, t1Dims.size());

    // Resample T1 to MNI resolution
    float* t1InMNI = changeVolumesResolutionAndSize(
        t1Buf, t1Dims.W, t1Dims.H, t1Dims.D,
        {t1Vox.x, t1Vox.y, t1Vox.z},
        mniDims.W, mniDims.H, mniDims.D,
        {mniVox.x, mniVox.y, mniVox.z}, mmZCut);

    // Center-of-mass alignment
    std::vector<float> hostMNI(mniVol), hostT1(mniVol);
    c.readBuffer(hostMNI.data(), mniBuf, mniVol);
    c.readBuffer(hostT1.data(), t1InMNI, mniVol);

    float cx1, cy1, cz1, cx2, cy2, cz2;
    centerOfMass(hostMNI.data(), mniDims.W, mniDims.H, mniDims.D, cx1, cy1, cz1);
    centerOfMass(hostT1.data(), mniDims.W, mniDims.H, mniDims.D, cx2, cy2, cz2);

    float initParams[12] = {0};
    initParams[0] = roundf(cx2 - cx1);
    initParams[1] = roundf(cy2 - cy1);
    initParams[2] = roundf(cz2 - cz1);

    interpolateLinear(t1InMNI, t1InMNI, initParams, mniDims.W, mniDims.H, mniDims.D);

    std::vector<float> interpResult(mniVol);
    c.readBuffer(interpResult.data(), t1InMNI, mniVol);

    // Linear registration
    float regParams[12] = {0};

    if (verbose) printf("Running linear registration (%d iterations)...\n", linearIterations);

    alignTwoVolumesLinearSeveralScales(
        t1InMNI, mniBuf, filters,
        mniDims.W, mniDims.H, mniDims.D,
        7, linearIterations, coarsestScale,
        regParams, verbose);

    composeAffineParams(regParams, initParams);

    // Save linear result with single-step interpolation
    T1MNIResult result;
    {
        float* freshT1 = changeVolumesResolutionAndSize(
            t1Buf, t1Dims.W, t1Dims.H, t1Dims.D,
            {t1Vox.x, t1Vox.y, t1Vox.z},
            mniDims.W, mniDims.H, mniDims.D,
            {mniVox.x, mniVox.y, mniVox.z}, mmZCut);
        interpolateLinear(freshT1, freshT1, regParams, mniDims.W, mniDims.H, mniDims.D);
        result.alignedLinear.resize(mniVol);
        c.readBuffer(result.alignedLinear.data(), freshT1, mniVol);
        c.freeBuffer(freshT1);
    }
    for (int i = 0; i < 12; i++) result.params[i] = regParams[i];

    // Nonlinear registration
    if (nonlinearIterations > 0) {
        if (verbose) printf("Running nonlinear registration (%d iterations)...\n", nonlinearIterations);

        float* totalDispX, *totalDispY, *totalDispZ;

        alignTwoVolumesNonLinearSeveralScales(
            t1InMNI, mniBuf, filters,
            mniDims.W, mniDims.H, mniDims.D,
            nonlinearIterations, coarsestScale,
            totalDispX, totalDispY, totalDispZ, verbose);

        addLinearNonLinearDisplacement(totalDispX, totalDispY, totalDispZ,
                                       regParams, mniDims.W, mniDims.H, mniDims.D);

        result.dispX.resize(mniVol);
        result.dispY.resize(mniVol);
        result.dispZ.resize(mniVol);
        c.readBuffer(result.dispX.data(), totalDispX, mniVol);
        c.readBuffer(result.dispY.data(), totalDispY, mniVol);
        c.readBuffer(result.dispZ.data(), totalDispZ, mniVol);

        // Single-step nonlinear resample
        {
            float* freshT1 = changeVolumesResolutionAndSize(
                t1Buf, t1Dims.W, t1Dims.H, t1Dims.D,
                {t1Vox.x, t1Vox.y, t1Vox.z},
                mniDims.W, mniDims.H, mniDims.D,
                {mniVox.x, mniVox.y, mniVox.z}, mmZCut);
            interpolateNonLinear(freshT1, freshT1, totalDispX, totalDispY, totalDispZ,
                                 mniDims.W, mniDims.H, mniDims.D);
            result.alignedNonLinear.resize(mniVol);
            c.readBuffer(result.alignedNonLinear.data(), freshT1, mniVol);

            multiplyVolumes(freshT1, mniMaskBuf, mniVol);
            result.skullstripped.resize(mniVol);
            c.readBuffer(result.skullstripped.data(), freshT1, mniVol);
            c.freeBuffer(freshT1);
        }

        c.freeBuffer(totalDispX); c.freeBuffer(totalDispY); c.freeBuffer(totalDispZ);
    } else {
        result.alignedNonLinear = result.alignedLinear;
        result.skullstripped.resize(mniVol, 0);
    }

    result.interpolated = std::move(interpResult);

    // Cleanup
    c.freeBuffer(mniBuf); c.freeBuffer(mniBrainBuf);
    c.freeBuffer(mniMaskBuf); c.freeBuffer(t1Buf);
    c.freeBuffer(t1InMNI);

    return result;

    } catch (const std::runtime_error& e) {
        fprintf(stderr, "%s\n", e.what());
        return T1MNIResult{};
    }
}

} // namespace cuda_reg
