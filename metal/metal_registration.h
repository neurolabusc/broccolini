// metal_registration.h — Public C++ API for Metal-based image registration
#pragma once

#include <vector>
#include <array>
#include <cstdint>

namespace metal_reg {

struct VolumeDims {
    int W, H, D;
    int size() const { return W * H * D; }
};

struct VoxelSize {
    float x, y, z;
};

// Registration results for EPI -> T1
struct EPIT1Result {
    std::vector<float> aligned;       // EPI aligned in T1 space
    std::vector<float> interpolated;  // EPI interpolated (before alignment)
    std::array<float, 6> params;      // 6 rigid-body parameters
};

// Registration results for T1 -> MNI
struct T1MNIResult {
    std::vector<float> alignedLinear;      // T1 after affine
    std::vector<float> alignedNonLinear;   // T1 after affine + nonlinear
    std::vector<float> skullstripped;      // Nonlinear masked by MNI brain
    std::vector<float> interpolated;       // T1 resliced to MNI space
    std::array<float, 12> params;          // 12 affine parameters
    std::vector<float> dispX, dispY, dispZ; // Displacement fields
};

// Quadrature filters for registration
struct QuadratureFilters {
    // 3 parametric filters (linear registration), each 7x7x7 complex
    // Stored as separate real/imag arrays, each of size 343 (7^3)
    std::vector<float> linearReal[3];
    std::vector<float> linearImag[3];

    // 6 nonparametric filters (nonlinear registration)
    std::vector<float> nonlinearReal[6];
    std::vector<float> nonlinearImag[6];

    // Projection tensors (6 symmetric 3x3 matrices, stored as m11,m12,m13,m22,m23,m33)
    float projectionTensors[6][6];

    // Filter directions (3 components per filter)
    float filterDirectionsX[6];
    float filterDirectionsY[6];
    float filterDirectionsZ[6];
};

// Main registration functions
EPIT1Result registerEPIT1(
    const float* epiData, VolumeDims epiDims, VoxelSize epiVox,
    const float* t1Data, VolumeDims t1Dims, VoxelSize t1Vox,
    const QuadratureFilters& filters,
    int numIterations = 20,
    int coarsestScale = 8,
    int mmZCut = 30,
    bool verbose = false);

T1MNIResult registerT1MNI(
    const float* t1Data, VolumeDims t1Dims, VoxelSize t1Vox,
    const float* mniData, VolumeDims mniDims, VoxelSize mniVox,
    const float* mniBrainData,
    const float* mniMaskData,
    const QuadratureFilters& filters,
    int linearIterations = 10,
    int nonlinearIterations = 5,
    int coarsestScale = 4,
    int mmZCut = 30,
    bool verbose = false);

} // namespace metal_reg
