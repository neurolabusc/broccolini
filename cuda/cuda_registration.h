// cuda_registration.h — Public C++ API for CUDA-based image registration
#pragma once

#include <vector>
#include <array>
#include <cstdint>

namespace cuda_reg {

struct VolumeDims {
    int W, H, D;
    int size() const { return W * H * D; }
};

struct VoxelSize {
    float x, y, z;
};

// Registration results for T1 -> MNI (universal path)
struct T1MNIResult {
    std::vector<float> alignedLinear;
    std::vector<float> alignedNonLinear;
    std::vector<float> skullstripped;
    std::vector<float> interpolated;
    std::array<float, 12> params;
    std::vector<float> dispX, dispY, dispZ;
};

// Quadrature filters for registration
struct QuadratureFilters {
    std::vector<float> linearReal[3];
    std::vector<float> linearImag[3];
    std::vector<float> nonlinearReal[6];
    std::vector<float> nonlinearImag[6];
    float projectionTensors[6][6];
    float filterDirectionsX[6];
    float filterDirectionsY[6];
    float filterDirectionsZ[6];
};

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

} // namespace cuda_reg
