/* opencl_registration.h — OpenCL compute backend for phase-based registration
 *
 * Mirrors the metal/cuda/webgpu registration API using OpenCL.
 * Kernel source files are loaded from $BROCCOLI_DIR at runtime.
 */

#ifndef OPENCL_REGISTRATION_H
#define OPENCL_REGISTRATION_H

#include <vector>
#include <array>

namespace opencl_reg {

struct VolumeDims {
    int W, H, D;
    int size() const { return W * H * D; }
};

struct VoxelSize {
    float x, y, z;
};

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

struct T1MNIResult {
    std::vector<float> alignedLinear;
    std::vector<float> alignedNonLinear;
    std::vector<float> skullstripped;
    std::vector<float> interpolated;
    std::array<float, 12> params;
    std::vector<float> dispX, dispY, dispZ;
};

T1MNIResult registerT1MNI(
    const float *t1Data,       VolumeDims t1Dims,  VoxelSize t1Vox,
    const float *mniData,      VolumeDims mniDims, VoxelSize mniVox,
    const float *mniBrainData,
    const float *mniMaskData,
    const QuadratureFilters &filters,
    int linearIterations,
    int nonlinearIterations,
    int coarsestScale,
    int mmZCut,
    bool verbose);

} // namespace opencl_reg

#endif /* OPENCL_REGISTRATION_H */
