/* webgpu_registration.h — WebGPU compute backend for BROCCOLI registration
 *
 * Mirrors metal_registration.h API using wgpu-native as the WebGPU implementation.
 * WGSL shaders are embedded; no external shader files needed.
 */

#ifndef WEBGPU_REGISTRATION_H
#define WEBGPU_REGISTRATION_H

#include <vector>
#include <array>

namespace webgpu_reg {

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
    std::vector<float> params;       // 12 affine parameters
    std::vector<float> dispX, dispY, dispZ;
};

/* Universal registration entry point.
 * When nonlinearIterations == 0, performs linear-only registration. */
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

} // namespace webgpu_reg

#endif /* WEBGPU_REGISTRATION_H */
