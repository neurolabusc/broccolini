/* opencl_registration.cpp — OpenCL compute backend for phase-based registration
 *
 * Extracted from broccoli_lib.cpp (BROCCOLI by Anders Eklund).
 * Converts the BROCCOLI_LIB class into free functions in the opencl_reg namespace,
 * following the same architecture as the Metal and WebGPU backends.
 *
 * Original BROCCOLI code Copyright (C) 2013 Anders Eklund
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "Eigen/Eigenvalues"
#include <limits>
#include "Eigen/Dense"
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include <vector>
#include <cfloat>
#include <cassert>

#include <sys/time.h>

#include <opencl.h>

#include "opencl_registration.h"

#define EIGEN_DONT_PARALLELIZE

// ============================================================
//  Constants
// ============================================================

#define TRANSLATION 0
#define RIGID       1
#define AFFINE      2

#define NEAREST     0
#define LINEAR      1
#define CUBIC       2

// Valid filter responses for separable convolution shared-memory kernels
#define VALID_FILTER_RESPONSES_X_SEPARABLE_CONVOLUTION_ROWS    32
#define VALID_FILTER_RESPONSES_Y_SEPARABLE_CONVOLUTION_ROWS     8
#define VALID_FILTER_RESPONSES_Z_SEPARABLE_CONVOLUTION_ROWS     8

#define VALID_FILTER_RESPONSES_X_SEPARABLE_CONVOLUTION_COLUMNS 24
#define VALID_FILTER_RESPONSES_Y_SEPARABLE_CONVOLUTION_COLUMNS 16
#define VALID_FILTER_RESPONSES_Z_SEPARABLE_CONVOLUTION_COLUMNS  8

#define VALID_FILTER_RESPONSES_X_SEPARABLE_CONVOLUTION_RODS    32
#define VALID_FILTER_RESPONSES_Y_SEPARABLE_CONVOLUTION_RODS     8
#define VALID_FILTER_RESPONSES_Z_SEPARABLE_CONVOLUTION_RODS     8

#define VALID_FILTER_RESPONSES_X_CONVOLUTION_2D_24KB  90
#define VALID_FILTER_RESPONSES_Y_CONVOLUTION_2D_24KB  58

#define VALID_FILTER_RESPONSES_X_CONVOLUTION_2D_32KB 122
#define VALID_FILTER_RESPONSES_Y_CONVOLUTION_2D_32KB  58

// ============================================================
//  Internal helpers
// ============================================================

namespace opencl_reg {
namespace {

static int mymax(int a, int b) { return a > b ? a : b; }
static float mymax(float a, float b) { return a > b ? a : b; }
static float myround(float a) { return floorf(a + 0.5f); }

// ============================================================
//  OpenCLContext definition
// ============================================================

struct OpenCLContext {
    cl_context   context;
    cl_command_queue queue;
    cl_device_id device;

    cl_ulong localMemorySize;   // in KB (after dividing by 1024)
    size_t   maxThreadsPerBlock;
    size_t   maxThreadsPerDimension[3];

    // Programs (3 kernel files)
    cl_program programs[3];

    // --- Kernels ---
    // Convolution
    cl_kernel nonsepConv3DKernel;
    cl_kernel separableConvRowsKernel;
    cl_kernel separableConvColumnsKernel;
    cl_kernel separableConvRodsKernel;

    // Linear registration
    cl_kernel calculatePhaseDiffKernel;
    cl_kernel calculatePhaseGradientsXKernel;
    cl_kernel calculatePhaseGradientsYKernel;
    cl_kernel calculatePhaseGradientsZKernel;
    cl_kernel calculateAMatrixAndHVector2DValuesXKernel;
    cl_kernel calculateAMatrixAndHVector2DValuesYKernel;
    cl_kernel calculateAMatrixAndHVector2DValuesZKernel;
    cl_kernel calculateAMatrix1DValuesKernel;
    cl_kernel calculateHVector1DValuesKernel;
    cl_kernel calculateAMatrixKernel;
    cl_kernel calculateHVectorKernel;

    // Non-linear registration
    cl_kernel calculateTensorComponentsKernel;
    cl_kernel calculateTensorNormsKernel;
    cl_kernel calculateAMatricesAndHVectorsKernel;
    cl_kernel calculateDisplacementUpdateKernel;
    cl_kernel addLinearAndNonLinearDisplacementKernel;

    // Misc / helper kernels
    cl_kernel calculateMagnitudesKernel;
    cl_kernel calculateColumnSumsKernel;
    cl_kernel calculateRowSumsKernel;
    cl_kernel calculateColumnMaxsKernel;
    cl_kernel calculateRowMaxsKernel;
    cl_kernel calculateMaxAtomicKernel;
    cl_kernel thresholdVolumeKernel;
    cl_kernel memsetKernel;
    cl_kernel memsetFloat2Kernel;
    cl_kernel multiplyVolumeKernel;
    cl_kernel multiplyVolumesOverwriteKernel;
    cl_kernel addVolumesOverwriteKernel;

    // Interpolation kernels
    cl_kernel interpolateVolumeNearestLinearKernel;
    cl_kernel interpolateVolumeLinearLinearKernel;
    cl_kernel interpolateVolumeCubicLinearKernel;
    cl_kernel interpolateVolumeNearestNonLinearKernel;
    cl_kernel interpolateVolumeLinearNonLinearKernel;
    cl_kernel interpolateVolumeCubicNonLinearKernel;

    // Rescale kernels
    cl_kernel rescaleVolumeLinearKernel;
    cl_kernel rescaleVolumeCubicKernel;
    cl_kernel rescaleVolumeNearestKernel;

    // Copy kernels
    cl_kernel copyT1VolumeToMNIKernel;
    cl_kernel copyVolumeToNewKernel;

    bool inited;

    OpenCLContext() : inited(false), context(NULL), queue(NULL), device(NULL) {
        for (int i = 0; i < 3; i++) programs[i] = NULL;
    }

    bool init();
    void cleanup();
};

// ============================================================
//  GetBROCCOLIDirectory
// ============================================================

static std::string GetBROCCOLIDirectory()
{
    const char* dir = getenv("BROCCOLI_DIR");
    if (dir != NULL)
        return std::string(dir);
    else
        return "ERROR";
}

// ============================================================
//  OpenCLContext::init — create context, queue, compile kernels
// ============================================================

bool OpenCLContext::init()
{
    cl_int error;

    // Get platforms
    cl_uint platformIdCount = 0;
    error = clGetPlatformIDs(0, NULL, &platformIdCount);
    if (error != CL_SUCCESS || platformIdCount == 0) {
        fprintf(stderr, "OpenCL: No platforms found (error %d)\n", error);
        return false;
    }

    std::vector<cl_platform_id> platformIds(platformIdCount);
    clGetPlatformIDs(platformIdCount, platformIds.data(), NULL);

    // Try each platform/device until we find one
    bool found = false;
    for (cl_uint p = 0; p < platformIdCount && !found; p++) {
        cl_uint deviceIdCount = 0;
        clGetDeviceIDs(platformIds[p], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceIdCount);
        if (deviceIdCount == 0) continue;

        std::vector<cl_device_id> deviceIds(deviceIdCount);
        clGetDeviceIDs(platformIds[p], CL_DEVICE_TYPE_ALL, deviceIdCount, deviceIds.data(), NULL);

        // Prefer GPU, fall back to first device
        cl_device_id chosenDevice = deviceIds[0];
        for (cl_uint d = 0; d < deviceIdCount; d++) {
            cl_device_type dtype;
            clGetDeviceInfo(deviceIds[d], CL_DEVICE_TYPE, sizeof(dtype), &dtype, NULL);
            if (dtype & CL_DEVICE_TYPE_GPU) {
                chosenDevice = deviceIds[d];
                break;
            }
        }

        const cl_context_properties contextProperties[] = {
            CL_CONTEXT_PLATFORM, (cl_context_properties)platformIds[p], 0
        };
        context = clCreateContext(contextProperties, 1, &chosenDevice, NULL, NULL, &error);
        if (error == CL_SUCCESS) {
            device = chosenDevice;
            found = true;
        }
    }

    if (!found) {
        fprintf(stderr, "OpenCL: Failed to create context\n");
        return false;
    }

    // Create command queue
    queue = clCreateCommandQueue(context, device, 0, &error);
    if (error != CL_SUCCESS) {
        fprintf(stderr, "OpenCL: Failed to create command queue\n");
        return false;
    }

    // Query device properties
    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(localMemorySize), &localMemorySize, NULL);
    localMemorySize /= 1024;  // Convert to KB

    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxThreadsPerBlock), &maxThreadsPerBlock, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(maxThreadsPerDimension), maxThreadsPerDimension, NULL);

    // Print device info
    {
        char name[256];
        clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, NULL);
        printf("[OpenCL] Device: %s  localMem=%luKB  maxThreads=%zu  maxDim=[%zu,%zu,%zu]\n",
               name, (unsigned long)localMemorySize, maxThreadsPerBlock,
               maxThreadsPerDimension[0], maxThreadsPerDimension[1], maxThreadsPerDimension[2]);
    }

    // Compile kernel source files
    if (GetBROCCOLIDirectory() == "ERROR") {
        fprintf(stderr, "OpenCL: BROCCOLI_DIR environment variable not set\n");
        return false;
    }

    std::string kernelPath = GetBROCCOLIDirectory() + "kernels/";
    const char* kernelFileNames[3] = {
        "kernelConvolution.cpp",
        "kernelRegistration.cpp",
        "kernelMisc.cpp"
    };

    for (int k = 0; k < 3; k++) {
        std::string fullPath = kernelPath + kernelFileNames[k];
        std::ifstream file(fullPath.c_str());
        if (!file.good()) {
            fprintf(stderr, "OpenCL: Cannot open kernel file %s\n", fullPath.c_str());
            return false;
        }

        std::ostringstream oss;
        oss << file.rdbuf();
        std::string src = oss.str();
        const char* srcstr = src.c_str();

        programs[k] = clCreateProgramWithSource(context, 1, &srcstr, NULL, &error);
        if (error != CL_SUCCESS) {
            fprintf(stderr, "OpenCL: Failed to create program for %s (error %d)\n", kernelFileNames[k], error);
            return false;
        }

        error = clBuildProgram(programs[k], 1, &device, NULL, NULL, NULL);
        if (error != CL_SUCCESS) {
            fprintf(stderr, "OpenCL: Build failed for %s (error %d)\n", kernelFileNames[k], error);
            // Get build log
            size_t logSize = 0;
            clGetProgramBuildInfo(programs[k], device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
            if (logSize > 1) {
                char* log = (char*)malloc(logSize);
                clGetProgramBuildInfo(programs[k], device, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);
                fprintf(stderr, "Build log:\n%s\n", log);
                free(log);
            }
            return false;
        }
    }

    // Create kernels with device-capability-dependent selection

    // Non-separable convolution kernel
    cl_int kerr;
    if (localMemorySize >= 32 && maxThreadsPerBlock >= 512 &&
        maxThreadsPerDimension[0] >= 32 && maxThreadsPerDimension[1] >= 16) {
        nonsepConv3DKernel = clCreateKernel(programs[0], "Nonseparable3DConvolutionComplexThreeQuadratureFilters_32KB_512threads", &kerr);
    } else if (localMemorySize >= 24 && maxThreadsPerBlock >= 1024 &&
               maxThreadsPerDimension[0] >= 32 && maxThreadsPerDimension[1] >= 32) {
        nonsepConv3DKernel = clCreateKernel(programs[0], "Nonseparable3DConvolutionComplexThreeQuadratureFilters_24KB_1024threads", &kerr);
    } else if (localMemorySize >= 32 && maxThreadsPerBlock >= 256 &&
               maxThreadsPerDimension[0] >= 16 && maxThreadsPerDimension[1] >= 16) {
        nonsepConv3DKernel = clCreateKernel(programs[0], "Nonseparable3DConvolutionComplexThreeQuadratureFilters_32KB_256threads", &kerr);
    } else {
        nonsepConv3DKernel = clCreateKernel(programs[0], "Nonseparable3DConvolutionComplexThreeQuadratureFiltersGlobalMemory", &kerr);
    }

    // Separable convolution kernels
    if (localMemorySize >= 16 && maxThreadsPerBlock >= 512 &&
        maxThreadsPerDimension[0] >= 32 && maxThreadsPerDimension[1] >= 8 && maxThreadsPerDimension[2] >= 8) {
        separableConvRowsKernel    = clCreateKernel(programs[0], "SeparableConvolutionRows_16KB_512threads", &kerr);
        separableConvColumnsKernel = clCreateKernel(programs[0], "SeparableConvolutionColumns_16KB_512threads", &kerr);
        separableConvRodsKernel    = clCreateKernel(programs[0], "SeparableConvolutionRods_16KB_512threads", &kerr);
    } else if (localMemorySize >= 16 && maxThreadsPerBlock >= 256 &&
               maxThreadsPerDimension[0] >= 32 && maxThreadsPerDimension[1] >= 8 && maxThreadsPerDimension[2] >= 8) {
        separableConvRowsKernel    = clCreateKernel(programs[0], "SeparableConvolutionRows_16KB_256threads", &kerr);
        separableConvColumnsKernel = clCreateKernel(programs[0], "SeparableConvolutionColumns_16KB_256threads", &kerr);
        separableConvRodsKernel    = clCreateKernel(programs[0], "SeparableConvolutionRods_16KB_256threads", &kerr);
    } else {
        separableConvRowsKernel    = clCreateKernel(programs[0], "SeparableConvolutionRowsGlobalMemory", &kerr);
        separableConvColumnsKernel = clCreateKernel(programs[0], "SeparableConvolutionColumnsGlobalMemory", &kerr);
        separableConvRodsKernel    = clCreateKernel(programs[0], "SeparableConvolutionRodsGlobalMemory", &kerr);
    }

    // Linear registration kernels
    calculatePhaseDiffKernel                    = clCreateKernel(programs[1], "CalculatePhaseDifferencesAndCertainties", &kerr);
    calculatePhaseGradientsXKernel              = clCreateKernel(programs[1], "CalculatePhaseGradientsX", &kerr);
    calculatePhaseGradientsYKernel              = clCreateKernel(programs[1], "CalculatePhaseGradientsY", &kerr);
    calculatePhaseGradientsZKernel              = clCreateKernel(programs[1], "CalculatePhaseGradientsZ", &kerr);
    calculateAMatrixAndHVector2DValuesXKernel   = clCreateKernel(programs[1], "CalculateAMatrixAndHVector2DValuesX", &kerr);
    calculateAMatrixAndHVector2DValuesYKernel   = clCreateKernel(programs[1], "CalculateAMatrixAndHVector2DValuesY", &kerr);
    calculateAMatrixAndHVector2DValuesZKernel   = clCreateKernel(programs[1], "CalculateAMatrixAndHVector2DValuesZ", &kerr);
    calculateAMatrix1DValuesKernel              = clCreateKernel(programs[1], "CalculateAMatrix1DValues", &kerr);
    calculateHVector1DValuesKernel              = clCreateKernel(programs[1], "CalculateHVector1DValues", &kerr);
    calculateAMatrixKernel                      = clCreateKernel(programs[1], "CalculateAMatrix", &kerr);
    calculateHVectorKernel                      = clCreateKernel(programs[1], "CalculateHVector", &kerr);

    // Non-linear registration kernels
    calculateTensorComponentsKernel       = clCreateKernel(programs[1], "CalculateTensorComponents", &kerr);
    calculateTensorNormsKernel            = clCreateKernel(programs[1], "CalculateTensorNorms", &kerr);
    calculateAMatricesAndHVectorsKernel   = clCreateKernel(programs[1], "CalculateAMatricesAndHVectors", &kerr);
    calculateDisplacementUpdateKernel     = clCreateKernel(programs[1], "CalculateDisplacementUpdate", &kerr);
    addLinearAndNonLinearDisplacementKernel= clCreateKernel(programs[1], "AddLinearAndNonLinearDisplacement", &kerr);

    // Interpolation kernels
    interpolateVolumeNearestLinearKernel   = clCreateKernel(programs[1], "InterpolateVolumeNearestLinear", &kerr);
    interpolateVolumeLinearLinearKernel    = clCreateKernel(programs[1], "InterpolateVolumeLinearLinear", &kerr);
    interpolateVolumeCubicLinearKernel     = clCreateKernel(programs[1], "InterpolateVolumeCubicLinear", &kerr);
    interpolateVolumeNearestNonLinearKernel= clCreateKernel(programs[1], "InterpolateVolumeNearestNonLinear", &kerr);
    interpolateVolumeLinearNonLinearKernel = clCreateKernel(programs[1], "InterpolateVolumeLinearNonLinear", &kerr);
    interpolateVolumeCubicNonLinearKernel  = clCreateKernel(programs[1], "InterpolateVolumeCubicNonLinear", &kerr);

    // Rescale kernels
    rescaleVolumeLinearKernel  = clCreateKernel(programs[1], "RescaleVolumeLinear", &kerr);
    rescaleVolumeCubicKernel   = clCreateKernel(programs[1], "RescaleVolumeCubic", &kerr);
    rescaleVolumeNearestKernel = clCreateKernel(programs[1], "RescaleVolumeNearest", &kerr);

    // Copy kernels
    copyT1VolumeToMNIKernel = clCreateKernel(programs[1], "CopyT1VolumeToMNI", &kerr);
    copyVolumeToNewKernel   = clCreateKernel(programs[1], "CopyVolumeToNew", &kerr);

    // Misc kernels
    calculateMagnitudesKernel = clCreateKernel(programs[2], "CalculateMagnitudes", &kerr);
    calculateColumnSumsKernel = clCreateKernel(programs[2], "CalculateColumnSums", &kerr);
    calculateRowSumsKernel    = clCreateKernel(programs[2], "CalculateRowSums", &kerr);
    calculateColumnMaxsKernel = clCreateKernel(programs[2], "CalculateColumnMaxs", &kerr);
    calculateRowMaxsKernel    = clCreateKernel(programs[2], "CalculateRowMaxs", &kerr);
    calculateMaxAtomicKernel  = clCreateKernel(programs[2], "CalculateMaxAtomic", &kerr);
    thresholdVolumeKernel     = clCreateKernel(programs[2], "ThresholdVolume", &kerr);
    memsetKernel              = clCreateKernel(programs[2], "Memset", &kerr);
    memsetFloat2Kernel        = clCreateKernel(programs[2], "MemsetFloat2", &kerr);
    multiplyVolumeKernel      = clCreateKernel(programs[2], "MultiplyVolume", &kerr);
    multiplyVolumesOverwriteKernel = clCreateKernel(programs[2], "MultiplyVolumesOverwrite", &kerr);
    addVolumesOverwriteKernel = clCreateKernel(programs[2], "AddVolumesOverwrite", &kerr);

    inited = true;
    printf("[OpenCL] All kernels created successfully\n");
    return true;
}

void OpenCLContext::cleanup()
{
    if (!inited) return;

    // Release kernels
    cl_kernel* allKernels[] = {
        &nonsepConv3DKernel, &separableConvRowsKernel, &separableConvColumnsKernel, &separableConvRodsKernel,
        &calculatePhaseDiffKernel, &calculatePhaseGradientsXKernel, &calculatePhaseGradientsYKernel, &calculatePhaseGradientsZKernel,
        &calculateAMatrixAndHVector2DValuesXKernel, &calculateAMatrixAndHVector2DValuesYKernel, &calculateAMatrixAndHVector2DValuesZKernel,
        &calculateAMatrix1DValuesKernel, &calculateHVector1DValuesKernel, &calculateAMatrixKernel, &calculateHVectorKernel,
        &calculateTensorComponentsKernel, &calculateTensorNormsKernel, &calculateAMatricesAndHVectorsKernel,
        &calculateDisplacementUpdateKernel, &addLinearAndNonLinearDisplacementKernel,
        &calculateMagnitudesKernel, &calculateColumnSumsKernel, &calculateRowSumsKernel,
        &calculateColumnMaxsKernel, &calculateRowMaxsKernel, &calculateMaxAtomicKernel,
        &thresholdVolumeKernel, &memsetKernel, &memsetFloat2Kernel,
        &multiplyVolumeKernel, &multiplyVolumesOverwriteKernel, &addVolumesOverwriteKernel,
        &interpolateVolumeNearestLinearKernel, &interpolateVolumeLinearLinearKernel, &interpolateVolumeCubicLinearKernel,
        &interpolateVolumeNearestNonLinearKernel, &interpolateVolumeLinearNonLinearKernel, &interpolateVolumeCubicNonLinearKernel,
        &rescaleVolumeLinearKernel, &rescaleVolumeCubicKernel, &rescaleVolumeNearestKernel,
        &copyT1VolumeToMNIKernel, &copyVolumeToNewKernel,
    };
    for (auto* kp : allKernels) {
        if (*kp) { clReleaseKernel(*kp); *kp = NULL; }
    }
    for (int k = 0; k < 3; k++) {
        if (programs[k]) { clReleaseProgram(programs[k]); programs[k] = NULL; }
    }
    if (queue) { clReleaseCommandQueue(queue); queue = NULL; }
    if (context) { clReleaseContext(context); context = NULL; }
    inited = false;
}

// Singleton accessor
OpenCLContext& ctx() {
    static OpenCLContext c;
    if (!c.inited) {
        bool ok = c.init();
        assert(ok && "OpenCL initialization failed");
    }
    return c;
}

// ============================================================
//  Registration constants
// ============================================================

static const int NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS = 12;
static const int NUMBER_OF_NON_ZERO_A_MATRIX_ELEMENTS = 30;
static const int IMAGE_REGISTRATION_FILTER_SIZE = 7;
static const int SMOOTHING_FILTER_SIZE = 9;

// Smoothing sigmas for non-linear registration
static const double TSIGMA = 5.0;
static const double ESIGMA = 5.0;
static const double DSIGMA = 5.0;

// ============================================================
//  Work size calculation helpers
// ============================================================

struct WorkSizes3D {
    size_t global[3];
    size_t local[3];
};

static WorkSizes3D calcWorkSizes3D(OpenCLContext& c, int W, int H, int D, int localX, int localY, int localZ)
{
    WorkSizes3D ws;
    ws.local[0] = localX;
    ws.local[1] = localY;
    ws.local[2] = localZ;
    ws.global[0] = ((size_t)ceil((float)W / (float)localX)) * localX;
    ws.global[1] = ((size_t)ceil((float)H / (float)localY)) * localY;
    ws.global[2] = ((size_t)ceil((float)D / (float)localZ)) * localZ;
    return ws;
}

static WorkSizes3D calcWorkSizes16x16(OpenCLContext& c, int W, int H, int D)
{
    if (c.maxThreadsPerDimension[1] >= 16)
        return calcWorkSizes3D(c, W, H, D, 16, 16, 1);
    else
        return calcWorkSizes3D(c, W, H, D, 64, 1, 1);
}

struct SepConvWorkSizes {
    size_t globalRows[3], localRows[3];
    size_t globalColumns[3], localColumns[3];
    size_t globalRods[3], localRods[3];
};

static SepConvWorkSizes calcSepConvWorkSizes(OpenCLContext& c, int W, int H, int D)
{
    SepConvWorkSizes ws;
    size_t xb, yb, zb;

    if (c.maxThreadsPerBlock >= 512 && c.maxThreadsPerDimension[0] >= 32 &&
        c.maxThreadsPerDimension[1] >= 8 && c.maxThreadsPerDimension[2] >= 8) {
        // Rows
        ws.localRows[0] = 32; ws.localRows[1] = 8; ws.localRows[2] = 2;
        xb = (size_t)ceil((float)W / (float)VALID_FILTER_RESPONSES_X_SEPARABLE_CONVOLUTION_ROWS);
        yb = (size_t)ceil((float)H / (float)VALID_FILTER_RESPONSES_Y_SEPARABLE_CONVOLUTION_ROWS);
        zb = (size_t)ceil((float)D / (float)VALID_FILTER_RESPONSES_Z_SEPARABLE_CONVOLUTION_ROWS);
        ws.globalRows[0] = xb * ws.localRows[0];
        ws.globalRows[1] = yb * ws.localRows[1];
        ws.globalRows[2] = zb * ws.localRows[2];
        // Columns
        ws.localColumns[0] = 32; ws.localColumns[1] = 8; ws.localColumns[2] = 2;
        xb = (size_t)ceil((float)W / (float)VALID_FILTER_RESPONSES_X_SEPARABLE_CONVOLUTION_COLUMNS);
        yb = (size_t)ceil((float)H / (float)VALID_FILTER_RESPONSES_Y_SEPARABLE_CONVOLUTION_COLUMNS);
        zb = (size_t)ceil((float)D / (float)VALID_FILTER_RESPONSES_Z_SEPARABLE_CONVOLUTION_COLUMNS);
        ws.globalColumns[0] = xb * ws.localColumns[0];
        ws.globalColumns[1] = yb * ws.localColumns[1];
        ws.globalColumns[2] = zb * ws.localColumns[2];
        // Rods
        ws.localRods[0] = 32; ws.localRods[1] = 2; ws.localRods[2] = 8;
        xb = (size_t)ceil((float)W / (float)VALID_FILTER_RESPONSES_X_SEPARABLE_CONVOLUTION_RODS);
        yb = (size_t)ceil((float)H / (float)VALID_FILTER_RESPONSES_Y_SEPARABLE_CONVOLUTION_RODS);
        zb = (size_t)ceil((float)D / (float)VALID_FILTER_RESPONSES_Z_SEPARABLE_CONVOLUTION_RODS);
        ws.globalRods[0] = xb * ws.localRods[0];
        ws.globalRods[1] = yb * ws.localRods[1];
        ws.globalRods[2] = zb * ws.localRods[2];
    } else if (c.maxThreadsPerBlock >= 256 && c.maxThreadsPerDimension[0] >= 32 &&
               c.maxThreadsPerDimension[1] >= 8 && c.maxThreadsPerDimension[2] >= 8) {
        // Rows
        ws.localRows[0] = 32; ws.localRows[1] = 8; ws.localRows[2] = 1;
        xb = (size_t)ceil((float)W / (float)VALID_FILTER_RESPONSES_X_SEPARABLE_CONVOLUTION_ROWS);
        yb = (size_t)ceil((float)H / (float)VALID_FILTER_RESPONSES_Y_SEPARABLE_CONVOLUTION_ROWS);
        zb = (size_t)ceil((float)D / (float)VALID_FILTER_RESPONSES_Z_SEPARABLE_CONVOLUTION_ROWS);
        ws.globalRows[0] = xb * ws.localRows[0];
        ws.globalRows[1] = yb * ws.localRows[1];
        ws.globalRows[2] = zb * ws.localRows[2];
        // Columns
        ws.localColumns[0] = 32; ws.localColumns[1] = 8; ws.localColumns[2] = 1;
        xb = (size_t)ceil((float)W / (float)VALID_FILTER_RESPONSES_X_SEPARABLE_CONVOLUTION_COLUMNS);
        yb = (size_t)ceil((float)H / (float)VALID_FILTER_RESPONSES_Y_SEPARABLE_CONVOLUTION_COLUMNS);
        zb = (size_t)ceil((float)D / (float)VALID_FILTER_RESPONSES_Z_SEPARABLE_CONVOLUTION_COLUMNS);
        ws.globalColumns[0] = xb * ws.localColumns[0];
        ws.globalColumns[1] = yb * ws.localColumns[1];
        ws.globalColumns[2] = zb * ws.localColumns[2];
        // Rods
        ws.localRods[0] = 32; ws.localRods[1] = 1; ws.localRods[2] = 8;
        xb = (size_t)ceil((float)W / (float)VALID_FILTER_RESPONSES_X_SEPARABLE_CONVOLUTION_RODS);
        yb = (size_t)ceil((float)H / (float)VALID_FILTER_RESPONSES_Y_SEPARABLE_CONVOLUTION_RODS);
        zb = (size_t)ceil((float)D / (float)VALID_FILTER_RESPONSES_Z_SEPARABLE_CONVOLUTION_RODS);
        ws.globalRods[0] = xb * ws.localRods[0];
        ws.globalRods[1] = yb * ws.localRods[1];
        ws.globalRods[2] = zb * ws.localRods[2];
    } else {
        // Global memory fallback
        ws.localRows[0] = 64; ws.localRows[1] = 1; ws.localRows[2] = 1;
        ws.globalRows[0] = ((size_t)ceil((float)W / 64.0f)) * 64;
        ws.globalRows[1] = H; ws.globalRows[2] = D;
        ws.localColumns[0] = 64; ws.localColumns[1] = 1; ws.localColumns[2] = 1;
        ws.globalColumns[0] = ((size_t)ceil((float)W / 64.0f)) * 64;
        ws.globalColumns[1] = H; ws.globalColumns[2] = D;
        ws.localRods[0] = 64; ws.localRods[1] = 1; ws.localRods[2] = 1;
        ws.globalRods[0] = ((size_t)ceil((float)W / 64.0f)) * 64;
        ws.globalRods[1] = H; ws.globalRods[2] = D;
    }
    return ws;
}

struct NonsepConvWorkSizes {
    size_t global[3], local[3];
};

static NonsepConvWorkSizes calcNonsepConvWorkSizes(OpenCLContext& c, int W, int H, int D)
{
    NonsepConvWorkSizes ws;
    size_t xb, yb, zb;

    if (c.maxThreadsPerBlock >= 512 && c.maxThreadsPerDimension[0] >= 32 && c.maxThreadsPerDimension[1] >= 16) {
        ws.local[0] = 32; ws.local[1] = 16; ws.local[2] = 1;
        xb = (size_t)ceil((float)W / (float)VALID_FILTER_RESPONSES_X_CONVOLUTION_2D_32KB);
        yb = (size_t)ceil((float)H / (float)VALID_FILTER_RESPONSES_Y_CONVOLUTION_2D_32KB);
        zb = (size_t)ceil((float)D / (float)ws.local[2]);
    } else if (c.maxThreadsPerBlock >= 1024 && c.maxThreadsPerDimension[0] >= 32 && c.maxThreadsPerDimension[1] >= 32) {
        ws.local[0] = 32; ws.local[1] = 32; ws.local[2] = 1;
        xb = (size_t)ceil((float)W / (float)VALID_FILTER_RESPONSES_X_CONVOLUTION_2D_24KB);
        yb = (size_t)ceil((float)H / (float)VALID_FILTER_RESPONSES_Y_CONVOLUTION_2D_24KB);
        zb = (size_t)ceil((float)D / (float)ws.local[2]);
    } else if (c.maxThreadsPerBlock >= 256 && c.maxThreadsPerDimension[0] >= 16 && c.maxThreadsPerDimension[1] >= 16) {
        ws.local[0] = 16; ws.local[1] = 16; ws.local[2] = 1;
        xb = (size_t)ceil((float)W / (float)VALID_FILTER_RESPONSES_X_CONVOLUTION_2D_32KB);
        yb = (size_t)ceil((float)H / (float)VALID_FILTER_RESPONSES_Y_CONVOLUTION_2D_32KB);
        zb = (size_t)ceil((float)D / (float)ws.local[2]);
    } else {
        ws.local[0] = 64; ws.local[1] = 1; ws.local[2] = 1;
        xb = (size_t)ceil((float)W / (float)ws.local[0]);
        yb = (size_t)ceil((float)H / (float)ws.local[1]);
        zb = (size_t)ceil((float)D / (float)ws.local[2]);
    }

    ws.global[0] = xb * ws.local[0];
    ws.global[1] = yb * ws.local[1];
    ws.global[2] = zb * ws.local[2];
    return ws;
}

// ============================================================
//  GPU primitive operations
// ============================================================

static void setMemory(OpenCLContext& c, cl_mem memory, float value, int N)
{
    size_t localSize = (c.maxThreadsPerDimension[1] >= 64) ? 256 : 64;
    size_t globalSize = ((size_t)ceil((float)N / (float)localSize)) * localSize;

    clSetKernelArg(c.memsetKernel, 0, sizeof(cl_mem), &memory);
    clSetKernelArg(c.memsetKernel, 1, sizeof(float), &value);
    clSetKernelArg(c.memsetKernel, 2, sizeof(int), &N);
    clEnqueueNDRangeKernel(c.queue, c.memsetKernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
    clFinish(c.queue);
}

static void setMemoryFloat2(OpenCLContext& c, cl_mem memory, float value, int N)
{
    size_t localSize = (c.maxThreadsPerDimension[1] >= 64) ? 256 : 64;
    size_t globalSize = ((size_t)ceil((float)N / (float)localSize)) * localSize;

    clSetKernelArg(c.memsetFloat2Kernel, 0, sizeof(cl_mem), &memory);
    clSetKernelArg(c.memsetFloat2Kernel, 1, sizeof(float), &value);
    clSetKernelArg(c.memsetFloat2Kernel, 2, sizeof(int), &N);
    clEnqueueNDRangeKernel(c.queue, c.memsetFloat2Kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
    clFinish(c.queue);
}

static void multiplyVolume(OpenCLContext& c, cl_mem d_Volume, float factor, int W, int H, int D)
{
    WorkSizes3D ws = calcWorkSizes16x16(c, W, H, D);
    clSetKernelArg(c.multiplyVolumeKernel, 0, sizeof(cl_mem), &d_Volume);
    clSetKernelArg(c.multiplyVolumeKernel, 1, sizeof(float), &factor);
    clSetKernelArg(c.multiplyVolumeKernel, 2, sizeof(int), &W);
    clSetKernelArg(c.multiplyVolumeKernel, 3, sizeof(int), &H);
    clSetKernelArg(c.multiplyVolumeKernel, 4, sizeof(int), &D);
    clEnqueueNDRangeKernel(c.queue, c.multiplyVolumeKernel, 3, NULL, ws.global, ws.local, 0, NULL, NULL);
    clFinish(c.queue);
}

static void multiplyVolumes(OpenCLContext& c, cl_mem d_Vol1, cl_mem d_Vol2, int W, int H, int D)
{
    WorkSizes3D ws = calcWorkSizes16x16(c, W, H, D);
    int zero = 0;
    clSetKernelArg(c.multiplyVolumesOverwriteKernel, 0, sizeof(cl_mem), &d_Vol1);
    clSetKernelArg(c.multiplyVolumesOverwriteKernel, 1, sizeof(cl_mem), &d_Vol2);
    clSetKernelArg(c.multiplyVolumesOverwriteKernel, 2, sizeof(int), &W);
    clSetKernelArg(c.multiplyVolumesOverwriteKernel, 3, sizeof(int), &H);
    clSetKernelArg(c.multiplyVolumesOverwriteKernel, 4, sizeof(int), &D);
    clSetKernelArg(c.multiplyVolumesOverwriteKernel, 5, sizeof(int), &zero);
    clEnqueueNDRangeKernel(c.queue, c.multiplyVolumesOverwriteKernel, 3, NULL, ws.global, ws.local, 0, NULL, NULL);
    clFinish(c.queue);
}

static void addVolumes(OpenCLContext& c, cl_mem d_Vol1, cl_mem d_Vol2, int W, int H, int D)
{
    WorkSizes3D ws = calcWorkSizes16x16(c, W, H, D);
    clSetKernelArg(c.addVolumesOverwriteKernel, 0, sizeof(cl_mem), &d_Vol1);
    clSetKernelArg(c.addVolumesOverwriteKernel, 1, sizeof(cl_mem), &d_Vol2);
    clSetKernelArg(c.addVolumesOverwriteKernel, 2, sizeof(int), &W);
    clSetKernelArg(c.addVolumesOverwriteKernel, 3, sizeof(int), &H);
    clSetKernelArg(c.addVolumesOverwriteKernel, 4, sizeof(int), &D);
    clEnqueueNDRangeKernel(c.queue, c.addVolumesOverwriteKernel, 3, NULL, ws.global, ws.local, 0, NULL, NULL);
    clFinish(c.queue);
}

static float calculateMax(OpenCLContext& c, cl_mem d_Volume, int W, int H, int D)
{
    // Column maxs
    WorkSizes3D wsCol = calcWorkSizes16x16(c, H, D, 1);
    cl_mem d_Column_Maxs = clCreateBuffer(c.context, CL_MEM_READ_WRITE, H * D * sizeof(float), NULL, NULL);
    cl_mem d_Maxs = clCreateBuffer(c.context, CL_MEM_READ_WRITE, D * sizeof(float), NULL, NULL);

    clSetKernelArg(c.calculateColumnMaxsKernel, 0, sizeof(cl_mem), &d_Column_Maxs);
    clSetKernelArg(c.calculateColumnMaxsKernel, 1, sizeof(cl_mem), &d_Volume);
    clSetKernelArg(c.calculateColumnMaxsKernel, 2, sizeof(int), &W);
    clSetKernelArg(c.calculateColumnMaxsKernel, 3, sizeof(int), &H);
    clSetKernelArg(c.calculateColumnMaxsKernel, 4, sizeof(int), &D);
    clEnqueueNDRangeKernel(c.queue, c.calculateColumnMaxsKernel, 2, NULL, wsCol.global, wsCol.local, 0, NULL, NULL);
    clFinish(c.queue);

    size_t localRowMaxs = 32;
    size_t globalRowMaxs = ((size_t)ceil((float)D / 32.0f)) * 32;
    clSetKernelArg(c.calculateRowMaxsKernel, 0, sizeof(cl_mem), &d_Maxs);
    clSetKernelArg(c.calculateRowMaxsKernel, 1, sizeof(cl_mem), &d_Column_Maxs);
    clSetKernelArg(c.calculateRowMaxsKernel, 2, sizeof(int), &H);
    clSetKernelArg(c.calculateRowMaxsKernel, 3, sizeof(int), &D);
    clEnqueueNDRangeKernel(c.queue, c.calculateRowMaxsKernel, 1, NULL, &globalRowMaxs, &localRowMaxs, 0, NULL, NULL);
    clFinish(c.queue);

    float* h_Maxs = (float*)malloc(D * sizeof(float));
    clEnqueueReadBuffer(c.queue, d_Maxs, CL_TRUE, 0, D * sizeof(float), h_Maxs, 0, NULL, NULL);

    float maxVal = -FLT_MAX;
    for (int z = 0; z < D; z++)
        maxVal = mymax(maxVal, h_Maxs[z]);
    free(h_Maxs);

    clReleaseMemObject(d_Column_Maxs);
    clReleaseMemObject(d_Maxs);
    return maxVal;
}

// ============================================================
//  Non-separable 3D convolution (3 complex quadrature filters)
// ============================================================

static void copyThreeFiltersToConstant(OpenCLContext& c,
    cl_mem c_F1R, cl_mem c_F1I, cl_mem c_F2R, cl_mem c_F2I, cl_mem c_F3R, cl_mem c_F3I,
    float* h_F1R, float* h_F1I, float* h_F2R, float* h_F2I, float* h_F3R, float* h_F3I,
    int z, int FILTER_SIZE)
{
    int sliceSize = FILTER_SIZE * FILTER_SIZE;
    clEnqueueWriteBuffer(c.queue, c_F1R, CL_TRUE, 0, sliceSize * sizeof(float), &h_F1R[z * sliceSize], 0, NULL, NULL);
    clEnqueueWriteBuffer(c.queue, c_F1I, CL_TRUE, 0, sliceSize * sizeof(float), &h_F1I[z * sliceSize], 0, NULL, NULL);
    clEnqueueWriteBuffer(c.queue, c_F2R, CL_TRUE, 0, sliceSize * sizeof(float), &h_F2R[z * sliceSize], 0, NULL, NULL);
    clEnqueueWriteBuffer(c.queue, c_F2I, CL_TRUE, 0, sliceSize * sizeof(float), &h_F2I[z * sliceSize], 0, NULL, NULL);
    clEnqueueWriteBuffer(c.queue, c_F3R, CL_TRUE, 0, sliceSize * sizeof(float), &h_F3R[z * sliceSize], 0, NULL, NULL);
    clEnqueueWriteBuffer(c.queue, c_F3I, CL_TRUE, 0, sliceSize * sizeof(float), &h_F3I[z * sliceSize], 0, NULL, NULL);
}

static void nonseparableConvolution3D(OpenCLContext& c,
    cl_mem d_q1, cl_mem d_q2, cl_mem d_q3, cl_mem d_Volume,
    cl_mem c_F1R, cl_mem c_F1I, cl_mem c_F2R, cl_mem c_F2I, cl_mem c_F3R, cl_mem c_F3I,
    float* h_F1R, float* h_F1I, float* h_F2R, float* h_F2I, float* h_F3R, float* h_F3I,
    int W, int H, int D)
{
    NonsepConvWorkSizes ws = calcNonsepConvWorkSizes(c, W, H, D);

    clSetKernelArg(c.nonsepConv3DKernel, 0, sizeof(cl_mem), &d_q1);
    clSetKernelArg(c.nonsepConv3DKernel, 1, sizeof(cl_mem), &d_q2);
    clSetKernelArg(c.nonsepConv3DKernel, 2, sizeof(cl_mem), &d_q3);
    clSetKernelArg(c.nonsepConv3DKernel, 3, sizeof(cl_mem), &d_Volume);
    clSetKernelArg(c.nonsepConv3DKernel, 4, sizeof(cl_mem), &c_F1R);
    clSetKernelArg(c.nonsepConv3DKernel, 5, sizeof(cl_mem), &c_F1I);
    clSetKernelArg(c.nonsepConv3DKernel, 6, sizeof(cl_mem), &c_F2R);
    clSetKernelArg(c.nonsepConv3DKernel, 7, sizeof(cl_mem), &c_F2I);
    clSetKernelArg(c.nonsepConv3DKernel, 8, sizeof(cl_mem), &c_F3R);
    clSetKernelArg(c.nonsepConv3DKernel, 9, sizeof(cl_mem), &c_F3I);
    clSetKernelArg(c.nonsepConv3DKernel, 11, sizeof(int), &W);
    clSetKernelArg(c.nonsepConv3DKernel, 12, sizeof(int), &H);
    clSetKernelArg(c.nonsepConv3DKernel, 13, sizeof(int), &D);

    // Reset complex valued filter responses
    setMemoryFloat2(c, d_q1, 0.0f, W * H * D);
    setMemoryFloat2(c, d_q2, 0.0f, W * H * D);
    setMemoryFloat2(c, d_q3, 0.0f, W * H * D);

    // Do 3D convolution by summing 2D convolutions
    int z_offset = -(IMAGE_REGISTRATION_FILTER_SIZE - 1) / 2;
    for (int zz = IMAGE_REGISTRATION_FILTER_SIZE - 1; zz >= 0; zz--) {
        copyThreeFiltersToConstant(c, c_F1R, c_F1I, c_F2R, c_F2I, c_F3R, c_F3I,
                                   h_F1R, h_F1I, h_F2R, h_F2I, h_F3R, h_F3I,
                                   zz, IMAGE_REGISTRATION_FILTER_SIZE);
        clSetKernelArg(c.nonsepConv3DKernel, 10, sizeof(int), &z_offset);
        clEnqueueNDRangeKernel(c.queue, c.nonsepConv3DKernel, 3, NULL, ws.global, ws.local, 0, NULL, NULL);
        clFinish(c.queue);
        z_offset++;
    }
}

// ============================================================
//  Separable convolution (smoothing)
// ============================================================

static void createSmoothingFilters(float* fx, float* fy, float* fz, int size, double sigma)
{
    int halfSize = (size - 1) / 2;
    double sigma_2 = 2.0 * sigma * sigma;
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        double u = (double)(i - halfSize);
        fx[i] = (float)exp(-u * u / sigma_2);
        fy[i] = fx[i];
        fz[i] = fx[i];
        sum += fx[i];
    }
    for (int i = 0; i < size; i++) {
        fx[i] /= sum;
        fy[i] /= sum;
        fz[i] /= sum;
    }
}

static void performSmoothing(OpenCLContext& c, cl_mem d_Volumes,
    float* h_SmoothX, float* h_SmoothY, float* h_SmoothZ,
    int W, int H, int D, int T)
{
    SepConvWorkSizes ws = calcSepConvWorkSizes(c, W, H, D);

    cl_mem c_SmoothX = clCreateBuffer(c.context, CL_MEM_READ_ONLY, SMOOTHING_FILTER_SIZE * sizeof(float), NULL, NULL);
    cl_mem c_SmoothY = clCreateBuffer(c.context, CL_MEM_READ_ONLY, SMOOTHING_FILTER_SIZE * sizeof(float), NULL, NULL);
    cl_mem c_SmoothZ = clCreateBuffer(c.context, CL_MEM_READ_ONLY, SMOOTHING_FILTER_SIZE * sizeof(float), NULL, NULL);

    clEnqueueWriteBuffer(c.queue, c_SmoothX, CL_TRUE, 0, SMOOTHING_FILTER_SIZE * sizeof(float), h_SmoothX, 0, NULL, NULL);
    clEnqueueWriteBuffer(c.queue, c_SmoothY, CL_TRUE, 0, SMOOTHING_FILTER_SIZE * sizeof(float), h_SmoothY, 0, NULL, NULL);
    clEnqueueWriteBuffer(c.queue, c_SmoothZ, CL_TRUE, 0, SMOOTHING_FILTER_SIZE * sizeof(float), h_SmoothZ, 0, NULL, NULL);

    cl_mem d_Convolved_Rows = clCreateBuffer(c.context, CL_MEM_READ_WRITE, W * H * D * sizeof(float), NULL, NULL);
    cl_mem d_Convolved_Columns = clCreateBuffer(c.context, CL_MEM_READ_WRITE, W * H * D * sizeof(float), NULL, NULL);
    cl_mem d_Certainty = clCreateBuffer(c.context, CL_MEM_READ_WRITE, W * H * D * sizeof(float), NULL, NULL);
    setMemory(c, d_Certainty, 1.0f, W * H * D);

    clSetKernelArg(c.separableConvRowsKernel, 0, sizeof(cl_mem), &d_Convolved_Rows);
    clSetKernelArg(c.separableConvRowsKernel, 1, sizeof(cl_mem), &d_Volumes);
    clSetKernelArg(c.separableConvRowsKernel, 2, sizeof(cl_mem), &d_Certainty);
    clSetKernelArg(c.separableConvRowsKernel, 3, sizeof(cl_mem), &c_SmoothY);
    clSetKernelArg(c.separableConvRowsKernel, 5, sizeof(int), &W);
    clSetKernelArg(c.separableConvRowsKernel, 6, sizeof(int), &H);
    clSetKernelArg(c.separableConvRowsKernel, 7, sizeof(int), &D);
    clSetKernelArg(c.separableConvRowsKernel, 8, sizeof(int), &T);

    clSetKernelArg(c.separableConvColumnsKernel, 0, sizeof(cl_mem), &d_Convolved_Columns);
    clSetKernelArg(c.separableConvColumnsKernel, 1, sizeof(cl_mem), &d_Convolved_Rows);
    clSetKernelArg(c.separableConvColumnsKernel, 2, sizeof(cl_mem), &c_SmoothX);
    clSetKernelArg(c.separableConvColumnsKernel, 4, sizeof(int), &W);
    clSetKernelArg(c.separableConvColumnsKernel, 5, sizeof(int), &H);
    clSetKernelArg(c.separableConvColumnsKernel, 6, sizeof(int), &D);
    clSetKernelArg(c.separableConvColumnsKernel, 7, sizeof(int), &T);

    clSetKernelArg(c.separableConvRodsKernel, 0, sizeof(cl_mem), &d_Volumes);
    clSetKernelArg(c.separableConvRodsKernel, 1, sizeof(cl_mem), &d_Convolved_Columns);
    clSetKernelArg(c.separableConvRodsKernel, 2, sizeof(cl_mem), &d_Certainty);
    clSetKernelArg(c.separableConvRodsKernel, 3, sizeof(cl_mem), &c_SmoothZ);
    clSetKernelArg(c.separableConvRodsKernel, 5, sizeof(int), &W);
    clSetKernelArg(c.separableConvRodsKernel, 6, sizeof(int), &H);
    clSetKernelArg(c.separableConvRodsKernel, 7, sizeof(int), &D);
    clSetKernelArg(c.separableConvRodsKernel, 8, sizeof(int), &T);

    for (int v = 0; v < T; v++) {
        clSetKernelArg(c.separableConvRowsKernel, 4, sizeof(int), &v);
        clEnqueueNDRangeKernel(c.queue, c.separableConvRowsKernel, 3, NULL, ws.globalRows, ws.localRows, 0, NULL, NULL);
        clFinish(c.queue);

        clSetKernelArg(c.separableConvColumnsKernel, 3, sizeof(int), &v);
        clEnqueueNDRangeKernel(c.queue, c.separableConvColumnsKernel, 3, NULL, ws.globalColumns, ws.localColumns, 0, NULL, NULL);
        clFinish(c.queue);

        clSetKernelArg(c.separableConvRodsKernel, 4, sizeof(int), &v);
        clEnqueueNDRangeKernel(c.queue, c.separableConvRodsKernel, 3, NULL, ws.globalRods, ws.localRods, 0, NULL, NULL);
        clFinish(c.queue);
    }

    clReleaseMemObject(c_SmoothX);
    clReleaseMemObject(c_SmoothY);
    clReleaseMemObject(c_SmoothZ);
    clReleaseMemObject(d_Convolved_Rows);
    clReleaseMemObject(d_Convolved_Columns);
    clReleaseMemObject(d_Certainty);
}

// ============================================================
//  Eigen-based linear algebra helpers
// ============================================================

static void solveEquationSystem(float* h_Params, float* h_A, float* h_h, int N)
{
    Eigen::MatrixXd A(N, N);
    Eigen::VectorXd h(N);
    for (int i = 0; i < N; i++) {
        h(i) = (double)h_h[i];
        for (int j = 0; j < N; j++)
            A(i, j) = (double)h_A[i + j * N];
    }
    Eigen::VectorXd x = A.fullPivHouseholderQr().solve(h);
    for (int i = 0; i < N; i++)
        h_Params[i] = (float)x(i);
}

static void removeTransformationScaling(float* h_Params)
{
    Eigen::MatrixXd T(3, 3);
    T(0, 0) = (double)h_Params[3] + 1.0;  T(0, 1) = (double)h_Params[4];        T(0, 2) = (double)h_Params[5];
    T(1, 0) = (double)h_Params[6];         T(1, 1) = (double)h_Params[7] + 1.0;  T(1, 2) = (double)h_Params[8];
    T(2, 0) = (double)h_Params[9];         T(2, 1) = (double)h_Params[10];        T(2, 2) = (double)h_Params[11] + 1.0;

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(T, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd R = svd.matrixU() * svd.matrixV().transpose();

    h_Params[3]  = (float)(R(0, 0) - 1.0);  h_Params[4]  = (float)R(0, 1);         h_Params[5]  = (float)R(0, 2);
    h_Params[6]  = (float)R(1, 0);           h_Params[7]  = (float)(R(1, 1) - 1.0); h_Params[8]  = (float)R(1, 2);
    h_Params[9]  = (float)R(2, 0);           h_Params[10] = (float)R(2, 1);         h_Params[11] = (float)(R(2, 2) - 1.0);
}

// Compose two affine transformations via 4x4 matrix multiplication
static void addAffineRegistrationParameters(float* h_Old, float* h_New)
{
    Eigen::MatrixXd O(4, 4), N_(4, 4), T(4, 4);
    O(0,0) = h_Old[3]+1; O(0,1) = h_Old[4];   O(0,2) = h_Old[5];   O(0,3) = h_Old[0];
    O(1,0) = h_Old[6];   O(1,1) = h_Old[7]+1;  O(1,2) = h_Old[8];   O(1,3) = h_Old[1];
    O(2,0) = h_Old[9];   O(2,1) = h_Old[10];   O(2,2) = h_Old[11]+1; O(2,3) = h_Old[2];
    O(3,0) = 0; O(3,1) = 0; O(3,2) = 0; O(3,3) = 1;

    N_(0,0) = h_New[3]+1; N_(0,1) = h_New[4];   N_(0,2) = h_New[5];   N_(0,3) = h_New[0];
    N_(1,0) = h_New[6];   N_(1,1) = h_New[7]+1;  N_(1,2) = h_New[8];   N_(1,3) = h_New[1];
    N_(2,0) = h_New[9];   N_(2,1) = h_New[10];   N_(2,2) = h_New[11]+1; N_(2,3) = h_New[2];
    N_(3,0) = 0; N_(3,1) = 0; N_(3,2) = 0; N_(3,3) = 1;

    T = N_ * O;
    h_Old[0] = (float)T(0,3); h_Old[1] = (float)T(1,3); h_Old[2] = (float)T(2,3);
    h_Old[3] = (float)(T(0,0)-1); h_Old[4] = (float)T(0,1); h_Old[5] = (float)T(0,2);
    h_Old[6] = (float)T(1,0); h_Old[7] = (float)(T(1,1)-1); h_Old[8] = (float)T(1,2);
    h_Old[9] = (float)T(2,0); h_Old[10] = (float)T(2,1); h_Old[11] = (float)(T(2,2)-1);
}

// Next-scale variant: multiply translations by 2 before composing
static void addAffineRegistrationParametersNextScale(float* h_Old, float* h_New)
{
    Eigen::MatrixXd O(4,4), N_(4,4), T(4,4);
    O(0,0) = h_Old[3]+1; O(0,1) = h_Old[4];   O(0,2) = h_Old[5];   O(0,3) = h_Old[0]*2.0;
    O(1,0) = h_Old[6];   O(1,1) = h_Old[7]+1;  O(1,2) = h_Old[8];   O(1,3) = h_Old[1]*2.0;
    O(2,0) = h_Old[9];   O(2,1) = h_Old[10];   O(2,2) = h_Old[11]+1; O(2,3) = h_Old[2]*2.0;
    O(3,0) = 0; O(3,1) = 0; O(3,2) = 0; O(3,3) = 1;

    N_(0,0) = h_New[3]+1; N_(0,1) = h_New[4];   N_(0,2) = h_New[5];   N_(0,3) = h_New[0]*2.0;
    N_(1,0) = h_New[6];   N_(1,1) = h_New[7]+1;  N_(1,2) = h_New[8];   N_(1,3) = h_New[1]*2.0;
    N_(2,0) = h_New[9];   N_(2,1) = h_New[10];   N_(2,2) = h_New[11]+1; N_(2,3) = h_New[2]*2.0;
    N_(3,0) = 0; N_(3,1) = 0; N_(3,2) = 0; N_(3,3) = 1;

    T = N_ * O;
    h_Old[0] = (float)T(0,3); h_Old[1] = (float)T(1,3); h_Old[2] = (float)T(2,3);
    h_Old[3] = (float)(T(0,0)-1); h_Old[4] = (float)T(0,1); h_Old[5] = (float)T(0,2);
    h_Old[6] = (float)T(1,0); h_Old[7] = (float)(T(1,1)-1); h_Old[8] = (float)T(1,2);
    h_Old[9] = (float)T(2,0); h_Old[10] = (float)T(2,1); h_Old[11] = (float)(T(2,2)-1);
}

static void invertAffineRegistrationParameters(float* h_Inv, float* h_Params)
{
    Eigen::MatrixXd A(4,4);
    A(0,0) = h_Params[3]+1; A(0,1) = h_Params[4];   A(0,2) = h_Params[5];   A(0,3) = h_Params[0];
    A(1,0) = h_Params[6];   A(1,1) = h_Params[7]+1;  A(1,2) = h_Params[8];   A(1,3) = h_Params[1];
    A(2,0) = h_Params[9];   A(2,1) = h_Params[10];   A(2,2) = h_Params[11]+1; A(2,3) = h_Params[2];
    A(3,0) = 0; A(3,1) = 0; A(3,2) = 0; A(3,3) = 1;

    Eigen::MatrixXd I = A.inverse();
    h_Inv[0] = (float)I(0,3); h_Inv[1] = (float)I(1,3); h_Inv[2] = (float)I(2,3);
    h_Inv[3] = (float)(I(0,0)-1); h_Inv[4] = (float)I(0,1); h_Inv[5] = (float)I(0,2);
    h_Inv[6] = (float)I(1,0); h_Inv[7] = (float)(I(1,1)-1); h_Inv[8] = (float)I(1,2);
    h_Inv[9] = (float)I(2,0); h_Inv[10] = (float)I(2,1); h_Inv[11] = (float)(I(2,2)-1);
}

// ============================================================
//  Center of mass / volume matching
// ============================================================

static void calculateCenterOfMass(OpenCLContext& c, float &rx, float &ry, float &rz, cl_mem d_Vol, int W, int H, int D)
{
    float* h_Temp = (float*)malloc(W * H * D * sizeof(float));
    clEnqueueReadBuffer(c.queue, d_Vol, CL_TRUE, 0, W * H * D * sizeof(float), h_Temp, 0, NULL, NULL);

    float totalMass = 0.0f;
    rx = ry = rz = 0.0f;
    for (int z = 0; z < D; z++)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                float mass = h_Temp[x + y * W + z * W * H];
                rx += mass * (float)x;
                ry += mass * (float)y;
                rz += mass * (float)z;
                totalMass += mass;
            }
    if (totalMass > 0.0f) {
        rx /= totalMass;
        ry /= totalMass;
        rz /= totalMass;
    }
    free(h_Temp);
}

// ============================================================
//  Transform volumes (linear and non-linear)
// ============================================================

static void transformVolumesLinear(OpenCLContext& c, cl_mem d_Volumes, float* h_Params, int W, int H, int D, int numVols, int interpMode)
{
    cl_mem c_Params = clCreateBuffer(c.context, CL_MEM_READ_ONLY, NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS * sizeof(float), NULL, NULL);
    clEnqueueWriteBuffer(c.queue, c_Params, CL_TRUE, 0, NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS * sizeof(float), h_Params, 0, NULL, NULL);

    cl_image_format format;
    format.image_channel_data_type = CL_FLOAT;
    format.image_channel_order = CL_INTENSITY;
    cl_mem d_Tex = clCreateImage3D(c.context, CL_MEM_READ_ONLY, &format, W, H, D, 0, 0, NULL, NULL);

    WorkSizes3D ws = calcWorkSizes16x16(c, W, H, D);

    for (int vol = 0; vol < numVols; vol++) {
        size_t origin[3] = {0, 0, 0};
        size_t region[3] = {(size_t)W, (size_t)H, (size_t)D};
        clEnqueueCopyBufferToImage(c.queue, d_Volumes, d_Tex, vol * W * H * D * sizeof(float), origin, region, 0, NULL, NULL);

        cl_kernel kern;
        if (interpMode == LINEAR)      kern = c.interpolateVolumeLinearLinearKernel;
        else if (interpMode == CUBIC)  kern = c.interpolateVolumeCubicLinearKernel;
        else                           kern = c.interpolateVolumeNearestLinearKernel;

        clSetKernelArg(kern, 0, sizeof(cl_mem), &d_Volumes);
        clSetKernelArg(kern, 1, sizeof(cl_mem), &d_Tex);
        clSetKernelArg(kern, 2, sizeof(cl_mem), &c_Params);
        clSetKernelArg(kern, 3, sizeof(int), &W);
        clSetKernelArg(kern, 4, sizeof(int), &H);
        clSetKernelArg(kern, 5, sizeof(int), &D);
        clSetKernelArg(kern, 6, sizeof(int), &vol);
        clEnqueueNDRangeKernel(c.queue, kern, 3, NULL, ws.global, ws.local, 0, NULL, NULL);
        clFinish(c.queue);
    }

    clReleaseMemObject(d_Tex);
    clReleaseMemObject(c_Params);
}

static void transformVolumesNonLinear(OpenCLContext& c, cl_mem d_Volumes,
    cl_mem d_DispX, cl_mem d_DispY, cl_mem d_DispZ,
    int W, int H, int D, int numVols, int interpMode)
{
    cl_image_format format;
    format.image_channel_data_type = CL_FLOAT;
    format.image_channel_order = CL_INTENSITY;
    cl_mem d_Tex = clCreateImage3D(c.context, CL_MEM_READ_ONLY, &format, W, H, D, 0, 0, NULL, NULL);

    WorkSizes3D ws = calcWorkSizes16x16(c, W, H, D);

    for (int vol = 0; vol < numVols; vol++) {
        size_t origin[3] = {0, 0, 0};
        size_t region[3] = {(size_t)W, (size_t)H, (size_t)D};
        clEnqueueCopyBufferToImage(c.queue, d_Volumes, d_Tex, vol * W * H * D * sizeof(float), origin, region, 0, NULL, NULL);

        cl_kernel kern = c.interpolateVolumeLinearNonLinearKernel;
        clSetKernelArg(kern, 0, sizeof(cl_mem), &d_Volumes);
        clSetKernelArg(kern, 1, sizeof(cl_mem), &d_Tex);
        clSetKernelArg(kern, 2, sizeof(cl_mem), &d_DispX);
        clSetKernelArg(kern, 3, sizeof(cl_mem), &d_DispY);
        clSetKernelArg(kern, 4, sizeof(cl_mem), &d_DispZ);
        clSetKernelArg(kern, 5, sizeof(int), &W);
        clSetKernelArg(kern, 6, sizeof(int), &H);
        clSetKernelArg(kern, 7, sizeof(int), &D);
        clSetKernelArg(kern, 8, sizeof(int), &vol);
        clEnqueueNDRangeKernel(c.queue, kern, 3, NULL, ws.global, ws.local, 0, NULL, NULL);
        clFinish(c.queue);
    }

    clReleaseMemObject(d_Tex);
}

// ============================================================
//  Change volume size (rescale)
// ============================================================

// Out-of-place version
static void changeVolumeSize(OpenCLContext& c, cl_mem d_Changed, cl_mem d_Orig,
    int origW, int origH, int origD, int newW, int newH, int newD, int interpMode)
{
    cl_image_format format;
    format.image_channel_data_type = CL_FLOAT;
    format.image_channel_order = CL_INTENSITY;
    cl_mem d_Tex = clCreateImage3D(c.context, CL_MEM_READ_ONLY, &format, origW, origH, origD, 0, 0, NULL, NULL);

    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {(size_t)origW, (size_t)origH, (size_t)origD};
    clEnqueueCopyBufferToImage(c.queue, d_Orig, d_Tex, 0, origin, region, 0, NULL, NULL);

    float vdx = (float)(origW - 1) / (float)(newW - 1);
    float vdy = (float)(origH - 1) / (float)(newH - 1);
    float vdz = (float)(origD - 1) / (float)(newD - 1);

    WorkSizes3D ws = calcWorkSizes16x16(c, newW, newH, newD);
    cl_kernel kern = (interpMode == CUBIC) ? c.rescaleVolumeCubicKernel : c.rescaleVolumeLinearKernel;

    clSetKernelArg(kern, 0, sizeof(cl_mem), &d_Changed);
    clSetKernelArg(kern, 1, sizeof(cl_mem), &d_Tex);
    clSetKernelArg(kern, 2, sizeof(float), &vdx);
    clSetKernelArg(kern, 3, sizeof(float), &vdy);
    clSetKernelArg(kern, 4, sizeof(float), &vdz);
    clSetKernelArg(kern, 5, sizeof(int), &newW);
    clSetKernelArg(kern, 6, sizeof(int), &newH);
    clSetKernelArg(kern, 7, sizeof(int), &newD);
    clEnqueueNDRangeKernel(c.queue, kern, 3, NULL, ws.global, ws.local, 0, NULL, NULL);
    clFinish(c.queue);

    clReleaseMemObject(d_Tex);
}

// In-place version (reallocates)
static void changeVolumeSizeInPlace(OpenCLContext& c, cl_mem& d_Vol,
    int origW, int origH, int origD, int newW, int newH, int newD, int interpMode)
{
    cl_image_format format;
    format.image_channel_data_type = CL_FLOAT;
    format.image_channel_order = CL_INTENSITY;
    cl_mem d_Tex = clCreateImage3D(c.context, CL_MEM_READ_ONLY, &format, origW, origH, origD, 0, 0, NULL, NULL);

    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {(size_t)origW, (size_t)origH, (size_t)origD};
    clEnqueueCopyBufferToImage(c.queue, d_Vol, d_Tex, 0, origin, region, 0, NULL, NULL);

    clReleaseMemObject(d_Vol);
    d_Vol = clCreateBuffer(c.context, CL_MEM_READ_WRITE, newW * newH * newD * sizeof(float), NULL, NULL);

    float vdx = (float)(origW - 1) / (float)(newW - 1);
    float vdy = (float)(origH - 1) / (float)(newH - 1);
    float vdz = (float)(origD - 1) / (float)(newD - 1);

    WorkSizes3D ws = calcWorkSizes16x16(c, newW, newH, newD);
    cl_kernel kern = (interpMode == CUBIC) ? c.rescaleVolumeCubicKernel : c.rescaleVolumeLinearKernel;

    clSetKernelArg(kern, 0, sizeof(cl_mem), &d_Vol);
    clSetKernelArg(kern, 1, sizeof(cl_mem), &d_Tex);
    clSetKernelArg(kern, 2, sizeof(float), &vdx);
    clSetKernelArg(kern, 3, sizeof(float), &vdy);
    clSetKernelArg(kern, 4, sizeof(float), &vdz);
    clSetKernelArg(kern, 5, sizeof(int), &newW);
    clSetKernelArg(kern, 6, sizeof(int), &newH);
    clSetKernelArg(kern, 7, sizeof(int), &newD);
    clEnqueueNDRangeKernel(c.queue, kern, 3, NULL, ws.global, ws.local, 0, NULL, NULL);
    clFinish(c.queue);

    clReleaseMemObject(d_Tex);
}

// ============================================================
//  Change volume resolution and size (T1 -> MNI space)
// ============================================================

static void changeVolumesResolutionAndSize(OpenCLContext& c,
    cl_mem d_New, cl_mem d_Vol,
    int origW, int origH, int origD,
    int newW, int newH, int newD,
    float voxX, float voxY, float voxZ,
    float newVoxX, float newVoxY, float newVoxZ,
    int mmZCut, int interpMode)
{
    int interpW = (int)myround((float)origW * voxX / newVoxX);
    int interpH = (int)myround((float)origH * voxY / newVoxY);
    int interpD = (int)myround((float)origD * voxZ / newVoxZ);

    cl_mem d_Interp = clCreateBuffer(c.context, CL_MEM_READ_WRITE, interpW * interpH * interpD * sizeof(float), NULL, NULL);

    cl_image_format format;
    format.image_channel_data_type = CL_FLOAT;
    format.image_channel_order = CL_INTENSITY;
    cl_mem d_Tex = clCreateImage3D(c.context, CL_MEM_READ_ONLY, &format, origW, origH, origD, 0, 0, NULL, NULL);

    float vdx = (float)(origW - 1) / (float)(interpW - 1);
    float vdy = (float)(origH - 1) / (float)(interpH - 1);
    float vdz = (float)(origD - 1) / (float)(interpD - 1);

    WorkSizes3D wsInterp = calcWorkSizes16x16(c, interpW, interpH, interpD);

    cl_kernel rescaleKern = (interpMode == CUBIC) ? c.rescaleVolumeCubicKernel :
                            (interpMode == NEAREST) ? c.rescaleVolumeNearestKernel :
                            c.rescaleVolumeLinearKernel;

    clSetKernelArg(rescaleKern, 0, sizeof(cl_mem), &d_Interp);
    clSetKernelArg(rescaleKern, 1, sizeof(cl_mem), &d_Tex);
    clSetKernelArg(rescaleKern, 2, sizeof(float), &vdx);
    clSetKernelArg(rescaleKern, 3, sizeof(float), &vdy);
    clSetKernelArg(rescaleKern, 4, sizeof(float), &vdz);
    clSetKernelArg(rescaleKern, 5, sizeof(int), &interpW);
    clSetKernelArg(rescaleKern, 6, sizeof(int), &interpH);
    clSetKernelArg(rescaleKern, 7, sizeof(int), &interpD);

    int x_diff = interpW - newW;
    int y_diff = interpH - newH;
    int z_diff = interpD - newD;

    int maxW_ = mymax(newW, interpW);
    int maxH_ = mymax(newH, interpH);
    int maxD_ = mymax(newD, interpD);
    WorkSizes3D wsCopy = calcWorkSizes16x16(c, maxW_, maxH_, maxD_);

    clSetKernelArg(c.copyVolumeToNewKernel, 0, sizeof(cl_mem), &d_New);
    clSetKernelArg(c.copyVolumeToNewKernel, 1, sizeof(cl_mem), &d_Interp);
    clSetKernelArg(c.copyVolumeToNewKernel, 2, sizeof(int), &newW);
    clSetKernelArg(c.copyVolumeToNewKernel, 3, sizeof(int), &newH);
    clSetKernelArg(c.copyVolumeToNewKernel, 4, sizeof(int), &newD);
    clSetKernelArg(c.copyVolumeToNewKernel, 5, sizeof(int), &interpW);
    clSetKernelArg(c.copyVolumeToNewKernel, 6, sizeof(int), &interpH);
    clSetKernelArg(c.copyVolumeToNewKernel, 7, sizeof(int), &interpD);
    clSetKernelArg(c.copyVolumeToNewKernel, 8, sizeof(int), &x_diff);
    clSetKernelArg(c.copyVolumeToNewKernel, 9, sizeof(int), &y_diff);
    clSetKernelArg(c.copyVolumeToNewKernel, 10, sizeof(int), &z_diff);
    clSetKernelArg(c.copyVolumeToNewKernel, 11, sizeof(int), &mmZCut);
    clSetKernelArg(c.copyVolumeToNewKernel, 12, sizeof(float), &newVoxZ);

    setMemory(c, d_New, 0.0f, newW * newH * newD);
    setMemory(c, d_Interp, 0.0f, interpW * interpH * interpD);

    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {(size_t)origW, (size_t)origH, (size_t)origD};
    clEnqueueCopyBufferToImage(c.queue, d_Vol, d_Tex, 0, origin, region, 0, NULL, NULL);

    clEnqueueNDRangeKernel(c.queue, rescaleKern, 3, NULL, wsInterp.global, wsInterp.local, 0, NULL, NULL);
    clFinish(c.queue);

    int volume = 0;
    clSetKernelArg(c.copyVolumeToNewKernel, 13, sizeof(int), &volume);
    clEnqueueNDRangeKernel(c.queue, c.copyVolumeToNewKernel, 3, NULL, wsCopy.global, wsCopy.local, 0, NULL, NULL);
    clFinish(c.queue);

    clReleaseMemObject(d_Interp);
    clReleaseMemObject(d_Tex);
}

// ============================================================
//  Match volume masses (center-of-mass alignment)
// ============================================================

static void matchVolumeMasses(OpenCLContext& c, cl_mem d_Vol1, cl_mem d_Vol2,
    float* h_Params, int W, int H, int D)
{
    float x1, y1, z1, x2, y2, z2;
    calculateCenterOfMass(c, x1, y1, z1, d_Vol1, W, H, D);
    calculateCenterOfMass(c, x2, y2, z2, d_Vol2, W, H, D);

    h_Params[0] = -myround(x2 - x1);
    h_Params[1] = -myround(y2 - y1);
    h_Params[2] = -myround(z2 - z1);
    for (int i = 3; i < 12; i++) h_Params[i] = 0.0f;

    transformVolumesLinear(c, d_Vol1, h_Params, W, H, D, 1, LINEAR);
}

// ============================================================
//  Combined displacement field
// ============================================================

static void createCombinedDisplacementField(OpenCLContext& c, float* h_Params,
    cl_mem d_DispX, cl_mem d_DispY, cl_mem d_DispZ, int W, int H, int D)
{
    WorkSizes3D ws = calcWorkSizes16x16(c, W, H, D);

    cl_mem c_Params = clCreateBuffer(c.context, CL_MEM_READ_ONLY, NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS * sizeof(float), NULL, NULL);
    clEnqueueWriteBuffer(c.queue, c_Params, CL_TRUE, 0, NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS * sizeof(float), h_Params, 0, NULL, NULL);

    clSetKernelArg(c.addLinearAndNonLinearDisplacementKernel, 0, sizeof(cl_mem), &d_DispX);
    clSetKernelArg(c.addLinearAndNonLinearDisplacementKernel, 1, sizeof(cl_mem), &d_DispY);
    clSetKernelArg(c.addLinearAndNonLinearDisplacementKernel, 2, sizeof(cl_mem), &d_DispZ);
    clSetKernelArg(c.addLinearAndNonLinearDisplacementKernel, 3, sizeof(cl_mem), &c_Params);
    clSetKernelArg(c.addLinearAndNonLinearDisplacementKernel, 4, sizeof(int), &W);
    clSetKernelArg(c.addLinearAndNonLinearDisplacementKernel, 5, sizeof(int), &H);
    clSetKernelArg(c.addLinearAndNonLinearDisplacementKernel, 6, sizeof(int), &D);
    clEnqueueNDRangeKernel(c.queue, c.addLinearAndNonLinearDisplacementKernel, 3, NULL, ws.global, ws.local, 0, NULL, NULL);
    clFinish(c.queue);

    clReleaseMemObject(c_Params);
}

// ============================================================
//  Linear registration: single scale
// ============================================================

static void alignTwoVolumesLinear(OpenCLContext& c,
    float* h_RegParams, int W, int H, int D, int numIter, int alignType,
    cl_mem d_Aligned, cl_mem d_Reference, cl_mem d_Original_Volume,
    cl_mem c_QF1R, cl_mem c_QF1I, cl_mem c_QF2R, cl_mem c_QF2I, cl_mem c_QF3R, cl_mem c_QF3I,
    float* h_QF1R, float* h_QF1I, float* h_QF2R, float* h_QF2I, float* h_QF3R, float* h_QF3I,
    cl_mem d_q11, cl_mem d_q12, cl_mem d_q13, cl_mem d_q21, cl_mem d_q22, cl_mem d_q23,
    cl_mem d_PhaseDiff, cl_mem d_PhaseCert, cl_mem d_PhaseGrad,
    cl_mem d_AMat2D, cl_mem d_AMat1D, cl_mem d_AMat,
    cl_mem d_hVec2D, cl_mem d_hVec1D, cl_mem d_hVec,
    cl_mem c_RegParams)
{
    // Work sizes for image registration
    WorkSizes3D wsPD = calcWorkSizes16x16(c, W, H, D);
    WorkSizes3D wsInterp = calcWorkSizes16x16(c, W, H, D);

    // Work sizes for A-matrix/h-vector 2D values
    size_t globalAH2DX[3] = {(size_t)H, (size_t)D, 1};
    size_t localAH2DX[3]  = {(size_t)H, 1, 1};
    size_t globalAH2DY[3] = {(size_t)H, (size_t)D, 1};
    size_t localAH2DY[3]  = {(size_t)H, 1, 1};
    size_t globalAH2DZ[3] = {(size_t)H, (size_t)D, 1};
    size_t localAH2DZ[3]  = {(size_t)H, 1, 1};

    // Work sizes for 1D reduction
    size_t globalA1D[3] = {(size_t)D, (size_t)NUMBER_OF_NON_ZERO_A_MATRIX_ELEMENTS, 1};
    size_t localA1D[3]  = {(size_t)D, 1, 1};
    size_t globalH1D[3] = {(size_t)D, (size_t)NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS, 1};
    size_t localH1D[3]  = {(size_t)D, 1, 1};

    // Work sizes for final A and h
    size_t globalA[1] = {(size_t)NUMBER_OF_NON_ZERO_A_MATRIX_ELEMENTS};
    size_t localA[1]  = {(size_t)NUMBER_OF_NON_ZERO_A_MATRIX_ELEMENTS};
    size_t globalH[1] = {(size_t)NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS};
    size_t localH[1]  = {(size_t)NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS};

    // Set kernel arguments that don't change per iteration
    clSetKernelArg(c.calculatePhaseDiffKernel, 0, sizeof(cl_mem), &d_PhaseDiff);
    clSetKernelArg(c.calculatePhaseDiffKernel, 1, sizeof(cl_mem), &d_PhaseCert);
    clSetKernelArg(c.calculatePhaseDiffKernel, 4, sizeof(int), &W);
    clSetKernelArg(c.calculatePhaseDiffKernel, 5, sizeof(int), &H);
    clSetKernelArg(c.calculatePhaseDiffKernel, 6, sizeof(int), &D);

    clSetKernelArg(c.calculatePhaseGradientsXKernel, 0, sizeof(cl_mem), &d_PhaseGrad);
    clSetKernelArg(c.calculatePhaseGradientsXKernel, 1, sizeof(cl_mem), &d_q11);
    clSetKernelArg(c.calculatePhaseGradientsXKernel, 2, sizeof(cl_mem), &d_q21);
    clSetKernelArg(c.calculatePhaseGradientsXKernel, 3, sizeof(int), &W);
    clSetKernelArg(c.calculatePhaseGradientsXKernel, 4, sizeof(int), &H);
    clSetKernelArg(c.calculatePhaseGradientsXKernel, 5, sizeof(int), &D);

    clSetKernelArg(c.calculatePhaseGradientsYKernel, 0, sizeof(cl_mem), &d_PhaseGrad);
    clSetKernelArg(c.calculatePhaseGradientsYKernel, 1, sizeof(cl_mem), &d_q12);
    clSetKernelArg(c.calculatePhaseGradientsYKernel, 2, sizeof(cl_mem), &d_q22);
    clSetKernelArg(c.calculatePhaseGradientsYKernel, 3, sizeof(int), &W);
    clSetKernelArg(c.calculatePhaseGradientsYKernel, 4, sizeof(int), &H);
    clSetKernelArg(c.calculatePhaseGradientsYKernel, 5, sizeof(int), &D);

    clSetKernelArg(c.calculatePhaseGradientsZKernel, 0, sizeof(cl_mem), &d_PhaseGrad);
    clSetKernelArg(c.calculatePhaseGradientsZKernel, 1, sizeof(cl_mem), &d_q13);
    clSetKernelArg(c.calculatePhaseGradientsZKernel, 2, sizeof(cl_mem), &d_q23);
    clSetKernelArg(c.calculatePhaseGradientsZKernel, 3, sizeof(int), &W);
    clSetKernelArg(c.calculatePhaseGradientsZKernel, 4, sizeof(int), &H);
    clSetKernelArg(c.calculatePhaseGradientsZKernel, 5, sizeof(int), &D);

    int filterSize = IMAGE_REGISTRATION_FILTER_SIZE;
    // AMatrix/HVector 2D Values kernels (X, Y, Z share the same argument pattern)
    cl_kernel ahKernels[3] = {c.calculateAMatrixAndHVector2DValuesXKernel,
                              c.calculateAMatrixAndHVector2DValuesYKernel,
                              c.calculateAMatrixAndHVector2DValuesZKernel};
    for (int k = 0; k < 3; k++) {
        clSetKernelArg(ahKernels[k], 0, sizeof(cl_mem), &d_AMat2D);
        clSetKernelArg(ahKernels[k], 1, sizeof(cl_mem), &d_hVec2D);
        clSetKernelArg(ahKernels[k], 2, sizeof(cl_mem), &d_PhaseDiff);
        clSetKernelArg(ahKernels[k], 3, sizeof(cl_mem), &d_PhaseGrad);
        clSetKernelArg(ahKernels[k], 4, sizeof(cl_mem), &d_PhaseCert);
        clSetKernelArg(ahKernels[k], 5, sizeof(int), &W);
        clSetKernelArg(ahKernels[k], 6, sizeof(int), &H);
        clSetKernelArg(ahKernels[k], 7, sizeof(int), &D);
        clSetKernelArg(ahKernels[k], 8, sizeof(int), &filterSize);
    }

    clSetKernelArg(c.calculateAMatrix1DValuesKernel, 0, sizeof(cl_mem), &d_AMat1D);
    clSetKernelArg(c.calculateAMatrix1DValuesKernel, 1, sizeof(cl_mem), &d_AMat2D);
    clSetKernelArg(c.calculateAMatrix1DValuesKernel, 2, sizeof(int), &W);
    clSetKernelArg(c.calculateAMatrix1DValuesKernel, 3, sizeof(int), &H);
    clSetKernelArg(c.calculateAMatrix1DValuesKernel, 4, sizeof(int), &D);
    clSetKernelArg(c.calculateAMatrix1DValuesKernel, 5, sizeof(int), &filterSize);

    clSetKernelArg(c.calculateHVector1DValuesKernel, 0, sizeof(cl_mem), &d_hVec1D);
    clSetKernelArg(c.calculateHVector1DValuesKernel, 1, sizeof(cl_mem), &d_hVec2D);
    clSetKernelArg(c.calculateHVector1DValuesKernel, 2, sizeof(int), &W);
    clSetKernelArg(c.calculateHVector1DValuesKernel, 3, sizeof(int), &H);
    clSetKernelArg(c.calculateHVector1DValuesKernel, 4, sizeof(int), &D);
    clSetKernelArg(c.calculateHVector1DValuesKernel, 5, sizeof(int), &filterSize);

    clSetKernelArg(c.calculateAMatrixKernel, 0, sizeof(cl_mem), &d_AMat);
    clSetKernelArg(c.calculateAMatrixKernel, 1, sizeof(cl_mem), &d_AMat1D);
    clSetKernelArg(c.calculateAMatrixKernel, 2, sizeof(int), &W);
    clSetKernelArg(c.calculateAMatrixKernel, 3, sizeof(int), &H);
    clSetKernelArg(c.calculateAMatrixKernel, 4, sizeof(int), &D);
    clSetKernelArg(c.calculateAMatrixKernel, 5, sizeof(int), &filterSize);

    clSetKernelArg(c.calculateHVectorKernel, 0, sizeof(cl_mem), &d_hVec);
    clSetKernelArg(c.calculateHVectorKernel, 1, sizeof(cl_mem), &d_hVec1D);
    clSetKernelArg(c.calculateHVectorKernel, 2, sizeof(int), &W);
    clSetKernelArg(c.calculateHVectorKernel, 3, sizeof(int), &H);
    clSetKernelArg(c.calculateHVectorKernel, 4, sizeof(int), &D);
    clSetKernelArg(c.calculateHVectorKernel, 5, sizeof(int), &filterSize);

    int volume = 0;
    clSetKernelArg(c.interpolateVolumeLinearLinearKernel, 0, sizeof(cl_mem), &d_Aligned);
    clSetKernelArg(c.interpolateVolumeLinearLinearKernel, 1, sizeof(cl_mem), &d_Original_Volume);
    clSetKernelArg(c.interpolateVolumeLinearLinearKernel, 2, sizeof(cl_mem), &c_RegParams);
    clSetKernelArg(c.interpolateVolumeLinearLinearKernel, 3, sizeof(int), &W);
    clSetKernelArg(c.interpolateVolumeLinearLinearKernel, 4, sizeof(int), &H);
    clSetKernelArg(c.interpolateVolumeLinearLinearKernel, 5, sizeof(int), &D);
    clSetKernelArg(c.interpolateVolumeLinearLinearKernel, 6, sizeof(int), &volume);

    // Calculate reference filter responses (once)
    nonseparableConvolution3D(c, d_q11, d_q12, d_q13, d_Reference,
        c_QF1R, c_QF1I, c_QF2R, c_QF2I, c_QF3R, c_QF3I,
        h_QF1R, h_QF1I, h_QF2R, h_QF2I, h_QF3R, h_QF3I, W, H, D);

    // Reset parameters
    float h_Params[12];
    for (int p = 0; p < 12; p++) {
        h_RegParams[p] = 0.0f;
        h_Params[p] = 0.0f;
    }

    float h_A_Matrix[144];
    float h_h_Vector[12];

    // Iterative registration
    for (int it = 0; it < numIter; it++) {
        // Filter responses for aligned volume
        nonseparableConvolution3D(c, d_q21, d_q22, d_q23, d_Aligned,
            c_QF1R, c_QF1I, c_QF2R, c_QF2I, c_QF3R, c_QF3I,
            h_QF1R, h_QF1I, h_QF2R, h_QF2I, h_QF3R, h_QF3I, W, H, D);

        // X direction
        clSetKernelArg(c.calculatePhaseDiffKernel, 2, sizeof(cl_mem), &d_q11);
        clSetKernelArg(c.calculatePhaseDiffKernel, 3, sizeof(cl_mem), &d_q21);
        clEnqueueNDRangeKernel(c.queue, c.calculatePhaseDiffKernel, 3, NULL, wsPD.global, wsPD.local, 0, NULL, NULL);
        clFinish(c.queue);

        clEnqueueNDRangeKernel(c.queue, c.calculatePhaseGradientsXKernel, 3, NULL, wsPD.global, wsPD.local, 0, NULL, NULL);
        clFinish(c.queue);

        clEnqueueNDRangeKernel(c.queue, c.calculateAMatrixAndHVector2DValuesXKernel, 3, NULL, globalAH2DX, localAH2DX, 0, NULL, NULL);
        clFinish(c.queue);

        // Y direction
        clSetKernelArg(c.calculatePhaseDiffKernel, 2, sizeof(cl_mem), &d_q12);
        clSetKernelArg(c.calculatePhaseDiffKernel, 3, sizeof(cl_mem), &d_q22);
        clEnqueueNDRangeKernel(c.queue, c.calculatePhaseDiffKernel, 3, NULL, wsPD.global, wsPD.local, 0, NULL, NULL);
        clFinish(c.queue);

        clEnqueueNDRangeKernel(c.queue, c.calculatePhaseGradientsYKernel, 3, NULL, wsPD.global, wsPD.local, 0, NULL, NULL);
        clFinish(c.queue);

        clEnqueueNDRangeKernel(c.queue, c.calculateAMatrixAndHVector2DValuesYKernel, 3, NULL, globalAH2DY, localAH2DY, 0, NULL, NULL);
        clFinish(c.queue);

        // Z direction
        clSetKernelArg(c.calculatePhaseDiffKernel, 2, sizeof(cl_mem), &d_q13);
        clSetKernelArg(c.calculatePhaseDiffKernel, 3, sizeof(cl_mem), &d_q23);
        clEnqueueNDRangeKernel(c.queue, c.calculatePhaseDiffKernel, 3, NULL, wsPD.global, wsPD.local, 0, NULL, NULL);
        clFinish(c.queue);

        clEnqueueNDRangeKernel(c.queue, c.calculatePhaseGradientsZKernel, 3, NULL, wsPD.global, wsPD.local, 0, NULL, NULL);
        clFinish(c.queue);

        clEnqueueNDRangeKernel(c.queue, c.calculateAMatrixAndHVector2DValuesZKernel, 3, NULL, globalAH2DZ, localAH2DZ, 0, NULL, NULL);
        clFinish(c.queue);

        // Reduce to 1D
        clEnqueueNDRangeKernel(c.queue, c.calculateAMatrix1DValuesKernel, 3, NULL, globalA1D, localA1D, 0, NULL, NULL);
        clFinish(c.queue);

        clEnqueueNDRangeKernel(c.queue, c.calculateHVector1DValuesKernel, 3, NULL, globalH1D, localH1D, 0, NULL, NULL);
        clFinish(c.queue);

        // Final A and h
        setMemory(c, d_AMat, 0.0f, NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS * NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS);

        clEnqueueNDRangeKernel(c.queue, c.calculateAMatrixKernel, 1, NULL, globalA, localA, 0, NULL, NULL);
        clFinish(c.queue);

        clEnqueueNDRangeKernel(c.queue, c.calculateHVectorKernel, 1, NULL, globalH, localH, 0, NULL, NULL);
        clFinish(c.queue);

        // Read back
        clEnqueueReadBuffer(c.queue, d_AMat, CL_TRUE, 0, 144 * sizeof(float), h_A_Matrix, 0, NULL, NULL);
        clEnqueueReadBuffer(c.queue, d_hVec, CL_TRUE, 0, 12 * sizeof(float), h_h_Vector, 0, NULL, NULL);

        // Mirror the upper-triangular part
        for (int j = 0; j < 12; j++)
            for (int i = 0; i < 12; i++)
                h_A_Matrix[j + i * 12] = h_A_Matrix[i + j * 12];

        // Solve
        solveEquationSystem(h_Params, h_A_Matrix, h_h_Vector, 12);

        if (alignType == RIGID) {
            removeTransformationScaling(h_Params);
            addAffineRegistrationParameters(h_RegParams, h_Params);
        } else if (alignType == AFFINE) {
            addAffineRegistrationParameters(h_RegParams, h_Params);
        } else { // TRANSLATION
            h_RegParams[0] += h_Params[0];
            h_RegParams[1] += h_Params[1];
            h_RegParams[2] += h_Params[2];
            for (int i = 3; i < 12; i++) h_RegParams[i] = 0.0f;
        }

        // Copy parameters to constant memory and interpolate
        clEnqueueWriteBuffer(c.queue, c_RegParams, CL_TRUE, 0, 12 * sizeof(float), h_RegParams, 0, NULL, NULL);
        clEnqueueNDRangeKernel(c.queue, c.interpolateVolumeLinearLinearKernel, 3, NULL, wsInterp.global, wsInterp.local, 0, NULL, NULL);
        clFinish(c.queue);
    }
}

// ============================================================
//  Linear registration: multi-scale
// ============================================================

static void alignTwoVolumesLinearSeveralScales(OpenCLContext& c,
    float* h_RegParams, cl_mem d_OrigAligned, cl_mem d_OrigReference,
    int W, int H, int D, int coarsestScale, int numIter,
    float* h_QF1R, float* h_QF1I, float* h_QF2R, float* h_QF2I, float* h_QF3R, float* h_QF3I)
{
    float h_ParamsTemp[12];
    for (int i = 0; i < 12; i++) { h_RegParams[i] = 0.0f; h_ParamsTemp[i] = 0.0f; }

    int curW = (int)myround((float)W / (float)coarsestScale);
    int curH = (int)myround((float)H / (float)coarsestScale);
    int curD = (int)myround((float)D / (float)coarsestScale);

    for (int scale = coarsestScale; scale >= 1; scale /= 2) {
        // Allocate GPU buffers for this scale
        cl_image_format fmt; fmt.image_channel_data_type = CL_FLOAT; fmt.image_channel_order = CL_INTENSITY;
        cl_mem d_OrigTex = clCreateImage3D(c.context, CL_MEM_READ_ONLY, &fmt, curW, curH, curD, 0, 0, NULL, NULL);
        cl_mem d_Aligned = clCreateBuffer(c.context, CL_MEM_READ_WRITE, curW * curH * curD * sizeof(float), NULL, NULL);
        cl_mem d_Ref     = clCreateBuffer(c.context, CL_MEM_READ_WRITE, curW * curH * curD * sizeof(float), NULL, NULL);

        int fsqr = IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE;
        cl_mem c_QF1R = clCreateBuffer(c.context, CL_MEM_READ_ONLY, fsqr * sizeof(float), NULL, NULL);
        cl_mem c_QF1I = clCreateBuffer(c.context, CL_MEM_READ_ONLY, fsqr * sizeof(float), NULL, NULL);
        cl_mem c_QF2R = clCreateBuffer(c.context, CL_MEM_READ_ONLY, fsqr * sizeof(float), NULL, NULL);
        cl_mem c_QF2I = clCreateBuffer(c.context, CL_MEM_READ_ONLY, fsqr * sizeof(float), NULL, NULL);
        cl_mem c_QF3R = clCreateBuffer(c.context, CL_MEM_READ_ONLY, fsqr * sizeof(float), NULL, NULL);
        cl_mem c_QF3I = clCreateBuffer(c.context, CL_MEM_READ_ONLY, fsqr * sizeof(float), NULL, NULL);
        cl_mem c_RegParams = clCreateBuffer(c.context, CL_MEM_READ_ONLY, 12 * sizeof(float), NULL, NULL);

        cl_mem d_q11 = clCreateBuffer(c.context, CL_MEM_READ_WRITE, curW * curH * curD * sizeof(cl_float2), NULL, NULL);
        cl_mem d_q12 = clCreateBuffer(c.context, CL_MEM_READ_WRITE, curW * curH * curD * sizeof(cl_float2), NULL, NULL);
        cl_mem d_q13 = clCreateBuffer(c.context, CL_MEM_READ_WRITE, curW * curH * curD * sizeof(cl_float2), NULL, NULL);
        cl_mem d_q21 = clCreateBuffer(c.context, CL_MEM_READ_WRITE, curW * curH * curD * sizeof(cl_float2), NULL, NULL);
        cl_mem d_q22 = clCreateBuffer(c.context, CL_MEM_READ_WRITE, curW * curH * curD * sizeof(cl_float2), NULL, NULL);
        cl_mem d_q23 = clCreateBuffer(c.context, CL_MEM_READ_WRITE, curW * curH * curD * sizeof(cl_float2), NULL, NULL);

        cl_mem d_PD = clCreateBuffer(c.context, CL_MEM_READ_WRITE, curW * curH * curD * sizeof(float), NULL, NULL);
        cl_mem d_PC = clCreateBuffer(c.context, CL_MEM_READ_WRITE, curW * curH * curD * sizeof(float), NULL, NULL);
        cl_mem d_PG = clCreateBuffer(c.context, CL_MEM_READ_WRITE, curW * curH * curD * sizeof(float), NULL, NULL);

        cl_mem d_AMat   = clCreateBuffer(c.context, CL_MEM_READ_WRITE, 144 * sizeof(float), NULL, NULL);
        cl_mem d_hVec   = clCreateBuffer(c.context, CL_MEM_READ_WRITE, 12 * sizeof(float), NULL, NULL);
        cl_mem d_AMat2D = clCreateBuffer(c.context, CL_MEM_READ_WRITE, curH * curD * NUMBER_OF_NON_ZERO_A_MATRIX_ELEMENTS * sizeof(float), NULL, NULL);
        cl_mem d_AMat1D = clCreateBuffer(c.context, CL_MEM_READ_WRITE, curD * NUMBER_OF_NON_ZERO_A_MATRIX_ELEMENTS * sizeof(float), NULL, NULL);
        cl_mem d_hVec2D = clCreateBuffer(c.context, CL_MEM_READ_WRITE, curH * curD * 12 * sizeof(float), NULL, NULL);
        cl_mem d_hVec1D = clCreateBuffer(c.context, CL_MEM_READ_WRITE, curD * 12 * sizeof(float), NULL, NULL);

        // Rescale original volumes
        changeVolumeSize(c, d_Aligned, d_OrigAligned, W, H, D, curW, curH, curD, LINEAR);
        changeVolumeSize(c, d_Ref, d_OrigReference, W, H, D, curW, curH, curD, LINEAR);

        // Copy to texture
        size_t origin[3] = {0, 0, 0};
        size_t region[3] = {(size_t)curW, (size_t)curH, (size_t)curD};
        clEnqueueCopyBufferToImage(c.queue, d_Aligned, d_OrigTex, 0, origin, region, 0, NULL, NULL);

        // Apply accumulated transform
        if (scale != coarsestScale) {
            clEnqueueWriteBuffer(c.queue, c_RegParams, CL_TRUE, 0, 12 * sizeof(float), h_RegParams, 0, NULL, NULL);
            WorkSizes3D wsI = calcWorkSizes16x16(c, curW, curH, curD);
            int vol = 0;
            clSetKernelArg(c.interpolateVolumeLinearLinearKernel, 0, sizeof(cl_mem), &d_Aligned);
            clSetKernelArg(c.interpolateVolumeLinearLinearKernel, 1, sizeof(cl_mem), &d_OrigTex);
            clSetKernelArg(c.interpolateVolumeLinearLinearKernel, 2, sizeof(cl_mem), &c_RegParams);
            clSetKernelArg(c.interpolateVolumeLinearLinearKernel, 3, sizeof(int), &curW);
            clSetKernelArg(c.interpolateVolumeLinearLinearKernel, 4, sizeof(int), &curH);
            clSetKernelArg(c.interpolateVolumeLinearLinearKernel, 5, sizeof(int), &curD);
            clSetKernelArg(c.interpolateVolumeLinearLinearKernel, 6, sizeof(int), &vol);
            clEnqueueNDRangeKernel(c.queue, c.interpolateVolumeLinearLinearKernel, 3, NULL, wsI.global, wsI.local, 0, NULL, NULL);
            clFinish(c.queue);
            clEnqueueCopyBufferToImage(c.queue, d_Aligned, d_OrigTex, 0, origin, region, 0, NULL, NULL);
        }

        int iters = (scale == 1) ? (int)ceil((float)numIter / 5.0f) : numIter;

        for (int i = 0; i < 12; i++) h_ParamsTemp[i] = 0.0f;

        alignTwoVolumesLinear(c, h_ParamsTemp, curW, curH, curD, iters, AFFINE,
            d_Aligned, d_Ref, d_OrigTex,
            c_QF1R, c_QF1I, c_QF2R, c_QF2I, c_QF3R, c_QF3I,
            h_QF1R, h_QF1I, h_QF2R, h_QF2I, h_QF3R, h_QF3I,
            d_q11, d_q12, d_q13, d_q21, d_q22, d_q23,
            d_PD, d_PC, d_PG,
            d_AMat2D, d_AMat1D, d_AMat,
            d_hVec2D, d_hVec1D, d_hVec,
            c_RegParams);

        // Cleanup this scale
        clReleaseMemObject(d_OrigTex); clReleaseMemObject(d_Aligned); clReleaseMemObject(d_Ref);
        clReleaseMemObject(c_QF1R); clReleaseMemObject(c_QF1I);
        clReleaseMemObject(c_QF2R); clReleaseMemObject(c_QF2I);
        clReleaseMemObject(c_QF3R); clReleaseMemObject(c_QF3I);
        clReleaseMemObject(c_RegParams);
        clReleaseMemObject(d_q11); clReleaseMemObject(d_q12); clReleaseMemObject(d_q13);
        clReleaseMemObject(d_q21); clReleaseMemObject(d_q22); clReleaseMemObject(d_q23);
        clReleaseMemObject(d_PD); clReleaseMemObject(d_PC); clReleaseMemObject(d_PG);
        clReleaseMemObject(d_AMat); clReleaseMemObject(d_hVec);
        clReleaseMemObject(d_AMat2D); clReleaseMemObject(d_AMat1D);
        clReleaseMemObject(d_hVec2D); clReleaseMemObject(d_hVec1D);

        if (scale != 1) {
            addAffineRegistrationParametersNextScale(h_RegParams, h_ParamsTemp);
            curW = (int)myround((float)W / ((float)scale / 2.0f));
            curH = (int)myround((float)H / ((float)scale / 2.0f));
            curD = (int)myround((float)D / ((float)scale / 2.0f));
        } else {
            addAffineRegistrationParameters(h_RegParams, h_ParamsTemp);
            // Final transform
            transformVolumesLinear(c, d_OrigAligned, h_RegParams, W, H, D, 1, LINEAR);
        }
    }
}

// ============================================================
//  Non-linear registration: single scale
// ============================================================

static void alignTwoVolumesNonLinear(OpenCLContext& c,
    cl_mem d_Aligned, cl_mem d_Reference, cl_mem d_OrigTex,
    cl_mem d_UpdDispX, cl_mem d_UpdDispY, cl_mem d_UpdDispZ,
    float* h_NLF1R, float* h_NLF1I, float* h_NLF2R, float* h_NLF2I, float* h_NLF3R, float* h_NLF3I,
    float* h_NLF4R, float* h_NLF4I, float* h_NLF5R, float* h_NLF5I, float* h_NLF6R, float* h_NLF6I,
    const float projTensors[6][6],
    const float* filterDirX, const float* filterDirY, const float* filterDirZ,
    int W, int H, int D, int numIter)
{
    int fsqr = IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE;

    cl_mem c_QF1R = clCreateBuffer(c.context, CL_MEM_READ_ONLY, fsqr * sizeof(float), NULL, NULL);
    cl_mem c_QF1I = clCreateBuffer(c.context, CL_MEM_READ_ONLY, fsqr * sizeof(float), NULL, NULL);
    cl_mem c_QF2R = clCreateBuffer(c.context, CL_MEM_READ_ONLY, fsqr * sizeof(float), NULL, NULL);
    cl_mem c_QF2I = clCreateBuffer(c.context, CL_MEM_READ_ONLY, fsqr * sizeof(float), NULL, NULL);
    cl_mem c_QF3R = clCreateBuffer(c.context, CL_MEM_READ_ONLY, fsqr * sizeof(float), NULL, NULL);
    cl_mem c_QF3I = clCreateBuffer(c.context, CL_MEM_READ_ONLY, fsqr * sizeof(float), NULL, NULL);

    int volSize = W * H * D;

    // 12 complex filter responses (6 for ref, 6 for aligned)
    cl_mem d_q11 = clCreateBuffer(c.context, CL_MEM_READ_WRITE, volSize * sizeof(cl_float2), NULL, NULL);
    cl_mem d_q12 = clCreateBuffer(c.context, CL_MEM_READ_WRITE, volSize * sizeof(cl_float2), NULL, NULL);
    cl_mem d_q13 = clCreateBuffer(c.context, CL_MEM_READ_WRITE, volSize * sizeof(cl_float2), NULL, NULL);
    cl_mem d_q14 = clCreateBuffer(c.context, CL_MEM_READ_WRITE, volSize * sizeof(cl_float2), NULL, NULL);
    cl_mem d_q15 = clCreateBuffer(c.context, CL_MEM_READ_WRITE, volSize * sizeof(cl_float2), NULL, NULL);
    cl_mem d_q16 = clCreateBuffer(c.context, CL_MEM_READ_WRITE, volSize * sizeof(cl_float2), NULL, NULL);
    cl_mem d_q21 = clCreateBuffer(c.context, CL_MEM_READ_WRITE, volSize * sizeof(cl_float2), NULL, NULL);
    cl_mem d_q22 = clCreateBuffer(c.context, CL_MEM_READ_WRITE, volSize * sizeof(cl_float2), NULL, NULL);
    cl_mem d_q23 = clCreateBuffer(c.context, CL_MEM_READ_WRITE, volSize * sizeof(cl_float2), NULL, NULL);
    cl_mem d_q24 = clCreateBuffer(c.context, CL_MEM_READ_WRITE, volSize * sizeof(cl_float2), NULL, NULL);
    cl_mem d_q25 = clCreateBuffer(c.context, CL_MEM_READ_WRITE, volSize * sizeof(cl_float2), NULL, NULL);
    cl_mem d_q26 = clCreateBuffer(c.context, CL_MEM_READ_WRITE, volSize * sizeof(cl_float2), NULL, NULL);

    // Tensor components
    cl_mem d_t11 = clCreateBuffer(c.context, CL_MEM_READ_WRITE, volSize * sizeof(float), NULL, NULL);
    cl_mem d_t12 = clCreateBuffer(c.context, CL_MEM_READ_WRITE, volSize * sizeof(float), NULL, NULL);
    cl_mem d_t13 = clCreateBuffer(c.context, CL_MEM_READ_WRITE, volSize * sizeof(float), NULL, NULL);
    cl_mem d_t22 = clCreateBuffer(c.context, CL_MEM_READ_WRITE, volSize * sizeof(float), NULL, NULL);
    cl_mem d_t23 = clCreateBuffer(c.context, CL_MEM_READ_WRITE, volSize * sizeof(float), NULL, NULL);
    cl_mem d_t33 = clCreateBuffer(c.context, CL_MEM_READ_WRITE, volSize * sizeof(float), NULL, NULL);

    // A-matrix and h-vector per voxel
    cl_mem d_a11 = clCreateBuffer(c.context, CL_MEM_READ_WRITE, volSize * sizeof(float), NULL, NULL);
    cl_mem d_a12 = clCreateBuffer(c.context, CL_MEM_READ_WRITE, volSize * sizeof(float), NULL, NULL);
    cl_mem d_a13 = clCreateBuffer(c.context, CL_MEM_READ_WRITE, volSize * sizeof(float), NULL, NULL);
    cl_mem d_a22 = clCreateBuffer(c.context, CL_MEM_READ_WRITE, volSize * sizeof(float), NULL, NULL);
    cl_mem d_a23 = clCreateBuffer(c.context, CL_MEM_READ_WRITE, volSize * sizeof(float), NULL, NULL);
    cl_mem d_a33 = clCreateBuffer(c.context, CL_MEM_READ_WRITE, volSize * sizeof(float), NULL, NULL);
    cl_mem d_h1  = clCreateBuffer(c.context, CL_MEM_READ_WRITE, volSize * sizeof(float), NULL, NULL);
    cl_mem d_h2  = clCreateBuffer(c.context, CL_MEM_READ_WRITE, volSize * sizeof(float), NULL, NULL);
    cl_mem d_h3  = clCreateBuffer(c.context, CL_MEM_READ_WRITE, volSize * sizeof(float), NULL, NULL);

    cl_mem d_TempDispX = clCreateBuffer(c.context, CL_MEM_READ_WRITE, volSize * sizeof(float), NULL, NULL);
    cl_mem d_TempDispY = clCreateBuffer(c.context, CL_MEM_READ_WRITE, volSize * sizeof(float), NULL, NULL);
    cl_mem d_TempDispZ = clCreateBuffer(c.context, CL_MEM_READ_WRITE, volSize * sizeof(float), NULL, NULL);

    // Filter directions
    cl_mem c_DirX = clCreateBuffer(c.context, CL_MEM_READ_ONLY, 6 * sizeof(float), NULL, NULL);
    cl_mem c_DirY = clCreateBuffer(c.context, CL_MEM_READ_ONLY, 6 * sizeof(float), NULL, NULL);
    cl_mem c_DirZ = clCreateBuffer(c.context, CL_MEM_READ_ONLY, 6 * sizeof(float), NULL, NULL);
    clEnqueueWriteBuffer(c.queue, c_DirX, CL_TRUE, 0, 6 * sizeof(float), filterDirX, 0, NULL, NULL);
    clEnqueueWriteBuffer(c.queue, c_DirY, CL_TRUE, 0, 6 * sizeof(float), filterDirY, 0, NULL, NULL);
    clEnqueueWriteBuffer(c.queue, c_DirZ, CL_TRUE, 0, 6 * sizeof(float), filterDirZ, 0, NULL, NULL);

    WorkSizes3D wsPD = calcWorkSizes16x16(c, W, H, D);

    // Setup tensor/displacement kernel args (persistent across iterations)
    clSetKernelArg(c.calculateTensorComponentsKernel, 0, sizeof(cl_mem), &d_t11);
    clSetKernelArg(c.calculateTensorComponentsKernel, 1, sizeof(cl_mem), &d_t12);
    clSetKernelArg(c.calculateTensorComponentsKernel, 2, sizeof(cl_mem), &d_t13);
    clSetKernelArg(c.calculateTensorComponentsKernel, 3, sizeof(cl_mem), &d_t22);
    clSetKernelArg(c.calculateTensorComponentsKernel, 4, sizeof(cl_mem), &d_t23);
    clSetKernelArg(c.calculateTensorComponentsKernel, 5, sizeof(cl_mem), &d_t33);
    clSetKernelArg(c.calculateTensorComponentsKernel, 14, sizeof(int), &W);
    clSetKernelArg(c.calculateTensorComponentsKernel, 15, sizeof(int), &H);
    clSetKernelArg(c.calculateTensorComponentsKernel, 16, sizeof(int), &D);

    clSetKernelArg(c.calculateTensorNormsKernel, 0, sizeof(cl_mem), &d_a11);
    clSetKernelArg(c.calculateTensorNormsKernel, 1, sizeof(cl_mem), &d_t11);
    clSetKernelArg(c.calculateTensorNormsKernel, 2, sizeof(cl_mem), &d_t12);
    clSetKernelArg(c.calculateTensorNormsKernel, 3, sizeof(cl_mem), &d_t13);
    clSetKernelArg(c.calculateTensorNormsKernel, 4, sizeof(cl_mem), &d_t22);
    clSetKernelArg(c.calculateTensorNormsKernel, 5, sizeof(cl_mem), &d_t23);
    clSetKernelArg(c.calculateTensorNormsKernel, 6, sizeof(cl_mem), &d_t33);
    clSetKernelArg(c.calculateTensorNormsKernel, 7, sizeof(int), &W);
    clSetKernelArg(c.calculateTensorNormsKernel, 8, sizeof(int), &H);
    clSetKernelArg(c.calculateTensorNormsKernel, 9, sizeof(int), &D);

    clSetKernelArg(c.calculateAMatricesAndHVectorsKernel, 0, sizeof(cl_mem), &d_a11);
    clSetKernelArg(c.calculateAMatricesAndHVectorsKernel, 1, sizeof(cl_mem), &d_a12);
    clSetKernelArg(c.calculateAMatricesAndHVectorsKernel, 2, sizeof(cl_mem), &d_a13);
    clSetKernelArg(c.calculateAMatricesAndHVectorsKernel, 3, sizeof(cl_mem), &d_a22);
    clSetKernelArg(c.calculateAMatricesAndHVectorsKernel, 4, sizeof(cl_mem), &d_a23);
    clSetKernelArg(c.calculateAMatricesAndHVectorsKernel, 5, sizeof(cl_mem), &d_a33);
    clSetKernelArg(c.calculateAMatricesAndHVectorsKernel, 6, sizeof(cl_mem), &d_h1);
    clSetKernelArg(c.calculateAMatricesAndHVectorsKernel, 7, sizeof(cl_mem), &d_h2);
    clSetKernelArg(c.calculateAMatricesAndHVectorsKernel, 8, sizeof(cl_mem), &d_h3);
    clSetKernelArg(c.calculateAMatricesAndHVectorsKernel, 11, sizeof(cl_mem), &d_t11);
    clSetKernelArg(c.calculateAMatricesAndHVectorsKernel, 12, sizeof(cl_mem), &d_t12);
    clSetKernelArg(c.calculateAMatricesAndHVectorsKernel, 13, sizeof(cl_mem), &d_t13);
    clSetKernelArg(c.calculateAMatricesAndHVectorsKernel, 14, sizeof(cl_mem), &d_t22);
    clSetKernelArg(c.calculateAMatricesAndHVectorsKernel, 15, sizeof(cl_mem), &d_t23);
    clSetKernelArg(c.calculateAMatricesAndHVectorsKernel, 16, sizeof(cl_mem), &d_t33);
    clSetKernelArg(c.calculateAMatricesAndHVectorsKernel, 17, sizeof(cl_mem), &c_DirX);
    clSetKernelArg(c.calculateAMatricesAndHVectorsKernel, 18, sizeof(cl_mem), &c_DirY);
    clSetKernelArg(c.calculateAMatricesAndHVectorsKernel, 19, sizeof(cl_mem), &c_DirZ);
    clSetKernelArg(c.calculateAMatricesAndHVectorsKernel, 20, sizeof(int), &W);
    clSetKernelArg(c.calculateAMatricesAndHVectorsKernel, 21, sizeof(int), &H);
    clSetKernelArg(c.calculateAMatricesAndHVectorsKernel, 22, sizeof(int), &D);

    clSetKernelArg(c.calculateDisplacementUpdateKernel, 0, sizeof(cl_mem), &d_TempDispX);
    clSetKernelArg(c.calculateDisplacementUpdateKernel, 1, sizeof(cl_mem), &d_TempDispY);
    clSetKernelArg(c.calculateDisplacementUpdateKernel, 2, sizeof(cl_mem), &d_TempDispZ);
    clSetKernelArg(c.calculateDisplacementUpdateKernel, 3, sizeof(cl_mem), &d_a11);
    clSetKernelArg(c.calculateDisplacementUpdateKernel, 4, sizeof(cl_mem), &d_a12);
    clSetKernelArg(c.calculateDisplacementUpdateKernel, 5, sizeof(cl_mem), &d_a13);
    clSetKernelArg(c.calculateDisplacementUpdateKernel, 6, sizeof(cl_mem), &d_a22);
    clSetKernelArg(c.calculateDisplacementUpdateKernel, 7, sizeof(cl_mem), &d_a23);
    clSetKernelArg(c.calculateDisplacementUpdateKernel, 8, sizeof(cl_mem), &d_a33);
    clSetKernelArg(c.calculateDisplacementUpdateKernel, 9, sizeof(cl_mem), &d_h1);
    clSetKernelArg(c.calculateDisplacementUpdateKernel, 10, sizeof(cl_mem), &d_h2);
    clSetKernelArg(c.calculateDisplacementUpdateKernel, 11, sizeof(cl_mem), &d_h3);
    clSetKernelArg(c.calculateDisplacementUpdateKernel, 12, sizeof(int), &W);
    clSetKernelArg(c.calculateDisplacementUpdateKernel, 13, sizeof(int), &H);
    clSetKernelArg(c.calculateDisplacementUpdateKernel, 14, sizeof(int), &D);

    int vol = 0;
    clSetKernelArg(c.interpolateVolumeLinearNonLinearKernel, 0, sizeof(cl_mem), &d_Aligned);
    clSetKernelArg(c.interpolateVolumeLinearNonLinearKernel, 1, sizeof(cl_mem), &d_OrigTex);
    clSetKernelArg(c.interpolateVolumeLinearNonLinearKernel, 5, sizeof(int), &W);
    clSetKernelArg(c.interpolateVolumeLinearNonLinearKernel, 6, sizeof(int), &H);
    clSetKernelArg(c.interpolateVolumeLinearNonLinearKernel, 7, sizeof(int), &D);
    clSetKernelArg(c.interpolateVolumeLinearNonLinearKernel, 8, sizeof(int), &vol);

    // Reference filter responses (once)
    nonseparableConvolution3D(c, d_q11, d_q12, d_q13, d_Reference, c_QF1R, c_QF1I, c_QF2R, c_QF2I, c_QF3R, c_QF3I,
        h_NLF1R, h_NLF1I, h_NLF2R, h_NLF2I, h_NLF3R, h_NLF3I, W, H, D);
    nonseparableConvolution3D(c, d_q14, d_q15, d_q16, d_Reference, c_QF1R, c_QF1I, c_QF2R, c_QF2I, c_QF3R, c_QF3I,
        h_NLF4R, h_NLF4I, h_NLF5R, h_NLF5I, h_NLF6R, h_NLF6I, W, H, D);

    setMemory(c, d_UpdDispX, 0.0f, volSize);
    setMemory(c, d_UpdDispY, 0.0f, volSize);
    setMemory(c, d_UpdDispZ, 0.0f, volSize);

    float h_SmoothX[SMOOTHING_FILTER_SIZE], h_SmoothY[SMOOTHING_FILTER_SIZE], h_SmoothZ[SMOOTHING_FILTER_SIZE];

    // Projection tensor values for 6 filters
    float M[6][6];
    for (int f = 0; f < 6; f++)
        for (int j = 0; j < 6; j++)
            M[f][j] = projTensors[f][j];

    cl_mem qRef[6]  = {d_q11, d_q12, d_q13, d_q14, d_q15, d_q16};
    cl_mem qAln[6]  = {d_q21, d_q22, d_q23, d_q24, d_q25, d_q26};

    WorkSizes3D wsTN = calcWorkSizes16x16(c, W, H, D);
    WorkSizes3D wsAH = calcWorkSizes16x16(c, W, H, D);
    WorkSizes3D wsDU = calcWorkSizes16x16(c, W, H, D);

    for (int it = 0; it < numIter; it++) {
        nonseparableConvolution3D(c, d_q21, d_q22, d_q23, d_Aligned, c_QF1R, c_QF1I, c_QF2R, c_QF2I, c_QF3R, c_QF3I,
            h_NLF1R, h_NLF1I, h_NLF2R, h_NLF2I, h_NLF3R, h_NLF3I, W, H, D);
        nonseparableConvolution3D(c, d_q24, d_q25, d_q26, d_Aligned, c_QF1R, c_QF1I, c_QF2R, c_QF2I, c_QF3R, c_QF3I,
            h_NLF4R, h_NLF4I, h_NLF5R, h_NLF5I, h_NLF6R, h_NLF6I, W, H, D);

        // Reset tensors and equation system
        setMemory(c, d_t11, 0.0f, volSize); setMemory(c, d_t12, 0.0f, volSize);
        setMemory(c, d_t13, 0.0f, volSize); setMemory(c, d_t22, 0.0f, volSize);
        setMemory(c, d_t23, 0.0f, volSize); setMemory(c, d_t33, 0.0f, volSize);
        setMemory(c, d_a11, 0.0f, volSize); setMemory(c, d_a12, 0.0f, volSize);
        setMemory(c, d_a13, 0.0f, volSize); setMemory(c, d_a22, 0.0f, volSize);
        setMemory(c, d_a23, 0.0f, volSize); setMemory(c, d_a33, 0.0f, volSize);
        setMemory(c, d_h1, 0.0f, volSize); setMemory(c, d_h2, 0.0f, volSize); setMemory(c, d_h3, 0.0f, volSize);

        // Calculate tensor components for all 6 filters
        for (int f = 0; f < 6; f++) {
            clSetKernelArg(c.calculateTensorComponentsKernel, 6, sizeof(cl_mem), &qRef[f]);
            clSetKernelArg(c.calculateTensorComponentsKernel, 7, sizeof(cl_mem), &qAln[f]);
            clSetKernelArg(c.calculateTensorComponentsKernel, 8,  sizeof(float), &M[f][0]);
            clSetKernelArg(c.calculateTensorComponentsKernel, 9,  sizeof(float), &M[f][1]);
            clSetKernelArg(c.calculateTensorComponentsKernel, 10, sizeof(float), &M[f][2]);
            clSetKernelArg(c.calculateTensorComponentsKernel, 11, sizeof(float), &M[f][3]);
            clSetKernelArg(c.calculateTensorComponentsKernel, 12, sizeof(float), &M[f][4]);
            clSetKernelArg(c.calculateTensorComponentsKernel, 13, sizeof(float), &M[f][5]);
            clEnqueueNDRangeKernel(c.queue, c.calculateTensorComponentsKernel, 3, NULL, wsPD.global, wsPD.local, 0, NULL, NULL);
        }

        // Tensor norms
        clEnqueueNDRangeKernel(c.queue, c.calculateTensorNormsKernel, 3, NULL, wsTN.global, wsTN.local, 0, NULL, NULL);

        // Smooth tensor components
        createSmoothingFilters(h_SmoothX, h_SmoothY, h_SmoothZ, SMOOTHING_FILTER_SIZE, TSIGMA);
        performSmoothing(c, d_t11, h_SmoothX, h_SmoothY, h_SmoothZ, W, H, D, 1);
        performSmoothing(c, d_t12, h_SmoothX, h_SmoothY, h_SmoothZ, W, H, D, 1);
        performSmoothing(c, d_t13, h_SmoothX, h_SmoothY, h_SmoothZ, W, H, D, 1);
        performSmoothing(c, d_t22, h_SmoothX, h_SmoothY, h_SmoothZ, W, H, D, 1);
        performSmoothing(c, d_t23, h_SmoothX, h_SmoothY, h_SmoothZ, W, H, D, 1);
        performSmoothing(c, d_t33, h_SmoothX, h_SmoothY, h_SmoothZ, W, H, D, 1);

        clEnqueueNDRangeKernel(c.queue, c.calculateTensorNormsKernel, 3, NULL, wsTN.global, wsTN.local, 0, NULL, NULL);

        float maxNorm = calculateMax(c, d_a11, W, H, D);
        if (maxNorm > 0.0f) {
            float invMax = 1.0f / maxNorm;
            multiplyVolume(c, d_t11, invMax, W, H, D);
            multiplyVolume(c, d_t12, invMax, W, H, D);
            multiplyVolume(c, d_t13, invMax, W, H, D);
            multiplyVolume(c, d_t22, invMax, W, H, D);
            multiplyVolume(c, d_t23, invMax, W, H, D);
            multiplyVolume(c, d_t33, invMax, W, H, D);
        }

        // A-matrices and h-vectors for 6 filters
        for (int f = 0; f < 6; f++) {
            clSetKernelArg(c.calculateAMatricesAndHVectorsKernel, 9, sizeof(cl_mem), &qRef[f]);
            clSetKernelArg(c.calculateAMatricesAndHVectorsKernel, 10, sizeof(cl_mem), &qAln[f]);
            clSetKernelArg(c.calculateAMatricesAndHVectorsKernel, 23, sizeof(int), &f);
            clEnqueueNDRangeKernel(c.queue, c.calculateAMatricesAndHVectorsKernel, 3, NULL, wsAH.global, wsAH.local, 0, NULL, NULL);
        }

        // Smooth A and h
        createSmoothingFilters(h_SmoothX, h_SmoothY, h_SmoothZ, SMOOTHING_FILTER_SIZE, ESIGMA);
        performSmoothing(c, d_a11, h_SmoothX, h_SmoothY, h_SmoothZ, W, H, D, 1);
        performSmoothing(c, d_a12, h_SmoothX, h_SmoothY, h_SmoothZ, W, H, D, 1);
        performSmoothing(c, d_a13, h_SmoothX, h_SmoothY, h_SmoothZ, W, H, D, 1);
        performSmoothing(c, d_a22, h_SmoothX, h_SmoothY, h_SmoothZ, W, H, D, 1);
        performSmoothing(c, d_a23, h_SmoothX, h_SmoothY, h_SmoothZ, W, H, D, 1);
        performSmoothing(c, d_a33, h_SmoothX, h_SmoothY, h_SmoothZ, W, H, D, 1);
        performSmoothing(c, d_h1,  h_SmoothX, h_SmoothY, h_SmoothZ, W, H, D, 1);
        performSmoothing(c, d_h2,  h_SmoothX, h_SmoothY, h_SmoothZ, W, H, D, 1);
        performSmoothing(c, d_h3,  h_SmoothX, h_SmoothY, h_SmoothZ, W, H, D, 1);

        // Displacement update
        clEnqueueNDRangeKernel(c.queue, c.calculateDisplacementUpdateKernel, 3, NULL, wsDU.global, wsDU.local, 0, NULL, NULL);

        // Smooth displacement update
        createSmoothingFilters(h_SmoothX, h_SmoothY, h_SmoothZ, SMOOTHING_FILTER_SIZE, DSIGMA);
        performSmoothing(c, d_TempDispX, h_SmoothX, h_SmoothY, h_SmoothZ, W, H, D, 1);
        performSmoothing(c, d_TempDispY, h_SmoothX, h_SmoothY, h_SmoothZ, W, H, D, 1);
        performSmoothing(c, d_TempDispZ, h_SmoothX, h_SmoothY, h_SmoothZ, W, H, D, 1);

        // Accumulate
        addVolumes(c, d_UpdDispX, d_TempDispX, W, H, D);
        addVolumes(c, d_UpdDispY, d_TempDispY, W, H, D);
        addVolumes(c, d_UpdDispZ, d_TempDispZ, W, H, D);

        // Interpolate
        clSetKernelArg(c.interpolateVolumeLinearNonLinearKernel, 2, sizeof(cl_mem), &d_UpdDispX);
        clSetKernelArg(c.interpolateVolumeLinearNonLinearKernel, 3, sizeof(cl_mem), &d_UpdDispY);
        clSetKernelArg(c.interpolateVolumeLinearNonLinearKernel, 4, sizeof(cl_mem), &d_UpdDispZ);
        clEnqueueNDRangeKernel(c.queue, c.interpolateVolumeLinearNonLinearKernel, 3, NULL, wsPD.global, wsPD.local, 0, NULL, NULL);
        clFinish(c.queue);
    }

    // Cleanup
    clReleaseMemObject(c_QF1R); clReleaseMemObject(c_QF1I);
    clReleaseMemObject(c_QF2R); clReleaseMemObject(c_QF2I);
    clReleaseMemObject(c_QF3R); clReleaseMemObject(c_QF3I);
    clReleaseMemObject(d_q11); clReleaseMemObject(d_q12); clReleaseMemObject(d_q13);
    clReleaseMemObject(d_q14); clReleaseMemObject(d_q15); clReleaseMemObject(d_q16);
    clReleaseMemObject(d_q21); clReleaseMemObject(d_q22); clReleaseMemObject(d_q23);
    clReleaseMemObject(d_q24); clReleaseMemObject(d_q25); clReleaseMemObject(d_q26);
    clReleaseMemObject(d_t11); clReleaseMemObject(d_t12); clReleaseMemObject(d_t13);
    clReleaseMemObject(d_t22); clReleaseMemObject(d_t23); clReleaseMemObject(d_t33);
    clReleaseMemObject(d_a11); clReleaseMemObject(d_a12); clReleaseMemObject(d_a13);
    clReleaseMemObject(d_a22); clReleaseMemObject(d_a23); clReleaseMemObject(d_a33);
    clReleaseMemObject(d_h1); clReleaseMemObject(d_h2); clReleaseMemObject(d_h3);
    clReleaseMemObject(d_TempDispX); clReleaseMemObject(d_TempDispY); clReleaseMemObject(d_TempDispZ);
    clReleaseMemObject(c_DirX); clReleaseMemObject(c_DirY); clReleaseMemObject(c_DirZ);
}

// ============================================================
//  Non-linear registration: multi-scale
// ============================================================

static void alignTwoVolumesNonLinearSeveralScales(OpenCLContext& c,
    cl_mem d_OrigAligned, cl_mem d_OrigReference,
    cl_mem& d_TotalDispX, cl_mem& d_TotalDispY, cl_mem& d_TotalDispZ,
    float* h_NLF1R, float* h_NLF1I, float* h_NLF2R, float* h_NLF2I, float* h_NLF3R, float* h_NLF3I,
    float* h_NLF4R, float* h_NLF4I, float* h_NLF5R, float* h_NLF5I, float* h_NLF6R, float* h_NLF6I,
    const float projTensors[6][6],
    const float* filterDirX, const float* filterDirY, const float* filterDirZ,
    int W, int H, int D, int coarsestScale, int numIter)
{
    int curW = (int)myround((float)W / (float)coarsestScale);
    int curH = (int)myround((float)H / (float)coarsestScale);
    int curD = (int)myround((float)D / (float)coarsestScale);
    int prevW = curW, prevH = curH, prevD = curD;

    d_TotalDispX = clCreateBuffer(c.context, CL_MEM_READ_WRITE, curW * curH * curD * sizeof(float), NULL, NULL);
    d_TotalDispY = clCreateBuffer(c.context, CL_MEM_READ_WRITE, curW * curH * curD * sizeof(float), NULL, NULL);
    d_TotalDispZ = clCreateBuffer(c.context, CL_MEM_READ_WRITE, curW * curH * curD * sizeof(float), NULL, NULL);
    setMemory(c, d_TotalDispX, 0.0f, curW * curH * curD);
    setMemory(c, d_TotalDispY, 0.0f, curW * curH * curD);
    setMemory(c, d_TotalDispZ, 0.0f, curW * curH * curD);

    for (int scale = coarsestScale; scale >= 1; scale /= 2) {
        cl_image_format fmt; fmt.image_channel_data_type = CL_FLOAT; fmt.image_channel_order = CL_INTENSITY;
        cl_mem d_OrigTex = clCreateImage3D(c.context, CL_MEM_READ_ONLY, &fmt, curW, curH, curD, 0, 0, NULL, NULL);
        cl_mem d_Aligned = clCreateBuffer(c.context, CL_MEM_READ_WRITE, curW * curH * curD * sizeof(float), NULL, NULL);
        cl_mem d_Ref     = clCreateBuffer(c.context, CL_MEM_READ_WRITE, curW * curH * curD * sizeof(float), NULL, NULL);
        cl_mem d_UpdDispX = clCreateBuffer(c.context, CL_MEM_READ_WRITE, curW * curH * curD * sizeof(float), NULL, NULL);
        cl_mem d_UpdDispY = clCreateBuffer(c.context, CL_MEM_READ_WRITE, curW * curH * curD * sizeof(float), NULL, NULL);
        cl_mem d_UpdDispZ = clCreateBuffer(c.context, CL_MEM_READ_WRITE, curW * curH * curD * sizeof(float), NULL, NULL);

        changeVolumeSize(c, d_Aligned, d_OrigAligned, W, H, D, curW, curH, curD, LINEAR);
        changeVolumeSize(c, d_Ref, d_OrigReference, W, H, D, curW, curH, curD, LINEAR);

        size_t origin[3] = {0, 0, 0};
        size_t region[3] = {(size_t)curW, (size_t)curH, (size_t)curD};
        clEnqueueCopyBufferToImage(c.queue, d_Aligned, d_OrigTex, 0, origin, region, 0, NULL, NULL);

        // Apply accumulated displacement from previous scales
        if (scale != coarsestScale) {
            WorkSizes3D wsI = calcWorkSizes16x16(c, curW, curH, curD);
            int vol = 0;
            clSetKernelArg(c.interpolateVolumeLinearNonLinearKernel, 0, sizeof(cl_mem), &d_Aligned);
            clSetKernelArg(c.interpolateVolumeLinearNonLinearKernel, 1, sizeof(cl_mem), &d_OrigTex);
            clSetKernelArg(c.interpolateVolumeLinearNonLinearKernel, 2, sizeof(cl_mem), &d_TotalDispX);
            clSetKernelArg(c.interpolateVolumeLinearNonLinearKernel, 3, sizeof(cl_mem), &d_TotalDispY);
            clSetKernelArg(c.interpolateVolumeLinearNonLinearKernel, 4, sizeof(cl_mem), &d_TotalDispZ);
            clSetKernelArg(c.interpolateVolumeLinearNonLinearKernel, 5, sizeof(int), &curW);
            clSetKernelArg(c.interpolateVolumeLinearNonLinearKernel, 6, sizeof(int), &curH);
            clSetKernelArg(c.interpolateVolumeLinearNonLinearKernel, 7, sizeof(int), &curD);
            clSetKernelArg(c.interpolateVolumeLinearNonLinearKernel, 8, sizeof(int), &vol);
            clEnqueueNDRangeKernel(c.queue, c.interpolateVolumeLinearNonLinearKernel, 3, NULL, wsI.global, wsI.local, 0, NULL, NULL);
            clFinish(c.queue);
            clEnqueueCopyBufferToImage(c.queue, d_Aligned, d_OrigTex, 0, origin, region, 0, NULL, NULL);
        }

        int iters = (scale == 1) ? (int)ceil((float)numIter / 2.0f) : numIter;

        alignTwoVolumesNonLinear(c, d_Aligned, d_Ref, d_OrigTex,
            d_UpdDispX, d_UpdDispY, d_UpdDispZ,
            h_NLF1R, h_NLF1I, h_NLF2R, h_NLF2I, h_NLF3R, h_NLF3I,
            h_NLF4R, h_NLF4I, h_NLF5R, h_NLF5I, h_NLF6R, h_NLF6I,
            projTensors, filterDirX, filterDirY, filterDirZ,
            curW, curH, curD, iters);

        // Accumulate
        addVolumes(c, d_TotalDispX, d_UpdDispX, curW, curH, curD);
        addVolumes(c, d_TotalDispY, d_UpdDispY, curW, curH, curD);
        addVolumes(c, d_TotalDispZ, d_UpdDispZ, curW, curH, curD);

        clReleaseMemObject(d_OrigTex); clReleaseMemObject(d_Aligned); clReleaseMemObject(d_Ref);
        clReleaseMemObject(d_UpdDispX); clReleaseMemObject(d_UpdDispY); clReleaseMemObject(d_UpdDispZ);

        if (scale != 1) {
            prevW = curW; prevH = curH; prevD = curD;
            curW = (int)myround((float)W / ((float)scale / 2.0f));
            curH = (int)myround((float)H / ((float)scale / 2.0f));
            curD = (int)myround((float)D / ((float)scale / 2.0f));

            changeVolumeSizeInPlace(c, d_TotalDispX, prevW, prevH, prevD, curW, curH, curD, LINEAR);
            changeVolumeSizeInPlace(c, d_TotalDispY, prevW, prevH, prevD, curW, curH, curD, LINEAR);
            changeVolumeSizeInPlace(c, d_TotalDispZ, prevW, prevH, prevD, curW, curH, curD, LINEAR);

            multiplyVolume(c, d_TotalDispX, 2.0f, curW, curH, curD);
            multiplyVolume(c, d_TotalDispY, 2.0f, curW, curH, curD);
            multiplyVolume(c, d_TotalDispZ, 2.0f, curW, curH, curD);
        } else {
            // Final: transform original volume with total displacement
            transformVolumesNonLinear(c, d_OrigAligned, d_TotalDispX, d_TotalDispY, d_TotalDispZ, W, H, D, 1, LINEAR);
        }
    }
}

} // anonymous namespace

// ============================================================
//  Public API: registerT1MNI
// ============================================================

T1MNIResult registerT1MNI(
    const float* t1Data,       VolumeDims t1Dims,  VoxelSize t1Vox,
    const float* mniData,      VolumeDims mniDims,  VoxelSize mniVox,
    const float* mniBrainData,
    const float* mniMaskData,
    const QuadratureFilters& filters,
    int linearIterations,
    int nonlinearIterations,
    int coarsestScale,
    int mmZCut,
    bool verbose)
{
    OpenCLContext& c = ctx();

    T1MNIResult result;
    int mniSize = mniDims.size();
    int t1Size  = t1Dims.size();

    result.alignedLinear.resize(mniSize, 0.0f);
    result.alignedNonLinear.resize(mniSize, 0.0f);
    result.skullstripped.resize(mniSize, 0.0f);
    result.interpolated.resize(mniSize, 0.0f);
    result.params.fill(0.0f);
    result.dispX.resize(mniSize, 0.0f);
    result.dispY.resize(mniSize, 0.0f);
    result.dispZ.resize(mniSize, 0.0f);

    // Upload volumes
    cl_mem d_Input = clCreateBuffer(c.context, CL_MEM_READ_WRITE, t1Size * sizeof(float), NULL, NULL);
    cl_mem d_Ref   = clCreateBuffer(c.context, CL_MEM_READ_WRITE, mniSize * sizeof(float), NULL, NULL);
    cl_mem d_InputMNI = clCreateBuffer(c.context, CL_MEM_READ_WRITE, mniSize * sizeof(float), NULL, NULL);

    clEnqueueWriteBuffer(c.queue, d_Input, CL_TRUE, 0, t1Size * sizeof(float), t1Data, 0, NULL, NULL);
    clEnqueueWriteBuffer(c.queue, d_Ref,   CL_TRUE, 0, mniSize * sizeof(float), mniBrainData, 0, NULL, NULL);

    float h_MatchParams[12] = {0};
    float h_RegParams[12] = {0};

    if (linearIterations > 0) {
        // Resample T1 to MNI space
        changeVolumesResolutionAndSize(c, d_InputMNI, d_Input,
            t1Dims.W, t1Dims.H, t1Dims.D,
            mniDims.W, mniDims.H, mniDims.D,
            t1Vox.x, t1Vox.y, t1Vox.z,
            mniVox.x, mniVox.y, mniVox.z,
            mmZCut, LINEAR);

        // Center-of-mass alignment
        matchVolumeMasses(c, d_InputMNI, d_Ref, h_MatchParams, mniDims.W, mniDims.H, mniDims.D);

        // Save interpolated volume
        clEnqueueReadBuffer(c.queue, d_InputMNI, CL_TRUE, 0, mniSize * sizeof(float), result.interpolated.data(), 0, NULL, NULL);

        // Multi-scale linear registration
        alignTwoVolumesLinearSeveralScales(c, h_RegParams, d_InputMNI, d_Ref,
            mniDims.W, mniDims.H, mniDims.D, coarsestScale, linearIterations,
            const_cast<float*>(filters.linearReal[0].data()),
            const_cast<float*>(filters.linearImag[0].data()),
            const_cast<float*>(filters.linearReal[1].data()),
            const_cast<float*>(filters.linearImag[1].data()),
            const_cast<float*>(filters.linearReal[2].data()),
            const_cast<float*>(filters.linearImag[2].data()));

        // Read back linearly aligned volume
        clEnqueueReadBuffer(c.queue, d_InputMNI, CL_TRUE, 0, mniSize * sizeof(float), result.alignedLinear.data(), 0, NULL, NULL);
    }

    // Combine match + registration parameters
    addAffineRegistrationParameters(h_RegParams, h_MatchParams);
    for (int i = 0; i < 12; i++) result.params[i] = h_RegParams[i];

    // Non-linear registration
    if (nonlinearIterations > 0) {
        cl_mem d_TotalDispX = NULL, d_TotalDispY = NULL, d_TotalDispZ = NULL;

        alignTwoVolumesNonLinearSeveralScales(c, d_InputMNI, d_Ref,
            d_TotalDispX, d_TotalDispY, d_TotalDispZ,
            const_cast<float*>(filters.nonlinearReal[0].data()),
            const_cast<float*>(filters.nonlinearImag[0].data()),
            const_cast<float*>(filters.nonlinearReal[1].data()),
            const_cast<float*>(filters.nonlinearImag[1].data()),
            const_cast<float*>(filters.nonlinearReal[2].data()),
            const_cast<float*>(filters.nonlinearImag[2].data()),
            const_cast<float*>(filters.nonlinearReal[3].data()),
            const_cast<float*>(filters.nonlinearImag[3].data()),
            const_cast<float*>(filters.nonlinearReal[4].data()),
            const_cast<float*>(filters.nonlinearImag[4].data()),
            const_cast<float*>(filters.nonlinearReal[5].data()),
            const_cast<float*>(filters.nonlinearImag[5].data()),
            filters.projectionTensors,
            filters.filterDirectionsX,
            filters.filterDirectionsY,
            filters.filterDirectionsZ,
            mniDims.W, mniDims.H, mniDims.D, coarsestScale, nonlinearIterations);

        // Create combined displacement field if we also did linear
        if (linearIterations > 0) {
            createCombinedDisplacementField(c, h_RegParams, d_TotalDispX, d_TotalDispY, d_TotalDispZ,
                mniDims.W, mniDims.H, mniDims.D);

            // Re-interpolate T1 cleanly
            changeVolumesResolutionAndSize(c, d_InputMNI, d_Input,
                t1Dims.W, t1Dims.H, t1Dims.D,
                mniDims.W, mniDims.H, mniDims.D,
                t1Vox.x, t1Vox.y, t1Vox.z,
                mniVox.x, mniVox.y, mniVox.z,
                mmZCut, LINEAR);

            transformVolumesNonLinear(c, d_InputMNI, d_TotalDispX, d_TotalDispY, d_TotalDispZ,
                mniDims.W, mniDims.H, mniDims.D, 1, LINEAR);
        }

        // Read back results
        clEnqueueReadBuffer(c.queue, d_InputMNI, CL_TRUE, 0, mniSize * sizeof(float), result.alignedNonLinear.data(), 0, NULL, NULL);
        clEnqueueReadBuffer(c.queue, d_TotalDispX, CL_TRUE, 0, mniSize * sizeof(float), result.dispX.data(), 0, NULL, NULL);
        clEnqueueReadBuffer(c.queue, d_TotalDispY, CL_TRUE, 0, mniSize * sizeof(float), result.dispY.data(), 0, NULL, NULL);
        clEnqueueReadBuffer(c.queue, d_TotalDispZ, CL_TRUE, 0, mniSize * sizeof(float), result.dispZ.data(), 0, NULL, NULL);

        clReleaseMemObject(d_TotalDispX);
        clReleaseMemObject(d_TotalDispY);
        clReleaseMemObject(d_TotalDispZ);
    }

    // Skullstripping
    if (mniMaskData) {
        cl_mem d_Mask = clCreateBuffer(c.context, CL_MEM_READ_WRITE, mniSize * sizeof(float), NULL, NULL);
        clEnqueueWriteBuffer(c.queue, d_Mask, CL_TRUE, 0, mniSize * sizeof(float), mniMaskData, 0, NULL, NULL);
        multiplyVolumes(c, d_InputMNI, d_Mask, mniDims.W, mniDims.H, mniDims.D);
        clEnqueueReadBuffer(c.queue, d_InputMNI, CL_TRUE, 0, mniSize * sizeof(float), result.skullstripped.data(), 0, NULL, NULL);
        clReleaseMemObject(d_Mask);
    }

    // Cleanup
    clReleaseMemObject(d_Input);
    clReleaseMemObject(d_Ref);
    clReleaseMemObject(d_InputMNI);

    if (verbose) {
        printf("[OpenCL] Registration complete. Parameters: [%.3f %.3f %.3f | %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f]\n",
               result.params[0], result.params[1], result.params[2],
               result.params[3], result.params[4], result.params[5],
               result.params[6], result.params[7], result.params[8],
               result.params[9], result.params[10], result.params[11]);
    }

    return result;
}

} // namespace opencl_reg
