/* opencl_backend.cpp — Adapts BROCCOLI_LIB (OpenCL) to the broccolini C vtable.
 *
 * Requires BROCCOLI_DIR env var pointing to src/opencl/ so the OpenCL
 * kernels can be found at $BROCCOLI_DIR/code/Kernels/*.cpp (we symlink
 * kernels/ → code/Kernels/ for this).
 *
 * Alternatively, the kernel .cpp files can sit at opencl/kernels/ and we
 * set BROCCOLI_DIR so the library finds them.
 */

#include "opencl_backend.h"
#include "broccoli_lib.h"
#include "broccoli_constants.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>

/* ---------- helpers ---------- */

/* Convert BROCCOLI's 12-parameter representation to 4×4 row-major affine.
 * params = [tx,ty,tz, r11-1,r12,r13, r21,r22-1,r23, r31,r32,r33-1] */
static void params12_to_mat44(const float p[12], float m[16])
{
    m[ 0] = p[3] + 1.0f;  m[ 1] = p[4];         m[ 2] = p[5];         m[ 3] = p[0];
    m[ 4] = p[6];         m[ 5] = p[7] + 1.0f;  m[ 6] = p[8];         m[ 7] = p[1];
    m[ 8] = p[9];         m[ 9] = p[10];        m[10] = p[11] + 1.0f; m[11] = p[2];
    m[12] = 0.0f;         m[13] = 0.0f;         m[14] = 0.0f;         m[15] = 1.0f;
}

/* ---------- backend implementation ---------- */

struct opencl_priv {
    /* We store the kernel directory path so we can set BROCCOLI_DIR */
    std::string kernel_dir;
};

static broc_result opencl_register_volumes(
    broc_backend *self,
    const float *input, broc_dims in_dims, broc_voxsize in_vox,
    const float *ref,   broc_dims ref_dims, broc_voxsize ref_vox,
    const float *ref_mask,
    const broc_filters *filters,
    const broc_reg_params *params)
{
    broc_result result;
    memset(&result, 0, sizeof(result));

    int64_t ref_nvox = (int64_t)ref_dims.W * ref_dims.H * ref_dims.D;

    /* --- Set BROCCOLI_DIR so the library finds kernel source files ---
     * The library looks for kernels at $BROCCOLI_DIR/code/Kernels/
     * We create a symlink: opencl/code/Kernels -> opencl/kernels
     * Or the user can set BROCCOLI_DIR themselves. */
    opencl_priv *priv = (opencl_priv *)self->priv;
    if (!priv->kernel_dir.empty()) {
        setenv("BROCCOLI_DIR", priv->kernel_dir.c_str(), 1);
    }

    /* Create BROCCOLI_LIB with wrapper=BASH (reads BROCCOLI_DIR) */
    BROCCOLI_LIB broc(0, 0, BASH, params->verbose != 0);

    if (!broc.GetOpenCLInitiated()) {
        fprintf(stderr, "OpenCL initialization failed\n");
        return result;
    }

    /* --- Set input/reference volumes --- */
    broc.SetInputT1Volume((float *)input);
    broc.SetT1Width(in_dims.W);
    broc.SetT1Height(in_dims.H);
    broc.SetT1Depth(in_dims.D);
    broc.SetT1Timepoints(1);
    broc.SetT1VoxelSizeX(in_vox.x);
    broc.SetT1VoxelSizeY(in_vox.y);
    broc.SetT1VoxelSizeZ(in_vox.z);

    broc.SetInputMNIVolume((float *)ref);
    broc.SetInputMNIBrainVolume((float *)ref);
    broc.SetMNIWidth(ref_dims.W);
    broc.SetMNIHeight(ref_dims.H);
    broc.SetMNIDepth(ref_dims.D);
    broc.SetMNIVoxelSizeX(ref_vox.x);
    broc.SetMNIVoxelSizeY(ref_vox.y);
    broc.SetMNIVoxelSizeZ(ref_vox.z);

    /* Mask: use provided or derive from ref > 0 */
    float *auto_mask = NULL;
    if (ref_mask) {
        broc.SetInputMNIBrainMask((float *)ref_mask);
    } else {
        auto_mask = (float *)malloc(ref_nvox * sizeof(float));
        if (auto_mask) {
            for (int64_t i = 0; i < ref_nvox; i++)
                auto_mask[i] = (ref[i] > 0.0f) ? 1.0f : 0.0f;
            broc.SetInputMNIBrainMask(auto_mask);
        }
    }

    /* --- Registration parameters --- */
    broc.SetInterpolationMode(params->interp);
    broc.SetNumberOfIterationsForLinearImageRegistration(params->linear_iterations);
    broc.SetNumberOfIterationsForNonLinearImageRegistration(params->nonlinear_iterations);
    broc.SetCoarsestScaleT1MNI(params->coarsest_scale);
    broc.SetMMT1ZCUT(params->zcut_mm);
    broc.SetDoSkullstrip(false);
    broc.SetSaveDisplacementField(params->nonlinear_iterations > 0);

    /* --- Filters --- */
    broc.SetImageRegistrationFilterSize(BROC_FILTER_SIZE);

    broc.SetLinearImageRegistrationFilters(
        (float *)filters->linear_real[0], (float *)filters->linear_imag[0],
        (float *)filters->linear_real[1], (float *)filters->linear_imag[1],
        (float *)filters->linear_real[2], (float *)filters->linear_imag[2]);

    broc.SetNonLinearImageRegistrationFilters(
        (float *)filters->nonlinear_real[0], (float *)filters->nonlinear_imag[0],
        (float *)filters->nonlinear_real[1], (float *)filters->nonlinear_imag[1],
        (float *)filters->nonlinear_real[2], (float *)filters->nonlinear_imag[2],
        (float *)filters->nonlinear_real[3], (float *)filters->nonlinear_imag[3],
        (float *)filters->nonlinear_real[4], (float *)filters->nonlinear_imag[4],
        (float *)filters->nonlinear_real[5], (float *)filters->nonlinear_imag[5]);

    /* Projection tensors (6 symmetric 3×3 stored as 6 unique elements) */
    broc.SetProjectionTensorMatrixFirstFilter(
        filters->projection_tensors[0][0], filters->projection_tensors[0][1],
        filters->projection_tensors[0][2], filters->projection_tensors[0][3],
        filters->projection_tensors[0][4], filters->projection_tensors[0][5]);
    broc.SetProjectionTensorMatrixSecondFilter(
        filters->projection_tensors[1][0], filters->projection_tensors[1][1],
        filters->projection_tensors[1][2], filters->projection_tensors[1][3],
        filters->projection_tensors[1][4], filters->projection_tensors[1][5]);
    broc.SetProjectionTensorMatrixThirdFilter(
        filters->projection_tensors[2][0], filters->projection_tensors[2][1],
        filters->projection_tensors[2][2], filters->projection_tensors[2][3],
        filters->projection_tensors[2][4], filters->projection_tensors[2][5]);
    broc.SetProjectionTensorMatrixFourthFilter(
        filters->projection_tensors[3][0], filters->projection_tensors[3][1],
        filters->projection_tensors[3][2], filters->projection_tensors[3][3],
        filters->projection_tensors[3][4], filters->projection_tensors[3][5]);
    broc.SetProjectionTensorMatrixFifthFilter(
        filters->projection_tensors[4][0], filters->projection_tensors[4][1],
        filters->projection_tensors[4][2], filters->projection_tensors[4][3],
        filters->projection_tensors[4][4], filters->projection_tensors[4][5]);
    broc.SetProjectionTensorMatrixSixthFilter(
        filters->projection_tensors[5][0], filters->projection_tensors[5][1],
        filters->projection_tensors[5][2], filters->projection_tensors[5][3],
        filters->projection_tensors[5][4], filters->projection_tensors[5][5]);

    broc.SetFilterDirections(
        (float *)filters->filter_directions_x,
        (float *)filters->filter_directions_y,
        (float *)filters->filter_directions_z);

    /* --- Output buffers --- */
    float *h_aligned_linear = (float *)calloc(ref_nvox, sizeof(float));
    float *h_aligned_nonlinear = (float *)calloc(ref_nvox, sizeof(float));
    float *h_interpolated = (float *)calloc(ref_nvox, sizeof(float));
    float *h_skullstripped = (float *)calloc(ref_nvox, sizeof(float));
    float h_reg_params[12] = {0};

    /* Dummy outputs required by the wrapper */
    float *h_phase_diff = (float *)calloc(ref_nvox, sizeof(float));
    float *h_phase_cert = (float *)calloc(ref_nvox, sizeof(float));
    float *h_phase_grad = (float *)calloc(ref_nvox, sizeof(float));
    float *h_slice_sums = (float *)calloc(ref_dims.D, sizeof(float));
    float h_top_slice[1] = {0};
    float h_a_matrix[144] = {0};
    float h_h_vector[12] = {0};

    broc.SetOutputAlignedT1VolumeLinear(h_aligned_linear);
    broc.SetOutputAlignedT1VolumeNonLinear(h_aligned_nonlinear);
    broc.SetOutputInterpolatedT1Volume(h_interpolated);
    broc.SetOutputSkullstrippedT1Volume(h_skullstripped);
    broc.SetOutputT1MNIRegistrationParameters(h_reg_params);
    broc.SetOutputPhaseDifferences(h_phase_diff);
    broc.SetOutputPhaseCertainties(h_phase_cert);
    broc.SetOutputPhaseGradients(h_phase_grad);
    broc.SetOutputSliceSums(h_slice_sums);
    broc.SetOutputTopSlice(h_top_slice);
    broc.SetOutputAMatrix(h_a_matrix);
    broc.SetOutputHVector(h_h_vector);

    /* Displacement field outputs */
    float *h_disp_x = NULL, *h_disp_y = NULL, *h_disp_z = NULL;
    if (params->nonlinear_iterations > 0) {
        h_disp_x = (float *)calloc(ref_nvox, sizeof(float));
        h_disp_y = (float *)calloc(ref_nvox, sizeof(float));
        h_disp_z = (float *)calloc(ref_nvox, sizeof(float));
        broc.SetOutputDisplacementField(h_disp_x, h_disp_y, h_disp_z);
    }

    /* --- Run registration --- */
    broc.PerformRegistrationTwoVolumesWrapper();

    /* --- Pack results --- */
    /* Use nonlinear output if nonlinear was run, otherwise linear */
    if (params->nonlinear_iterations > 0) {
        result.aligned = h_aligned_nonlinear;
        free(h_aligned_linear);
    } else {
        result.aligned = h_aligned_linear;
        free(h_aligned_nonlinear);
    }

    result.out_W = ref_dims.W;
    result.out_H = ref_dims.H;
    result.out_D = ref_dims.D;

    params12_to_mat44(h_reg_params, result.affine);

    result.disp_x = h_disp_x;
    result.disp_y = h_disp_y;
    result.disp_z = h_disp_z;

    /* Cleanup temporaries */
    free(h_interpolated);
    free(h_skullstripped);
    free(h_phase_diff);
    free(h_phase_cert);
    free(h_phase_grad);
    free(h_slice_sums);
    free(auto_mask);

    return result;
}

static void opencl_destroy(broc_backend *self)
{
    if (self) {
        delete (opencl_priv *)self->priv;
        free(self);
    }
}

extern "C" broc_backend *broc_opencl_create_backend(void)
{
    broc_backend *b = (broc_backend *)calloc(1, sizeof(broc_backend));
    if (!b) return NULL;

    b->register_volumes = opencl_register_volumes;
    b->destroy = opencl_destroy;
    b->name = "OpenCL (BROCCOLI)";

    opencl_priv *priv = new opencl_priv();

    /* Resolve kernel directory: look for kernels/ relative to executable.
     * The user needs: src/opencl/code/Kernels/ containing the kernel .cpp files.
     * We create this path structure with a symlink in the Makefile. */
    const char *env = getenv("BROCCOLI_DIR");
    if (env) {
        priv->kernel_dir = env;
    }
    /* If BROCCOLI_DIR not set, the user must set it or the library will fail
     * at OpenCLInitiate() with an error message. */

    b->priv = priv;
    return b;
}
