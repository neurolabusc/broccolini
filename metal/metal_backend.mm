/* metal_backend.mm — Adapts metal_registration.mm to the C backend vtable */

#include "metal_backend.h"
#include "metal_registration.h"
#include <cstring>

/* ========== Convert broc_filters → metal_reg::QuadratureFilters ========== */

static metal_reg::QuadratureFilters convert_filters(const broc_filters *f)
{
    metal_reg::QuadratureFilters qf;

    for (int i = 0; i < BROC_NUM_LINEAR_FILTERS; i++) {
        qf.linearReal[i].assign(f->linear_real[i],
                                f->linear_real[i] + BROC_FILTER_ELEMENTS);
        qf.linearImag[i].assign(f->linear_imag[i],
                                f->linear_imag[i] + BROC_FILTER_ELEMENTS);
    }

    for (int i = 0; i < BROC_NUM_NONLINEAR_FILTERS; i++) {
        qf.nonlinearReal[i].assign(f->nonlinear_real[i],
                                   f->nonlinear_real[i] + BROC_FILTER_ELEMENTS);
        qf.nonlinearImag[i].assign(f->nonlinear_imag[i],
                                   f->nonlinear_imag[i] + BROC_FILTER_ELEMENTS);
    }

    for (int i = 0; i < BROC_NUM_NONLINEAR_FILTERS; i++)
        for (int j = 0; j < 6; j++)
            qf.projectionTensors[i][j] = f->projection_tensors[i][j];

    memcpy(qf.filterDirectionsX, f->filter_directions_x, 6 * sizeof(float));
    memcpy(qf.filterDirectionsY, f->filter_directions_y, 6 * sizeof(float));
    memcpy(qf.filterDirectionsZ, f->filter_directions_z, 6 * sizeof(float));

    return qf;
}

/* ========== Convert 12 BROCCOLI params → 4x4 row-major affine ========== */

static void params12_to_mat44(const float *p, float mat[16])
{
    /* BROCCOLI 12-param layout:
     *   p[0..2] = tx, ty, tz
     *   p[3..11] = rotation/scale matrix - identity
     *
     *   | p3+1  p4    p5    tx |
     *   | p6    p7+1  p8    ty |
     *   | p9    p10   p11+1 tz |
     *   | 0     0     0     1  |
     */
    mat[0]  = p[3] + 1.0f;  mat[1]  = p[4];         mat[2]  = p[5];         mat[3]  = p[0];
    mat[4]  = p[6];         mat[5]  = p[7] + 1.0f;  mat[6]  = p[8];         mat[7]  = p[1];
    mat[8]  = p[9];         mat[9]  = p[10];        mat[10] = p[11] + 1.0f; mat[11] = p[2];
    mat[12] = 0.0f;         mat[13] = 0.0f;         mat[14] = 0.0f;         mat[15] = 1.0f;
}

/* ========== Backend register_volumes implementation ========== */

static broc_result metal_register_volumes(
    broc_backend       *self,
    const float        *input,    broc_dims in_dims,  broc_voxsize in_vox,
    const float        *ref,      broc_dims ref_dims, broc_voxsize ref_vox,
    const float        *ref_mask,
    const broc_filters *filters,
    const broc_reg_params *params)
{
    (void)self;
    broc_result result;
    memset(&result, 0, sizeof(result));

    metal_reg::QuadratureFilters qf = convert_filters(filters);
    metal_reg::VolumeDims idims = { in_dims.W, in_dims.H, in_dims.D };
    metal_reg::VoxelSize  ivox  = { in_vox.x, in_vox.y, in_vox.z };
    metal_reg::VolumeDims rdims = { ref_dims.W, ref_dims.H, ref_dims.D };
    metal_reg::VoxelSize  rvox  = { ref_vox.x, ref_vox.y, ref_vox.z };

    int vol_size = ref_dims.W * ref_dims.H * ref_dims.D;

    /* Use registerT1MNI as the universal path (superset of EPI-T1).
     * When nonlinear_iterations == 0, it does linear-only registration.
     *
     * ref_mask: if NULL, derive from ref > 0 (same as Python default).
     * We pass ref as both mniData and mniBrainData; mask is ref_mask or derived. */

    /* Derive mask if not provided */
    float *derived_mask = NULL;
    if (!ref_mask) {
        derived_mask = (float *)malloc(vol_size * sizeof(float));
        if (derived_mask) {
            for (int i = 0; i < vol_size; i++)
                derived_mask[i] = (ref[i] > 0.0f) ? 1.0f : 0.0f;
        }
        ref_mask = derived_mask;
    }

    metal_reg::T1MNIResult mr = metal_reg::registerT1MNI(
        input, idims, ivox,
        ref, rdims, rvox,
        ref,       /* mniBrainData = ref (skull-stripped assumption) */
        ref_mask,  /* mniMaskData */
        qf,
        params->linear_iterations,
        params->nonlinear_iterations,
        params->coarsest_scale,
        params->zcut_mm,
        params->verbose);

    free(derived_mask);

    /* Pick the best available output */
    const std::vector<float> &out_vol =
        (params->nonlinear_iterations > 0) ? mr.alignedNonLinear : mr.alignedLinear;

    result.out_W = ref_dims.W;
    result.out_H = ref_dims.H;
    result.out_D = ref_dims.D;

    /* Copy aligned volume */
    if ((int)out_vol.size() != vol_size) {
        /* Size mismatch — return failure (aligned == NULL) */
        return result;
    }
    result.aligned = (float *)malloc(vol_size * sizeof(float));
    if (result.aligned)
        memcpy(result.aligned, out_vol.data(), vol_size * sizeof(float));

    /* Convert 12 params to 4x4 matrix */
    params12_to_mat44(mr.params.data(), result.affine);

    /* Copy displacement fields if nonlinear */
    if (params->nonlinear_iterations > 0 && (int)mr.dispX.size() == vol_size) {
        result.disp_x = (float *)malloc(vol_size * sizeof(float));
        result.disp_y = (float *)malloc(vol_size * sizeof(float));
        result.disp_z = (float *)malloc(vol_size * sizeof(float));
        if (!result.disp_x || !result.disp_y || !result.disp_z) {
            free(result.disp_x); free(result.disp_y); free(result.disp_z);
            result.disp_x = result.disp_y = result.disp_z = NULL;
        } else {
            memcpy(result.disp_x, mr.dispX.data(), vol_size * sizeof(float));
            memcpy(result.disp_y, mr.dispY.data(), vol_size * sizeof(float));
            memcpy(result.disp_z, mr.dispZ.data(), vol_size * sizeof(float));
        }
    }

    return result;
}

/* ========== Backend destroy ========== */

static void metal_destroy(broc_backend *self)
{
    free(self);
}

/* ========== Factory function ========== */

extern "C" broc_backend *broc_metal_create_backend(void)
{
    broc_backend *b = (broc_backend *)calloc(1, sizeof(broc_backend));
    if (!b) return NULL;

    b->register_volumes = metal_register_volumes;
    b->destroy          = metal_destroy;
    b->name             = "Metal";
    b->priv             = NULL;

    return b;
}
