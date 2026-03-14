/* registration.c — Shared utility functions for BROCCOLI registration
 *
 * Filter loading, volume packing/unpacking, datatype conversion,
 * matrix I/O, and parameter defaults.
 */

#include "registration.h"
#include "nifti_io.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

/* ========== Parameter defaults ========== */

void broc_reg_params_defaults(broc_reg_params *p)
{
    p->dof                  = 12;
    p->linear_iterations    = 10;
    p->nonlinear_iterations = 0;
    p->coarsest_scale       = 4;
    p->zcut_mm              = 0;
    p->interp               = BROC_INTERP_LINEAR;
    p->verbose              = 0;
}

/* ========== Filter loading ========== */

/* Read exactly `count` floats from a binary file. Returns 0 on success. */
static int read_bin_floats(const char *path, float *buf, int count)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "Error: cannot open filter file: %s\n", path);
        return -1;
    }
    size_t n = fread(buf, sizeof(float), count, fp);
    fclose(fp);
    if ((int)n != count) {
        fprintf(stderr, "Error: expected %d floats from %s, got %d\n",
                count, path, (int)n);
        return -1;
    }
    return 0;
}

int broc_load_filters(const char *filter_dir, broc_filters *out)
{
    char path[1024];
    int i;

    /* Linear registration filters (3 filters, real + imag) */
    for (i = 0; i < BROC_NUM_LINEAR_FILTERS; i++) {
        snprintf(path, sizeof(path), "%s/filter%d_real_linear_registration.bin",
                 filter_dir, i + 1);
        if (read_bin_floats(path, out->linear_real[i], BROC_FILTER_ELEMENTS))
            return -1;

        snprintf(path, sizeof(path), "%s/filter%d_imag_linear_registration.bin",
                 filter_dir, i + 1);
        if (read_bin_floats(path, out->linear_imag[i], BROC_FILTER_ELEMENTS))
            return -1;
    }

    /* Nonlinear registration filters (6 filters, real + imag) */
    for (i = 0; i < BROC_NUM_NONLINEAR_FILTERS; i++) {
        snprintf(path, sizeof(path), "%s/filter%d_real_nonlinear_registration.bin",
                 filter_dir, i + 1);
        if (read_bin_floats(path, out->nonlinear_real[i], BROC_FILTER_ELEMENTS))
            return -1;

        snprintf(path, sizeof(path), "%s/filter%d_imag_nonlinear_registration.bin",
                 filter_dir, i + 1);
        if (read_bin_floats(path, out->nonlinear_imag[i], BROC_FILTER_ELEMENTS))
            return -1;
    }

    /* Projection tensors (6 tensors, 6 floats each) */
    for (i = 0; i < BROC_NUM_NONLINEAR_FILTERS; i++) {
        snprintf(path, sizeof(path), "%s/projection_tensor%d.bin",
                 filter_dir, i + 1);
        if (read_bin_floats(path, out->projection_tensors[i], 6))
            return -1;
    }

    /* Filter directions (3 files, 6 floats each) */
    snprintf(path, sizeof(path), "%s/filter_directions_x.bin", filter_dir);
    if (read_bin_floats(path, out->filter_directions_x, BROC_NUM_NONLINEAR_FILTERS))
        return -1;

    snprintf(path, sizeof(path), "%s/filter_directions_y.bin", filter_dir);
    if (read_bin_floats(path, out->filter_directions_y, BROC_NUM_NONLINEAR_FILTERS))
        return -1;

    snprintf(path, sizeof(path), "%s/filter_directions_z.bin", filter_dir);
    if (read_bin_floats(path, out->filter_directions_z, BROC_NUM_NONLINEAR_FILTERS))
        return -1;

    return 0;
}

/* ========== Volume packing ========== */

/*
 * Pack NIfTI (i,j,k) → BROCCOLI layout.
 *
 * Python equivalent (on C-order array from nibabel):
 *   packed = np.flipud(data).transpose(2, 0, 1)
 *
 * NIfTI raw data is stored in Fortran order: x(=i) varies fastest.
 *   nifti_data[i + j*ni + k*ni*nj]  for voxel (i,j,k)
 *
 * BROCCOLI packed layout: (D=nk, H=ni, W=nj)
 *   packed[k * (ni * nj) + (ni-1-i) * nj + j]
 */
float *broc_pack_volume(const float *nifti_data, int ni, int nj, int nk,
                        broc_dims *out_dims)
{
    int64_t nvox = (int64_t)ni * nj * nk;
    float *packed = (float *)malloc(nvox * sizeof(float));
    if (!packed) return NULL;

    for (int i = 0; i < ni; i++) {
        int fi = ni - 1 - i; /* flipud */
        for (int j = 0; j < nj; j++) {
            for (int k = 0; k < nk; k++) {
                /* Source: NIfTI Fortran order (x=i fastest) */
                int64_t src = (int64_t)i + (int64_t)j * ni + (int64_t)k * ni * nj;
                /* Dest: packed[k][fi][j] in (nk, ni, nj) */
                int64_t dst = (int64_t)k * (ni * nj) + fi * nj + j;
                packed[dst] = nifti_data[src];
            }
        }
    }

    if (out_dims) {
        out_dims->W = nj;
        out_dims->H = ni;
        out_dims->D = nk;
    }
    return packed;
}

/*
 * Unpack BROCCOLI layout → NIfTI Fortran-order (i,j,k).
 *
 * Inverse of broc_pack_volume.
 * packed[k][fi][j] where fi = ni-1-i → nifti_data[i + j*ni + k*ni*nj]
 */
float *broc_unpack_volume(const float *packed_data, broc_dims dims,
                          int ni, int nj, int nk)
{
    int64_t nvox = (int64_t)ni * nj * nk;
    float *data = (float *)malloc(nvox * sizeof(float));
    if (!data) return NULL;

    /* dims: W=nj, H=ni, D=nk */
    for (int k = 0; k < nk; k++) {
        for (int fi = 0; fi < ni; fi++) {
            int i = ni - 1 - fi; /* reverse flipud */
            for (int j = 0; j < nj; j++) {
                int64_t src = (int64_t)k * (ni * nj) + fi * nj + j;
                /* NIfTI Fortran order: x=i fastest */
                int64_t dst = (int64_t)i + (int64_t)j * ni + (int64_t)k * ni * nj;
                data[dst] = packed_data[src];
            }
        }
    }
    return data;
}

/* ========== Voxel size conversion ========== */

broc_voxsize broc_voxsize_from_nifti(double pixdim1, double pixdim2, double pixdim3)
{
    broc_voxsize v;
    /* BROCCOLI: x = Width (j-axis) = pixdim[2],
     *           y = Height (i-axis) = pixdim[1],
     *           z = Depth (k-axis) = pixdim[3] */
    v.x = (float)fabs(pixdim2);
    v.y = (float)fabs(pixdim1);
    v.z = (float)fabs(pixdim3);
    return v;
}

/* ========== Result cleanup ========== */

void broc_result_free(broc_result *r)
{
    if (!r) return;
    free(r->aligned);  r->aligned = NULL;
    free(r->disp_x);   r->disp_x = NULL;
    free(r->disp_y);   r->disp_y = NULL;
    free(r->disp_z);   r->disp_z = NULL;
}

/* ========== NIfTI data type conversion ========== */

float *broc_nifti_to_float(const void *data, int datatype, int64_t nvox,
                           double scl_slope, double scl_inter)
{
    float *out = (float *)malloc(nvox * sizeof(float));
    if (!out) return NULL;

    /* If slope is 0 or NaN, treat as unset (slope=1, inter=0) */
    if (scl_slope == 0.0 || isnan(scl_slope) || isnan(scl_inter)) {
        scl_slope = 1.0;
        scl_inter = 0.0;
    }

    int need_scale = (scl_slope != 1.0 || scl_inter != 0.0);

    switch (datatype) {
    case DT_FLOAT32:
        memcpy(out, data, nvox * sizeof(float));
        break;
    case DT_FLOAT64: {
        const double *src = (const double *)data;
        for (int64_t i = 0; i < nvox; i++) out[i] = (float)src[i];
        break;
    }
    case DT_INT16: {
        const int16_t *src = (const int16_t *)data;
        for (int64_t i = 0; i < nvox; i++) out[i] = (float)src[i];
        break;
    }
    case DT_UINT16: {
        const uint16_t *src = (const uint16_t *)data;
        for (int64_t i = 0; i < nvox; i++) out[i] = (float)src[i];
        break;
    }
    case DT_INT32: {
        const int32_t *src = (const int32_t *)data;
        for (int64_t i = 0; i < nvox; i++) out[i] = (float)src[i];
        break;
    }
    case DT_UINT32: {
        const uint32_t *src = (const uint32_t *)data;
        for (int64_t i = 0; i < nvox; i++) out[i] = (float)src[i];
        break;
    }
    case DT_UINT8: {
        const uint8_t *src = (const uint8_t *)data;
        for (int64_t i = 0; i < nvox; i++) out[i] = (float)src[i];
        break;
    }
    case DT_INT8: {
        const int8_t *src = (const int8_t *)data;
        for (int64_t i = 0; i < nvox; i++) out[i] = (float)src[i];
        break;
    }
    case DT_INT64: {
        const int64_t *src = (const int64_t *)data;
        for (int64_t i = 0; i < nvox; i++) out[i] = (float)src[i];
        break;
    }
    case DT_UINT64: {
        const uint64_t *src = (const uint64_t *)data;
        for (int64_t i = 0; i < nvox; i++) out[i] = (float)src[i];
        break;
    }
    default:
        fprintf(stderr, "Error: unsupported NIfTI datatype %d\n", datatype);
        free(out);
        return NULL;
    }

    if (need_scale) {
        float s = (float)scl_slope;
        float o = (float)scl_inter;
        for (int64_t i = 0; i < nvox; i++)
            out[i] = out[i] * s + o;
    }

    return out;
}

/* ========== Matrix I/O ========== */

int broc_write_matrix(const char *path, const float affine[16])
{
    FILE *fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "Error: cannot open %s for writing\n", path);
        return -1;
    }
    for (int row = 0; row < 4; row++) {
        fprintf(fp, "%e %e %e %e\n",
                affine[row * 4 + 0], affine[row * 4 + 1],
                affine[row * 4 + 2], affine[row * 4 + 3]);
    }
    fclose(fp);
    return 0;
}

int broc_read_matrix(const char *path, float affine[16])
{
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "Error: cannot open %s for reading\n", path);
        return -1;
    }
    for (int row = 0; row < 4; row++) {
        if (fscanf(fp, "%f %f %f %f",
                   &affine[row * 4 + 0], &affine[row * 4 + 1],
                   &affine[row * 4 + 2], &affine[row * 4 + 3]) != 4) {
            fprintf(stderr, "Error: failed to read row %d from %s\n", row, path);
            fclose(fp);
            return -1;
        }
    }
    fclose(fp);
    return 0;
}
