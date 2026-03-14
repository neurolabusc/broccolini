/* registration.h — Backend-agnostic C API for BROCCOLI registration
 *
 * Defines types, backend vtable, and shared utility functions.
 * Backends (Metal, OpenCL, CUDA, etc.) implement the vtable and
 * expose a single C-linkage factory function.
 */

#ifndef REGISTRATION_H
#define REGISTRATION_H

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========== Constants ========== */

#define BROC_FILTER_SIZE     7
#define BROC_FILTER_ELEMENTS (BROC_FILTER_SIZE * BROC_FILTER_SIZE * BROC_FILTER_SIZE) /* 343 */
#define BROC_NUM_LINEAR_FILTERS    3
#define BROC_NUM_NONLINEAR_FILTERS 6

/* ========== Core types ========== */

typedef struct {
    int W, H, D;
} broc_dims;

typedef struct {
    float x, y, z;
} broc_voxsize;

typedef enum {
    BROC_INTERP_NEAREST  = 0,
    BROC_INTERP_LINEAR   = 1,
    BROC_INTERP_CUBIC    = 2
} broc_interp;

/* ========== Registration parameters ========== */

typedef struct {
    int         dof;                  /* 6 (rigid) or 12 (affine) */
    int         linear_iterations;    /* iterations per scale (default 10) */
    int         nonlinear_iterations; /* 0 = linear only (default 0) */
    int         coarsest_scale;       /* 1, 2, 4, or 8 (default 4) */
    int         zcut_mm;              /* z-axis crop in mm (default 0) */
    broc_interp interp;               /* final reslice interpolation */
    int         verbose;
} broc_reg_params;

/* ========== Quadrature filters (loaded from .bin files) ========== */

typedef struct {
    float linear_real[BROC_NUM_LINEAR_FILTERS][BROC_FILTER_ELEMENTS];
    float linear_imag[BROC_NUM_LINEAR_FILTERS][BROC_FILTER_ELEMENTS];
    float nonlinear_real[BROC_NUM_NONLINEAR_FILTERS][BROC_FILTER_ELEMENTS];
    float nonlinear_imag[BROC_NUM_NONLINEAR_FILTERS][BROC_FILTER_ELEMENTS];
    float projection_tensors[BROC_NUM_NONLINEAR_FILTERS][6]; /* m11,m12,m13,m22,m23,m33 */
    float filter_directions_x[BROC_NUM_NONLINEAR_FILTERS];
    float filter_directions_y[BROC_NUM_NONLINEAR_FILTERS];
    float filter_directions_z[BROC_NUM_NONLINEAR_FILTERS];
} broc_filters;

/* ========== Registration result ========== */

typedef struct {
    float  *aligned;              /* output volume in ref space (caller frees) */
    int     out_W, out_H, out_D;  /* dimensions in packed layout */
    float   affine[16];           /* 4x4 row-major affine matrix */
    float  *disp_x;              /* displacement field X (NULL if linear only) */
    float  *disp_y;              /* displacement field Y */
    float  *disp_z;              /* displacement field Z */
} broc_result;

/* ========== Backend vtable ========== */

typedef struct broc_backend broc_backend;
struct broc_backend {
    /* Register input volume to reference volume */
    broc_result (*register_volumes)(
        broc_backend       *self,
        const float        *input,    broc_dims in_dims,  broc_voxsize in_vox,
        const float        *ref,      broc_dims ref_dims, broc_voxsize ref_vox,
        const float        *ref_mask, /* optional brain mask, NULL if none */
        const broc_filters *filters,
        const broc_reg_params *params);

    /* Free backend resources */
    void (*destroy)(broc_backend *self);

    /* Backend name for diagnostics */
    const char *name;

    /* Private data (backend stores its context here) */
    void *priv;
};

/* ========== Backend factory functions ========== */

#ifdef HAVE_METAL
broc_backend *broc_metal_create_backend(void);
#endif

#ifdef HAVE_OPENCL
broc_backend *broc_opencl_create_backend(void);
#endif

#ifdef HAVE_WEBGPU
broc_backend *broc_webgpu_create_backend(void);
#endif

/* ========== Shared utility functions ========== */

/* Set default registration parameters */
void broc_reg_params_defaults(broc_reg_params *p);

/* Load filters from directory of .bin files.
 * Returns 0 on success, -1 on error. */
int broc_load_filters(const char *filter_dir, broc_filters *out);

/* Pack a NIfTI volume (i,j,k) into BROCCOLI layout.
 * Equivalent to Python: flipud + transpose(2,0,1).
 * Input: nifti_data with shape (ni, nj, nk) in row-major order.
 * Output: newly allocated float array; caller frees.
 * out_dims is set to (W=nj, H=ni, D=nk). */
float *broc_pack_volume(const float *nifti_data, int ni, int nj, int nk,
                        broc_dims *out_dims);

/* Unpack a BROCCOLI-layout volume back to NIfTI (i,j,k) order.
 * Inverse of broc_pack_volume.
 * Output: newly allocated float array; caller frees. */
float *broc_unpack_volume(const float *packed_data, broc_dims dims,
                          int ni, int nj, int nk);

/* Convert NIfTI pixdim to BROCCOLI voxel sizes.
 * NIfTI: pixdim[1]=i-spacing, pixdim[2]=j-spacing, pixdim[3]=k-spacing.
 * BROCCOLI: x=j-spacing (width), y=i-spacing (height), z=k-spacing (depth). */
broc_voxsize broc_voxsize_from_nifti(double pixdim1, double pixdim2, double pixdim3);

/* Free all allocated fields in a registration result */
void broc_result_free(broc_result *r);

/* Convert NIfTI image data of any type to a newly allocated float32 array.
 * Applies scl_slope and scl_inter if set.
 * Returns NULL on unsupported datatype. Caller frees. */
float *broc_nifti_to_float(const void *data, int datatype, int64_t nvox,
                           double scl_slope, double scl_inter);

/* Write 4x4 affine matrix to text file (FSL-compatible format).
 * Returns 0 on success. */
int broc_write_matrix(const char *path, const float affine[16]);

/* Read 4x4 affine matrix from text file.
 * Returns 0 on success. */
int broc_read_matrix(const char *path, float affine[16]);

#ifdef __cplusplus
}
#endif

#endif /* REGISTRATION_H */
