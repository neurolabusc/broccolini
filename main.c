/* main.c — broccolini: standalone spatial normalization using BROCCOLI
 *
 * Usage: broccolini [options] -in <inputvol> -ref <refvol> -out <outputvol>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libgen.h>
#include <unistd.h>

#include "nifti_io.h"
#include "registration.h"

#include <math.h>

#define BROCCOLINI_VERSION "0.1.0"

/* High-frequency variance metric using Welford's online algorithm.
 * Scans the volume as a 1D array; for each consecutive pair of non-zero
 * voxels, computes the absolute intensity difference.  Reports the mean
 * and standard deviation of these differences as a proxy for retained
 * high-frequency content. */
static void print_hf_variance(const char *label, const float *data, int n)
{
    int64_t count = 0;
    double mean = 0.0, m2 = 0.0;

    for (int i = 1; i < n; i++) {
        if (data[i - 1] == 0.0f || data[i] == 0.0f)
            continue;
        double diff = fabs((double)data[i] - (double)data[i - 1]);
        count++;
        double delta = diff - mean;
        mean += delta / (double)count;
        double delta2 = diff - mean;
        m2 += delta * delta2;
    }

    if (count < 2) {
        printf("  HF variance %-30s: insufficient non-zero pairs\n", label);
        return;
    }
    double stddev = sqrt(m2 / (double)(count - 1));
    printf("  HF variance %-30s: mean=%.4f  sd=%.4f  (n=%lld)\n",
           label, mean, stddev, (long long)count);
}

static void print_usage(void)
{
    printf(
        "broccolini — GPU-accelerated spatial normalization (BROCCOLI)\n\n"
        "Usage: broccolini [options] -in <input> -ref <reference> -out <output>\n\n"
        "Required:\n"
        "  -in  <file>             Input volume (.nii/.nii.gz)\n"
        "  -ref <file>             Reference volume\n"
        "  -out, -o <file>         Output aligned volume\n\n"
        "Registration:\n"
        "  -dof <6|12>             Degrees of freedom: 6=rigid, 12=affine (default 12)\n"
        "  -lineariter <N>         Linear iterations per scale (default 10)\n"
        "  -nonlineariter <N>      Nonlinear iterations per scale (default 0 = off)\n"
        "  -coarsestscale <N>      Coarsest scale: 1,2,4,8 (default 4)\n"
        "  -zcut <mm>              Z-axis crop in mm (default 0)\n"
        "  -interp <mode>          nearestneighbour, trilinear, cubic (default trilinear)\n\n"
        "I/O:\n"
        "  -mask <file>            Brain mask for reference (default: derive from ref > 0)\n"
        "  -omat <file>            Save affine matrix (4x4 text)\n"
        "  -ofield <prefix>        Save displacement field as 3 NIfTI volumes\n\n"
        "Filters:\n"
        "  -filters <dir>          Directory with .bin filter files\n"
        "                          (default: ../filters/ relative to executable)\n\n"
        "Other:\n"
        "  -verbose                Verbose output\n"
        "  -v, -version            Print version\n"
        "  -h, -help               Print this help\n\n"
    );
}

/* Resolve filter directory: try -filters arg, then ../filters/ relative to exe */
static int resolve_filter_dir(const char *arg, const char *argv0, char *out, size_t out_size)
{
    if (arg) {
        snprintf(out, out_size, "%s", arg);
        return 0;
    }

    /* Try relative to executable */
    char exe_path[1024];
    strncpy(exe_path, argv0, sizeof(exe_path) - 1);
    exe_path[sizeof(exe_path) - 1] = '\0';
    char *dir = dirname(exe_path);
    snprintf(out, out_size, "%s/../filters", dir);

    /* Check if the directory has at least one expected file */
    char test_path[1200];
    snprintf(test_path, sizeof(test_path),
             "%s/filter1_real_linear_registration.bin", out);
    if (access(test_path, R_OK) == 0)
        return 0;

    /* Try ./filters/ as fallback */
    snprintf(out, out_size, "filters");
    snprintf(test_path, sizeof(test_path),
             "%s/filter1_real_linear_registration.bin", out);
    if (access(test_path, R_OK) == 0)
        return 0;

    fprintf(stderr, "Error: cannot find filter .bin files. Use -filters <dir>.\n");
    return -1;
}

/* Load a NIfTI volume and convert to float32.
 * Returns packed BROCCOLI-layout data; sets dims and vox. */
static float *load_and_pack(const char *path, broc_dims *dims, broc_voxsize *vox,
                            int *ni_out, int *nj_out, int *nk_out, int verbose)
{
    nifti_image *nim = nifti_image_read(path, 1);
    if (!nim) {
        fprintf(stderr, "Error: cannot read %s\n", path);
        return NULL;
    }
    if (nim->ndim < 3) {
        fprintf(stderr, "Error: %s is not a 3D volume (ndim=%lld)\n", path, (long long)nim->ndim);
        nifti_image_free(nim);
        return NULL;
    }

    int ni = (int)nim->nx;
    int nj = (int)nim->ny;
    int nk = (int)nim->nz;

    if (verbose)
        printf("  %s: %dx%dx%d, voxel %.2fx%.2fx%.2f mm, datatype %d\n",
               path, ni, nj, nk, nim->dx, nim->dy, nim->dz, nim->datatype);

    /* Convert to float32 */
    float *fdata = broc_nifti_to_float(nim->data, nim->datatype, nim->nvox,
                                        nim->scl_slope, nim->scl_inter);
    if (!fdata) {
        nifti_image_free(nim);
        return NULL;
    }

    /* For 4D, take only the first volume */
    int64_t vol_vox = (int64_t)ni * nj * nk;
    if (nim->nvox > vol_vox && verbose)
        printf("  (using first volume of 4D data)\n");

    *vox = broc_voxsize_from_nifti(nim->dx, nim->dy, nim->dz);
    if (ni_out) *ni_out = ni;
    if (nj_out) *nj_out = nj;
    if (nk_out) *nk_out = nk;

    nifti_image_free(nim);

    /* Pack to BROCCOLI layout */
    float *packed = broc_pack_volume(fdata, ni, nj, nk, dims);
    free(fdata);
    return packed;
}

/* Create output NIfTI by cloning the reference header */
static int save_output(const char *out_path, const char *ref_path,
                       const float *packed_data, broc_dims dims,
                       int ni, int nj, int nk, int verbose)
{
    /* Read reference header only (no data) */
    nifti_image *ref = nifti_image_read(ref_path, 0);
    if (!ref) {
        fprintf(stderr, "Error: cannot re-read reference header from %s\n", ref_path);
        return -1;
    }

    /* Unpack from BROCCOLI layout to NIfTI order */
    float *nifti_data = broc_unpack_volume(packed_data, dims, ni, nj, nk);
    if (!nifti_data) {
        nifti_image_free(ref);
        return -1;
    }

    /* Update header for 3D float32 output */
    ref->ndim = 3;
    ref->dim[0] = 3;
    ref->nx = ni; ref->dim[1] = ni;
    ref->ny = nj; ref->dim[2] = nj;
    ref->nz = nk; ref->dim[3] = nk;
    ref->nt = 1;  ref->dim[4] = 1;
    ref->nvox = (int64_t)ni * nj * nk;
    ref->datatype = DT_FLOAT32;
    ref->nbyper = 4;
    ref->scl_slope = 0.0;
    ref->scl_inter = 0.0;

    /* Set output filename */
    nifti_set_filenames(ref, out_path, 0, 0);

    /* Attach data and write */
    ref->data = nifti_data;
    nifti_image_write(ref);

    if (verbose)
        printf("  Saved: %s (%dx%dx%d)\n", out_path, ni, nj, nk);

    /* nifti_image_free will free ref->data (our nifti_data) */
    nifti_image_free(ref);
    return 0;
}

/* Save a displacement field component as NIfTI */
static int save_displacement(const char *prefix, const char *suffix,
                             const char *ref_path,
                             const float *packed_data, broc_dims dims,
                             int ni, int nj, int nk)
{
    char path[1024];
    snprintf(path, sizeof(path), "%s_%s.nii.gz", prefix, suffix);
    return save_output(path, ref_path, packed_data, dims, ni, nj, nk, 0);
}

int main(int argc, char *argv[])
{
    /* CLI arguments */
    const char *in_file     = NULL;
    const char *ref_file    = NULL;
    const char *out_file    = NULL;
    const char *mask_file   = NULL;
    const char *omat_file   = NULL;
    const char *ofield_prefix = NULL;
    const char *filter_arg  = NULL;

    broc_reg_params params;
    broc_reg_params_defaults(&params);

    /* Parse arguments */
    if (argc < 2) {
        print_usage();
        return EXIT_SUCCESS;
    }

    int i = 1;
    while (i < argc) {
        const char *arg = argv[i];

        if (strcmp(arg, "-h") == 0 || strcmp(arg, "-help") == 0 ||
            strcmp(arg, "--help") == 0) {
            print_usage();
            return EXIT_SUCCESS;
        }
        else if (strcmp(arg, "-v") == 0 || strcmp(arg, "-version") == 0 ||
                 strcmp(arg, "--version") == 0) {
            printf("broccolini %s\n", BROCCOLINI_VERSION);
            return EXIT_SUCCESS;
        }
        else if (strcmp(arg, "-in") == 0 && i + 1 < argc) {
            in_file = argv[++i];
        }
        else if (strcmp(arg, "-ref") == 0 && i + 1 < argc) {
            ref_file = argv[++i];
        }
        else if ((strcmp(arg, "-out") == 0 || strcmp(arg, "-o") == 0) && i + 1 < argc) {
            out_file = argv[++i];
        }
        else if (strcmp(arg, "-mask") == 0 && i + 1 < argc) {
            mask_file = argv[++i];
        }
        else if (strcmp(arg, "-omat") == 0 && i + 1 < argc) {
            omat_file = argv[++i];
        }
        else if (strcmp(arg, "-ofield") == 0 && i + 1 < argc) {
            ofield_prefix = argv[++i];
        }
        else if (strcmp(arg, "-filters") == 0 && i + 1 < argc) {
            filter_arg = argv[++i];
        }
        else if (strcmp(arg, "-dof") == 0 && i + 1 < argc) {
            params.dof = atoi(argv[++i]);
            if (params.dof != 6 && params.dof != 12) {
                fprintf(stderr, "Error: -dof must be 6 or 12\n");
                return EXIT_FAILURE;
            }
        }
        else if (strcmp(arg, "-lineariter") == 0 && i + 1 < argc) {
            params.linear_iterations = atoi(argv[++i]);
        }
        else if (strcmp(arg, "-nonlineariter") == 0 && i + 1 < argc) {
            params.nonlinear_iterations = atoi(argv[++i]);
        }
        else if (strcmp(arg, "-coarsestscale") == 0 && i + 1 < argc) {
            params.coarsest_scale = atoi(argv[++i]);
        }
        else if (strcmp(arg, "-zcut") == 0 && i + 1 < argc) {
            params.zcut_mm = atoi(argv[++i]);
        }
        else if (strcmp(arg, "-interp") == 0 && i + 1 < argc) {
            const char *mode = argv[++i];
            if (strcmp(mode, "nearestneighbour") == 0 || strcmp(mode, "nn") == 0)
                params.interp = BROC_INTERP_NEAREST;
            else if (strcmp(mode, "trilinear") == 0)
                params.interp = BROC_INTERP_LINEAR;
            else if (strcmp(mode, "cubic") == 0 || strcmp(mode, "spline") == 0)
                params.interp = BROC_INTERP_CUBIC;
            else {
                fprintf(stderr, "Error: unknown interpolation mode '%s'\n", mode);
                return EXIT_FAILURE;
            }
        }
        else if (strcmp(arg, "-verbose") == 0) {
            params.verbose = 1;
        }
        else {
            fprintf(stderr, "Error: unrecognized option '%s'\n", arg);
            return EXIT_FAILURE;
        }
        i++;
    }

    /* Validate required arguments */
    if (!in_file || !ref_file) {
        fprintf(stderr, "Error: -in and -ref are required\n");
        return EXIT_FAILURE;
    }
    if (!out_file && !omat_file) {
        fprintf(stderr, "Error: at least one of -out or -omat is required\n");
        return EXIT_FAILURE;
    }

    /* Resolve filter directory */
    char filter_dir[1024];
    if (resolve_filter_dir(filter_arg, argv[0], filter_dir, sizeof(filter_dir)) != 0)
        return EXIT_FAILURE;

    if (params.verbose)
        printf("Filter directory: %s\n", filter_dir);

    /* Load filters */
    broc_filters filters;
    if (broc_load_filters(filter_dir, &filters) != 0)
        return EXIT_FAILURE;

    if (params.verbose)
        printf("Filters loaded.\n");

    /* Load input volume */
    broc_dims in_dims;
    broc_voxsize in_vox;
    int in_ni, in_nj, in_nk;
    if (params.verbose)
        printf("Loading input volume...\n");
    float *in_packed = load_and_pack(in_file, &in_dims, &in_vox,
                                     &in_ni, &in_nj, &in_nk, params.verbose);
    if (!in_packed)
        return EXIT_FAILURE;

    /* Load reference volume */
    broc_dims ref_dims;
    broc_voxsize ref_vox;
    int ref_ni, ref_nj, ref_nk;
    if (params.verbose)
        printf("Loading reference volume...\n");
    float *ref_packed = load_and_pack(ref_file, &ref_dims, &ref_vox,
                                      &ref_ni, &ref_nj, &ref_nk, params.verbose);
    if (!ref_packed) {
        free(in_packed);
        return EXIT_FAILURE;
    }


    /* Load mask if provided */
    float *mask_packed = NULL;
    if (mask_file) {
        broc_dims mask_dims;
        broc_voxsize mask_vox;
        if (params.verbose)
            printf("Loading mask...\n");
        mask_packed = load_and_pack(mask_file, &mask_dims, &mask_vox,
                                    NULL, NULL, NULL, params.verbose);
        if (!mask_packed) {
            free(in_packed);
            free(ref_packed);
            return EXIT_FAILURE;
        }
        /* Verify mask dimensions match reference */
        if (mask_dims.W != ref_dims.W || mask_dims.H != ref_dims.H ||
            mask_dims.D != ref_dims.D) {
            fprintf(stderr, "Error: mask dimensions (%dx%dx%d) don't match "
                    "reference (%dx%dx%d)\n",
                    mask_dims.W, mask_dims.H, mask_dims.D,
                    ref_dims.W, ref_dims.H, ref_dims.D);
            free(in_packed); free(ref_packed); free(mask_packed);
            return EXIT_FAILURE;
        }
    }

    /* Create backend */
    broc_backend *backend = NULL;
#ifdef HAVE_METAL
    backend = broc_metal_create_backend();
#elif defined(HAVE_WEBGPU)
    backend = broc_webgpu_create_backend();
#elif defined(HAVE_OPENCL)
    backend = broc_opencl_create_backend();
#endif

    if (!backend) {
        fprintf(stderr, "Error: no GPU backend available\n");
        free(in_packed); free(ref_packed); free(mask_packed);
        return EXIT_FAILURE;
    }

    if (params.verbose)
        printf("Backend: %s\n", backend->name);

    /* Run registration */
    if (params.verbose)
        printf("Starting registration...\n");

    broc_result result = backend->register_volumes(
        backend,
        in_packed, in_dims, in_vox,
        ref_packed, ref_dims, ref_vox,
        mask_packed,
        &filters,
        &params);

    free(in_packed);
    free(ref_packed);
    free(mask_packed);

    if (!result.aligned) {
        fprintf(stderr, "Error: registration failed\n");
        backend->destroy(backend);
        return EXIT_FAILURE;
    }

    if (params.verbose)
        printf("Registration complete.\n");

    /* Print high-frequency variance metric */
    {
        int vol_size = result.out_W * result.out_H * result.out_D;
        print_hf_variance("(aligned output)", result.aligned, vol_size);
    }

    /* Save output volume */
    if (out_file) {
        if (params.verbose)
            printf("Saving output...\n");
        broc_dims out_dims = { result.out_W, result.out_H, result.out_D };
        if (save_output(out_file, ref_file, result.aligned, out_dims,
                        ref_ni, ref_nj, ref_nk, params.verbose) != 0) {
            broc_result_free(&result);
            backend->destroy(backend);
            return EXIT_FAILURE;
        }
    }

    /* Save affine matrix */
    if (omat_file) {
        if (broc_write_matrix(omat_file, result.affine) != 0) {
            broc_result_free(&result);
            backend->destroy(backend);
            return EXIT_FAILURE;
        }
        if (params.verbose)
            printf("  Saved matrix: %s\n", omat_file);
    }

    /* Save displacement field */
    if (ofield_prefix && result.disp_x) {
        broc_dims out_dims = { result.out_W, result.out_H, result.out_D };
        save_displacement(ofield_prefix, "dx", ref_file,
                          result.disp_x, out_dims, ref_ni, ref_nj, ref_nk);
        save_displacement(ofield_prefix, "dy", ref_file,
                          result.disp_y, out_dims, ref_ni, ref_nj, ref_nk);
        save_displacement(ofield_prefix, "dz", ref_file,
                          result.disp_z, out_dims, ref_ni, ref_nj, ref_nk);
        if (params.verbose)
            printf("  Saved displacement fields: %s_{dx,dy,dz}.nii.gz\n", ofield_prefix);
    }

    broc_result_free(&result);
    backend->destroy(backend);

    return EXIT_SUCCESS;
}
