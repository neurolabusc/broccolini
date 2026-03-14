/* cuda_backend.h — CUDA backend factory for BROCCOLI registration */

#ifndef CUDA_BACKEND_H
#define CUDA_BACKEND_H

#include "../registration.h"

#ifdef __cplusplus
extern "C" {
#endif

broc_backend *broc_cuda_create_backend(void);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_BACKEND_H */
