/* webgpu_backend.h — Factory function for the WebGPU backend */

#ifndef WEBGPU_BACKEND_H
#define WEBGPU_BACKEND_H

#include "registration.h"

#ifdef __cplusplus
extern "C" {
#endif

broc_backend *broc_webgpu_create_backend(void);

#ifdef __cplusplus
}
#endif

#endif /* WEBGPU_BACKEND_H */
