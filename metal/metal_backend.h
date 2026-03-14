/* metal_backend.h — Metal backend factory for BROCCOLI registration */

#ifndef METAL_BACKEND_H
#define METAL_BACKEND_H

#include "../registration.h"

#ifdef __cplusplus
extern "C" {
#endif

broc_backend *broc_metal_create_backend(void);

#ifdef __cplusplus
}
#endif

#endif /* METAL_BACKEND_H */
