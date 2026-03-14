/* opencl_backend.h — OpenCL backend factory for broccolini */
#ifndef OPENCL_BACKEND_H
#define OPENCL_BACKEND_H

#include "../registration.h"

#ifdef __cplusplus
extern "C" {
#endif

broc_backend *broc_opencl_create_backend(void);

#ifdef __cplusplus
}
#endif

#endif /* OPENCL_BACKEND_H */
