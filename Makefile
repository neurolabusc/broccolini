# Makefile for broccolini — standalone spatial normalization executable
#
# Usage:
#   make                # auto-detect (Metal on macOS, OpenCL elsewhere)
#   make BACKEND=metal
#   make BACKEND=opencl
#   make BACKEND=webgpu
#   make clean

# ---------- Auto-detect platform and backend ----------

UNAME := $(shell uname)
ifeq ($(UNAME),Darwin)
  BACKEND ?= metal
else
  BACKEND ?= opencl
endif

# ---------- Common settings ----------

CC       = clang
CXX      = clang++
CFLAGS   = -O2 -std=c11 -Wall -Wextra -Wno-unused-parameter
CXXFLAGS = -O2 -std=c++17 -Wall -Wextra -Wno-unused-parameter -Wno-narrowing -w
LDFLAGS  =

TARGET   = broccolini
BUILDDIR = build

# Common C objects
C_SRCS    = main.c registration.c nifti_io.c
C_OBJS    = $(patsubst %.c,$(BUILDDIR)/%.o,$(C_SRCS))

# ---------- Metal backend (macOS only) ----------

ifeq ($(BACKEND),metal)
  DEFINES  = -DHAVE_METAL -DHAVE_ZLIB
  SHADER_PATH ?= $(CURDIR)/metal/shaders/registration.metal

  METAL_SRCS = metal/metal_backend.mm metal/metal_registration.mm
  METAL_OBJS = $(BUILDDIR)/metal_backend.o $(BUILDDIR)/metal_registration.o

  METAL_CXXFLAGS = $(CXXFLAGS) $(DEFINES) \
    -DMETAL_SHADER_DEFAULT_PATH="\"$(SHADER_PATH)\"" \
    -I. -Imetal

  FRAMEWORKS = -framework Metal -framework Foundation -framework Accelerate
  LDFLAGS   += -lz $(FRAMEWORKS)

  ALL_OBJS = $(C_OBJS) $(METAL_OBJS)
endif

# ---------- OpenCL backend ----------

ifeq ($(BACKEND),opencl)
  DEFINES  = -DHAVE_OPENCL -DHAVE_ZLIB -DEIGEN_DONT_VECTORIZE

  # OpenCL headers location (macOS SDK or system)
  ifeq ($(UNAME),Darwin)
    OPENCL_HEADERS ?= $(shell xcrun --show-sdk-path)/System/Library/Frameworks/OpenCL.framework/Headers
    OPENCL_LDFLAGS  = -framework OpenCL
  else
    OPENCL_HEADERS ?= /usr/include/CL
    OPENCL_LDFLAGS  = -lOpenCL
  endif

  OPENCL_SRCS = opencl/opencl_backend.cpp opencl/broccoli_lib.cpp
  OPENCL_OBJS = $(BUILDDIR)/opencl_backend.o $(BUILDDIR)/broccoli_lib.o

  ifeq ($(UNAME),Darwin)
    CLBLAS_DIR ?= $(CURDIR)/../code/BROCCOLI_LIB/clBLASMac
  else
    CLBLAS_DIR ?= $(CURDIR)/../code/BROCCOLI_LIB/clBLASLinux
  endif

  OPENCL_CXXFLAGS = -O2 -std=c++14 -Wall -Wextra -Wno-unused-parameter -Wno-narrowing -w $(DEFINES) \
    -I. -Iopencl -I$(OPENCL_HEADERS) -I$(CLBLAS_DIR)

  LDFLAGS += -lz $(OPENCL_LDFLAGS)
  ifeq ($(UNAME),Darwin)
    LDFLAGS += -framework Accelerate
  endif

  ALL_OBJS = $(C_OBJS) $(OPENCL_OBJS)
endif

# ---------- WebGPU backend (cross-platform via wgpu-native) ----------

ifeq ($(BACKEND),webgpu)
  DEFINES  = -DHAVE_WEBGPU -DHAVE_ZLIB

  # wgpu-native installation: headers + library
  # Download from https://github.com/gfx-rs/wgpu-native/releases
  # Set WGPU_DIR to the extracted directory (contains include/ and lib/)
  WGPU_DIR ?= /usr/local
  WGPU_INCLUDE = $(WGPU_DIR)/include
  WGPU_LIB     = $(WGPU_DIR)/lib

  WEBGPU_SRCS = webgpu/webgpu_backend.cpp webgpu/webgpu_registration.cpp
  WEBGPU_OBJS = $(BUILDDIR)/webgpu_backend.o $(BUILDDIR)/webgpu_registration.o

  WEBGPU_CXXFLAGS = $(CXXFLAGS) $(DEFINES) \
    -I. -Iwebgpu -I$(WGPU_INCLUDE)

  LDFLAGS += -lz -L$(WGPU_LIB) -lwgpu_native -Wl,-rpath,$(WGPU_LIB)
  ifeq ($(UNAME),Darwin)
    LDFLAGS += -framework Metal -framework QuartzCore -framework CoreGraphics
  endif
  ifeq ($(UNAME),Linux)
    LDFLAGS += -lm -ldl -lpthread
  endif

  ALL_OBJS = $(C_OBJS) $(WEBGPU_OBJS)
endif

# ---------- CUDA backend ----------

ifeq ($(BACKEND),cuda)
  NVCC     ?= nvcc
  DEFINES  = -DHAVE_CUDA -DHAVE_ZLIB

  CUDA_SRCS = cuda/cuda_backend.cpp cuda/cuda_registration.cu
  CUDA_OBJS = $(BUILDDIR)/cuda_backend.o $(BUILDDIR)/cuda_registration.o

  CUDA_CXXFLAGS = $(CXXFLAGS) $(DEFINES) -I. -Icuda
  NVCC_FLAGS    = -O2 -std=c++17 --extended-lambda -w $(DEFINES) -I. -Icuda

  # Auto-detect GPU architecture if not specified
  # --list-gpu-code gives sm_XX codes; fallback to sm_70
  CUDA_ARCH ?= $(shell nvcc --list-gpu-code 2>/dev/null | tail -1)
  ifeq ($(CUDA_ARCH),)
    CUDA_ARCH = sm_70
  endif
  NVCC_FLAGS += -arch=$(CUDA_ARCH)

  LDFLAGS += -lz -lcudart
  # Link with nvcc to resolve CUDA runtime
  CXX_LINK = $(NVCC) -arch=$(CUDA_ARCH)

  ALL_OBJS = $(C_OBJS) $(CUDA_OBJS)
endif

# ---------- Rules ----------

.PHONY: all clean setup

all: setup $(TARGET)

setup:
	@mkdir -p $(BUILDDIR)
ifeq ($(BACKEND),opencl)
	@# Create symlink structure that BROCCOLI_LIB expects: code/Kernels/
	@mkdir -p opencl/code
	@ln -sfn ../kernels opencl/code/Kernels
endif

$(TARGET): $(ALL_OBJS)
ifeq ($(BACKEND),cuda)
	$(CXX_LINK) -o $@ $^ $(LDFLAGS)
else
	$(CXX) -o $@ $^ $(LDFLAGS)
endif
	@echo ""
	@echo "=== Build successful: $(TARGET) ($(BACKEND) backend) ==="
	@echo ""
ifeq ($(BACKEND),opencl)
	@echo "NOTE: Set BROCCOLI_DIR before running:"
	@echo "  export BROCCOLI_DIR=$(CURDIR)/opencl/"
	@echo ""
endif
	@echo "Quick test:"
	@echo "  ./$(TARGET) -in examples/t1_brain.nii.gz \\"
	@echo "    -ref examples/MNI152_T1_1mm_brain.nii.gz \\"
	@echo "    -out /tmp/t1_aligned.nii.gz -verbose"

# Common C sources
$(BUILDDIR)/%.o: %.c
	$(CC) $(CFLAGS) $(DEFINES) -I. -c $< -o $@

# Metal backend
$(BUILDDIR)/metal_backend.o: metal/metal_backend.mm
	$(CXX) $(METAL_CXXFLAGS) -c $< -o $@

$(BUILDDIR)/metal_registration.o: metal/metal_registration.mm
	$(CXX) $(METAL_CXXFLAGS) -c $< -o $@

# OpenCL backend
$(BUILDDIR)/opencl_backend.o: opencl/opencl_backend.cpp
	$(CXX) $(OPENCL_CXXFLAGS) -c $< -o $@

$(BUILDDIR)/broccoli_lib.o: opencl/broccoli_lib.cpp
	$(CXX) $(OPENCL_CXXFLAGS) -fPIC -c $< -o $@

# WebGPU backend
$(BUILDDIR)/webgpu_backend.o: webgpu/webgpu_backend.cpp
	$(CXX) $(WEBGPU_CXXFLAGS) -c $< -o $@

$(BUILDDIR)/webgpu_registration.o: webgpu/webgpu_registration.cpp
	$(CXX) $(WEBGPU_CXXFLAGS) -c $< -o $@

# CUDA backend
$(BUILDDIR)/cuda_backend.o: cuda/cuda_backend.cpp
	$(CXX) $(CUDA_CXXFLAGS) -c $< -o $@

$(BUILDDIR)/cuda_registration.o: cuda/cuda_registration.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -rf $(BUILDDIR) $(TARGET)
	rm -f opencl/code/Kernels
	rmdir opencl/code 2>/dev/null || true
