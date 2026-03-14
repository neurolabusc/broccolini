// metal_registration.mm — Metal compute backend for BROCCOLI image registration
// Objective-C++ required for Metal API

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#ifdef USE_MPSGRAPH_FFT
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <cassert>
#include <algorithm>
#include <CoreFoundation/CoreFoundation.h>

#include "metal_registration.h"

// Timing helper macro
#define TIMER_START(name) CFAbsoluteTime _t_##name = CFAbsoluteTimeGetCurrent()
#define TIMER_END(name, label) printf("[TIMING] %-50s %8.3f ms\n", label, (CFAbsoluteTimeGetCurrent() - _t_##name) * 1000.0)

// ============================================================
//  Internal helpers
// ============================================================

namespace metal_reg {
namespace {

struct Dims {
    int W, H, D;
};

struct MetalContext {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;

    // Pipeline states (lazily created)
    NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* pipelines;

    MetalContext() : pipelines(nil) {}

    void init() {
        device = MTLCreateSystemDefaultDevice();
        assert(device && "No Metal device found");
        queue = [device newCommandQueue];

        // Load shader library from metallib file next to executable,
        // or compile from source
        NSError* err = nil;
        NSString* libPath = [[NSBundle mainBundle] pathForResource:@"registration" ofType:@"metallib"];
        if (libPath) {
            NSURL* libURL = [NSURL fileURLWithPath:libPath];
            library = [device newLibraryWithURL:libURL error:&err];
        }

        if (!library) {
            // Try loading from the same directory as this code
            // Look for the .metal source and compile it
            NSString* srcPath = nil;

            // Search in several candidate locations
            NSArray* candidates = @[
                @"src/metal/shaders/registration.metal",
                @"metal/shaders/registration.metal",
                @"shaders/registration.metal",
                @"registration.metal",
            ];

            NSFileManager* fm = [NSFileManager defaultManager];

            // Try relative to the process working directory
            for (NSString* cand in candidates) {
                if ([fm fileExistsAtPath:cand]) { srcPath = cand; break; }
            }

            // Try relative to the executable
            if (!srcPath) {
                NSString* execDir = [[[NSProcessInfo processInfo] arguments][0] stringByDeletingLastPathComponent];
                for (NSString* cand in candidates) {
                    NSString* full = [execDir stringByAppendingPathComponent:cand];
                    if ([fm fileExistsAtPath:full]) { srcPath = full; break; }
                }
            }

            // Try from environment variable
            if (!srcPath) {
                const char* envPath = getenv("METAL_SHADER_PATH");
                if (envPath) {
                    srcPath = [NSString stringWithUTF8String:envPath];
                }
            }

            if (srcPath) {
                NSString* src = [NSString stringWithContentsOfFile:srcPath
                                                         encoding:NSUTF8StringEncoding error:&err];
                if (src) {
                    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
                    opts.mathMode = MTLMathModeSafe;
                    library = [device newLibraryWithSource:src options:opts error:&err];
                }
            }
        }

        if (!library) {
            NSLog(@"Failed to create Metal library: %@", err);
            assert(false && "Could not load Metal shaders");
        }

        pipelines = [NSMutableDictionary dictionary];
    }

    id<MTLComputePipelineState> getPipeline(const char* name) {
        NSString* key = [NSString stringWithUTF8String:name];
        id<MTLComputePipelineState> ps = pipelines[key];
        if (!ps) {
            id<MTLFunction> fn = [library newFunctionWithName:key];
            assert(fn && "Shader function not found");
            NSError* err = nil;
            ps = [device newComputePipelineStateWithFunction:fn error:&err];
            assert(ps && "Failed to create pipeline state");
            pipelines[key] = ps;
        }
        return ps;
    }

    id<MTLBuffer> newBuffer(size_t bytes) {
        return [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    }

    id<MTLBuffer> newBuffer(const void* data, size_t bytes) {
        return [device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
    }

    id<MTLTexture> newTexture3D(int W, int H, int D) {
        MTLTextureDescriptor* desc = [MTLTextureDescriptor new];
        desc.textureType = MTLTextureType3D;
        desc.pixelFormat = MTLPixelFormatR32Float;
        desc.width = W;
        desc.height = H;
        desc.depth = D;
        desc.storageMode = MTLStorageModeShared;
        desc.usage = MTLTextureUsageShaderRead;
        return [device newTextureWithDescriptor:desc];
    }

    void copyBufferToTexture(id<MTLBuffer> buf, id<MTLTexture> tex, int W, int H, int D) {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:buf
               sourceOffset:0
          sourceBytesPerRow:W * sizeof(float)
        sourceBytesPerImage:W * H * sizeof(float)
                 sourceSize:MTLSizeMake(W, H, D)
                  toTexture:tex
           destinationSlice:0
           destinationLevel:0
          destinationOrigin:MTLOriginMake(0, 0, 0)];
        [blit endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }
};

// Singleton context
MetalContext& ctx() {
    static MetalContext c;
    static bool inited = false;
    if (!inited) { c.init(); inited = true; }
    return c;
}

// ============================================================
//  Dispatch helpers
// ============================================================

void dispatch3D(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> ps,
                int W, int H, int D) {
    NSUInteger tw = ps.threadExecutionWidth;
    NSUInteger th = ps.maxTotalThreadsPerThreadgroup / tw;
    MTLSize threads = MTLSizeMake(W, H, D);
    MTLSize tgSize = MTLSizeMake(tw, std::min(th, (NSUInteger)H), 1);
    [enc setComputePipelineState:ps];
    [enc dispatchThreads:threads threadsPerThreadgroup:tgSize];
}

void dispatch1D(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> ps,
                int count) {
    NSUInteger tw = std::min((NSUInteger)count, ps.maxTotalThreadsPerThreadgroup);
    [enc setComputePipelineState:ps];
    [enc dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
}

void dispatch2D(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> ps,
                int W, int H) {
    NSUInteger tw = ps.threadExecutionWidth;
    NSUInteger th = std::min(ps.maxTotalThreadsPerThreadgroup / tw, (NSUInteger)H);
    [enc setComputePipelineState:ps];
    [enc dispatchThreads:MTLSizeMake(W, H, 1) threadsPerThreadgroup:MTLSizeMake(tw, th, 1)];
}

// ============================================================
//  GPU operations
// ============================================================

// ============================================================
//  Encoder-level helpers (encode into existing command encoder)
//  These allow batching multiple operations into a single CB.
// ============================================================

void encodeFill(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> buf, float value, int count) {
    auto& c = ctx();
    auto ps = c.getPipeline("fillFloat");
    id<MTLBuffer> valBuf = c.newBuffer(&value, sizeof(float));
    [enc setComputePipelineState:ps];
    [enc setBuffer:buf offset:0 atIndex:0];
    [enc setBuffer:valBuf offset:0 atIndex:1];
    dispatch1D(enc, ps, count);
}

void encodeFillFloat2(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> buf, int count) {
    auto& c = ctx();
    auto ps = c.getPipeline("fillFloat2");
    float zero[2] = {0, 0};
    id<MTLBuffer> valBuf = c.newBuffer(zero, sizeof(float) * 2);
    [enc setComputePipelineState:ps];
    [enc setBuffer:buf offset:0 atIndex:0];
    [enc setBuffer:valBuf offset:0 atIndex:1];
    dispatch1D(enc, ps, count);
}

void encodeAdd(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> A, id<MTLBuffer> B, int count) {
    auto& c = ctx();
    auto ps = c.getPipeline("addVolumes");
    [enc setComputePipelineState:ps];
    [enc setBuffer:A offset:0 atIndex:0];
    [enc setBuffer:B offset:0 atIndex:1];
    dispatch1D(enc, ps, count);
}

void encodeMultiply(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> vol, float factor, int count) {
    auto& c = ctx();
    auto ps = c.getPipeline("multiplyVolume");
    id<MTLBuffer> fBuf = c.newBuffer(&factor, sizeof(float));
    [enc setComputePipelineState:ps];
    [enc setBuffer:vol offset:0 atIndex:0];
    [enc setBuffer:fBuf offset:0 atIndex:1];
    dispatch1D(enc, ps, count);
}

void encodeMultiplyVolumes(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> A, id<MTLBuffer> B, int count) {
    auto& c = ctx();
    auto ps = c.getPipeline("multiplyVolumes");
    [enc setComputePipelineState:ps];
    [enc setBuffer:A offset:0 atIndex:0];
    [enc setBuffer:B offset:0 atIndex:1];
    dispatch1D(enc, ps, count);
}

// Standalone versions (own command buffer) — used where batching isn't beneficial
void fillBuffer(id<MTLBuffer> buf, float value, int count) {
    auto& c = ctx();
    id<MTLCommandBuffer> cb = [c.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    encodeFill(enc, buf, value, count);
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

void fillFloat2Buffer(id<MTLBuffer> buf, int count) {
    auto& c = ctx();
    id<MTLCommandBuffer> cb = [c.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    encodeFillFloat2(enc, buf, count);
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

void addVolumes(id<MTLBuffer> A, id<MTLBuffer> B, int count) {
    auto& c = ctx();
    id<MTLCommandBuffer> cb = [c.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    encodeAdd(enc, A, B, count);
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

void multiplyVolume(id<MTLBuffer> vol, float factor, int count) {
    auto& c = ctx();
    id<MTLCommandBuffer> cb = [c.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    encodeMultiply(enc, vol, factor, count);
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

void multiplyVolumes(id<MTLBuffer> A, id<MTLBuffer> B, int count) {
    auto& c = ctx();
    id<MTLCommandBuffer> cb = [c.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    encodeMultiplyVolumes(enc, A, B, count);
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

float calculateMax(id<MTLBuffer> volume, int W, int H, int D) {
    auto& c = ctx();
    Dims dims = {W, H, D};

    id<MTLBuffer> dimBuf = c.newBuffer(&dims, sizeof(Dims));
    id<MTLBuffer> colMaxs = c.newBuffer(H * D * sizeof(float));
    id<MTLBuffer> rowMaxs = c.newBuffer(D * sizeof(float));

    id<MTLCommandBuffer> cb = [c.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

    auto ps1 = c.getPipeline("calculateColumnMaxs");
    [enc setComputePipelineState:ps1];
    [enc setBuffer:colMaxs offset:0 atIndex:0];
    [enc setBuffer:volume offset:0 atIndex:1];
    [enc setBuffer:dimBuf offset:0 atIndex:2];
    dispatch2D(enc, ps1, H, D);

    auto ps2 = c.getPipeline("calculateRowMaxs");
    [enc setComputePipelineState:ps2];
    [enc setBuffer:rowMaxs offset:0 atIndex:0];
    [enc setBuffer:colMaxs offset:0 atIndex:1];
    [enc setBuffer:dimBuf offset:0 atIndex:2];
    dispatch1D(enc, ps2, D);

    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    float* rowData = (float*)[rowMaxs contents];
    float mx = rowData[0];
    for (int i = 1; i < D; i++) mx = std::max(mx, rowData[i]);
    return mx;
}

// ============================================================
//  3D Nonseparable Convolution (3 quadrature filters)
// ============================================================

#ifdef USE_MPSGRAPH_FFT

// FFT-based convolution via MPSGraph (macOS 14.0+).
// Batches all 6 filter components (3 filters × real/imag) into a single
// graph execution: 1 volume FFT + 6 filter FFTs + 6 IFFTs.
void nonseparableConvolution3D(
    id<MTLBuffer> resp1, id<MTLBuffer> resp2, id<MTLBuffer> resp3,
    id<MTLBuffer> volume,
    const float* filterReal1, const float* filterImag1,
    const float* filterReal2, const float* filterImag2,
    const float* filterReal3, const float* filterImag3,
    int W, int H, int D)
{
    TIMER_START(conv3d_fft);
    auto& c = ctx();
    int vol = W * H * D;

    // Pad dimensions to avoid circular wrap-around (filter radius = 3)
    int padW = W + 6, padH = H + 6, padD = D + 6;
    int padVol = padW * padH * padD;

    // Pad volume (zero beyond original extent)
    TIMER_START(conv3d_pad);
    std::vector<float> pVol(padVol, 0.0f);
    const float* vp = (const float*)[volume contents];
    for (int z = 0; z < D; z++)
        for (int y = 0; y < H; y++)
            memcpy(&pVol[y * padW + z * padW * padH],
                   &vp[y * W + z * W * H], W * sizeof(float));

    // Pad and center 6 filter components (wrap-around for circular conv)
    const float* fps[6] = {filterReal1, filterImag1,
                           filterReal2, filterImag2,
                           filterReal3, filterImag3};
    std::vector<float> pFilt(6 * padVol, 0.0f);
    for (int fi = 0; fi < 6; fi++) {
        float* dst = &pFilt[fi * padVol];
        for (int fz = 0; fz < 7; fz++)
            for (int fy = 0; fy < 7; fy++)
                for (int fx = 0; fx < 7; fx++) {
                    int px = ((fx - 3) + padW) % padW;
                    int py = ((fy - 3) + padH) % padH;
                    int pz = ((fz - 3) + padD) % padD;
                    dst[px + py * padW + pz * padW * padH] =
                        fps[fi][fx + fy * 7 + fz * 49];
                }
    }
    TIMER_END(conv3d_pad, "  conv3D FFT: pad");

    // Upload padded data to GPU
    id<MTLBuffer> volBuf = c.newBuffer(pVol.data(), padVol * sizeof(float));
    id<MTLBuffer> filtBuf = c.newBuffer(pFilt.data(), 6 * padVol * sizeof(float));

    // Build/cache MPSGraph per volume dimensions
    static MPSGraph* cachedGraph = nil;
    static MPSGraphTensor* cachedVolIn = nil;
    static MPSGraphTensor* cachedFiltIn = nil;
    static MPSGraphTensor* cachedResult = nil;
    static int cachedW = 0, cachedH = 0, cachedD = 0;

    if (!cachedGraph || cachedW != W || cachedH != H || cachedD != D) {
        cachedGraph = [[MPSGraph alloc] init];
        cachedW = W; cachedH = H; cachedD = D;

        NSArray<NSNumber*>* vShape = @[@1, @(padD), @(padH), @(padW)];
        NSArray<NSNumber*>* fShape = @[@6, @(padD), @(padH), @(padW)];

        cachedVolIn = [cachedGraph placeholderWithShape:vShape
                                              dataType:MPSDataTypeFloat32 name:@"v"];
        cachedFiltIn = [cachedGraph placeholderWithShape:fShape
                                               dataType:MPSDataTypeFloat32 name:@"f"];

        NSArray<NSNumber*>* axes = @[@1, @2, @3];

        MPSGraphFFTDescriptor* fwd = [MPSGraphFFTDescriptor descriptor];
        fwd.inverse = NO;
        fwd.scalingMode = MPSGraphFFTScalingModeNone;

        MPSGraphFFTDescriptor* inv = [MPSGraphFFTDescriptor descriptor];
        inv.inverse = YES;
        inv.scalingMode = MPSGraphFFTScalingModeSize;

        // Forward FFTs (volume broadcasts across batch dim of filters)
        MPSGraphTensor* vFFT = [cachedGraph realToHermiteanFFTWithTensor:cachedVolIn
                                    axes:axes descriptor:fwd name:@"vfft"];
        MPSGraphTensor* fFFT = [cachedGraph realToHermiteanFFTWithTensor:cachedFiltIn
                                    axes:axes descriptor:fwd name:@"ffft"];

        // Complex multiply: [1,...] × [6,...] → [6,...] via broadcast
        MPSGraphTensor* prod = [cachedGraph multiplicationWithPrimaryTensor:vFFT
                                    secondaryTensor:fFFT name:@"prod"];

        // Inverse FFT → [6, padD, padH, padW] real
        MPSGraphTensor* conv = [cachedGraph HermiteanToRealFFTWithTensor:prod
                                    axes:axes descriptor:inv name:@"ifft"];

        // Crop to [6, D, H, W]
        cachedResult = [cachedGraph sliceTensor:conv
                            starts:@[@0, @0, @0, @0]
                            ends:@[@6, @(D), @(H), @(W)]
                            strides:@[@1, @1, @1, @1]
                            name:@"crop"];
    }

    // Create tensor data from MTLBuffers
    MPSGraphTensorData* volTD = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:volBuf
                    shape:@[@1, @(padD), @(padH), @(padW)]
                 dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* filtTD = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:filtBuf
                    shape:@[@6, @(padD), @(padH), @(padW)]
                 dataType:MPSDataTypeFloat32];

    // Execute graph (single call does all 6 convolutions)
    TIMER_START(conv3d_graph);
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
        [cachedGraph runWithMTLCommandQueue:c.queue
            feeds:@{cachedVolIn: volTD, cachedFiltIn: filtTD}
            targetTensors:@[cachedResult]
            targetOperations:nil];
    TIMER_END(conv3d_graph, "  conv3D FFT: graph exec");

    // Read results [6, D, H, W] → interleave into 3 float2 response buffers
    TIMER_START(conv3d_readback);
    MPSGraphTensorData* resTD = results[cachedResult];
    std::vector<float> resData(6 * vol);
    [[resTD mpsndarray] readBytes:resData.data() strideBytes:nil];

    float* r1 = (float*)[resp1 contents];
    float* r2 = (float*)[resp2 contents];
    float* r3 = (float*)[resp3 contents];
    for (int i = 0; i < vol; i++) {
        r1[2*i]   = resData[0 * vol + i];  // filter1 real
        r1[2*i+1] = resData[1 * vol + i];  // filter1 imag
        r2[2*i]   = resData[2 * vol + i];  // filter2 real
        r2[2*i+1] = resData[3 * vol + i];  // filter2 imag
        r3[2*i]   = resData[4 * vol + i];  // filter3 real
        r3[2*i+1] = resData[5 * vol + i];  // filter3 imag
    }
    TIMER_END(conv3d_readback, "  conv3D FFT: readback");
    TIMER_END(conv3d_fft, "  conv3D FFT total");
}

#else  // Spatial convolution via texture3D (default)

void nonseparableConvolution3D(
    id<MTLBuffer> resp1, id<MTLBuffer> resp2, id<MTLBuffer> resp3,
    id<MTLBuffer> volume,
    const float* filterReal1, const float* filterImag1,
    const float* filterReal2, const float* filterImag2,
    const float* filterReal3, const float* filterImag3,
    int W, int H, int D)
{
    TIMER_START(conv3d);
    auto& c = ctx();

    // Create texture3D from volume buffer (hardware-cached reads)
    id<MTLTexture> tex = c.newTexture3D(W, H, D);
    c.copyBufferToTexture(volume, tex, W, H, D);

    // Upload full 7x7x7 filter coefficients (343 floats each)
    id<MTLBuffer> f1r = c.newBuffer(filterReal1, 343 * sizeof(float));
    id<MTLBuffer> f1i = c.newBuffer(filterImag1, 343 * sizeof(float));
    id<MTLBuffer> f2r = c.newBuffer(filterReal2, 343 * sizeof(float));
    id<MTLBuffer> f2i = c.newBuffer(filterImag2, 343 * sizeof(float));
    id<MTLBuffer> f3r = c.newBuffer(filterReal3, 343 * sizeof(float));
    id<MTLBuffer> f3i = c.newBuffer(filterImag3, 343 * sizeof(float));

    Dims dims = {W, H, D};
    id<MTLBuffer> dimBuf = c.newBuffer(&dims, sizeof(Dims));

    // Single dispatch — full 7x7x7 convolution per thread, texture-cached
    auto ps = c.getPipeline("nonseparableConv3D_Full");
    id<MTLCommandBuffer> cb = [c.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ps];
    [enc setBuffer:resp1 offset:0 atIndex:0];
    [enc setBuffer:resp2 offset:0 atIndex:1];
    [enc setBuffer:resp3 offset:0 atIndex:2];
    [enc setTexture:tex atIndex:0];
    [enc setBuffer:f1r offset:0 atIndex:3];
    [enc setBuffer:f1i offset:0 atIndex:4];
    [enc setBuffer:f2r offset:0 atIndex:5];
    [enc setBuffer:f2i offset:0 atIndex:6];
    [enc setBuffer:f3r offset:0 atIndex:7];
    [enc setBuffer:f3i offset:0 atIndex:8];
    [enc setBuffer:dimBuf offset:0 atIndex:9];
    dispatch3D(enc, ps, W, H, D);

    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    TIMER_END(conv3d, "  conv3D total");
}

#endif  // USE_MPSGRAPH_FFT

// ============================================================
//  Separable smoothing (3-pass: rows, columns, rods)
// ============================================================

void performSmoothing(id<MTLBuffer> output, id<MTLBuffer> input, int W, int H, int D,
                      const float* smoothingFilter) {
    TIMER_START(smooth);
    auto& c = ctx();
    Dims dims = {W, H, D};
    id<MTLBuffer> dimBuf = c.newBuffer(&dims, sizeof(Dims));
    id<MTLBuffer> filterBuf = c.newBuffer(smoothingFilter, 9 * sizeof(float));
    id<MTLBuffer> temp1 = c.newBuffer(W * H * D * sizeof(float));
    id<MTLBuffer> temp2 = c.newBuffer(W * H * D * sizeof(float));

    id<MTLCommandBuffer> cb = [c.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

    // Rows (along y)
    auto ps1 = c.getPipeline("separableConvRows");
    [enc setComputePipelineState:ps1];
    [enc setBuffer:temp1 offset:0 atIndex:0];
    [enc setBuffer:input offset:0 atIndex:1];
    [enc setBuffer:filterBuf offset:0 atIndex:2];
    [enc setBuffer:dimBuf offset:0 atIndex:3];
    dispatch3D(enc, ps1, W, H, D);

    // Barrier: pass 2 reads temp1 written by pass 1
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // Columns (along x)
    auto ps2 = c.getPipeline("separableConvColumns");
    [enc setComputePipelineState:ps2];
    [enc setBuffer:temp2 offset:0 atIndex:0];
    [enc setBuffer:temp1 offset:0 atIndex:1];
    [enc setBuffer:filterBuf offset:0 atIndex:2];
    [enc setBuffer:dimBuf offset:0 atIndex:3];
    dispatch3D(enc, ps2, W, H, D);

    // Barrier: pass 3 reads temp2 written by pass 2
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // Rods (along z)
    auto ps3 = c.getPipeline("separableConvRods");
    [enc setComputePipelineState:ps3];
    [enc setBuffer:output offset:0 atIndex:0];
    [enc setBuffer:temp2 offset:0 atIndex:1];
    [enc setBuffer:filterBuf offset:0 atIndex:2];
    [enc setBuffer:dimBuf offset:0 atIndex:3];
    dispatch3D(enc, ps3, W, H, D);

    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    TIMER_END(smooth, "  smoothing 3-pass");
}

// In-place smoothing
void performSmoothingInPlace(id<MTLBuffer> volume, int W, int H, int D,
                             const float* smoothingFilter) {
    auto& c = ctx();
    id<MTLBuffer> output = c.newBuffer(W * H * D * sizeof(float));
    performSmoothing(output, volume, W, H, D, smoothingFilter);
    // GPU-to-GPU copy (avoids CPU roundtrip)
    NSUInteger bytes = W * H * D * sizeof(float);
    id<MTLCommandBuffer> cb = [c.queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:output sourceOffset:0 toBuffer:volume destinationOffset:0 size:bytes];
    [blit endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

// Encode smoothing + blit-copy-back into an existing command buffer.
// Uses pre-allocated temp buffers to avoid per-call allocation.
void encodeSmoothingInPlace(id<MTLCommandBuffer> cb,
                            id<MTLBuffer> volume, int W, int H, int D,
                            id<MTLBuffer> filterBuf, id<MTLBuffer> dimBuf,
                            id<MTLBuffer> temp1, id<MTLBuffer> temp2,
                            id<MTLBuffer> output) {
    auto& c = ctx();

    // Compute encoder: 3-pass separable convolution
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

    auto ps1 = c.getPipeline("separableConvRows");
    [enc setComputePipelineState:ps1];
    [enc setBuffer:temp1 offset:0 atIndex:0];
    [enc setBuffer:volume offset:0 atIndex:1];
    [enc setBuffer:filterBuf offset:0 atIndex:2];
    [enc setBuffer:dimBuf offset:0 atIndex:3];
    dispatch3D(enc, ps1, W, H, D);

    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    auto ps2 = c.getPipeline("separableConvColumns");
    [enc setComputePipelineState:ps2];
    [enc setBuffer:temp2 offset:0 atIndex:0];
    [enc setBuffer:temp1 offset:0 atIndex:1];
    [enc setBuffer:filterBuf offset:0 atIndex:2];
    [enc setBuffer:dimBuf offset:0 atIndex:3];
    dispatch3D(enc, ps2, W, H, D);

    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    auto ps3 = c.getPipeline("separableConvRods");
    [enc setComputePipelineState:ps3];
    [enc setBuffer:output offset:0 atIndex:0];
    [enc setBuffer:temp2 offset:0 atIndex:1];
    [enc setBuffer:filterBuf offset:0 atIndex:2];
    [enc setBuffer:dimBuf offset:0 atIndex:3];
    dispatch3D(enc, ps3, W, H, D);

    [enc endEncoding];

    // Blit encoder: copy output back to volume
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    NSUInteger bytes = W * H * D * sizeof(float);
    [blit copyFromBuffer:output sourceOffset:0 toBuffer:volume destinationOffset:0 size:bytes];
    [blit endEncoding];
}

// Batch multiple in-place smoothings into a single command buffer.
// Dramatically reduces CB creation overhead for the 6/9/3 smoothing batches.
void batchSmoothInPlace(std::initializer_list<id<MTLBuffer>> volumes,
                        int W, int H, int D, const float* smoothingFilter) {
    auto& c = ctx();
    int vol = W * H * D;

    // Pre-allocate shared temp buffers (reused across all smoothings in this batch)
    id<MTLBuffer> temp1 = c.newBuffer(vol * sizeof(float));
    id<MTLBuffer> temp2 = c.newBuffer(vol * sizeof(float));
    id<MTLBuffer> output = c.newBuffer(vol * sizeof(float));

    Dims dims = {W, H, D};
    id<MTLBuffer> dimBuf = c.newBuffer(&dims, sizeof(Dims));
    id<MTLBuffer> filterBuf = c.newBuffer(smoothingFilter, 9 * sizeof(float));

    id<MTLCommandBuffer> cb = [c.queue commandBuffer];
    for (id<MTLBuffer> vol_buf : volumes) {
        encodeSmoothingInPlace(cb, vol_buf, W, H, D,
                               filterBuf, dimBuf, temp1, temp2, output);
    }
    [cb commit];
    [cb waitUntilCompleted];
}

// ============================================================
//  Create smoothing filter (Gaussian, same as BROCCOLI)
// ============================================================

void createSmoothingFilter(float* filter, float sigma) {
    float sum = 0;
    for (int i = 0; i < 9; i++) {
        float x = float(i) - 4.0f;
        filter[i] = expf(-0.5f * x * x / (sigma * sigma));
        sum += filter[i];
    }
    for (int i = 0; i < 9; i++) filter[i] /= sum;
}

// ============================================================
//  Rescale volume (change voxel size)
// ============================================================

id<MTLBuffer> rescaleVolume(id<MTLBuffer> input, int srcW, int srcH, int srcD,
                            int dstW, int dstH, int dstD,
                            float scaleX, float scaleY, float scaleZ) {
    auto& c = ctx();
    id<MTLTexture> tex = c.newTexture3D(srcW, srcH, srcD);
    c.copyBufferToTexture(input, tex, srcW, srcH, srcD);

    id<MTLBuffer> output = c.newBuffer(dstW * dstH * dstD * sizeof(float));
    fillBuffer(output, 0.0f, dstW * dstH * dstD);

    Dims dims = {dstW, dstH, dstD};
    id<MTLBuffer> dimBuf = c.newBuffer(&dims, sizeof(Dims));
    id<MTLBuffer> sxBuf = c.newBuffer(&scaleX, sizeof(float));
    id<MTLBuffer> syBuf = c.newBuffer(&scaleY, sizeof(float));
    id<MTLBuffer> szBuf = c.newBuffer(&scaleZ, sizeof(float));

    auto ps = c.getPipeline("rescaleVolumeLinear");
    id<MTLCommandBuffer> cb = [c.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ps];
    [enc setBuffer:output offset:0 atIndex:0];
    [enc setTexture:tex atIndex:0];
    [enc setBuffer:sxBuf offset:0 atIndex:1];
    [enc setBuffer:syBuf offset:0 atIndex:2];
    [enc setBuffer:szBuf offset:0 atIndex:3];
    [enc setBuffer:dimBuf offset:0 atIndex:4];
    dispatch3D(enc, ps, dstW, dstH, dstD);
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    return output;
}

// ============================================================
//  Copy volume to new dimensions (crop/pad)
// ============================================================

id<MTLBuffer> copyVolumeToNew(id<MTLBuffer> src,
                               int srcW, int srcH, int srcD,
                               int dstW, int dstH, int dstD,
                               int mmZCut, float voxelSizeZ) {
    auto& c = ctx();
    id<MTLBuffer> dst = c.newBuffer(dstW * dstH * dstD * sizeof(float));
    fillBuffer(dst, 0.0f, dstW * dstH * dstD);

    struct CopyParams {
        int newW, newH, newD;
        int srcW, srcH, srcD;
        int xDiff, yDiff, zDiff;
        int mmZCut;
        float voxelSizeZ;
    };

    CopyParams cp = {
        dstW, dstH, dstD,
        srcW, srcH, srcD,
        srcW - dstW, srcH - dstH, srcD - dstD,
        mmZCut, voxelSizeZ
    };

    id<MTLBuffer> paramBuf = c.newBuffer(&cp, sizeof(CopyParams));

    // Dispatch over the larger of src/dst dimensions (matching OpenCL's mymax)
    // Kernel has bounds checks for both src and dst coordinates
    int dispW = std::max(srcW, dstW);
    int dispH = std::max(srcH, dstH);
    int dispD = std::max(srcD, dstD);

    auto ps = c.getPipeline("copyVolumeToNew");
    id<MTLCommandBuffer> cb = [c.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ps];
    [enc setBuffer:dst offset:0 atIndex:0];
    [enc setBuffer:src offset:0 atIndex:1];
    [enc setBuffer:paramBuf offset:0 atIndex:2];
    dispatch3D(enc, ps, dispW, dispH, dispD);
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    return dst;
}

// ============================================================
//  Change volume resolution and size
// ============================================================

id<MTLBuffer> changeVolumesResolutionAndSize(
    id<MTLBuffer> input, int srcW, int srcH, int srcD,
    VoxelSize srcVox, int dstW, int dstH, int dstD,
    VoxelSize dstVox, int mmZCut)
{
    // Step 1: Rescale to match voxel sizes
    float scaleX = srcVox.x / dstVox.x;
    float scaleY = srcVox.y / dstVox.y;
    float scaleZ = srcVox.z / dstVox.z;

    int interpW = (int)roundf(srcW * scaleX);
    int interpH = (int)roundf(srcH * scaleY);
    int interpD = (int)roundf(srcD * scaleZ);

    // Use fence-post corrected scale factors: (srcDim-1)/(dstDim-1)
    // This matches OpenCL's VOXEL_DIFFERENCE calculation and ensures
    // the last source voxel maps exactly to the last destination voxel
    float voxDiffX = (float)(srcW - 1) / (float)(interpW - 1);
    float voxDiffY = (float)(srcH - 1) / (float)(interpH - 1);
    float voxDiffZ = (float)(srcD - 1) / (float)(interpD - 1);

    id<MTLBuffer> interpolated = rescaleVolume(input, srcW, srcH, srcD,
                                                interpW, interpH, interpD,
                                                voxDiffX, voxDiffY, voxDiffZ);

    // Step 2: Copy to target dimensions (crop/pad)
    return copyVolumeToNew(interpolated, interpW, interpH, interpD,
                           dstW, dstH, dstD, mmZCut, dstVox.z);
}

// ============================================================
//  Center-of-mass calculation (CPU-side)
// ============================================================

void centerOfMass(const float* vol, int W, int H, int D,
                  float& cx, float& cy, float& cz) {
    double sum = 0, sx = 0, sy = 0, sz = 0;
    for (int z = 0; z < D; z++)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                float v = vol[x + y * W + z * W * H];
                if (v > 0) {
                    sum += v;
                    sx += v * x;
                    sy += v * y;
                    sz += v * z;
                }
            }
    if (sum > 0) {
        cx = sx / sum;
        cy = sy / sum;
        cz = sz / sum;
    } else {
        cx = W * 0.5f;
        cy = H * 0.5f;
        cz = D * 0.5f;
    }
}

// ============================================================
//  Affine interpolation (GPU)
// ============================================================

void interpolateLinear(id<MTLBuffer> output, id<MTLBuffer> volume,
                       const float* params, int W, int H, int D) {
    auto& c = ctx();
    id<MTLTexture> tex = c.newTexture3D(W, H, D);
    c.copyBufferToTexture(volume, tex, W, H, D);

    Dims dims = {W, H, D};
    id<MTLBuffer> dimBuf = c.newBuffer(&dims, sizeof(Dims));
    id<MTLBuffer> paramBuf = c.newBuffer(params, 12 * sizeof(float));

    auto ps = c.getPipeline("interpolateLinearLinear");
    id<MTLCommandBuffer> cb = [c.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ps];
    [enc setBuffer:output offset:0 atIndex:0];
    [enc setTexture:tex atIndex:0];
    [enc setBuffer:paramBuf offset:0 atIndex:1];
    [enc setBuffer:dimBuf offset:0 atIndex:2];
    dispatch3D(enc, ps, W, H, D);
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

// ============================================================
//  Nonlinear interpolation (GPU)
// ============================================================

void interpolateNonLinear(id<MTLBuffer> output, id<MTLBuffer> volume,
                          id<MTLBuffer> dispX, id<MTLBuffer> dispY, id<MTLBuffer> dispZ,
                          int W, int H, int D) {
    auto& c = ctx();
    id<MTLTexture> tex = c.newTexture3D(W, H, D);
    c.copyBufferToTexture(volume, tex, W, H, D);

    Dims dims = {W, H, D};
    id<MTLBuffer> dimBuf = c.newBuffer(&dims, sizeof(Dims));

    auto ps = c.getPipeline("interpolateLinearNonLinear");
    id<MTLCommandBuffer> cb = [c.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ps];
    [enc setBuffer:output offset:0 atIndex:0];
    [enc setTexture:tex atIndex:0];
    [enc setBuffer:dispX offset:0 atIndex:1];
    [enc setBuffer:dispY offset:0 atIndex:2];
    [enc setBuffer:dispZ offset:0 atIndex:3];
    [enc setBuffer:dimBuf offset:0 atIndex:4];
    dispatch3D(enc, ps, W, H, D);
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

// ============================================================
//  Add linear + nonlinear displacement (GPU)
// ============================================================

void addLinearNonLinearDisplacement(id<MTLBuffer> dispX, id<MTLBuffer> dispY, id<MTLBuffer> dispZ,
                                    const float* params, int W, int H, int D) {
    auto& c = ctx();
    Dims dims = {W, H, D};
    id<MTLBuffer> dimBuf = c.newBuffer(&dims, sizeof(Dims));
    id<MTLBuffer> paramBuf = c.newBuffer(params, 12 * sizeof(float));

    auto ps = c.getPipeline("addLinearAndNonLinearDisplacement");
    id<MTLCommandBuffer> cb = [c.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ps];
    [enc setBuffer:dispX offset:0 atIndex:0];
    [enc setBuffer:dispY offset:0 atIndex:1];
    [enc setBuffer:dispZ offset:0 atIndex:2];
    [enc setBuffer:paramBuf offset:0 atIndex:3];
    [enc setBuffer:dimBuf offset:0 atIndex:4];
    dispatch3D(enc, ps, W, H, D);
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

// ============================================================
//  Solve equation system (CPU, via Accelerate LAPACK)
// ============================================================

void solveEquationSystem(float* A, float* h, double* params, int n) {
    // Convert to double for precision
    double Ad[144], hd[12];
    for (int i = 0; i < n * n; i++) Ad[i] = A[i];
    for (int i = 0; i < n; i++) hd[i] = h[i];

    // Mirror symmetric matrix
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            Ad[i * n + j] = Ad[j * n + i];

    // Solve using simple Gaussian elimination (avoids LAPACK deprecation issues)
    // Make augmented matrix: [A | h]
    double aug[12][13];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            aug[i][j] = Ad[j * n + i]; // column-major to row-major
        aug[i][n] = hd[i];
    }

    // Forward elimination with partial pivoting
    for (int col = 0; col < n; col++) {
        // Find pivot
        int pivotRow = col;
        double pivotVal = fabs(aug[col][col]);
        for (int row = col + 1; row < n; row++) {
            if (fabs(aug[row][col]) > pivotVal) {
                pivotVal = fabs(aug[row][col]);
                pivotRow = row;
            }
        }
        if (pivotVal < 1e-30) {
            for (int i = 0; i < n; i++) params[i] = 0;
            return;
        }
        if (pivotRow != col) {
            for (int j = 0; j <= n; j++)
                std::swap(aug[col][j], aug[pivotRow][j]);
        }
        // Eliminate
        for (int row = col + 1; row < n; row++) {
            double factor = aug[row][col] / aug[col][col];
            for (int j = col; j <= n; j++)
                aug[row][j] -= factor * aug[col][j];
        }
    }

    // Back substitution
    for (int row = n - 1; row >= 0; row--) {
        double sum = aug[row][n];
        for (int j = row + 1; j < n; j++)
            sum -= aug[row][j] * params[j];
        params[row] = sum / aug[row][row];
    }
}

// ============================================================
//  Affine parameter composition via 4x4 matrix multiplication
//  Matches BROCCOLI's AddAffineRegistrationParameters
// ============================================================

// Build a 4x4 affine matrix from 12 parameters:
//   (p3+1  p4   p5   tx)
//   (p6    p7+1 p8   ty)
//   (p9    p10  p11+1 tz)
//   (0     0    0     1 )
static void paramsToMatrix(const float* p, double M[4][4], float translationScale = 1.0f) {
    M[0][0] = p[3] + 1.0; M[0][1] = p[4];       M[0][2] = p[5];       M[0][3] = p[0] * translationScale;
    M[1][0] = p[6];       M[1][1] = p[7] + 1.0;  M[1][2] = p[8];       M[1][3] = p[1] * translationScale;
    M[2][0] = p[9];       M[2][1] = p[10];       M[2][2] = p[11] + 1.0; M[2][3] = p[2] * translationScale;
    M[3][0] = 0;          M[3][1] = 0;           M[3][2] = 0;           M[3][3] = 1.0;
}

static void matrixToParams(const double M[4][4], float* p) {
    p[0] = (float)M[0][3];
    p[1] = (float)M[1][3];
    p[2] = (float)M[2][3];
    p[3] = (float)(M[0][0] - 1.0);
    p[4] = (float)M[0][1];
    p[5] = (float)M[0][2];
    p[6] = (float)M[1][0];
    p[7] = (float)(M[1][1] - 1.0);
    p[8] = (float)M[1][2];
    p[9] = (float)M[2][0];
    p[10] = (float)M[2][1];
    p[11] = (float)(M[2][2] - 1.0);
}

static void matMul4x4(const double A[4][4], const double B[4][4], double C[4][4]) {
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            C[i][j] = 0;
            for (int k = 0; k < 4; k++)
                C[i][j] += A[i][k] * B[k][j];
        }
}

// Compose: result = New * Old (result stored in oldParams)
void composeAffineParams(float* oldParams, const float* newParams) {
    double O[4][4], N[4][4], T[4][4];
    paramsToMatrix(oldParams, O);
    paramsToMatrix(newParams, N);
    matMul4x4(N, O, T);
    matrixToParams(T, oldParams);
}

// Compose for next scale: translations scaled by 2x before matrix multiply
void composeAffineParamsNextScale(float* oldParams, const float* newParams) {
    double O[4][4], N[4][4], T[4][4];
    paramsToMatrix(oldParams, O, 2.0f);  // old translations * 2
    paramsToMatrix(newParams, N, 2.0f);  // new translations * 2
    matMul4x4(N, O, T);
    matrixToParams(T, oldParams);
}

// 3-arg version: result = New * Old, stored in resultParams
void composeAffineParams3(float* resultParams, const float* oldParams, const float* newParams) {
    double O[4][4], N[4][4], T[4][4];
    paramsToMatrix(oldParams, O);
    paramsToMatrix(newParams, N);
    matMul4x4(N, O, T);
    matrixToParams(T, resultParams);
}

// ============================================================
//  Change volume size (multi-scale rescaling)
// ============================================================

id<MTLBuffer> changeVolumeSize(id<MTLBuffer> input, int srcW, int srcH, int srcD,
                                int dstW, int dstH, int dstD) {
    // Fence-post corrected scale: (srcDim-1)/(dstDim-1) matches OpenCL
    float scaleX = float(srcW - 1) / float(dstW - 1);
    float scaleY = float(srcH - 1) / float(dstH - 1);
    float scaleZ = float(srcD - 1) / float(dstD - 1);
    return rescaleVolume(input, srcW, srcH, srcD, dstW, dstH, dstD,
                         scaleX, scaleY, scaleZ);
}

// ============================================================
//  LINEAR REGISTRATION
// ============================================================

// Single scale, single iteration batch
void alignTwoVolumesLinear(
    id<MTLBuffer> alignedVolume,
    id<MTLBuffer> referenceVolume,
    const QuadratureFilters& filters,
    int W, int H, int D,
    int filterSize,
    int numIterations,
    float* registrationParams,  // 12 params, accumulated
    bool verbose)
{
    auto& c = ctx();
    int vol = W * H * D;

    // Save the original aligned volume — re-interpolate from this each iteration
    id<MTLBuffer> originalAligned = c.newBuffer(vol * sizeof(float));
    memcpy([originalAligned contents], [alignedVolume contents], vol * sizeof(float));

    // Reset params to zero — each scale solves independently
    // (BROCCOLI resets h_Registration_Parameters_Align_Two_Volumes to 0 at start)
    for (int i = 0; i < 12; i++) registrationParams[i] = 0.0f;

    // Allocate filter response buffers (complex, float2)
    id<MTLBuffer> q11 = c.newBuffer(vol * sizeof(float) * 2);
    id<MTLBuffer> q12 = c.newBuffer(vol * sizeof(float) * 2);
    id<MTLBuffer> q13 = c.newBuffer(vol * sizeof(float) * 2);
    id<MTLBuffer> q21 = c.newBuffer(vol * sizeof(float) * 2);
    id<MTLBuffer> q22 = c.newBuffer(vol * sizeof(float) * 2);
    id<MTLBuffer> q23 = c.newBuffer(vol * sizeof(float) * 2);

    // Phase/certainty buffers
    id<MTLBuffer> phaseDiff = c.newBuffer(vol * sizeof(float));
    id<MTLBuffer> certainties = c.newBuffer(vol * sizeof(float));
    id<MTLBuffer> phaseGrad = c.newBuffer(vol * sizeof(float));

    // A-matrix / h-vector buffers — must be zero-initialized
    int HD = H * D;
    id<MTLBuffer> A2D = c.newBuffer(30 * HD * sizeof(float));
    id<MTLBuffer> A1D = c.newBuffer(30 * D * sizeof(float));
    id<MTLBuffer> Amat = c.newBuffer(144 * sizeof(float));
    id<MTLBuffer> h2D = c.newBuffer(12 * HD * sizeof(float));
    id<MTLBuffer> h1D = c.newBuffer(12 * D * sizeof(float));
    id<MTLBuffer> hvec = c.newBuffer(12 * sizeof(float));
    memset([A2D contents], 0, 30 * HD * sizeof(float));
    memset([A1D contents], 0, 30 * D * sizeof(float));
    memset([h2D contents], 0, 12 * HD * sizeof(float));
    memset([h1D contents], 0, 12 * D * sizeof(float));

    // Filter reference volume once
    TIMER_START(lin_ref_conv);
    nonseparableConvolution3D(q11, q12, q13, referenceVolume,
        filters.linearReal[0].data(), filters.linearImag[0].data(),
        filters.linearReal[1].data(), filters.linearImag[1].data(),
        filters.linearReal[2].data(), filters.linearImag[2].data(),
        W, H, D);
    TIMER_END(lin_ref_conv, "  linear: ref convolution");

    Dims dims = {W, H, D};
    id<MTLBuffer> dimBuf = c.newBuffer(&dims, sizeof(Dims));

    for (int iter = 0; iter < numIterations; iter++) {
      @autoreleasepool {
        TIMER_START(lin_iter);
        // Filter aligned volume
        TIMER_START(lin_align_conv);
        nonseparableConvolution3D(q21, q22, q23, alignedVolume,
            filters.linearReal[0].data(), filters.linearImag[0].data(),
            filters.linearReal[1].data(), filters.linearImag[1].data(),
            filters.linearReal[2].data(), filters.linearImag[2].data(),
            W, H, D);
        TIMER_END(lin_align_conv, "  linear: aligned convolution");

        // Process each direction (X, Y, Z)
        struct { int dirOff; int hOff; id<MTLBuffer> q1; id<MTLBuffer> q2; const char* gradKernel; }
        dirs[3] = {
            {0, 0, q11, q21, "calculatePhaseGradientsX"},
            {10, 1, q12, q22, "calculatePhaseGradientsY"},
            {20, 2, q13, q23, "calculatePhaseGradientsZ"},
        };

        // Zero intermediate buffers + 3-direction phase/grad/Amat in single CB
        memset([h2D contents], 0, 12 * HD * sizeof(float));
        memset([A2D contents], 0, 30 * HD * sizeof(float));

        TIMER_START(lin_phase_amat);
        {
            struct AParams {
                int W, H, D, filterSize, dirOff, hOff;
            };

            id<MTLCommandBuffer> cb = [c.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

            // Zero phase/certainty/grad buffers
            encodeFill(enc, phaseDiff, 0.0f, vol);
            encodeFill(enc, certainties, 0.0f, vol);
            encodeFill(enc, phaseGrad, 0.0f, vol);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            auto psPhaseDiff = c.getPipeline("calculatePhaseDifferencesAndCertainties");
            auto psAmat2D = c.getPipeline("calculateAMatrixAndHVector2D");

            for (int d = 0; d < 3; d++) {
                // Phase differences + certainties
                [enc setComputePipelineState:psPhaseDiff];
                [enc setBuffer:phaseDiff offset:0 atIndex:0];
                [enc setBuffer:certainties offset:0 atIndex:1];
                [enc setBuffer:dirs[d].q1 offset:0 atIndex:2];
                [enc setBuffer:dirs[d].q2 offset:0 atIndex:3];
                [enc setBuffer:dimBuf offset:0 atIndex:4];
                dispatch3D(enc, psPhaseDiff, W, H, D);

                // Phase gradients (reads q1/q2, writes phaseGrad — independent of phaseDiff)
                auto psGrad = c.getPipeline(dirs[d].gradKernel);
                [enc setComputePipelineState:psGrad];
                [enc setBuffer:phaseGrad offset:0 atIndex:0];
                [enc setBuffer:dirs[d].q1 offset:0 atIndex:1];
                [enc setBuffer:dirs[d].q2 offset:0 atIndex:2];
                [enc setBuffer:dimBuf offset:0 atIndex:3];
                dispatch3D(enc, psGrad, W, H, D);

                // Barrier: Amat reads phaseDiff, phaseGrad, certainties
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                // A-matrix and h-vector 2D values
                AParams ap = {W, H, D, filterSize, dirs[d].dirOff, dirs[d].hOff};
                id<MTLBuffer> apBuf = c.newBuffer(&ap, sizeof(AParams));
                [enc setComputePipelineState:psAmat2D];
                [enc setBuffer:A2D offset:0 atIndex:0];
                [enc setBuffer:h2D offset:0 atIndex:1];
                [enc setBuffer:phaseDiff offset:0 atIndex:2];
                [enc setBuffer:phaseGrad offset:0 atIndex:3];
                [enc setBuffer:certainties offset:0 atIndex:4];
                [enc setBuffer:apBuf offset:0 atIndex:5];
                dispatch2D(enc, psAmat2D, H, D);

                // Barrier before next direction overwrites phaseDiff/phaseGrad/certainties
                if (d < 2) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }

            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }
        TIMER_END(lin_phase_amat, "  linear: phase/grad/Amat 3dirs");

        // Reduce A-matrix: 2D -> 1D -> final (single CB with barriers)
        TIMER_START(lin_reduce);
        {
            id<MTLBuffer> hBuf = c.newBuffer(&H, sizeof(int));
            id<MTLBuffer> dBuf = c.newBuffer(&D, sizeof(int));
            id<MTLBuffer> fsBuf = c.newBuffer(&filterSize, sizeof(int));

            id<MTLCommandBuffer> cb = [c.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

            // A-matrix 1D + h-vector 1D (independent, both read from 2D)
            auto psA1D = c.getPipeline("calculateAMatrix1D");
            [enc setComputePipelineState:psA1D];
            [enc setBuffer:A1D offset:0 atIndex:0];
            [enc setBuffer:A2D offset:0 atIndex:1];
            [enc setBuffer:hBuf offset:0 atIndex:2];
            [enc setBuffer:dBuf offset:0 atIndex:3];
            [enc setBuffer:fsBuf offset:0 atIndex:4];
            dispatch2D(enc, psA1D, D, 30);

            auto psH1D = c.getPipeline("calculateHVector1D");
            [enc setComputePipelineState:psH1D];
            [enc setBuffer:h1D offset:0 atIndex:0];
            [enc setBuffer:h2D offset:0 atIndex:1];
            [enc setBuffer:hBuf offset:0 atIndex:2];
            [enc setBuffer:dBuf offset:0 atIndex:3];
            [enc setBuffer:fsBuf offset:0 atIndex:4];
            dispatch2D(enc, psH1D, D, 12);

            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // Zero Amat, then compute final reduction
            encodeFill(enc, Amat, 0.0f, 144);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            auto psAFinal = c.getPipeline("calculateAMatrixFinal");
            [enc setComputePipelineState:psAFinal];
            [enc setBuffer:Amat offset:0 atIndex:0];
            [enc setBuffer:A1D offset:0 atIndex:1];
            [enc setBuffer:dBuf offset:0 atIndex:2];
            [enc setBuffer:fsBuf offset:0 atIndex:3];
            dispatch1D(enc, psAFinal, 30);

            // h-vector final (reads h1D which is already done)
            auto psHFinal = c.getPipeline("calculateHVectorFinal");
            [enc setComputePipelineState:psHFinal];
            [enc setBuffer:hvec offset:0 atIndex:0];
            [enc setBuffer:h1D offset:0 atIndex:1];
            [enc setBuffer:dBuf offset:0 atIndex:2];
            [enc setBuffer:fsBuf offset:0 atIndex:3];
            dispatch1D(enc, psHFinal, 12);

            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }

        TIMER_END(lin_reduce, "  linear: A-matrix reduction");

        // Read back A and h, solve on CPU
        TIMER_START(lin_solve);
        float hA[144], hh[12];
        memcpy(hA, [Amat contents], 144 * sizeof(float));

        // Do h-vector final reduction on CPU to bypass GPU kernel issue
        {
            int fhalf = (filterSize - 1) / 2;
            float* h1Dp = (float*)[h1D contents];
            for (int elem = 0; elem < 12; elem++) {
                float sum = 0;
                for (int z = fhalf; z < D - fhalf; z++) {
                    sum += h1Dp[elem * D + z];
                }
                hh[elem] = sum;
            }
        }

        double paramsDbl[12];
        solveEquationSystem(hA, hh, paramsDbl, 12);

        // Compose parameters via matrix multiplication (BROCCOLI pattern)
        float deltaParams[12];
        for (int i = 0; i < 12; i++) deltaParams[i] = (float)paramsDbl[i];
        composeAffineParams(registrationParams, deltaParams);

        TIMER_END(lin_solve, "  linear: solve+compose");

        // Apply affine transform from original volume (not already-transformed)
        TIMER_START(lin_interp);
        interpolateLinear(alignedVolume, originalAligned, registrationParams, W, H, D);
        TIMER_END(lin_interp, "  linear: interpolation");
        TIMER_END(lin_iter, "  linear: iteration total");
      }
    }
}

// Multi-scale linear registration
// Matches BROCCOLI's AlignTwoVolumesLinearSeveralScales pattern:
// - Reset total params to zero
// - At each scale: downscale from originals, pre-transform with accumulated params,
//   solve independently from zero, compose result via matrix multiplication
void alignTwoVolumesLinearSeveralScales(
    id<MTLBuffer>& alignedVolume,
    id<MTLBuffer> referenceVolume,
    const QuadratureFilters& filters,
    int W, int H, int D,
    int filterSize,
    int numIterations,
    int coarsestScale,
    float* registrationParams,
    bool verbose)
{
    auto& c = ctx();
    int vol = W * H * D;

    // Reset total parameters (BROCCOLI pattern)
    for (int i = 0; i < 12; i++) registrationParams[i] = 0.0f;

    // Keep a copy of the original full-resolution aligned volume
    id<MTLBuffer> originalAligned = c.newBuffer(vol * sizeof(float));
    memcpy([originalAligned contents], [alignedVolume contents], vol * sizeof(float));

    // Start from coarsest scale, work to finest
    TIMER_START(lin_all_scales);
    for (int scale = coarsestScale; scale >= 1; scale /= 2) {
      @autoreleasepool {
        TIMER_START(lin_scale);
        int sW = (int)roundf((float)W / (float)scale);
        int sH = (int)roundf((float)H / (float)scale);
        int sD = (int)roundf((float)D / (float)scale);

        if (sW < 8 || sH < 8 || sD < 8) continue;

        // Downscale both volumes from originals at each scale
        TIMER_START(lin_downscale);
        id<MTLBuffer> scaledRef = (scale == 1) ? referenceVolume :
            changeVolumeSize(referenceVolume, W, H, D, sW, sH, sD);
        id<MTLBuffer> scaledAligned = (scale == 1) ? alignedVolume :
            changeVolumeSize(originalAligned, W, H, D, sW, sH, sD);
        TIMER_END(lin_downscale, "  linear: downscale volumes");

        // For non-coarsest scales: pre-transform with accumulated params
        if (scale < coarsestScale) {
            interpolateLinear(scaledAligned, scaledAligned, registrationParams, sW, sH, sD);
        }

        // Fewer iterations at finest scale (BROCCOLI: ceil(N/5))
        int iters = (scale == 1) ? (int)ceilf((float)numIterations / 5.0f) : numIterations;

        if (verbose) {
            printf("  Linear scale %d: %dx%dx%d, %d iterations\n", scale, sW, sH, sD, iters);
        }

        // Temp params for this scale (starts at zero inside alignTwoVolumesLinear)
        float tempParams[12] = {0};

        alignTwoVolumesLinear(scaledAligned, scaledRef, filters,
                              sW, sH, sD, filterSize, iters,
                              tempParams, verbose);

        char scaleLabel[80];
        snprintf(scaleLabel, sizeof(scaleLabel), "LINEAR SCALE %d total", scale);
        TIMER_END(lin_scale, scaleLabel);

        // Compose this scale's params with accumulated total
        if (scale != 1) {
            // NextScale variant: translations * 2 before matrix multiply
            // After this, total params are at the next finer scale's resolution
            composeAffineParamsNextScale(registrationParams, tempParams);
        } else {
            // Final scale: standard composition (no 2x scaling)
            // Result is in full-resolution coordinates
            composeAffineParams(registrationParams, tempParams);
        }
      }
    }

    TIMER_END(lin_all_scales, "LINEAR REGISTRATION all scales total");

    // Final transform of original volume with complete parameters (full resolution)
    TIMER_START(lin_final_interp);
    interpolateLinear(alignedVolume, originalAligned, registrationParams, W, H, D);
    TIMER_END(lin_final_interp, "LINEAR: final interpolation");
}

// ============================================================
//  NONLINEAR REGISTRATION
// ============================================================

void alignTwoVolumesNonLinear(
    id<MTLBuffer> alignedVolume,
    id<MTLBuffer> referenceVolume,
    const QuadratureFilters& filters,
    int W, int H, int D,
    int numIterations,
    id<MTLBuffer> updateDispX, id<MTLBuffer> updateDispY, id<MTLBuffer> updateDispZ,
    bool verbose)
{
    auto& c = ctx();
    int vol = W * H * D;
    int filterSize = 7;

    // Allocate filter response buffers (6 filters, 2 volumes = 12 complex buffers)
    id<MTLBuffer> q1[6], q2[6];
    for (int i = 0; i < 6; i++) {
        q1[i] = c.newBuffer(vol * sizeof(float) * 2);
        q2[i] = c.newBuffer(vol * sizeof(float) * 2);
    }

    // Tensor components
    id<MTLBuffer> t11 = c.newBuffer(vol * sizeof(float));
    id<MTLBuffer> t12 = c.newBuffer(vol * sizeof(float));
    id<MTLBuffer> t13 = c.newBuffer(vol * sizeof(float));
    id<MTLBuffer> t22 = c.newBuffer(vol * sizeof(float));
    id<MTLBuffer> t23 = c.newBuffer(vol * sizeof(float));
    id<MTLBuffer> t33 = c.newBuffer(vol * sizeof(float));

    // A-matrix (6 unique) and h-vector (3)
    id<MTLBuffer> a11 = c.newBuffer(vol * sizeof(float));
    id<MTLBuffer> a12 = c.newBuffer(vol * sizeof(float));
    id<MTLBuffer> a13 = c.newBuffer(vol * sizeof(float));
    id<MTLBuffer> a22 = c.newBuffer(vol * sizeof(float));
    id<MTLBuffer> a23 = c.newBuffer(vol * sizeof(float));
    id<MTLBuffer> a33 = c.newBuffer(vol * sizeof(float));
    id<MTLBuffer> h1 = c.newBuffer(vol * sizeof(float));
    id<MTLBuffer> h2 = c.newBuffer(vol * sizeof(float));
    id<MTLBuffer> h3 = c.newBuffer(vol * sizeof(float));

    // Tensor norms
    id<MTLBuffer> tensorNorms = c.newBuffer(vol * sizeof(float));

    // Displacement update
    id<MTLBuffer> dux = c.newBuffer(vol * sizeof(float));
    id<MTLBuffer> duy = c.newBuffer(vol * sizeof(float));
    id<MTLBuffer> duz = c.newBuffer(vol * sizeof(float));

    // Save the original aligned volume — re-interpolate from this each iteration
    id<MTLBuffer> originalAligned = c.newBuffer(vol * sizeof(float));
    memcpy([originalAligned contents], [alignedVolume contents], vol * sizeof(float));

    // Filter directions
    id<MTLBuffer> fdxBuf = c.newBuffer(filters.filterDirectionsX, 6 * sizeof(float));
    id<MTLBuffer> fdyBuf = c.newBuffer(filters.filterDirectionsY, 6 * sizeof(float));
    id<MTLBuffer> fdzBuf = c.newBuffer(filters.filterDirectionsZ, 6 * sizeof(float));

    // Smoothing filters
    float smoothTensor[9], smoothEq[9], smoothDisp[9];
    createSmoothingFilter(smoothTensor, 1.0f);
    createSmoothingFilter(smoothEq, 2.0f);
    createSmoothingFilter(smoothDisp, 2.0f);

    Dims dims = {W, H, D};

    // Filter reference volume (once per scale)
    TIMER_START(nl_ref_conv);
    nonseparableConvolution3D(q1[0], q1[1], q1[2], referenceVolume,
        filters.nonlinearReal[0].data(), filters.nonlinearImag[0].data(),
        filters.nonlinearReal[1].data(), filters.nonlinearImag[1].data(),
        filters.nonlinearReal[2].data(), filters.nonlinearImag[2].data(),
        W, H, D);
    nonseparableConvolution3D(q1[3], q1[4], q1[5], referenceVolume,
        filters.nonlinearReal[3].data(), filters.nonlinearImag[3].data(),
        filters.nonlinearReal[4].data(), filters.nonlinearImag[4].data(),
        filters.nonlinearReal[5].data(), filters.nonlinearImag[5].data(),
        W, H, D);
    TIMER_END(nl_ref_conv, "  nonlinear: ref convolution (2x)");

    for (int iter = 0; iter < numIterations; iter++) {
      @autoreleasepool {
        TIMER_START(nl_iter);
        if (verbose) printf("    Nonlinear iter %d/%d\n", iter + 1, numIterations);

        // Filter aligned volume
        TIMER_START(nl_align_conv);
        nonseparableConvolution3D(q2[0], q2[1], q2[2], alignedVolume,
            filters.nonlinearReal[0].data(), filters.nonlinearImag[0].data(),
            filters.nonlinearReal[1].data(), filters.nonlinearImag[1].data(),
            filters.nonlinearReal[2].data(), filters.nonlinearImag[2].data(),
            W, H, D);
        nonseparableConvolution3D(q2[3], q2[4], q2[5], alignedVolume,
            filters.nonlinearReal[3].data(), filters.nonlinearImag[3].data(),
            filters.nonlinearReal[4].data(), filters.nonlinearImag[4].data(),
            filters.nonlinearReal[5].data(), filters.nonlinearImag[5].data(),
            W, H, D);
        TIMER_END(nl_align_conv, "  nonlinear: aligned convolution (2x)");

        // Reset tensors/displacement + compute tensor components (single CB)
        TIMER_START(nl_tensor);
        {
            id<MTLBuffer> dimBuf = c.newBuffer(&dims, sizeof(Dims));
            id<MTLCommandBuffer> cb = [c.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

            // Zero all 9 buffers
            for (auto buf : {t11, t12, t13, t22, t23, t33, dux, duy, duz})
                encodeFill(enc, buf, 0.0f, vol);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // 6 tensor component dispatches (each accumulates into t11..t33)
            auto psTensor = c.getPipeline("calculateTensorComponents");
            for (int f = 0; f < 6; f++) {
                const float* pt = filters.projectionTensors[f];
                id<MTLBuffer> mBufs[6];
                for (int k = 0; k < 6; k++)
                    mBufs[k] = c.newBuffer(&pt[k], sizeof(float));

                [enc setComputePipelineState:psTensor];
                [enc setBuffer:t11 offset:0 atIndex:0];
                [enc setBuffer:t12 offset:0 atIndex:1];
                [enc setBuffer:t13 offset:0 atIndex:2];
                [enc setBuffer:t22 offset:0 atIndex:3];
                [enc setBuffer:t23 offset:0 atIndex:4];
                [enc setBuffer:t33 offset:0 atIndex:5];
                [enc setBuffer:q1[f] offset:0 atIndex:6];
                [enc setBuffer:q2[f] offset:0 atIndex:7];
                for (int k = 0; k < 6; k++)
                    [enc setBuffer:mBufs[k] offset:0 atIndex:8 + k];
                [enc setBuffer:dimBuf offset:0 atIndex:14];
                dispatch3D(enc, psTensor, W, H, D);
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }

            // Tensor norms (pre-smooth)
            auto psNorms = c.getPipeline("calculateTensorNorms");
            [enc setComputePipelineState:psNorms];
            [enc setBuffer:tensorNorms offset:0 atIndex:0];
            [enc setBuffer:t11 offset:0 atIndex:1];
            [enc setBuffer:t12 offset:0 atIndex:2];
            [enc setBuffer:t13 offset:0 atIndex:3];
            [enc setBuffer:t22 offset:0 atIndex:4];
            [enc setBuffer:t23 offset:0 atIndex:5];
            [enc setBuffer:t33 offset:0 atIndex:6];
            [enc setBuffer:dimBuf offset:0 atIndex:7];
            dispatch3D(enc, psNorms, W, H, D);

            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }
        TIMER_END(nl_tensor, "  nonlinear: tensor+norms");

        // Smooth tensor components (batched: 6 smoothings in 1 CB)
        TIMER_START(nl_smooth_tensor);
        batchSmoothInPlace({t11, t12, t13, t22, t23, t33}, W, H, D, smoothTensor);
        TIMER_END(nl_smooth_tensor, "  nonlinear: smooth tensors (6x)");

        // Tensor norms (post-smooth) + normalize (single CB)
        TIMER_START(nl_normalize);
        {
            id<MTLBuffer> dimBuf = c.newBuffer(&dims, sizeof(Dims));
            id<MTLCommandBuffer> cb = [c.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

            auto psNorms = c.getPipeline("calculateTensorNorms");
            [enc setComputePipelineState:psNorms];
            [enc setBuffer:tensorNorms offset:0 atIndex:0];
            [enc setBuffer:t11 offset:0 atIndex:1];
            [enc setBuffer:t12 offset:0 atIndex:2];
            [enc setBuffer:t13 offset:0 atIndex:3];
            [enc setBuffer:t22 offset:0 atIndex:4];
            [enc setBuffer:t23 offset:0 atIndex:5];
            [enc setBuffer:t33 offset:0 atIndex:6];
            [enc setBuffer:dimBuf offset:0 atIndex:7];
            dispatch3D(enc, psNorms, W, H, D);

            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }

        float maxNorm = calculateMax(tensorNorms, W, H, D);
        if (maxNorm > 0) {
            float invMax = 1.0f / maxNorm;
            id<MTLCommandBuffer> cb = [c.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            for (auto buf : {t11, t12, t13, t22, t23, t33})
                encodeMultiply(enc, buf, invMax, vol);
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }
        TIMER_END(nl_normalize, "  nonlinear: norms+normalize");

        // A-matrices and h-vectors (6 filters, single CB with barriers)
        TIMER_START(nl_amat_hvec);
        {
            id<MTLBuffer> dimBuf = c.newBuffer(&dims, sizeof(Dims));
            auto ps = c.getPipeline("calculateAMatricesAndHVectors");

            id<MTLCommandBuffer> cb = [c.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

            for (int f = 0; f < 6; f++) {
                struct MorphonParams { int W, H, D, FILTER; };
                MorphonParams mp = {W, H, D, f};
                id<MTLBuffer> mpBuf = c.newBuffer(&mp, sizeof(MorphonParams));

                [enc setComputePipelineState:ps];
                [enc setBuffer:a11 offset:0 atIndex:0];
                [enc setBuffer:a12 offset:0 atIndex:1];
                [enc setBuffer:a13 offset:0 atIndex:2];
                [enc setBuffer:a22 offset:0 atIndex:3];
                [enc setBuffer:a23 offset:0 atIndex:4];
                [enc setBuffer:a33 offset:0 atIndex:5];
                [enc setBuffer:h1 offset:0 atIndex:6];
                [enc setBuffer:h2 offset:0 atIndex:7];
                [enc setBuffer:h3 offset:0 atIndex:8];
                [enc setBuffer:q1[f] offset:0 atIndex:9];
                [enc setBuffer:q2[f] offset:0 atIndex:10];
                [enc setBuffer:t11 offset:0 atIndex:11];
                [enc setBuffer:t12 offset:0 atIndex:12];
                [enc setBuffer:t13 offset:0 atIndex:13];
                [enc setBuffer:t22 offset:0 atIndex:14];
                [enc setBuffer:t23 offset:0 atIndex:15];
                [enc setBuffer:t33 offset:0 atIndex:16];
                [enc setBuffer:fdxBuf offset:0 atIndex:17];
                [enc setBuffer:fdyBuf offset:0 atIndex:18];
                [enc setBuffer:fdzBuf offset:0 atIndex:19];
                [enc setBuffer:mpBuf offset:0 atIndex:20];
                dispatch3D(enc, ps, W, H, D);
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }

            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }
        TIMER_END(nl_amat_hvec, "  nonlinear: A-matrices+h-vectors (6f)");

        // Smooth A-matrix and h-vector components (batched: 9 in 1 CB)
        TIMER_START(nl_smooth_eq);
        batchSmoothInPlace({a11, a12, a13, a22, a23, a33, h1, h2, h3}, W, H, D, smoothEq);
        TIMER_END(nl_smooth_eq, "  nonlinear: smooth A+h (9x)");

        // Displacement update + smooth + accumulate
        TIMER_START(nl_disp_update);
        {
            id<MTLBuffer> dimBuf = c.newBuffer(&dims, sizeof(Dims));
            auto ps = c.getPipeline("calculateDisplacementUpdate");
            id<MTLCommandBuffer> cb = [c.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:ps];
            [enc setBuffer:dux offset:0 atIndex:0];
            [enc setBuffer:duy offset:0 atIndex:1];
            [enc setBuffer:duz offset:0 atIndex:2];
            [enc setBuffer:a11 offset:0 atIndex:3];
            [enc setBuffer:a12 offset:0 atIndex:4];
            [enc setBuffer:a13 offset:0 atIndex:5];
            [enc setBuffer:a22 offset:0 atIndex:6];
            [enc setBuffer:a23 offset:0 atIndex:7];
            [enc setBuffer:a33 offset:0 atIndex:8];
            [enc setBuffer:h1 offset:0 atIndex:9];
            [enc setBuffer:h2 offset:0 atIndex:10];
            [enc setBuffer:h3 offset:0 atIndex:11];
            [enc setBuffer:dimBuf offset:0 atIndex:12];
            dispatch3D(enc, ps, W, H, D);
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }
        TIMER_END(nl_disp_update, "  nonlinear: displacement update");

        // Smooth displacement (batched: 3 in 1 CB)
        TIMER_START(nl_smooth_disp);
        batchSmoothInPlace({dux, duy, duz}, W, H, D, smoothDisp);
        TIMER_END(nl_smooth_disp, "  nonlinear: smooth displacement (3x)");

        // Accumulate displacement (single CB)
        {
            id<MTLCommandBuffer> cb = [c.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            encodeAdd(enc, updateDispX, dux, vol);
            encodeAdd(enc, updateDispY, duy, vol);
            encodeAdd(enc, updateDispZ, duz, vol);
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }

        // Interpolate from original volume with accumulated displacement
        TIMER_START(nl_final_interp);
        interpolateNonLinear(alignedVolume, originalAligned,
                             updateDispX, updateDispY, updateDispZ, W, H, D);
        TIMER_END(nl_final_interp, "  nonlinear: final interpolation");
        TIMER_END(nl_iter, "  nonlinear: iteration total");
      }
    }
}

// Multi-scale nonlinear registration
void alignTwoVolumesNonLinearSeveralScales(
    id<MTLBuffer>& alignedVolume,
    id<MTLBuffer> referenceVolume,
    const QuadratureFilters& filters,
    int W, int H, int D,
    int numIterations,
    int coarsestScale,
    id<MTLBuffer>& totalDispX, id<MTLBuffer>& totalDispY, id<MTLBuffer>& totalDispZ,
    bool verbose)
{
    auto& c = ctx();
    int vol = W * H * D;

    // Keep a copy of the original full-resolution aligned volume
    id<MTLBuffer> originalAligned = c.newBuffer(vol * sizeof(float));
    memcpy([originalAligned contents], [alignedVolume contents], vol * sizeof(float));

    totalDispX = c.newBuffer(vol * sizeof(float));
    totalDispY = c.newBuffer(vol * sizeof(float));
    totalDispZ = c.newBuffer(vol * sizeof(float));
    fillBuffer(totalDispX, 0.0f, vol);
    fillBuffer(totalDispY, 0.0f, vol);
    fillBuffer(totalDispZ, 0.0f, vol);

    TIMER_START(nl_all_scales);
    for (int scale = coarsestScale; scale >= 1; scale /= 2) {
      @autoreleasepool {
        TIMER_START(nl_scale);
        int sW = W / scale;
        int sH = H / scale;
        int sD = D / scale;

        if (sW < 8 || sH < 8 || sD < 8) continue;

        if (verbose) printf("  Nonlinear scale %d: %dx%dx%d\n", scale, sW, sH, sD);

        id<MTLBuffer> scaledRef = (scale == 1) ? referenceVolume :
            changeVolumeSize(referenceVolume, W, H, D, sW, sH, sD);
        // Always downscale from current aligned (which was transformed at full res)
        id<MTLBuffer> scaledAligned = (scale == 1) ? alignedVolume :
            changeVolumeSize(alignedVolume, W, H, D, sW, sH, sD);

        int sVol = sW * sH * sD;
        id<MTLBuffer> updateX = c.newBuffer(sVol * sizeof(float));
        id<MTLBuffer> updateY = c.newBuffer(sVol * sizeof(float));
        id<MTLBuffer> updateZ = c.newBuffer(sVol * sizeof(float));
        fillBuffer(updateX, 0.0f, sVol);
        fillBuffer(updateY, 0.0f, sVol);
        fillBuffer(updateZ, 0.0f, sVol);

        int iters = numIterations;

        alignTwoVolumesNonLinear(scaledAligned, scaledRef, filters,
                                 sW, sH, sD, iters,
                                 updateX, updateY, updateZ, verbose);

        // Accumulate into total displacement
        if (scale > 1) {
            // Rescale displacement to full resolution
            id<MTLBuffer> rescX = changeVolumeSize(updateX, sW, sH, sD, W, H, D);
            id<MTLBuffer> rescY = changeVolumeSize(updateY, sW, sH, sD, W, H, D);
            id<MTLBuffer> rescZ = changeVolumeSize(updateZ, sW, sH, sD, W, H, D);
            multiplyVolume(rescX, (float)scale, vol);
            multiplyVolume(rescY, (float)scale, vol);
            multiplyVolume(rescZ, (float)scale, vol);
            addVolumes(totalDispX, rescX, vol);
            addVolumes(totalDispY, rescY, vol);
            addVolumes(totalDispZ, rescZ, vol);

            // Re-interpolate from original aligned volume at full resolution
            interpolateNonLinear(alignedVolume, originalAligned,
                                 totalDispX, totalDispY, totalDispZ, W, H, D);
        } else {
            addVolumes(totalDispX, updateX, vol);
            addVolumes(totalDispY, updateY, vol);
            addVolumes(totalDispZ, updateZ, vol);
        }
        char nlScaleLabel[80];
        snprintf(nlScaleLabel, sizeof(nlScaleLabel), "NONLINEAR SCALE %d total", scale);
        TIMER_END(nl_scale, nlScaleLabel);
      }
    }
    TIMER_END(nl_all_scales, "NONLINEAR REGISTRATION all scales total");
}

} // anonymous namespace

// ============================================================
//  PUBLIC API
// ============================================================

EPIT1Result registerEPIT1(
    const float* epiData, VolumeDims epiDims, VoxelSize epiVox,
    const float* t1Data, VolumeDims t1Dims, VoxelSize t1Vox,
    const QuadratureFilters& filters,
    int numIterations, int coarsestScale, int mmZCut, bool verbose)
{
    TIMER_START(epit1_total);
    auto& c = ctx();

    if (verbose) printf("registerEPIT1: EPI %dx%dx%d -> T1 %dx%dx%d\n",
                        epiDims.W, epiDims.H, epiDims.D,
                        t1Dims.W, t1Dims.H, t1Dims.D);

    int t1Vol = t1Dims.size();

    // Upload volumes to GPU
    id<MTLBuffer> t1Buf = c.newBuffer(t1Data, t1Vol * sizeof(float));
    id<MTLBuffer> epiBuf = c.newBuffer(epiData, epiDims.size() * sizeof(float));

    // Resample EPI to T1 resolution and size
    TIMER_START(epit1_resamp);
    id<MTLBuffer> epiInT1 = changeVolumesResolutionAndSize(
        epiBuf, epiDims.W, epiDims.H, epiDims.D, epiVox,
        t1Dims.W, t1Dims.H, t1Dims.D, t1Vox, mmZCut);

    TIMER_END(epit1_resamp, "EPIT1: changeVolumesResolutionAndSize");

    // Center-of-mass alignment
    // OpenCL convention: params = input_center - ref_center
    // This shifts input toward the reference center
    TIMER_START(epit1_com);
    float cx1, cy1, cz1, cx2, cy2, cz2;
    centerOfMass((float*)[t1Buf contents], t1Dims.W, t1Dims.H, t1Dims.D, cx1, cy1, cz1);
    centerOfMass((float*)[epiInT1 contents], t1Dims.W, t1Dims.H, t1Dims.D, cx2, cy2, cz2);

    float initParams[12] = {0};
    // Round to integers to avoid interpolation blur (matches OpenCL's myround)
    initParams[0] = roundf(cx2 - cx1);
    initParams[1] = roundf(cy2 - cy1);
    initParams[2] = roundf(cz2 - cz1);

    // Apply initial translation
    interpolateLinear(epiInT1, epiInT1, initParams, t1Dims.W, t1Dims.H, t1Dims.D);
    TIMER_END(epit1_com, "EPIT1: center-of-mass + initial align");

    // Save interpolated volume after center-of-mass alignment (matches OpenCL)
    std::vector<float> interpResult(t1Vol);
    memcpy(interpResult.data(), [epiInT1 contents], t1Vol * sizeof(float));

    // Multi-scale linear registration (rigid body = 6 DOF subset of 12-param affine)
    float regParams[12] = {0};
    memcpy(regParams, initParams, 12 * sizeof(float));

    TIMER_START(epit1_linear);
    alignTwoVolumesLinearSeveralScales(
        epiInT1, t1Buf, filters,
        t1Dims.W, t1Dims.H, t1Dims.D,
        7, numIterations, coarsestScale,
        regParams, verbose);
    TIMER_END(epit1_linear, "EPIT1: LINEAR REGISTRATION total");

    // Read back result
    EPIT1Result result;
    result.aligned.resize(t1Vol);
    memcpy(result.aligned.data(), [epiInT1 contents], t1Vol * sizeof(float));
    result.interpolated = std::move(interpResult);

    // Extract 6 rigid body parameters (3 translation + 3 rotation)
    for (int i = 0; i < 6; i++) result.params[i] = regParams[i];

    TIMER_END(epit1_total, "EPIT1: TOTAL WALL CLOCK");

    return result;
}

T1MNIResult registerT1MNI(
    const float* t1Data, VolumeDims t1Dims, VoxelSize t1Vox,
    const float* mniData, VolumeDims mniDims, VoxelSize mniVox,
    const float* mniBrainData,
    const float* mniMaskData,
    const QuadratureFilters& filters,
    int linearIterations, int nonlinearIterations, int coarsestScale,
    int mmZCut, bool verbose)
{
    TIMER_START(t1mni_total);
    auto& c = ctx();

    if (verbose) printf("registerT1MNI: T1 %dx%dx%d -> MNI %dx%dx%d\n",
                        t1Dims.W, t1Dims.H, t1Dims.D,
                        mniDims.W, mniDims.H, mniDims.D);

    int mniVol = mniDims.size();

    // Upload volumes
    TIMER_START(t1mni_upload);
    id<MTLBuffer> mniBuf = c.newBuffer(mniData, mniVol * sizeof(float));
    id<MTLBuffer> mniBrainBuf = c.newBuffer(mniBrainData, mniVol * sizeof(float));
    id<MTLBuffer> mniMaskBuf = c.newBuffer(mniMaskData, mniVol * sizeof(float));
    id<MTLBuffer> t1Buf = c.newBuffer(t1Data, t1Dims.size() * sizeof(float));
    TIMER_END(t1mni_upload, "T1MNI: upload volumes");

    // Resample T1 to MNI resolution and size
    TIMER_START(t1mni_resamp);
    id<MTLBuffer> t1InMNI = changeVolumesResolutionAndSize(
        t1Buf, t1Dims.W, t1Dims.H, t1Dims.D, t1Vox,
        mniDims.W, mniDims.H, mniDims.D, mniVox, mmZCut);

    TIMER_END(t1mni_resamp, "T1MNI: changeVolumesResolutionAndSize");

    // Center-of-mass alignment
    // OpenCL convention: params = input_center - ref_center
    // This shifts input toward the reference center
    TIMER_START(t1mni_com);
    float cx1, cy1, cz1, cx2, cy2, cz2;
    centerOfMass((float*)[mniBuf contents], mniDims.W, mniDims.H, mniDims.D, cx1, cy1, cz1);
    centerOfMass((float*)[t1InMNI contents], mniDims.W, mniDims.H, mniDims.D, cx2, cy2, cz2);

    float initParams[12] = {0};
    // Round to integers to avoid interpolation blur (matches OpenCL's myround)
    initParams[0] = roundf(cx2 - cx1);
    initParams[1] = roundf(cy2 - cy1);
    initParams[2] = roundf(cz2 - cz1);

    interpolateLinear(t1InMNI, t1InMNI, initParams, mniDims.W, mniDims.H, mniDims.D);
    TIMER_END(t1mni_com, "T1MNI: center-of-mass + initial align");

    // Save interpolated volume after center-of-mass alignment (matches OpenCL)
    std::vector<float> interpResult(mniVol);
    memcpy(interpResult.data(), [t1InMNI contents], mniVol * sizeof(float));

    // Linear registration
    float regParams[12] = {0};

    if (verbose) printf("Running linear registration (%d iterations)...\n", linearIterations);

    TIMER_START(t1mni_linear);
    alignTwoVolumesLinearSeveralScales(
        t1InMNI, mniBuf, filters,
        mniDims.W, mniDims.H, mniDims.D,
        7, linearIterations, coarsestScale,
        regParams, verbose);
    TIMER_END(t1mni_linear, "T1MNI: LINEAR REGISTRATION total");

    // Compose COM shift into registration params (matches OpenCL line 8042:
    // AddAffineRegistrationParameters(regParams, matchParams))
    // composeAffineParams(old, new) = New * Old → stored in old
    // We want: T_com * T_aff → stored in regParams
    composeAffineParams(regParams, initParams);

    // Save linear result — re-rescale original T1 and apply combined
    // COM + affine in a single interpolation to avoid compounded blur
    // (matches OpenCL's "do total interpolation in one step" pattern)
    T1MNIResult result;
    {
        TIMER_START(t1mni_linear_resample);
        id<MTLBuffer> freshT1 = changeVolumesResolutionAndSize(
            t1Buf, t1Dims.W, t1Dims.H, t1Dims.D, t1Vox,
            mniDims.W, mniDims.H, mniDims.D, mniVox, mmZCut);
        interpolateLinear(freshT1, freshT1, regParams, mniDims.W, mniDims.H, mniDims.D);
        result.alignedLinear.resize(mniVol);
        memcpy(result.alignedLinear.data(), [freshT1 contents], mniVol * sizeof(float));
        TIMER_END(t1mni_linear_resample, "T1MNI: linear single-step resample");
    }
    for (int i = 0; i < 12; i++) result.params[i] = regParams[i];

    // Nonlinear registration
    if (nonlinearIterations > 0) {
        if (verbose) printf("Running nonlinear registration (%d iterations)...\n", nonlinearIterations);

        id<MTLBuffer> totalDispX, totalDispY, totalDispZ;

        TIMER_START(t1mni_nonlinear);
        alignTwoVolumesNonLinearSeveralScales(
            t1InMNI, mniBuf, filters,
            mniDims.W, mniDims.H, mniDims.D,
            nonlinearIterations, coarsestScale,
            totalDispX, totalDispY, totalDispZ, verbose);
        TIMER_END(t1mni_nonlinear, "T1MNI: NONLINEAR REGISTRATION total");

        // Combine linear + nonlinear displacement into single field
        addLinearNonLinearDisplacement(totalDispX, totalDispY, totalDispZ,
                                       regParams, mniDims.W, mniDims.H, mniDims.D);

        // Copy displacement fields
        result.dispX.resize(mniVol);
        result.dispY.resize(mniVol);
        result.dispZ.resize(mniVol);
        memcpy(result.dispX.data(), [totalDispX contents], mniVol * sizeof(float));
        memcpy(result.dispY.data(), [totalDispY contents], mniVol * sizeof(float));
        memcpy(result.dispZ.data(), [totalDispZ contents], mniVol * sizeof(float));

        // Re-rescale original T1 and apply combined displacement in one step
        // (rescale + single interpolation = 2 passes, not 4)
        TIMER_START(t1mni_nonlinear_resample);
        {
            id<MTLBuffer> freshT1 = changeVolumesResolutionAndSize(
                t1Buf, t1Dims.W, t1Dims.H, t1Dims.D, t1Vox,
                mniDims.W, mniDims.H, mniDims.D, mniVox, mmZCut);
            interpolateNonLinear(freshT1, freshT1, totalDispX, totalDispY, totalDispZ,
                                 mniDims.W, mniDims.H, mniDims.D);
            result.alignedNonLinear.resize(mniVol);
            memcpy(result.alignedNonLinear.data(), [freshT1 contents], mniVol * sizeof(float));

            // Skullstrip from the single-step result
            multiplyVolumes(freshT1, mniMaskBuf, mniVol);
            result.skullstripped.resize(mniVol);
            memcpy(result.skullstripped.data(), [freshT1 contents], mniVol * sizeof(float));
        }
        TIMER_END(t1mni_nonlinear_resample, "T1MNI: nonlinear single-step resample");
    } else {
        result.alignedNonLinear = result.alignedLinear;
        result.skullstripped.resize(mniVol, 0);
    }

    result.interpolated = std::move(interpResult);

    TIMER_END(t1mni_total, "T1MNI: TOTAL WALL CLOCK");

    return result;
}

} // namespace metal_reg
