/*
 * Minimal CUDA Driver API wrapper. Declares the handful of stable-ABI
 * types/prototypes needed directly (no cuda.h), dlopen's libcuda.so.1, and
 * resolves each symbol with dlsym - see gpu_cuda.h for why.
 */

#include "backends/gpu/gpu_cuda.h"
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;
typedef void *CUmodule;
typedef void *CUfunction;

typedef CUresult (*fn_cuInit)(unsigned int);
typedef CUresult (*fn_cuDeviceGet)(CUdevice *, int);
typedef CUresult (*fn_cuCtxCreate)(CUcontext *, unsigned int, CUdevice);
typedef CUresult (*fn_cuCtxDestroy)(CUcontext);
typedef CUresult (*fn_cuModuleLoadData)(CUmodule *, const void *);
typedef CUresult (*fn_cuModuleGetFunction)(CUfunction *, CUmodule, const char *);
typedef CUresult (*fn_cuMemAlloc)(uint64_t *, size_t);
typedef CUresult (*fn_cuMemFree)(uint64_t);
typedef CUresult (*fn_cuMemcpyHtoD)(uint64_t, const void *, size_t);
typedef CUresult (*fn_cuMemcpyDtoH)(void *, uint64_t, size_t);
typedef CUresult (*fn_cuLaunchKernel)(CUfunction, unsigned int, unsigned int, unsigned int,
                                      unsigned int, unsigned int, unsigned int,
                                      unsigned int, void *, void **, void **);
typedef CUresult (*fn_cuCtxSynchronize)(void);
typedef CUresult (*fn_cuGetErrorString)(CUresult, const char **);
typedef CUresult (*fn_cuDeviceGetAttribute)(int *, int, CUdevice);

struct gpu_cuda_ctx {
    void *lib;
    CUdevice device;
    CUcontext cu_ctx;
    CUmodule module;      /* most recently loaded module, if any */
    CUfunction kernel;    /* most recently resolved kernel, if any */

    fn_cuDeviceGetAttribute cuDeviceGetAttribute;
    fn_cuCtxDestroy cuCtxDestroy;
    fn_cuModuleLoadData cuModuleLoadData;
    fn_cuModuleGetFunction cuModuleGetFunction;
    fn_cuMemAlloc cuMemAlloc;
    fn_cuMemFree cuMemFree;
    fn_cuMemcpyHtoD cuMemcpyHtoD;
    fn_cuMemcpyDtoH cuMemcpyDtoH;
    fn_cuLaunchKernel cuLaunchKernel;
    fn_cuCtxSynchronize cuCtxSynchronize;
    fn_cuGetErrorString cuGetErrorString;
};

static void log_cuda_error(gpu_cuda_ctx_t *ctx, const char *label, CUresult rc) {
    const char *msg = "unknown";
    if (ctx->cuGetErrorString) ctx->cuGetErrorString(rc, &msg);
    fprintf(stderr, "gpu_cuda: %s failed: CUresult=%d (%s)\n", label, rc, msg);
}

gpu_cuda_ctx_t *gpu_cuda_init(void) {
    gpu_cuda_ctx_t *ctx = calloc(1, sizeof(gpu_cuda_ctx_t));
    if (!ctx) return NULL;

    ctx->lib = dlopen("libcuda.so.1", RTLD_NOW);
    if (!ctx->lib) ctx->lib = dlopen("libcuda.so", RTLD_NOW);
    if (!ctx->lib) {
        free(ctx);
        return NULL;
    }

#define LOAD(field, name) ctx->field = (fn_##name)dlsym(ctx->lib, #name); \
    if (!ctx->field) { dlclose(ctx->lib); free(ctx); return NULL; }

    fn_cuInit cuInit;
    fn_cuDeviceGet cuDeviceGet;
    fn_cuCtxCreate cuCtxCreate;
    cuInit = (fn_cuInit)dlsym(ctx->lib, "cuInit");
    cuDeviceGet = (fn_cuDeviceGet)dlsym(ctx->lib, "cuDeviceGet");
    cuCtxCreate = (fn_cuCtxCreate)dlsym(ctx->lib, "cuCtxCreate");
    if (!cuInit || !cuDeviceGet || !cuCtxCreate) {
        dlclose(ctx->lib);
        free(ctx);
        return NULL;
    }

    LOAD(cuDeviceGetAttribute, cuDeviceGetAttribute)
    LOAD(cuCtxDestroy, cuCtxDestroy)
    LOAD(cuModuleLoadData, cuModuleLoadData)
    LOAD(cuModuleGetFunction, cuModuleGetFunction)
    LOAD(cuMemAlloc, cuMemAlloc)
    LOAD(cuMemFree, cuMemFree)
    LOAD(cuMemcpyHtoD, cuMemcpyHtoD)
    LOAD(cuMemcpyDtoH, cuMemcpyDtoH)
    LOAD(cuLaunchKernel, cuLaunchKernel)
    LOAD(cuCtxSynchronize, cuCtxSynchronize)
    LOAD(cuGetErrorString, cuGetErrorString)
#undef LOAD

    if (cuInit(0) != 0) {
        dlclose(ctx->lib);
        free(ctx);
        return NULL;
    }

    CUdevice dev;
    if (cuDeviceGet(&dev, 0) != 0) {
        dlclose(ctx->lib);
        free(ctx);
        return NULL;
    }
    ctx->device = dev;

    if (cuCtxCreate(&ctx->cu_ctx, 0, dev) != 0) {
        dlclose(ctx->lib);
        free(ctx);
        return NULL;
    }

    return ctx;
}

void gpu_cuda_shutdown(gpu_cuda_ctx_t *ctx) {
    if (!ctx) return;
    if (ctx->cu_ctx && ctx->cuCtxDestroy) ctx->cuCtxDestroy(ctx->cu_ctx);
    if (ctx->lib) dlclose(ctx->lib);
    free(ctx);
}

int gpu_cuda_load_kernel(gpu_cuda_ctx_t *ctx, const char *ptx_src, const char *kernel_name) {
    CUresult rc = ctx->cuModuleLoadData(&ctx->module, ptx_src);
    if (rc != 0) {
        log_cuda_error(ctx, "cuModuleLoadData", rc);
        return -1;
    }

    rc = ctx->cuModuleGetFunction(&ctx->kernel, ctx->module, kernel_name);
    if (rc != 0) {
        log_cuda_error(ctx, "cuModuleGetFunction", rc);
        return -1;
    }

    return 0;
}

uint64_t gpu_cuda_alloc(gpu_cuda_ctx_t *ctx, size_t bytes) {
    uint64_t dptr = 0;
    CUresult rc = ctx->cuMemAlloc(&dptr, bytes);
    if (rc != 0) {
        log_cuda_error(ctx, "cuMemAlloc", rc);
        return 0;
    }
    return dptr;
}

void gpu_cuda_free(gpu_cuda_ctx_t *ctx, uint64_t dptr) {
    if (dptr) ctx->cuMemFree(dptr);
}

int gpu_cuda_upload(gpu_cuda_ctx_t *ctx, uint64_t dptr, const void *host, size_t bytes) {
    CUresult rc = ctx->cuMemcpyHtoD(dptr, host, bytes);
    if (rc != 0) {
        log_cuda_error(ctx, "cuMemcpyHtoD", rc);
        return -1;
    }
    return 0;
}

int gpu_cuda_download(gpu_cuda_ctx_t *ctx, void *host, uint64_t dptr, size_t bytes) {
    CUresult rc = ctx->cuMemcpyDtoH(host, dptr, bytes);
    if (rc != 0) {
        log_cuda_error(ctx, "cuMemcpyDtoH", rc);
        return -1;
    }
    return 0;
}

int gpu_cuda_launch_2d(gpu_cuda_ctx_t *ctx,
                        unsigned int grid_x, unsigned int grid_y,
                        unsigned int block_x, unsigned int block_y,
                        void **args) {
    CUresult rc = ctx->cuLaunchKernel(ctx->kernel, grid_x, grid_y, 1, block_x, block_y, 1,
                                       0, NULL, args, NULL);
    if (rc != 0) {
        log_cuda_error(ctx, "cuLaunchKernel", rc);
        return -1;
    }

    rc = ctx->cuCtxSynchronize();
    if (rc != 0) {
        log_cuda_error(ctx, "cuCtxSynchronize", rc);
        return -1;
    }

    return 0;
}

int gpu_cuda_device_attribute(gpu_cuda_ctx_t *ctx, int attrib, int *out) {
    int value = 0;
    CUresult rc = ctx->cuDeviceGetAttribute(&value, attrib, ctx->device);
    if (rc != 0) return -1; /* not a recognized/supported attribute on this device - expected, not logged */
    *out = value;
    return 0;
}
