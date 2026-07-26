/* OpenCL probe. Deliberately checks two DIFFERENT things:
 *
 *   1. The ICD loader (libOpenCL.so) - the vendor-neutral dispatch library
 *      that OpenCL programs actually link against.
 *   2. Vendor ICDs registered under /etc/OpenCL/vendors/ (*.icd files), and,
 *      correspondingly, whether clGetPlatformIDs() actually reports any
 *      platforms.
 *
 * The loader can be fully present and functional while reporting zero
 * platforms if no vendor has registered one (CL_PLATFORM_NOT_FOUND_KHR,
 * error -1001) - this is this exact sandbox's state, and looks
 * confusingly like "OpenCL isn't installed" unless the two are checked
 * separately, since `dlopen("libOpenCL.so")` alone would succeed either way.
 *
 * As with the other probes, the OpenCL types/prototypes needed are
 * declared directly (matching the stable public OpenCL 1.2 ABI) rather
 * than requiring the opencl-headers package to be installed. */

#include "backends/gpu/gpu_probe.h"
#include <dlfcn.h>
#include <dirent.h>
#include <stdio.h>
#include <string.h>

typedef int cl_int;
typedef unsigned int cl_uint;
typedef void *cl_platform_id;
typedef void *cl_device_id;
typedef unsigned long long cl_device_type;
typedef unsigned int cl_platform_info;
typedef unsigned int cl_device_info;

#define CL_DEVICE_TYPE_ALL 0xFFFFFFFFULL
#define CL_PLATFORM_NAME 0x0902
#define CL_DEVICE_NAME 0x102B

typedef cl_int (*fn_clGetPlatformIDs)(cl_uint, cl_platform_id *, cl_uint *);
typedef cl_int (*fn_clGetPlatformInfo)(cl_platform_id, cl_platform_info, size_t, void *, size_t *);
typedef cl_int (*fn_clGetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *);
typedef cl_int (*fn_clGetDeviceInfo)(cl_device_id, cl_device_info, size_t, void *, size_t *);

/* Registered vendor ICD files, independent of whether the loader can
 * currently see them - useful diagnostic even if clGetPlatformIDs fails. */
static int count_registered_icds(char *out_names, size_t out_names_size) {
    DIR *dir = opendir("/etc/OpenCL/vendors");
    if (!dir) return 0;

    int count = 0;
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (strstr(entry->d_name, ".icd")) {
            if (count > 0) strncat(out_names, ", ", out_names_size - strlen(out_names) - 1);
            strncat(out_names, entry->d_name, out_names_size - strlen(out_names) - 1);
            count++;
        }
    }
    closedir(dir);
    return count;
}

void gpu_backend_opencl_probe(gpu_backend_report_t *out, gpu_opencl_facts_t *facts) {
    out->status = GPU_BACKEND_UNAVAILABLE;
    out->message[0] = '\0';
    size_t len = 0;
    if (facts) memset(facts, 0, sizeof(*facts));

    void *lib = dlopen("libOpenCL.so.1", RTLD_NOW);
    if (!lib) lib = dlopen("libOpenCL.so", RTLD_NOW);
    if (!lib) {
        snprintf(out->message, sizeof(out->message),
                 "libOpenCL.so not found - no OpenCL ICD loader installed.\n"
                 "    Install with: sudo apt-get install ocl-icd-opencl-dev opencl-headers");
        return;
    }

    fn_clGetPlatformIDs clGetPlatformIDs = (fn_clGetPlatformIDs)dlsym(lib, "clGetPlatformIDs");
    fn_clGetPlatformInfo clGetPlatformInfo = (fn_clGetPlatformInfo)dlsym(lib, "clGetPlatformInfo");
    fn_clGetDeviceIDs clGetDeviceIDs = (fn_clGetDeviceIDs)dlsym(lib, "clGetDeviceIDs");
    fn_clGetDeviceInfo clGetDeviceInfo = (fn_clGetDeviceInfo)dlsym(lib, "clGetDeviceInfo");

    if (!clGetPlatformIDs || !clGetPlatformInfo || !clGetDeviceIDs || !clGetDeviceInfo) {
        snprintf(out->message, sizeof(out->message),
                 "libOpenCL.so loaded, but expected symbols are missing (unexpected loader version?).");
        dlclose(lib);
        return;
    }

    len += (size_t)snprintf(out->message + len, sizeof(out->message) - len,
                             "OpenCL ICD loader (libOpenCL.so): functional\n");
    if (facts) facts->loader_functional = 1;

    char icd_names[512] = "";
    int icd_count = count_registered_icds(icd_names, sizeof(icd_names));
    len += (size_t)snprintf(out->message + len, sizeof(out->message) - len,
                             "    Registered vendor ICDs (/etc/OpenCL/vendors/): %s\n",
                             icd_count > 0 ? icd_names : "none");

    cl_uint num_platforms = 0;
    cl_int rc = clGetPlatformIDs(0, NULL, &num_platforms);
    if (facts) facts->platform_count = (int)num_platforms;

    if (rc != 0 || num_platforms == 0) {
        len += (size_t)snprintf(out->message + len, sizeof(out->message) - len,
                                 "    clGetPlatformIDs() found 0 platforms (rc=%d) - the loader works, but no vendor\n"
                                 "    OpenCL implementation is registered, so there is nothing to run on yet.\n"
                                 "    For an NVIDIA GPU this typically needs the vendor's OpenCL ICD registered\n"
                                 "    (bundled with some NVIDIA driver packages, or via nvidia-opencl-icd where available);\n"
                                 "    a software fallback like POCL (sudo apt-get install pocl-opencl-icd) also works.\n",
                                 rc);
        dlclose(lib);
        out->status = GPU_BACKEND_PARTIAL;
        return;
    }

    cl_platform_id platforms[8];
    cl_uint got_platforms = num_platforms < 8 ? num_platforms : 8;
    clGetPlatformIDs(got_platforms, platforms, NULL);

    for (cl_uint p = 0; p < got_platforms && len < sizeof(out->message); p++) {
        char platform_name[256] = {0};
        clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);

        if (facts) {
            if (p > 0) strncat(facts->platform_names, ", ", sizeof(facts->platform_names) - strlen(facts->platform_names) - 1);
            strncat(facts->platform_names, platform_name, sizeof(facts->platform_names) - strlen(facts->platform_names) - 1);
        }

        cl_uint num_devices = 0;
        clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        if (facts) facts->total_device_count += (int)num_devices;

        len += (size_t)snprintf(out->message + len, sizeof(out->message) - len,
                                 "    Platform %u: %s (%u device(s))\n", p, platform_name, num_devices);

        if (num_devices > 0) {
            cl_device_id devices[8];
            cl_uint got_devices = num_devices < 8 ? num_devices : 8;
            clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, got_devices, devices, NULL);

            for (cl_uint d = 0; d < got_devices && len < sizeof(out->message); d++) {
                char device_name[256] = {0};
                clGetDeviceInfo(devices[d], CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
                len += (size_t)snprintf(out->message + len, sizeof(out->message) - len,
                                         "        [%u] %s\n", d, device_name);
            }
        }
    }

    dlclose(lib);
    out->status = GPU_BACKEND_READY;
}
