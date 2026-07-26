/* Generic GPU device node presence via /dev/dri - this is the Linux
 * kernel's DRM (Direct Rendering Manager) subsystem, populated for any GPU
 * with a loaded kernel driver regardless of vendor (NVIDIA/AMD/Intel) or
 * whether any userspace compute library (CUDA, OpenCL) is installed. It's
 * the most basic "is there a GPU here at all, as far as the kernel can
 * tell" signal, and needs nothing beyond standard directory listing. */

#include "include/gpu_probe.h"
#include <dirent.h>
#include <string.h>
#include <stdio.h>

void gpu_backend_drm_probe(gpu_backend_report_t *out) {
    out->status = GPU_BACKEND_UNAVAILABLE;
    out->message[0] = '\0';

    DIR *dir = opendir("/dev/dri");
    if (!dir) {
        snprintf(out->message, sizeof(out->message),
                 "/dev/dri not present - no GPU kernel driver (DRM) detected on this system.");
        return;
    }

    char nodes[512] = "";
    int count = 0;
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_name[0] == '.') continue;
        if (count > 0) strncat(nodes, ", ", sizeof(nodes) - strlen(nodes) - 1);
        strncat(nodes, entry->d_name, sizeof(nodes) - strlen(nodes) - 1);
        count++;
    }
    closedir(dir);

    if (count == 0) {
        snprintf(out->message, sizeof(out->message),
                 "/dev/dri exists but is empty - no GPU device nodes found.");
        return;
    }

    out->status = GPU_BACKEND_PARTIAL; /* presence only - says nothing about compute capability */
    snprintf(out->message, sizeof(out->message),
             "/dev/dri device nodes present: %s (a GPU kernel driver is loaded; this alone doesn't\n"
             "    imply any compute backend - see the NVML/CUDA/OpenCL probes below for that).",
             nodes);
}
