/* Invalid process configuration is returned as structured state and remains
 * silent. The CLI/application decides whether and where to report it. */

#include "backends/gpu/gpu_matmul.h"
#include "core/cpu_features.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(void) {
    int capture[2];
    if (pipe(capture) != 0) return 1;
    int saved_stdout = dup(STDOUT_FILENO);
    int saved_stderr = dup(STDERR_FILENO);
    if (saved_stdout < 0 || saved_stderr < 0 ||
        dup2(capture[1], STDOUT_FILENO) < 0 ||
        dup2(capture[1], STDERR_FILENO) < 0) return 1;

    setenv("DRANZER_CPU_ISA", "not-an-isa", 1);
    setenv("DRANZER_GPU_MATMUL", "not-a-kernel", 1);
    cpu_features_detect();
    int structured =
        cpu_features_config_status() == CPU_FEATURES_CONFIG_INVALID_ENV &&
        strcmp(cpu_features_invalid_environment_value(), "not-an-isa") == 0 &&
        gpu_matmul_config_status() == GPU_MATMUL_CONFIG_INVALID_ENV &&
        strcmp(gpu_matmul_invalid_environment_value(), "not-a-kernel") == 0;

    fflush(stdout);
    fflush(stderr);
    dup2(saved_stdout, STDOUT_FILENO);
    dup2(saved_stderr, STDERR_FILENO);
    close(saved_stdout);
    close(saved_stderr);
    close(capture[1]);
    char byte = 0;
    ssize_t captured = read(capture[0], &byte, 1);
    close(capture[0]);
    unsetenv("DRANZER_CPU_ISA");
    unsetenv("DRANZER_GPU_MATMUL");

    int failed = !structured || captured != 0;
    printf("%s\n", failed ? "LIBRARY DIAGNOSTIC CHECK FAILED"
                           : "LIBRARY DIAGNOSTIC CHECK PASSED");
    return failed ? 1 : 0;
}
