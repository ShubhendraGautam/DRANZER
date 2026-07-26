/* Statistical sanity check for dropout_forward/dropout_backward
 * (tensor_ops.c): with a constant input, the observed drop fraction should
 * match the configured rate, and inverted scaling should keep the mean
 * activation roughly unchanged. Also checks that is_training=0 is a true
 * no-op (inference must never apply dropout). */
#include "core/tensor_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 200000

int main(void) {
    srand(1);
    float *x = malloc(N * sizeof(float));
    float *mask = malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) x[i] = 2.0f; /* constant, so post-dropout mean should stay ~2.0 */

    float rate = 0.4f;
    dropout_forward(x, mask, N, rate, 1);

    size_t zeros = 0;
    double sum = 0.0;
    for (size_t i = 0; i < N; i++) {
        if (mask[i] == 0.0f) zeros++;
        sum += x[i];
    }
    double mean = sum / N;
    double drop_frac = (double)zeros / N;

    printf("rate=%.2f observed_drop_frac=%.4f mean_after=%.4f (expect ~2.0000)\n", rate, drop_frac, mean);

    /* is_training=0: must be a no-op identity regardless of rate */
    for (size_t i = 0; i < N; i++) x[i] = 3.0f;
    dropout_forward(x, mask, N, rate, 0);
    int all_unchanged = 1, all_mask_one = 1;
    for (size_t i = 0; i < N; i++) {
        if (x[i] != 3.0f) all_unchanged = 0;
        if (mask[i] != 1.0f) all_mask_one = 0;
    }
    printf("inference_noop=%s mask_all_ones=%s\n", all_unchanged ? "PASS" : "FAIL", all_mask_one ? "PASS" : "FAIL");

    int pass = fabs(drop_frac - rate) < 0.01 && fabs(mean - 2.0) < 0.02 && all_unchanged && all_mask_one;
    free(x);
    free(mask);
    printf("\n%s\n", pass ? "DROPOUT STATISTICAL CHECK PASSED" : "DROPOUT STATISTICAL CHECK FAILED");
    return pass ? 0 : 1;
}
