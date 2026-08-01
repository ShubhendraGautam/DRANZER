#include "cli/sampling.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    float logits[] = {-3.0f, 0.5f, 4.0f, 1.0f};
    const size_t vocab_size = sizeof(logits) / sizeof(logits[0]);

    if (sample_greedy(logits, vocab_size) != 2 ||
        sample_topk(logits, vocab_size, 1) != 2 ||
        sample_next_token(logits, vocab_size, SAMPLING_TOPK, 0.0f, 4, 0.9f) != 2) {
        fprintf(stderr, "greedy/top-k selection failed\n");
        return 1;
    }

    /* p=1 must retain the complete distribution. The previous cutoff
     * implementation accidentally retained only two candidates. */
    float uniform[] = {0.0f, 0.0f, 0.0f, 0.0f};
    int seen[4] = {0};
    srand(9);
    for (size_t i = 0; i < 256; i++) {
        uint32_t token = sample_topp(uniform, 4, 1.0f);
        if (token >= 4) {
            fprintf(stderr, "top-p returned an out-of-range token\n");
            return 1;
        }
        seen[token] = 1;
    }
    for (size_t i = 0; i < 4; i++) {
        if (!seen[i]) {
            fprintf(stderr, "top-p p=1 excluded token %zu\n", i);
            return 1;
        }
    }

    float top_only[] = {10.0f, 0.0f, -1.0f, -2.0f};
    srand(4);
    for (size_t i = 0; i < 32; i++) {
        if (sample_next_token(top_only, 4, SAMPLING_TOPP, 1.0f, 4, 0.0001f) != 0) {
            fprintf(stderr, "small top-p nucleus did not select the best token\n");
            return 1;
        }
    }

    float invalid_filter[] = {-2.0f, 5.0f, 1.0f};
    if (sample_next_token(invalid_filter, 3, SAMPLING_TOPK, 1.0f, 0, 0.9f) != 1 ||
        sample_next_token(invalid_filter, 3, SAMPLING_TOPP, 1.0f, 3, 0.0f) != 1) {
        fprintf(stderr, "invalid sampling filter did not fall back to greedy\n");
        return 1;
    }

    printf("\nSAMPLING CHECK PASSED\n");
    return 0;
}
