#include "cli/sampling.h"
#include "core/rng.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    /* An explicit stream, replacing two srand() calls: the samplers no longer
     * read a process-global generator, because a sampled token that only
     * reproduces on one C library is not reproducible (see core/rng.h). */
    uint64_t rng = dranzer_rng_stream(9, DRANZER_RNG_STREAM_SAMPLING);
    float logits[] = {-3.0f, 0.5f, 4.0f, 1.0f};
    const size_t vocab_size = sizeof(logits) / sizeof(logits[0]);

    if (sample_greedy(logits, vocab_size) != 2 ||
        sample_topk(logits, vocab_size, 1, &rng) != 2 ||
        sample_next_token(logits, vocab_size, SAMPLING_TOPK, 0.0f, 4, 0.9f, &rng) != 2) {
        fprintf(stderr, "greedy/top-k selection failed\n");
        return 1;
    }

    /* p=1 must retain the complete distribution. The previous cutoff
     * implementation accidentally retained only two candidates. */
    float uniform[] = {0.0f, 0.0f, 0.0f, 0.0f};
    int seen[4] = {0};
    for (size_t i = 0; i < 256; i++) {
        uint32_t token = sample_topp(uniform, 4, 1.0f, &rng);
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
    for (size_t i = 0; i < 32; i++) {
        if (sample_next_token(top_only, 4, SAMPLING_TOPP, 1.0f, 4, 0.0001f, &rng) != 0) {
            fprintf(stderr, "small top-p nucleus did not select the best token\n");
            return 1;
        }
    }

    float invalid_filter[] = {-2.0f, 5.0f, 1.0f};
    if (sample_next_token(invalid_filter, 3, SAMPLING_TOPK, 1.0f, 0, 0.9f, &rng) != 1 ||
        sample_next_token(invalid_filter, 3, SAMPLING_TOPP, 1.0f, 3, 0.0f, &rng) != 1) {
        fprintf(stderr, "invalid sampling filter did not fall back to greedy\n");
        return 1;
    }

    /* A NULL stream must decode greedily rather than reaching for a global
     * generator - the contract that keeps an unreproducible sample from being
     * produced by accident. */
    if (sample_next_token(logits, vocab_size, SAMPLING_TOPP, 1.0f, 4, 0.9f,
                          NULL) != 2) {
        fprintf(stderr, "a NULL RNG stream did not fall back to greedy\n");
        return 1;
    }

    printf("\nSAMPLING CHECK PASSED\n");
    return 0;
}
