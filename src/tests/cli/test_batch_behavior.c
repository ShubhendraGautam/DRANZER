#include "cli/batch.h"
#include "core/lm_head.h"
#include <stdio.h>

/* Each example carries a target per position now (core/lm_head.h), so the
 * shuffle has one more parallel array to keep aligned than it used to.
 * Targets are made distinguishable per example AND per position so a swap
 * that moved the right example but the wrong position would still show. */
static int fill(batch_t *batch) {
    for (uint32_t i = 0; i < 8; i++) {
        uint32_t tokens[2] = {i, i + 10};
        uint32_t targets[2] = {i + 100, i + 200};
        if (batch_add_sequence(batch, tokens, 2, targets) != 0) return -1;
    }
    return 0;
}

int main(void) {
    batch_t *first = batch_create(8, 2);
    batch_t *second = batch_create(8, 2);
    uint32_t too_long[3] = {1, 2, 3};
    uint32_t too_long_targets[3] = {4, 5, 6};
    int failed = !first || !second || batch_create(0, 2) != NULL ||
                 batch_create(2, 0) != NULL;

    if (!failed && (fill(first) != 0 || fill(second) != 0 ||
                    batch_add_sequence(first, too_long, 3, too_long_targets) == 0 ||
                    batch_add_sequence(first, too_long, 2, NULL) == 0)) failed = 1;
    batch_shuffle(first, 12345);
    batch_shuffle(second, 12345);
    for (size_t i = 0; !failed && i < 8; i++) {
        if (first->sequence_lengths[i] != second->sequence_lengths[i] ||
            first->token_sequences[i][0] != second->token_sequences[i][0] ||
            first->token_sequences[i][1] != second->token_sequences[i][1] ||
            first->target_sequences[i][0] != second->target_sequences[i][0] ||
            first->target_sequences[i][1] != second->target_sequences[i][1]) failed = 1;

        /* Tokens and targets must stay attached to each other through the
         * permutation: example n was built as tokens {n, n+10} with targets
         * {n+100, n+200}, so this relation holds wherever it landed. */
        uint32_t n = first->token_sequences[i][0];
        if (first->token_sequences[i][1] != n + 10 ||
            first->target_sequences[i][0] != n + 100 ||
            first->target_sequences[i][1] != n + 200) failed = 1;
    }

    if (failed) fprintf(stderr, "bounded deterministic batch behavior failed\n");
    else printf("\nDETERMINISTIC MINIBATCH SHUFFLE CHECK PASSED\n");
    batch_free(first);
    batch_free(second);
    return failed;
}
