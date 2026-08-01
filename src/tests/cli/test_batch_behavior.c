#include "cli/batch.h"
#include <stdio.h>

static int fill(batch_t *batch) {
    for (uint32_t i = 0; i < 8; i++) {
        uint32_t tokens[2] = {i, i + 10};
        if (batch_add_sequence(batch, tokens, 2, i + 100) != 0) return -1;
    }
    return 0;
}

int main(void) {
    batch_t *first = batch_create(8, 2);
    batch_t *second = batch_create(8, 2);
    uint32_t too_long[3] = {1, 2, 3};
    int failed = !first || !second || batch_create(0, 2) != NULL ||
                 batch_create(2, 0) != NULL;

    if (!failed && (fill(first) != 0 || fill(second) != 0 ||
                    batch_add_sequence(first, too_long, 3, 0) == 0)) failed = 1;
    batch_shuffle(first, 12345);
    batch_shuffle(second, 12345);
    for (size_t i = 0; !failed && i < 8; i++) {
        if (first->target_tokens[i] != second->target_tokens[i] ||
            first->sequence_lengths[i] != second->sequence_lengths[i] ||
            first->token_sequences[i][0] != second->token_sequences[i][0]) failed = 1;
    }

    if (failed) fprintf(stderr, "bounded deterministic batch behavior failed\n");
    else printf("\nDETERMINISTIC MINIBATCH SHUFFLE CHECK PASSED\n");
    batch_free(first);
    batch_free(second);
    return failed;
}
