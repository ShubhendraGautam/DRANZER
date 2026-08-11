#include "dranzer.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int report(const char *operation, dranzer_status_t status) {
    fprintf(stderr, "%s: %s\n", operation, dranzer_status_string(status));
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 3 || argv[2][0] == '\0') {
        fprintf(stderr, "usage: %s MODEL_BUNDLE PROMPT\n", argv[0]);
        return 2;
    }

    dranzer_model_t *model = NULL;
    dranzer_tokenizer_t *tokenizer = NULL;
    uint32_t *tokens = NULL;
    float *logits = NULL;
    int result = 1;

    dranzer_status_t status = dranzer_bundle_load(
        argv[1], DRANZER_LOAD_COPY, &model, &tokenizer, NULL);
    if (status != DRANZER_OK) return report("load", status);

    size_t token_count = 0;
    status = dranzer_tokenize(tokenizer, argv[2], strlen(argv[2]),
                              NULL, &token_count);
    if (status != DRANZER_BUFFER_TOO_SMALL || token_count == 0) {
        report("tokenize size", status);
        goto done;
    }
    tokens = malloc(token_count * sizeof(*tokens));
    if (!tokens) {
        fputs("allocate tokens: out of memory\n", stderr);
        goto done;
    }
    size_t token_capacity = token_count;
    status = dranzer_tokenize(tokenizer, argv[2], strlen(argv[2]),
                              tokens, &token_capacity);
    if (status != DRANZER_OK) {
        report("tokenize", status);
        goto done;
    }

    size_t vocab_size = dranzer_model_vocab_size(model);
    logits = malloc(vocab_size * sizeof(*logits));
    if (!logits) {
        fputs("allocate logits: out of memory\n", stderr);
        goto done;
    }
    status = dranzer_model_forward(model, tokens, token_count,
                                   logits, vocab_size);
    if (status != DRANZER_OK) {
        report("forward", status);
        goto done;
    }

    uint32_t next = 0;
    for (uint32_t i = 1; i < vocab_size; i++) {
        if (logits[i] > logits[next]) next = i;
    }
    printf("next_token=%u logit=%.9g\n", next, logits[next]);
    result = 0;

done:
    free(logits);
    free(tokens);
    dranzer_tokenizer_free(tokenizer);
    dranzer_model_free(model);
    return result;
}
