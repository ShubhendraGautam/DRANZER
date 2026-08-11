#include "dranzer.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int report(const char *operation, dranzer_status_t status) {
    fprintf(stderr, "%s: %s\n", operation, dranzer_status_string(status));
    return 1;
}

static int parse_count(const char *text, size_t *out) {
    char *end = NULL;
    errno = 0;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno || end == text || *end || value == 0 ||
        value > (unsigned long long)SIZE_MAX) return 0;
    *out = (size_t)value;
    return 1;
}

int main(int argc, char **argv) {
    size_t limit = 0;
    if (argc != 4 || !parse_count(argv[3], &limit)) {
        fprintf(stderr, "usage: %s MODEL_BUNDLE PROMPT TOKEN_COUNT\n", argv[0]);
        return 2;
    }

    dranzer_model_t *model = NULL;
    dranzer_tokenizer_t *tokenizer = NULL;
    dranzer_generation_t *generation = NULL;
    int result = 1;

    dranzer_status_t status = dranzer_bundle_load(
        argv[1], DRANZER_LOAD_COPY, &model, &tokenizer, NULL);
    if (status != DRANZER_OK) return report("load", status);
    status = dranzer_generation_create(model, tokenizer, 0, &generation);
    if (status != DRANZER_OK) {
        report("create generation", status);
        goto done;
    }
    status = dranzer_generation_reset(generation, argv[2], strlen(argv[2]));
    if (status != DRANZER_OK) {
        report("reset generation", status);
        goto done;
    }

    for (size_t i = 0; i < limit; i++) {
        uint32_t token = 0;
        size_t piece_size = 0;
        status = dranzer_generation_next_greedy(
            generation, &token, NULL, &piece_size);
        if (status == DRANZER_FINISHED) break;
        if (status != DRANZER_BUFFER_TOO_SMALL) {
            report("size generated piece", status);
            goto done;
        }

        unsigned char *piece = malloc(piece_size);
        if (!piece) {
            fputs("allocate generated piece: out of memory\n", stderr);
            goto done;
        }
        size_t piece_capacity = piece_size;
        status = dranzer_generation_next_greedy(
            generation, &token, piece, &piece_capacity);
        if (status != DRANZER_OK) {
            free(piece);
            report("generate", status);
            goto done;
        }
        if (fwrite(piece, 1, piece_capacity, stdout) != piece_capacity) {
            free(piece);
            fputs("write generated piece: I/O error\n", stderr);
            goto done;
        }
        free(piece);
    }
    fputc('\n', stdout);
    result = ferror(stdout) ? 1 : 0;

done:
    dranzer_generation_free(generation);
    dranzer_tokenizer_free(tokenizer);
    dranzer_model_free(model);
    return result;
}
