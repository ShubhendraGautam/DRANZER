/*
 * Print the weight fingerprint of a freshly initialized model.
 *
 * One line, one number, no timing: what a seed and an architecture produce.
 * That makes it the primitive the reproducibility contract is checked with
 * (docs/reproducibility.md) - build the tree two ways, run this, compare two
 * integers. Two compilers, two optimization levels, an ASan build, a
 * size-optimized build, and (via tests/integration/test_libc_independence.sh)
 * a process whose rand() has been replaced with garbage all have to agree.
 *
 * Deliberately not the CLI: `train` writes files, prints progress, and reads a
 * corpus, so comparing two of its runs compares much more than initialization.
 * Isolating the question is the point.
 */

#include "core/model.h"
#include "core/fingerprint.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_usage(const char *program) {
    printf("Usage: %s [--seed N] [--vocab N] [--embedding-dim N] [--heads N]\n"
           "          [--layers N] [--max-seq-len N] [--quiet]\n\n"
           "Prints the FNV-1a fingerprint of the initial weights for that\n"
           "(seed, architecture). Defaults match the tiny benchmark tier.\n"
           "  --quiet   print only the fingerprint, for use in a shell test\n",
           program);
}

static int parse_size(const char *text, size_t *out) {
    if (!text || !*text) return -1;
    char *end = NULL;
    unsigned long long value = strtoull(text, &end, 10);
    if (!end || *end != '\0' || value == 0) return -1;
    *out = (size_t)value;
    return 0;
}

int main(int argc, char **argv) {
    size_t vocab = 260, embedding_dim = 16, heads = 2, layers = 2, max_seq = 32;
    unsigned long long seed = 42;
    int quiet = 0;

    for (int i = 1; i < argc; i++) {
        const char *flag = argv[i];
        const char *value = (i + 1 < argc) ? argv[i + 1] : NULL;
        int consumed = 1;
        if (strcmp(flag, "--quiet") == 0) {
            quiet = 1;
        } else if (strcmp(flag, "--help") == 0 || strcmp(flag, "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (value == NULL) {
            fprintf(stderr, "Error: %s needs a value\n", flag);
            return 2;
        } else if (strcmp(flag, "--seed") == 0) {
            char *end = NULL;
            seed = strtoull(value, &end, 10);
            if (!end || *end != '\0') {
                fprintf(stderr, "Error: bad seed '%s'\n", value);
                return 2;
            }
            consumed = 2;
        } else {
            size_t *target = NULL;
            if (strcmp(flag, "--vocab") == 0) target = &vocab;
            else if (strcmp(flag, "--embedding-dim") == 0) target = &embedding_dim;
            else if (strcmp(flag, "--heads") == 0) target = &heads;
            else if (strcmp(flag, "--layers") == 0) target = &layers;
            else if (strcmp(flag, "--max-seq-len") == 0) target = &max_seq;
            if (!target || parse_size(value, target) != 0) {
                fprintf(stderr, "Error: unknown or invalid option '%s'\n", flag);
                print_usage(argv[0]);
                return 2;
            }
            consumed = 2;
        }
        i += consumed - 1;
    }

    neural_model_t model = {0};
    if (model_new_seeded(&model, vocab, embedding_dim, heads, layers, max_seq,
                         (uint64_t)seed) != MODEL_SUCCESS) {
        fprintf(stderr, "Error: model initialization failed\n");
        return 1;
    }

    uint64_t fingerprint = dranzer_weights_fingerprint(&model);
    if (quiet) {
        printf("%016" PRIX64 "\n", fingerprint);
    } else {
        printf("seed=%llu vocab=%zu emb=%zu heads=%zu layers=%zu max_seq=%zu\n",
               seed, vocab, embedding_dim, heads, layers, max_seq);
        printf("parameters=%zu\n", model.total_param_count);
        printf("weight_fingerprint=%016" PRIX64 "\n", fingerprint);
        printf("dropout_stream=%016" PRIX64 "\n", model.rng_state);
    }

    model_free(&model);
    return 0;
}
