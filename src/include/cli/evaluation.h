#ifndef CLI_EVALUATION_H
#define CLI_EVALUATION_H

#include "byte_pair_encoding.h"
#include "core/model_types.h"
#include <stddef.h>

typedef enum {
    EVALUATION_SUCCESS = 0,
    EVALUATION_INVALID_INPUT,
    EVALUATION_FILE_ERROR,
    EVALUATION_TOKENIZER_ERROR,
    EVALUATION_MODEL_ERROR,
    EVALUATION_ALLOCATION_FAILURE,
} evaluation_errors_t;

typedef struct {
    uint64_t corpus_fingerprint;
    size_t corpus_bytes;
    size_t token_count;
    size_t prediction_count;
    double total_cross_entropy;
    double average_cross_entropy;
    double perplexity;
    double elapsed_seconds;
} evaluation_report_t;

/**
 * Evaluate a corpus as one continuous token stream using a bounded
 * sliding context. The tokenizer and all persistent model/training state
 * remain unchanged.
 */
evaluation_errors_t evaluate_corpus_file(neural_model_t *model,
                                         bpe_encoder_t *encoder,
                                         const char *filename,
                                         size_t context_window,
                                         evaluation_report_t *out_report);

#endif /* CLI_EVALUATION_H */
