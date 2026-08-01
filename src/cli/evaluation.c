#include "cli/evaluation.h"
#include "cli/stream.h"
#include "core/evaluation.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define EVAL_FNV1A_OFFSET UINT64_C(14695981039346656037)
#define EVAL_FNV1A_PRIME UINT64_C(1099511628211)

static double monotonic_seconds(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return 0.0;
    return (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
}

static evaluation_errors_t evaluate_token(neural_model_t *model,
                                          uint32_t token,
                                          uint32_t *context,
                                          size_t *context_len,
                                          size_t context_window,
                                          evaluation_report_t *report) {
    report->token_count++;
    if (*context_len > 0) {
        double loss = 0.0;
        if (model_evaluate_step(model, context, token, *context_len, &loss) !=
            MODEL_SUCCESS) return EVALUATION_MODEL_ERROR;
        report->total_cross_entropy += loss;
        report->prediction_count++;
    }
    if (*context_len < context_window) {
        context[(*context_len)++] = token;
    } else {
        memmove(context, context + 1,
                (context_window - 1) * sizeof(*context));
        context[context_window - 1] = token;
    }
    return EVALUATION_SUCCESS;
}

evaluation_errors_t evaluate_corpus_file(neural_model_t *model,
                                         bpe_encoder_t *encoder,
                                         const char *filename,
                                         size_t context_window,
                                         evaluation_report_t *out_report) {
    if (!model || !encoder || !filename || !out_report || context_window == 0 ||
        context_window > model->max_seq_len ||
        encoder->max_vocab_size != model->vocab_size) {
        return EVALUATION_INVALID_INPUT;
    }

    memset(out_report, 0, sizeof(*out_report));
    out_report->corpus_fingerprint = EVAL_FNV1A_OFFSET;
    stream_reader_t *reader = stream_reader_create(filename, STREAM_CHUNK_SIZE);
    if (!reader) return EVALUATION_FILE_ERROR;

    uint32_t *context = malloc(context_window * sizeof(*context));
    char *chunk = malloc(STREAM_CHUNK_SIZE + 1);
    if (!context || !chunk) {
        free(context);
        free(chunk);
        stream_reader_free(reader);
        return EVALUATION_ALLOCATION_FAILURE;
    }

    evaluation_errors_t rc = EVALUATION_SUCCESS;
    size_t context_len = 0;
    double started = monotonic_seconds();
    if (bpe_encoder_has_special_tokens(encoder)) {
        rc = evaluate_token(model, BPE_BOS_TOKEN_ID, context, &context_len,
                            context_window, out_report);
    }

    while (rc == EVALUATION_SUCCESS && !stream_is_eof(reader)) {
        size_t chunk_size = stream_read_chunk(reader, chunk, STREAM_CHUNK_SIZE);
        if (chunk_size == 0) break;
        chunk[chunk_size] = '\0';
        out_report->corpus_bytes += chunk_size;
        for (size_t i = 0; i < chunk_size; i++) {
            out_report->corpus_fingerprint ^= (uint64_t)(unsigned char)chunk[i];
            out_report->corpus_fingerprint *= EVAL_FNV1A_PRIME;
        }

        bpe_tokens_t tokens = {0};
        if (bpe_encode(encoder, chunk, chunk_size, &tokens) != BPE_SUCCESS) {
            bpe_tokens_free(&tokens);
            rc = EVALUATION_TOKENIZER_ERROR;
            break;
        }

        for (size_t i = 0; i < tokens.token_count; i++) {
            rc = evaluate_token(model, tokens.token_ids[i], context,
                                &context_len, context_window, out_report);
            if (rc != EVALUATION_SUCCESS) break;
        }
        bpe_tokens_free(&tokens);
        if (rc != EVALUATION_SUCCESS) break;
    }

    if (rc == EVALUATION_SUCCESS && bpe_encoder_has_special_tokens(encoder)) {
        rc = evaluate_token(model, BPE_EOS_TOKEN_ID, context, &context_len,
                            context_window, out_report);
    }

    out_report->elapsed_seconds = monotonic_seconds() - started;
    if (rc == EVALUATION_SUCCESS && out_report->prediction_count == 0) {
        rc = EVALUATION_INVALID_INPUT;
    }
    if (rc == EVALUATION_SUCCESS) {
        out_report->average_cross_entropy =
            out_report->total_cross_entropy / (double)out_report->prediction_count;
        out_report->perplexity = exp(out_report->average_cross_entropy);
    }

    free(context);
    free(chunk);
    stream_reader_free(reader);
    return rc;
}
