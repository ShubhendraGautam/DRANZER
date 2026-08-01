#ifndef CLI_GENERATION_H
#define CLI_GENERATION_H

#include "byte_pair_encoding.h"
#include "cli/sampling.h"
#include "core/model_types.h"
#include <stddef.h>
#include <stdint.h>

/* Build the model-visible prompt, prepending BOS in special-token mode and
 * retaining the newest prompt tokens when capacity is exceeded. */
int generation_prepare_prompt(const bpe_encoder_t *encoder,
                              const uint32_t *prompt_tokens,
                              size_t prompt_count,
                              uint32_t *output,
                              size_t capacity,
                              size_t *out_count);

/* PAD/UNK/BOS and unassigned vocabulary slots cannot be sampled as generated
 * content. EOS remains eligible so it can terminate generation naturally. */
void generation_mask_control_logits(const bpe_encoder_t *encoder,
                                    float *logits,
                                    size_t vocab_size);

int generation_token_is_eos(const bpe_encoder_t *encoder, uint32_t token_id);

/* Apply the conventional sign-aware repetition penalty once per token ID
 * already present in sequence. A penalty of 1 is a no-op. */
void generation_apply_repetition_penalty(float *logits,
                                         size_t vocab_size,
                                         const uint32_t *sequence,
                                         size_t sequence_count,
                                         float penalty);

typedef enum {
    GENERATION_SUCCESS = 0,
    GENERATION_INVALID_INPUT,
    GENERATION_ALLOCATION_FAILURE,
    GENERATION_MODEL_ERROR,
} generation_errors_t;

typedef struct {
    size_t total_count;
    size_t new_count;
    size_t emitted_count;
    int stopped_on_eos;
    int stopped_on_stop_sequence;
    size_t stop_sequence_index;
    int stopped_by_callback;
} generation_result_t;

typedef struct {
    const uint32_t *token_ids;
    size_t token_count;
} generation_stop_sequence_t;

/* `text` points into the tokenizer and is valid for the duration of the call.
 * Return nonzero to stop after accepting this token. */
typedef int (*generation_token_callback_t)(uint32_t token_id,
                                           const char *text,
                                           size_t text_length,
                                           void *user_data);

typedef struct {
    sampling_strategy_t strategy;
    float temperature;
    size_t top_k;
    float top_p;
    float repetition_penalty;
    size_t minimum_new_tokens;
    size_t sequence_capacity; /* 0 uses model->max_seq_len */
    const generation_stop_sequence_t *stop_sequences;
    size_t stop_sequence_count;
    generation_token_callback_t on_token;
    void *callback_data;
} generation_options_t;

/* Populate conservative defaults: greedy decoding, no penalty/minimum/stops,
 * top-k 5, top-p 0.9, and temperature 0.8. */
void generation_options_init(generation_options_t *options);

/* Incremental decode interface. `sequence_capacity` must cover the prepared
 * prompt plus any tokens the caller wants retained in `sequence`; zero keeps
 * the compatibility bound of model->max_seq_len. Callbacks receive generated text tokens only:
 * EOS and matched stop sequences are withheld. Stop matching is token-based,
 * considers generated tokens (not the prompt), and uses registration order to
 * break ties. At most the longest stop sequence is held pending; the complete
 * decoded continuation is never buffered. */
generation_errors_t generation_decode_with_options(
    neural_model_t *model,
    const bpe_encoder_t *encoder,
    uint32_t *sequence,
    size_t prompt_count,
    size_t requested_new_tokens,
    const generation_options_t *options,
    generation_result_t *out_result);

/* Extend an already prepared prompt in sequence (capacity is the model's
 * max sequence length). This compatibility wrapper uses no callback, stop
 * sequences, repetition penalty, or minimum length. */
generation_errors_t generation_decode(neural_model_t *model,
                                      const bpe_encoder_t *encoder,
                                      uint32_t *sequence,
                                      size_t prompt_count,
                                      size_t requested_new_tokens,
                                      sampling_strategy_t strategy,
                                      float temperature,
                                      size_t top_k,
                                      float top_p,
                                      generation_result_t *out_result);

#endif /* CLI_GENERATION_H */
