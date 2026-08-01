#include "cli/generation.h"
#include "core/model.h"
#include <float.h>
#include <string.h>

int generation_prepare_prompt(const bpe_encoder_t *encoder,
                              const uint32_t *prompt_tokens,
                              size_t prompt_count,
                              uint32_t *output,
                              size_t capacity,
                              size_t *out_count) {
    if (!encoder || !prompt_tokens || prompt_count == 0 || !output ||
        capacity == 0 || !out_count) return -1;

    size_t offset = 0;
    size_t available = capacity;
    if (bpe_encoder_has_special_tokens(encoder)) {
        output[offset++] = BPE_BOS_TOKEN_ID;
        available--;
    }
    size_t retained = prompt_count < available ? prompt_count : available;
    if (retained > 0) {
        memcpy(output + offset, prompt_tokens + prompt_count - retained,
               retained * sizeof(*output));
    }
    *out_count = offset + retained;
    return 0;
}

void generation_mask_control_logits(const bpe_encoder_t *encoder,
                                    float *logits,
                                    size_t vocab_size) {
    if (!encoder || !logits) return;
    if (bpe_encoder_has_special_tokens(encoder)) {
        const uint32_t excluded[] = {
            BPE_PAD_TOKEN_ID, BPE_UNK_TOKEN_ID, BPE_BOS_TOKEN_ID,
        };
        for (size_t i = 0; i < sizeof(excluded) / sizeof(excluded[0]); i++) {
            if (excluded[i] < vocab_size) logits[excluded[i]] = -FLT_MAX;
        }
    }
    for (size_t i = encoder->vocab_size; i < vocab_size; i++) {
        logits[i] = -FLT_MAX;
    }
}

int generation_token_is_eos(const bpe_encoder_t *encoder, uint32_t token_id) {
    return bpe_encoder_has_special_tokens(encoder) && token_id == BPE_EOS_TOKEN_ID;
}

void generation_apply_repetition_penalty(float *logits,
                                         size_t vocab_size,
                                         const uint32_t *sequence,
                                         size_t sequence_count,
                                         float penalty) {
    if (!logits || !sequence || penalty <= 1.0f) return;
    for (size_t i = 0; i < sequence_count; i++) {
        uint32_t token_id = sequence[i];
        if (token_id >= vocab_size) continue;
        int already_penalized = 0;
        for (size_t j = 0; j < i; j++) {
            if (sequence[j] == token_id) {
                already_penalized = 1;
                break;
            }
        }
        if (already_penalized) continue;
        if (logits[token_id] < 0.0f) logits[token_id] *= penalty;
        else logits[token_id] /= penalty;
    }
}

static int generation_float_is_finite(float value) {
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
}

void generation_options_init(generation_options_t *options) {
    if (!options) return;
    memset(options, 0, sizeof(*options));
    options->strategy = SAMPLING_GREEDY;
    options->temperature = 0.8f;
    options->top_k = 5;
    options->top_p = 0.9f;
    options->repetition_penalty = 1.0f;
}

static int options_are_valid(const neural_model_t *model,
                             const bpe_encoder_t *encoder,
                             const generation_options_t *options) {
    if (!options || options->strategy < SAMPLING_GREEDY ||
        options->strategy > SAMPLING_TOPP ||
        !generation_float_is_finite(options->temperature) ||
        options->temperature < 0.0f ||
        !generation_float_is_finite(options->top_p) ||
        !generation_float_is_finite(options->repetition_penalty) ||
        options->repetition_penalty < 1.0f ||
        (options->strategy == SAMPLING_TOPK && options->top_k == 0) ||
        (options->strategy == SAMPLING_TOPP &&
         (options->top_p <= 0.0f || options->top_p > 1.0f)) ||
        (options->stop_sequence_count > 0 && !options->stop_sequences)) {
        return 0;
    }
    for (size_t i = 0; i < options->stop_sequence_count; i++) {
        const generation_stop_sequence_t *stop = &options->stop_sequences[i];
        if (!stop->token_ids || stop->token_count == 0) return 0;
        for (size_t j = 0; j < stop->token_count; j++) {
            uint32_t token_id = stop->token_ids[j];
            if (token_id >= encoder->vocab_size ||
                bpe_token_is_control(encoder, token_id) ||
                token_id >= model->vocab_size) {
                return 0;
            }
        }
    }
    return 1;
}

static int tokens_equal(const uint32_t *left, const uint32_t *right,
                        size_t count) {
    return count == 0 || memcmp(left, right, count * sizeof(*left)) == 0;
}

static int matching_stop_sequence(const uint32_t *generated,
                                  size_t generated_count,
                                  const generation_options_t *options,
                                  size_t *out_index,
                                  size_t *out_start) {
    for (size_t i = 0; i < options->stop_sequence_count; i++) {
        const generation_stop_sequence_t *stop = &options->stop_sequences[i];
        if (stop->token_count > generated_count) continue;
        size_t start = generated_count - stop->token_count;
        if (start < options->minimum_new_tokens) continue;
        if (tokens_equal(generated + start, stop->token_ids,
                         stop->token_count)) {
            *out_index = i;
            *out_start = start;
            return 1;
        }
    }
    return 0;
}

/* Return the longest not-yet-emitted suffix that can still grow into a stop
 * sequence. Everything before it is safe to deliver immediately. */
static size_t pending_stop_prefix(const uint32_t *generated,
                                  size_t generated_count,
                                  size_t delivered_count,
                                  const generation_options_t *options) {
    size_t best = 0;
    for (size_t i = 0; i < options->stop_sequence_count; i++) {
        const generation_stop_sequence_t *stop = &options->stop_sequences[i];
        size_t maximum = stop->token_count > 0 ? stop->token_count - 1 : 0;
        size_t pending = generated_count - delivered_count;
        if (maximum > pending) maximum = pending;
        for (size_t length = maximum; length > best; length--) {
            size_t start = generated_count - length;
            if (start < delivered_count ||
                start < options->minimum_new_tokens) continue;
            if (tokens_equal(generated + start, stop->token_ids, length)) {
                best = length;
                break;
            }
        }
    }
    return best;
}

/* Deliver generated tokens [*delivered_count, target_count). */
static int emit_until(const bpe_encoder_t *encoder,
                      uint32_t *sequence,
                      size_t prompt_count,
                      size_t target_count,
                      const generation_options_t *options,
                      size_t *delivered_count,
                      generation_result_t *result) {
    while (*delivered_count < target_count) {
        uint32_t token_id = sequence[prompt_count + *delivered_count];
        (*delivered_count)++;
        result->emitted_count = *delivered_count;
        if (!options->on_token) continue;

        const char *text = "";
        size_t text_length = 0;
        if (token_id < encoder->vocab_size &&
            !bpe_token_is_control(encoder, token_id) &&
            encoder->tokens[token_id].token) {
            text = encoder->tokens[token_id].token;
            text_length = strlen(text);
        }
        if (options->on_token(token_id, text, text_length,
                              options->callback_data) != 0) {
            result->stopped_by_callback = 1;
            result->new_count = *delivered_count;
            result->total_count = prompt_count + *delivered_count;
            return 1;
        }
    }
    return 0;
}

generation_errors_t generation_decode_with_options(
    neural_model_t *model,
    const bpe_encoder_t *encoder,
    uint32_t *sequence,
    size_t prompt_count,
    size_t requested_new_tokens,
    const generation_options_t *options,
    generation_result_t *out_result) {
    if (!model || !encoder || !sequence || !out_result || prompt_count == 0 ||
        prompt_count > model->max_seq_len ||
        encoder->max_vocab_size != model->vocab_size ||
        encoder->vocab_size > model->vocab_size ||
        !options_are_valid(model, encoder, options)) {
        return GENERATION_INVALID_INPUT;
    }
    memset(out_result, 0, sizeof(*out_result));
    out_result->total_count = prompt_count;
    out_result->stop_sequence_index = SIZE_MAX;

    size_t sequence_capacity = options->sequence_capacity
                                   ? options->sequence_capacity
                                   : model->max_seq_len;
    if (prompt_count > sequence_capacity) return GENERATION_INVALID_INPUT;

    model_kv_cache_t cache = {0};
    model_errors_t cache_rc = model_kv_cache_init(&cache, model);
    if (cache_rc != MODEL_SUCCESS) {
        return cache_rc == MODEL_ALLOCATION_FAILURE
                   ? GENERATION_ALLOCATION_FAILURE : GENERATION_MODEL_ERROR;
    }

    int prior_training = model->is_training;
    model->is_training = 0;
    for (size_t i = 0; i < prompt_count; i++) {
        if (model_forward_token(model, &cache, sequence[i], model->ws_logits) !=
            MODEL_SUCCESS) {
            model->is_training = prior_training;
            model_kv_cache_free(&cache);
            return GENERATION_MODEL_ERROR;
        }
    }

    size_t delivered_count = 0;
    size_t available = sequence_capacity - prompt_count;
    size_t limit = requested_new_tokens < available
                       ? requested_new_tokens : available;
    for (size_t i = 0; i < limit; i++) {
        generation_apply_repetition_penalty(
            model->ws_logits, model->vocab_size, sequence,
            out_result->total_count, options->repetition_penalty);
        generation_mask_control_logits(encoder, model->ws_logits,
                                       model->vocab_size);
        if (bpe_encoder_has_special_tokens(encoder) &&
            out_result->new_count < options->minimum_new_tokens &&
            BPE_EOS_TOKEN_ID < model->vocab_size) {
            model->ws_logits[BPE_EOS_TOKEN_ID] = -FLT_MAX;
        }

        uint32_t next = sample_next_token(
            model->ws_logits, model->vocab_size, options->strategy,
            options->temperature, options->top_k, options->top_p);
        sequence[out_result->total_count++] = next;
        out_result->new_count++;

        if (generation_token_is_eos(encoder, next)) {
            size_t content_count = out_result->new_count - 1;
            if (emit_until(encoder, sequence, prompt_count, content_count,
                           options, &delivered_count, out_result)) break;
            out_result->stopped_on_eos = 1;
            break;
        }

        size_t stop_index = 0, stop_start = 0;
        const uint32_t *generated = sequence + prompt_count;
        if (matching_stop_sequence(generated, out_result->new_count, options,
                                   &stop_index, &stop_start)) {
            if (emit_until(encoder, sequence, prompt_count, stop_start,
                           options, &delivered_count, out_result)) break;
            out_result->stopped_on_stop_sequence = 1;
            out_result->stop_sequence_index = stop_index;
            break;
        }

        size_t held = pending_stop_prefix(generated, out_result->new_count,
                                          delivered_count, options);
        if (emit_until(encoder, sequence, prompt_count,
                       out_result->new_count - held, options,
                       &delivered_count, out_result)) break;

        if (i + 1 < limit &&
            model_forward_token(model, &cache, next, model->ws_logits) !=
                MODEL_SUCCESS) {
            model->is_training = prior_training;
            model_kv_cache_free(&cache);
            return GENERATION_MODEL_ERROR;
        }
    }

    if (!out_result->stopped_on_eos &&
        !out_result->stopped_on_stop_sequence &&
        !out_result->stopped_by_callback) {
        (void)emit_until(encoder, sequence, prompt_count,
                         out_result->new_count, options,
                         &delivered_count, out_result);
    }

    model->is_training = prior_training;
    model_kv_cache_free(&cache);
    return GENERATION_SUCCESS;
}

generation_errors_t generation_decode(neural_model_t *model,
                                      const bpe_encoder_t *encoder,
                                      uint32_t *sequence,
                                      size_t prompt_count,
                                      size_t requested_new_tokens,
                                      sampling_strategy_t strategy,
                                      float temperature,
                                      size_t top_k,
                                      float top_p,
                                      generation_result_t *out_result) {
    generation_options_t options;
    generation_options_init(&options);
    options.strategy = strategy;
    options.temperature = temperature;
    options.top_k = top_k;
    options.top_p = top_p;
    return generation_decode_with_options(model, encoder, sequence,
                                          prompt_count, requested_new_tokens,
                                          &options, out_result);
}
