#include "byte_pair_encoding.h"
#include "cli/generation.h"
#include "cli/sampling.h"
#include "core/model.h"
#include <float.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    char text[64];
    size_t text_length;
    uint32_t token_ids[64];
    size_t token_count;
    size_t cancel_after;
} stream_capture_t;

static int capture_token(uint32_t token_id, const char *text,
                         size_t text_length, void *user_data) {
    stream_capture_t *capture = user_data;
    if (!capture || capture->token_count >= 64 ||
        text_length > sizeof(capture->text) - capture->text_length) return 1;
    capture->token_ids[capture->token_count++] = token_id;
    memcpy(capture->text + capture->text_length, text, text_length);
    capture->text_length += text_length;
    return capture->cancel_after > 0 &&
           capture->token_count >= capture->cancel_after;
}

static void force_bias(neural_model_t *model, uint32_t first, float first_bias,
                       uint32_t second, float second_bias) {
    memset(model->params, 0,
           model->total_param_count * sizeof(*model->params));
    model->output_bias[first] = first_bias;
    model->output_bias[second] = second_bias;
}

static int prepare(const bpe_encoder_t *encoder, uint32_t *sequence,
                   size_t capacity, size_t *prompt_count) {
    const uint32_t prompt[] = {'p'};
    return generation_prepare_prompt(encoder, prompt, 1, sequence, capacity,
                                     prompt_count);
}

int main(void) {
    bpe_encoder_t encoder = {0};
    neural_model_t model = {0};
    int failed = 0;

    if (bpe_encoder_new_with_special_tokens(&encoder, 264) != BPE_SUCCESS ||
        bpe_encoder_freeze(&encoder) != BPE_SUCCESS ||
        model_new(&model, 264, 8, 2, 1, 12) != MODEL_SUCCESS) {
        fprintf(stderr, "generation-control setup failed\n");
        failed = 1;
        goto cleanup;
    }

    generation_options_t defaults;
    generation_options_init(&defaults);
    if (defaults.strategy != SAMPLING_GREEDY || defaults.top_k != 5 ||
        defaults.top_p != 0.9f || defaults.temperature != 0.8f ||
        defaults.repetition_penalty != 1.0f) {
        fprintf(stderr, "generation option defaults changed\n");
        failed = 1;
        goto cleanup;
    }

    float mask_logits[264] = {0};
    mask_logits[260] = 100.0f; /* unassigned learned-token slot */
    mask_logits['x'] = 90.0f;
    mask_logits[BPE_EOS_TOKEN_ID] = 80.0f;
    generation_mask_control_logits(&encoder, mask_logits, 264);
    if (mask_logits[260] != -FLT_MAX ||
        sample_greedy(mask_logits, 264) != 'x') {
        fprintf(stderr, "unassigned vocabulary slot remained sampleable\n");
        failed = 1;
        goto cleanup;
    }

    float penalty_logits[264] = {0};
    const uint32_t repeated[] = {'x', 'x'};
    penalty_logits['x'] = 10.0f;
    penalty_logits['y'] = 8.0f;
    penalty_logits['z'] = -2.0f;
    const uint32_t penalty_context[] = {'x', 'x', 'z'};
    generation_apply_repetition_penalty(penalty_logits, 264,
                                        penalty_context, 3, 2.0f);
    if (penalty_logits['x'] != 5.0f || penalty_logits['z'] != -4.0f ||
        sample_greedy(penalty_logits, 264) != 'y') {
        fprintf(stderr, "sign-aware repetition penalty changed semantics\n");
        failed = 1;
        goto cleanup;
    }

    uint32_t sequence[12] = {0};
    size_t prompt_count = 0;
    generation_stop_sequence_t stop = {
        .token_ids = repeated,
        .token_count = 2,
    };
    stream_capture_t capture = {0};
    generation_options_t options = {
        .strategy = SAMPLING_GREEDY,
        .temperature = 0.0f,
        .top_k = 1,
        .top_p = 0.9f,
        .repetition_penalty = 1.0f,
        .minimum_new_tokens = 1,
        .stop_sequences = &stop,
        .stop_sequence_count = 1,
        .on_token = capture_token,
        .callback_data = &capture,
    };
    generation_result_t result = {0};
    force_bias(&model, 'x', 100.0f, BPE_EOS_TOKEN_ID, 90.0f);
    if (prepare(&encoder, sequence, 12, &prompt_count) != 0 ||
        generation_decode_with_options(&model, &encoder, sequence,
                                       prompt_count, 5, &options,
                                       &result) != GENERATION_SUCCESS ||
        !result.stopped_on_stop_sequence || result.stop_sequence_index != 0 ||
        result.new_count != 3 || result.emitted_count != 1 ||
        capture.token_count != 1 || capture.text_length != 1 ||
        capture.text[0] != 'x') {
        fprintf(stderr, "caller stop sequence was emitted or ignored\n");
        failed = 1;
        goto cleanup;
    }

    memset(&capture, 0, sizeof(capture));
    memset(&result, 0, sizeof(result));
    memset(sequence, 0, sizeof(sequence));
    options.stop_sequences = NULL;
    options.stop_sequence_count = 0;
    options.minimum_new_tokens = 2;
    force_bias(&model, BPE_EOS_TOKEN_ID, 100.0f, 'x', 90.0f);
    if (prepare(&encoder, sequence, 12, &prompt_count) != 0 ||
        generation_decode_with_options(&model, &encoder, sequence,
                                       prompt_count, 5, &options,
                                       &result) != GENERATION_SUCCESS ||
        !result.stopped_on_eos || result.new_count != 3 ||
        result.emitted_count != 2 || capture.token_count != 2 ||
        capture.text_length != 2 || memcmp(capture.text, "xx", 2) != 0) {
        fprintf(stderr, "minimum length did not defer EOS or stream text\n");
        failed = 1;
        goto cleanup;
    }

    memset(&capture, 0, sizeof(capture));
    memset(&result, 0, sizeof(result));
    memset(sequence, 0, sizeof(sequence));
    capture.cancel_after = 2;
    options.minimum_new_tokens = 10;
    force_bias(&model, 'x', 100.0f, BPE_EOS_TOKEN_ID, 90.0f);
    if (prepare(&encoder, sequence, 12, &prompt_count) != 0 ||
        generation_decode_with_options(&model, &encoder, sequence,
                                       prompt_count, 5, &options,
                                       &result) != GENERATION_SUCCESS ||
        !result.stopped_by_callback || result.new_count != 2 ||
        result.emitted_count != 2 || result.total_count != prompt_count + 2 ||
        capture.token_count != 2) {
        fprintf(stderr, "callback cancellation did not stop at its accepted prefix\n");
        failed = 1;
        goto cleanup;
    }

    const uint32_t partial_ids[] = {'x', 'x', 'x'};
    generation_stop_sequence_t partial_stop = {
        .token_ids = partial_ids,
        .token_count = 3,
    };
    memset(&capture, 0, sizeof(capture));
    memset(&result, 0, sizeof(result));
    memset(sequence, 0, sizeof(sequence));
    options.minimum_new_tokens = 0;
    options.stop_sequences = &partial_stop;
    options.stop_sequence_count = 1;
    if (prepare(&encoder, sequence, 12, &prompt_count) != 0 ||
        generation_decode_with_options(&model, &encoder, sequence,
                                       prompt_count, 2, &options,
                                       &result) != GENERATION_SUCCESS ||
        result.stopped_on_stop_sequence || result.emitted_count != 2 ||
        capture.text_length != 2 || memcmp(capture.text, "xx", 2) != 0) {
        fprintf(stderr, "unfinished stop prefix was not flushed at the length limit\n");
        failed = 1;
        goto cleanup;
    }

    uint32_t long_sequence[24] = {0};
    memset(&capture, 0, sizeof(capture));
    memset(&result, 0, sizeof(result));
    options.stop_sequences = NULL;
    options.stop_sequence_count = 0;
    options.minimum_new_tokens = 100;
    options.sequence_capacity = 24;
    if (prepare(&encoder, long_sequence, 12, &prompt_count) != 0 ||
        generation_decode_with_options(&model, &encoder, long_sequence,
                                       prompt_count, 18, &options,
                                       &result) != GENERATION_SUCCESS ||
        result.new_count != 18 || result.emitted_count != 18 ||
        result.total_count != prompt_count + 18 ||
        result.stopped_on_eos || result.stopped_by_callback ||
        capture.token_count != 18) {
        fprintf(stderr, "generation did not continue beyond the KV window\n");
        failed = 1;
        goto cleanup;
    }

    generation_options_t invalid = options;
    invalid.repetition_penalty = 0.5f;
    if (generation_decode_with_options(&model, &encoder, sequence,
                                       prompt_count, 1, &invalid,
                                       &result) != GENERATION_INVALID_INPUT) {
        fprintf(stderr, "invalid runtime controls were accepted\n");
        failed = 1;
    }

cleanup:
    model_free(&model);
    bpe_encoder_free(&encoder);
    printf("\n%s\n", failed ? "GENERATION CONTROL CHECK FAILED"
                              : "GENERATION CONTROL CHECK PASSED");
    return failed ? 1 : 0;
}
