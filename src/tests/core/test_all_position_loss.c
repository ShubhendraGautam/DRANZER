/* All-position supervision (core/lm_head.c) against the single-target path
 * it replaces (model_accumulate_gradients).
 *
 * The claim being tested is the one the whole change rests on: because
 * attention is causally masked, position i's final hidden state depends only
 * on tokens 0..i, so supervising position i inside a length-T window must
 * produce the same gradient as supervising the last position of a length-i+1
 * window. If that holds, one forward pass legitimately yields T training
 * signals instead of one; if it does not, the change is silently training on
 * leaked future context and every downstream number is worthless.
 *
 * Test 3 is the one that would catch a leak. Tests 1 and 2 are structural
 * checks that would pass even with a leak, and exist to localize a failure
 * rather than to establish correctness on their own.
 *
 * Dropout is forced off throughout: the two paths would otherwise draw from
 * model->rng_state a different number of times and diverge for a reason that
 * has nothing to do with what is being measured.
 */
#include "core/model.h"
#include "core/training.h"
#include "core/lm_head.h"
#include "core/cpu_features.h"
#include "core/matmul.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define VOCAB 23
#define EMB 8
#define HEADS 2
#define LAYERS 2
#define MAX_SEQ 7
#define SEQ_LEN 5

/* Agreement is checked as |a-b| <= ATOL + RTOL*max(|a|,|b|), not as a bare
 * relative difference. The two are needed for different reasons and one
 * alone gives a misleading answer here:
 *
 * The paths compute the same quantity by different routes - the
 * single-target path takes softmax() then -log(), the all-position path a
 * max-subtracted log-sum-exp in double - and the default build's
 * -ffast-math reassociates reductions differently when the trip count
 * differs. So the disagreement is real but tiny, and which term of the
 * criterion catches it depends entirely on magnitude:
 *
 * At full magnitude the absolute error stays near the float epsilon of the
 * values themselves. Worst seen anywhere is 3.0e-7 on a gradient of 0.91 -
 * a relative 3e-7, four orders inside RTOL.
 *
 * Near zero the relative view inflates. The worst relative figures printed
 * below reach 1.3e-4 - past RTOL - on entries around 1.6e-4, which is an
 * absolute difference of about 2e-8. ATOL admits those, not RTOL, and with
 * roughly 50x of margin. The extreme case is an entry that is
 * mathematically exactly zero: a longer reduction summing an unsupervised
 * position's zeros lands on -4.9e-8 while the shorter one that never sees
 * them returns exact 0.0.
 *
 * Those figures were swept over both compilers and every dispatched kernel
 * (DRANZER_CPU_ISA=baseline/avx2/avx512), because the kernel decides how the
 * reductions associate and the worst case is not on the widest one -
 * avx2 produces the 1.3e-4 above while avx512 stays at 7.8e-5. A tolerance
 * calibrated only on this machine's default kernel would be pinned to an
 * instruction set rather than to the arithmetic, so the test reports the
 * detected ISA below and any failure should be read against it first.
 *
 * A bare relative test therefore reports a ~5e-5 "failure" that is an
 * artifact of dividing noise by nothing - which is exactly what an earlier
 * version of this file did, on a gradient that was correct. Both terms are
 * here because each one is load-bearing over a different part of the range,
 * and the printout shows both so a real failure says which kind it is. */
#define RTOL 1e-4f
#define ATOL 1e-6f

static void init_model(neural_model_t *model) {
    srand(1234);
    if (model_new(model, VOCAB, EMB, HEADS, LAYERS, MAX_SEQ) != MODEL_SUCCESS) {
        fprintf(stderr, "model_new failed\n");
        exit(1);
    }
    model->dropout_rate = 0.0f;
    model->is_training = 1;
}

/* Compare two gradient buffers under |a-b| <= ATOL + RTOL*max(|a|,|b|).
 * Reports the worst absolute and worst relative entry separately so a
 * failure says which kind of disagreement it is, and returns the number of
 * entries that failed the combined criterion. */
static size_t compare_gradients(const char *label, const float *a, const float *b,
                                size_t n) {
    size_t violations = 0;
    size_t first_violation = 0;
    double worst_abs = 0.0;
    size_t worst_abs_at = 0;
    double worst_rel = 0.0;
    size_t worst_rel_at = 0;

    for (size_t i = 0; i < n; i++) {
        double av = a[i], bv = b[i];
        double diff = fabs(av - bv);
        double scale = fmax(fabs(av), fabs(bv));

        if (diff > worst_abs) {
            worst_abs = diff;
            worst_abs_at = i;
        }
        /* Only meaningful where there is a magnitude to be relative to; the
         * absolute term is what guards the rest. */
        if (scale > 1e-4) {
            double rel = diff / scale;
            if (rel > worst_rel) {
                worst_rel = rel;
                worst_rel_at = i;
            }
        }

        if (diff > (double)ATOL + (double)RTOL * scale) {
            if (violations == 0) first_violation = i;
            violations++;
        }
    }

    printf("  %s over %zu params:\n", label, n);
    printf("    worst absolute %.3e at %zu (%+.6e vs %+.6e)\n",
           worst_abs, worst_abs_at, a[worst_abs_at], b[worst_abs_at]);
    printf("    worst relative %.3e at %zu (%+.6e vs %+.6e)\n",
           worst_rel, worst_rel_at, a[worst_rel_at], b[worst_rel_at]);

    if (violations) {
        printf("    %zu entries exceed ATOL+RTOL*scale, first at %zu (%+.6e vs %+.6e)\n",
               violations, first_violation, a[first_violation], b[first_violation]);
    }
    return violations;
}

static float *snapshot_grads(neural_model_t *model) {
    float *copy = malloc(model->total_param_count * sizeof(float));
    if (!copy) {
        fprintf(stderr, "allocation failed\n");
        exit(1);
    }
    memcpy(copy, model->grads, model->total_param_count * sizeof(float));
    return copy;
}

/* ---- Test 1: one supervised position must reproduce the old path exactly.
 *
 * With every target but the last ignored, the all-position path supervises
 * exactly one position and divides by a count of one, so it is computing
 * precisely what model_accumulate_gradients computes. Any disagreement here
 * is a bug in the head itself, before causality enters the picture. */
static int test_single_supervised_position_matches_old_path(void) {
    printf("--- one supervised position vs the single-target path ---\n");

    neural_model_t model;
    init_model(&model);

    uint32_t tokens[SEQ_LEN] = { 3, 9, 1, 17, 5 };
    uint32_t final_target = 11;

    model_zero_gradients(&model);
    float old_loss = 0.0f;
    if (model_accumulate_gradients(&model, tokens, final_target, SEQ_LEN,
                                   &old_loss) != MODEL_SUCCESS) {
        fprintf(stderr, "model_accumulate_gradients failed\n");
        return 1;
    }
    float *old_grads = snapshot_grads(&model);

    uint32_t targets[SEQ_LEN];
    for (size_t i = 0; i < SEQ_LEN; i++) targets[i] = LM_HEAD_IGNORE_TARGET;
    targets[SEQ_LEN - 1] = final_target;

    model_zero_gradients(&model);
    float new_loss = 0.0f;
    size_t supervised = 0;
    if (model_accumulate_gradients_all(&model, tokens, targets, SEQ_LEN,
                                       &new_loss, &supervised) != MODEL_SUCCESS) {
        fprintf(stderr, "model_accumulate_gradients_all failed\n");
        free(old_grads);
        return 1;
    }

    int failed = 0;
    if (supervised != 1) {
        printf("  FAIL: expected 1 supervised position, got %zu\n", supervised);
        failed = 1;
    }

    float loss_diff = fabsf(old_loss - new_loss);
    printf("  loss: old=%.8f new=%.8f  |diff|=%.3e\n", old_loss, new_loss, loss_diff);
    if (loss_diff > 1e-5f) {
        printf("  FAIL: losses disagree\n");
        failed = 1;
    }

    if (compare_gradients("gradients", old_grads, model.grads,
                          model.total_param_count)) {
        printf("  FAIL: gradient mismatch\n");
        failed = 1;
    }

    if (!failed) printf("  PASS\n");
    free(old_grads);
    model_free(&model);
    return failed;
}

/* ---- Test 2: ignored positions contribute nothing.
 *
 * Supervising position 0 alone inside a length-SEQ_LEN window must give the
 * same gradient as supervising it inside a length-1 window. The later
 * positions are computed either way; the point is that their unsupervised
 * rows add exactly zero rather than leaking a contribution through the head.
 */
static int test_ignored_positions_contribute_nothing(void) {
    printf("--- ignored positions contribute no gradient ---\n");

    neural_model_t model;
    init_model(&model);

    uint32_t tokens[SEQ_LEN] = { 3, 9, 1, 17, 5 };
    uint32_t target_after_first = 9;

    /* Long window, only position 0 supervised. */
    uint32_t targets[SEQ_LEN];
    for (size_t i = 0; i < SEQ_LEN; i++) targets[i] = LM_HEAD_IGNORE_TARGET;
    targets[0] = target_after_first;

    model_zero_gradients(&model);
    float long_loss = 0.0f;
    if (model_accumulate_gradients_all(&model, tokens, targets, SEQ_LEN,
                                       &long_loss, NULL) != MODEL_SUCCESS) {
        fprintf(stderr, "long-window call failed\n");
        return 1;
    }
    float *long_grads = snapshot_grads(&model);

    /* Length-1 window over the same first token. */
    uint32_t short_target[1] = { target_after_first };
    model_zero_gradients(&model);
    float short_loss = 0.0f;
    if (model_accumulate_gradients_all(&model, tokens, short_target, 1,
                                       &short_loss, NULL) != MODEL_SUCCESS) {
        fprintf(stderr, "short-window call failed\n");
        free(long_grads);
        return 1;
    }

    int failed = 0;
    printf("  loss: long-window=%.8f short-window=%.8f\n", long_loss, short_loss);
    if (fabsf(long_loss - short_loss) > 1e-5f) {
        printf("  FAIL: losses disagree\n");
        failed = 1;
    }

    if (compare_gradients("gradients", long_grads, model.grads,
                          model.total_param_count)) {
        printf("  FAIL: unsupervised positions changed the gradient\n");
        failed = 1;
    }

    if (!failed) printf("  PASS\n");
    free(long_grads);
    model_free(&model);
    return failed;
}

/* ---- Test 3: the causality claim itself.
 *
 * One all-position pass over a length-T window, every position supervised,
 * must equal the MEAN of T single-target passes over windows of length
 * 1, 2, ... T. The single-target runs are the ground truth: each one only
 * ever sees the tokens up to its own target, so if the all-position pass
 * agrees with their mean, no position in it can have used a token that
 * follows it.
 *
 * A leak would show here and nowhere else in this file. If the causal mask
 * were dropped, position 0's gradient inside the long window would depend on
 * tokens 1..T-1 while its single-target counterpart could not possibly see
 * them, and the two would diverge far past any tolerance. */
static int test_matches_mean_of_single_target_runs(void) {
    printf("--- all-position pass vs the mean of per-position runs ---\n");

    neural_model_t model;
    init_model(&model);

    /* tokens[i] is the input at position i; targets[i] is what follows it,
     * so the target sequence is the input shifted left by one with a fresh
     * token appended - exactly what a training window supplies. */
    uint32_t tokens[SEQ_LEN] = { 3, 9, 1, 17, 5 };
    uint32_t targets[SEQ_LEN] = { 9, 1, 17, 5, 11 };

    /* Reference: accumulate T single-target passes, each over the prefix
     * ending at the position being supervised. model_accumulate_gradients
     * adds into model->grads, so T calls without an intervening zero give
     * the sum directly. */
    model_zero_gradients(&model);
    double summed_loss = 0.0;
    for (size_t i = 0; i < SEQ_LEN; i++) {
        float step_loss = 0.0f;
        if (model_accumulate_gradients(&model, tokens, targets[i], i + 1,
                                       &step_loss) != MODEL_SUCCESS) {
            fprintf(stderr, "reference pass %zu failed\n", i);
            return 1;
        }
        summed_loss += step_loss;
    }
    /* The all-position path divides by the supervised count, so scale the
     * reference the same way rather than comparing a sum against a mean. */
    for (size_t i = 0; i < model.total_param_count; i++) {
        model.grads[i] /= (float)SEQ_LEN;
    }
    float *reference_grads = snapshot_grads(&model);
    float reference_loss = (float)(summed_loss / (double)SEQ_LEN);

    /* One pass, every position supervised. */
    model_zero_gradients(&model);
    float all_loss = 0.0f;
    size_t supervised = 0;
    if (model_accumulate_gradients_all(&model, tokens, targets, SEQ_LEN,
                                       &all_loss, &supervised) != MODEL_SUCCESS) {
        fprintf(stderr, "all-position pass failed\n");
        free(reference_grads);
        return 1;
    }

    int failed = 0;
    if (supervised != SEQ_LEN) {
        printf("  FAIL: expected %d supervised positions, got %zu\n",
               SEQ_LEN, supervised);
        failed = 1;
    }

    printf("  loss: %d separate passes=%.8f  one pass=%.8f  |diff|=%.3e\n",
           SEQ_LEN, reference_loss, all_loss, fabsf(reference_loss - all_loss));
    if (fabsf(reference_loss - all_loss) > 1e-5f) {
        printf("  FAIL: mean loss disagrees\n");
        failed = 1;
    }

    if (compare_gradients("gradients", reference_grads, model.grads,
                          model.total_param_count)) {
        printf("  FAIL: one pass does not reproduce %d passes - check causality\n",
               SEQ_LEN);
        failed = 1;
    }

    if (!failed) {
        printf("  PASS: %d gradient signals from one forward pass, and they are\n"
               "        the same signals %d separate passes produce\n",
               SEQ_LEN, SEQ_LEN);
    }
    free(reference_grads);
    model_free(&model);
    return failed;
}

/* ---- Test 4: an all-ignored window is a no-op, not a crash or a NaN. */
static int test_no_supervised_positions(void) {
    printf("--- window with no supervised position ---\n");

    neural_model_t model;
    init_model(&model);

    uint32_t tokens[SEQ_LEN] = { 3, 9, 1, 17, 5 };
    uint32_t targets[SEQ_LEN];
    for (size_t i = 0; i < SEQ_LEN; i++) targets[i] = LM_HEAD_IGNORE_TARGET;

    model_zero_gradients(&model);
    float loss = -1.0f;
    size_t supervised = 99;
    model_errors_t rc = model_accumulate_gradients_all(&model, tokens, targets,
                                                       SEQ_LEN, &loss, &supervised);

    int failed = 0;
    if (rc != MODEL_SUCCESS) {
        printf("  FAIL: returned %d rather than succeeding with an empty result\n", rc);
        failed = 1;
    }
    if (supervised != 0) {
        printf("  FAIL: expected 0 supervised positions, got %zu\n", supervised);
        failed = 1;
    }
    if (loss != 0.0f) {
        printf("  FAIL: expected zero loss, got %f\n", loss);
        failed = 1;
    }
    for (size_t i = 0; i < model.total_param_count; i++) {
        if (model.grads[i] != 0.0f) {
            printf("  FAIL: gradient index %zu is %g, expected 0\n", i, model.grads[i]);
            failed = 1;
            break;
        }
    }

    if (!failed) printf("  PASS\n");
    model_free(&model);
    return failed;
}

int main(void) {
    /* Which kernel ran decides how the reductions associate, so a
     * tolerance failure is meaningless without it - see the note on RTOL. */
    printf("=== All-position supervision ===\n");
    printf("cpu: %s | kernel: %s\n\n",
           cpu_features_summary(), matmul_kernel_name(matmul_select(SEQ_LEN, EMB, VOCAB)));

    int failed = 0;
    failed |= test_single_supervised_position_matches_old_path();
    printf("\n");
    failed |= test_ignored_positions_contribute_nothing();
    printf("\n");
    failed |= test_matches_mean_of_single_target_runs();
    printf("\n");
    failed |= test_no_supervised_positions();
    printf("\n");

    if (failed) {
        printf("=== FAILED ===\n");
        return 1;
    }
    printf("=== ALL PASSED ===\n");
    return 0;
}
