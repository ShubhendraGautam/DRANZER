/* Numerical gradient check for the hand-rolled backprop in core/transformer.c
 * and core/training.c. There is no autodiff here, so this is the correctness
 * backstop for every gradient the project computes.
 *
 * What this replaced, and why
 * ---------------------------
 * The previous version had three defects that together let a wrong gradient
 * pass, and each one is easy to reintroduce, so they are written down.
 *
 * 1. It read the gradient out of an optimizer step: it called
 *    model_train_step() and recovered `(original - new_value) / lr`. That
 *    equals the gradient only for plain SGD with no weight decay, no clipping,
 *    and no schedule, so the test could not tell a wrong gradient from a wrong
 *    optimizer - and it would have silently stopped testing gradients at all
 *    the moment the default optimizer path changed. It now reads model->grads
 *    directly and never runs an optimizer.
 *
 * 2. Its pass criterion was `rel_error < 0.05 || diff < 1e-3`. The absolute
 *    clause was load-bearing rather than a nicety: the CI log showed
 *    `layer1.W_q[1] analytical=+0.000125 numerical=+0.000000 rel_err=1.0000
 *    PASS`. A gradient that disagreed completely passed because it was small.
 *    There is no absolute escape hatch here.
 *
 * 3. It perturbed 19 hand-picked scalars. Everything else - most of every
 *    tensor, and any tensor nobody thought to list - was unchecked. This walks
 *    the parameter inventory (core/model_params.h) and covers every tensor the
 *    model has, so a tensor added to model_new() cannot escape it.
 *
 * The method
 * ----------
 * A directional derivative rather than per-element differences. For each
 * tensor, draw a fixed random direction u, then compare
 *
 *     analytical:  g . u          (one dot product against model->grads)
 *     numerical:   (L(p + hu) - L(p - hu)) / 2h
 *
 * One comparison per tensor tests every element of it at once: an error in any
 * single element moves the dot product. It also has no small-gradient corner -
 * the quantity compared is a sum over the whole tensor, so it is not near zero
 * merely because one element's gradient is.
 *
 * u has entries of +1 and -1. The scale of u does not matter - it cancels in the
 * relative error - but the per-element step does, and it is chosen per tensor
 * rather than fixed. See compare_tensor() for the derivation; briefly, the step
 * must be large enough to clear the float rounding in the forward pass and small
 * enough that no ReLU unit switches sign between the two evaluations, and where
 * that window sits varies between tensors by orders of magnitude.
 *
 * The kink constraint is the one that matters and it is measured, not assumed: a
 * tensor with even one switched ReLU unit disagrees by 1e-2 or worse, one with
 * none agrees to 1e-4 or better, and the count is printed per tensor so the
 * reader can see which regime each row is in.
 *
 * What limits it
 * --------------
 * Losses are accumulated in double, but the forward pass is float, so the logits
 * carry about seven digits and the difference of two nearly equal losses cancels
 * away most of them. That is the floor here, and it is a real one: the same
 * fixture read between 3.62e-03 and 1.10e-02 across six toolchains (see
 * TOLERANCE), which is compilers reassociating float arithmetic differently, not
 * gradients changing.
 *
 * So the tolerance is set from that measured spread rather than from an ideal,
 * and because a tolerance chosen for headroom might be too loose to catch a real
 * error, the end of main() corrupts a gradient by 5% and requires the identical
 * comparison to reject it. The check's sensitivity is therefore measured rather
 * than assumed: it catches a systematic error of about 3% or larger.
 */
#include "core/model.h"
#include "core/model_params.h"
#include "core/lm_head.h"
#include "core/rng.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VOCAB 16
#define EMB 8
#define HEADS 2
#define LAYERS 2
#define MAX_SEQ 6
#define SEQ_LEN 5
#define SEED 42

/* The step for each tensor is chosen per tensor, from the loss change it
 * produces - see the derivation at the call site. These bound that search.
 *
 * PROBE_STEP is a perturbation norm small enough not to switch a ReLU unit on
 * this fixture (measured: zero kinks on all 27 tensors), used only to find out
 * how sensitive the loss is to this tensor.
 *
 * TARGET_LOSS_CHANGE is what |L(p+hu) - L(p-hu)| is then aimed at. The float
 * forward pass carries about seven digits, so the loss is known to roughly 1e-7;
 * targeting 1e-3 leaves four orders of magnitude of signal above that, which
 * caps the cancellation contribution to the relative error at about 1e-4. */
#define PROBE_STEP 3e-4
#define TARGET_LOSS_CHANGE 1e-3
#define MIN_STEP 1e-7
/* Capped so a tensor with a very small gradient cannot be given a step large
 * enough for the central difference's own O(h^2) truncation error to dominate.
 * Above about 1e-2 that error is what is being measured. */
#define MAX_STEP 3e-3
/* How many times the step may be halved chasing a kink-free evaluation. Twenty
 * halvings is a factor of a million, well past MIN_STEP from any starting
 * point, so this bounds the loop rather than shaping the result. */
#define MAX_HALVINGS 20

/* The largest relative disagreement any tensor may show.
 *
 * Measured, not chosen. The same fixture was run under six toolchains - clang
 * and gcc, at -O0, -O2, -O3, and -Os, with and without -ffast-math and OpenMP -
 * and the worst tensor read:
 *
 *     gcc -O3 -ffast-math          3.62e-03
 *     gcc -O3 -ffast-math -fopenmp 3.62e-03
 *     clang -O3 -ffast-math        8.07e-03
 *     clang -Os -ffast-math        5.59e-03
 *     gcc -O2                      1.10e-02
 *     clang -O0                    1.10e-02
 *
 * with zero switched ReLU units in every one, so all six are measuring a
 * derivative rather than a kink. The spread is float rounding in the forward
 * pass being reassociated differently by each compiler, and it is the floor on
 * this method here: the loss is computed in double from logits that are not.
 *
 * 3e-2 is 2.7x above the worst of those, so a correct tree passes on all six
 * with headroom, and the accompanying power check below measures what that
 * costs in sensitivity rather than leaving it to be assumed. */
#define TOLERANCE 3e-2

/* The power check: a gradient scaled by this factor must be caught. 1.05 is a
 * 5% error, which is far smaller than any plausible sign, transpose, or
 * missing-term bug and comfortably larger than the 1.1e-2 noise floor above. */
#define INJECTED_ERROR_FACTOR 1.05f

/* How many ReLU units the model has across all layers and positions, which is
 * the size of the activity pattern recorded below. */
#define RELU_UNITS (LAYERS * SEQ_LEN * EMB * 4)

/* Mean cross-entropy over all supervised positions, computed in double from the
 * logits the forward pass just wrote, plus (optionally) which ReLU units were on.
 *
 * This has to be the same function lm_head_loss_and_grad_all() differentiates or
 * the whole check is meaningless, so it is written in the same form - the
 * max-subtracted log-sum-exp, divided by the number of supervised positions -
 * and main() asserts the two agree before differentiating anything. */
static double forward_loss(neural_model_t *model, uint32_t *tokens,
                           const uint32_t *targets, size_t seq_len,
                           unsigned char *relu_active) {
    if (model_forward_hidden(model, tokens, seq_len) != MODEL_SUCCESS ||
        lm_head_forward_all(model, seq_len) != MODEL_SUCCESS) {
        fprintf(stderr, "forward pass failed inside the gradient check\n");
        exit(1);
    }

    /* Which ReLU units are on. cache_ff_hidden is post-ReLU, so a zero is an
     * inactive unit - see the kink discussion in main(). */
    if (relu_active) {
        size_t ffn_dim = EMB * 4;
        size_t k = 0;
        for (size_t l = 0; l < LAYERS; l++) {
            for (size_t i = 0; i < seq_len; i++) {
                for (size_t d = 0; d < ffn_dim; d++) {
                    relu_active[k++] =
                        model->cache_ff_hidden[l][i * ffn_dim + d] > 0.0f;
                }
            }
        }
    }

    const float *logits = model->ws_logits_all;
    double total = 0.0;
    size_t supervised = 0;

    for (size_t i = 0; i < seq_len; i++) {
        const float *row = &logits[i * VOCAB];
        uint32_t target = targets[i];
        if (target == LM_HEAD_IGNORE_TARGET || target >= VOCAB) continue;

        float max_logit = row[0];
        for (size_t v = 1; v < VOCAB; v++) {
            if (row[v] > max_logit) max_logit = row[v];
        }
        double exp_sum = 0.0;
        for (size_t v = 0; v < VOCAB; v++) {
            exp_sum += exp((double)row[v] - (double)max_logit);
        }
        total += log(exp_sum) + (double)max_logit - (double)row[target];
        supervised++;
    }

    return supervised > 0 ? total / (double)supervised : 0.0;
}

/* (L(p + h*dir) - L(p - h*dir)) / 2h over one tensor, restoring the parameters
 * afterwards from `backup` rather than by adding h back - the round trip
 * p += h; p -= h does not always land on the original float.
 *
 * `out_kinks`, if given, receives the number of ReLU units that were on at one
 * evaluation and off at the other. Those are the units whose derivative changed
 * between the two points, and they are the reason the agreement below is what it
 * is rather than machine precision. */
static double directional_numerical(neural_model_t *model, param_tensor_t *tensor,
                                    const float *backup, const float *direction,
                                    size_t count, double step,
                                    uint32_t *tokens, const uint32_t *targets,
                                    size_t *out_kinks) {
    static unsigned char active_plus[RELU_UNITS];
    static unsigned char active_minus[RELU_UNITS];

    for (size_t i = 0; i < count; i++) {
        tensor->values[i] = (float)((double)backup[i] + step * (double)direction[i]);
    }
    double loss_plus = forward_loss(model, tokens, targets, SEQ_LEN, active_plus);

    for (size_t i = 0; i < count; i++) {
        tensor->values[i] = (float)((double)backup[i] - step * (double)direction[i]);
    }
    double loss_minus = forward_loss(model, tokens, targets, SEQ_LEN, active_minus);

    memcpy(tensor->values, backup, count * sizeof(float));

    if (out_kinks) {
        size_t kinks = 0;
        for (size_t i = 0; i < RELU_UNITS; i++) {
            if (active_plus[i] != active_minus[i]) kinks++;
        }
        *out_kinks = kinks;
    }
    return (loss_plus - loss_minus) / (2.0 * step);
}

/* One tensor's comparison: the analytical directional derivative against the
 * numerical one, with the step chosen as described below.
 *
 * `grad` is passed in rather than derived from the tensor so that the power
 * check at the end of main() can run this identical comparison against a
 * deliberately corrupted gradient. If that returns a relative error below the
 * tolerance, the tolerance is too loose to catch a real error and the test says
 * so instead of quietly passing. */
typedef struct {
    double analytical;
    double numerical;
    double at_half_step;
    double rel_error;
    double half_rel_error;
    double step;
    size_t kinks;
    size_t halvings;
} comparison_t;

static comparison_t compare_tensor(neural_model_t *model, param_tensor_t *tensor,
                                   const float *grad, const float *direction,
                                   const float *backup, size_t count,
                                   uint32_t *tokens, const uint32_t *targets) {
    comparison_t out;
    memset(&out, 0, sizeof(out));

    for (size_t i = 0; i < count; i++) {
        out.analytical += (double)grad[i] * (double)direction[i];
    }

    /* Choose this tensor's step in two stages, both of which look only at loss
     * values and never at the analytical gradient, so neither can tune itself
     * towards agreement.
     *
     * Stage 1, signal. The step has to be large enough that
     * L(p+hu) - L(p-hu) clears the float rounding in the forward pass. How large
     * that is differs per tensor by orders of magnitude: a layer-norm gain with
     * a directional derivative of 0.009 needs a step a hundred times larger than
     * W_ff1's to move the loss as far. So probe small, then scale by how far the
     * observed change is from the target. The change is locally linear in the
     * step, so one correction lands close enough.
     *
     * Stage 2, kinks. The step also has to be small enough that no ReLU unit
     * switches between the two evaluations. A switch means the two sides of the
     * difference sit on different linear pieces of the function, so the quotient
     * is not an estimate of the derivative at all, and no amount of precision
     * repairs it.
     *
     * That is measured, not assumed. Before the halving loop existed, every
     * tensor with at least one switched unit disagreed by 1e-2 or worse and
     * every tensor with none agreed to 1e-4 or better - a clean split across all
     * 27, with the worst reading (W_o, five switched units) at 8.8e-1. That is
     * what the loop removes, and it is why the kink column is printed: a nonzero
     * entry means the number beside it is not a derivative comparison. */
    double probe_step = PROBE_STEP / sqrt((double)count);
    size_t probe_kinks = 0;
    double probe = directional_numerical(model, tensor, backup, direction, count,
                                         probe_step, tokens, targets, &probe_kinks);
    double observed_change = fabs(probe) * 2.0 * probe_step;
    out.step = probe_step;
    if (observed_change > 0.0) {
        out.step = probe_step * (TARGET_LOSS_CHANGE / observed_change);
        if (out.step < MIN_STEP) out.step = MIN_STEP;
        if (out.step > MAX_STEP) out.step = MAX_STEP;
    }

    out.numerical = directional_numerical(model, tensor, backup, direction, count,
                                          out.step, tokens, targets, &out.kinks);
    while (out.kinks > 0 && out.step > MIN_STEP && out.halvings < MAX_HALVINGS) {
        out.step /= 2.0;
        out.halvings++;
        out.numerical = directional_numerical(model, tensor, backup, direction,
                                              count, out.step, tokens, targets,
                                              &out.kinks);
    }

    size_t ignored = 0;
    out.at_half_step = directional_numerical(model, tensor, backup, direction,
                                             count, out.step / 2.0, tokens,
                                             targets, &ignored);

    double scale = fmax(fabs(out.analytical), fabs(out.numerical));
    out.rel_error = scale > 0.0
                        ? fabs(out.numerical - out.analytical) / scale : 0.0;
    double half_scale = fmax(fabs(out.analytical), fabs(out.at_half_step));
    out.half_rel_error = half_scale > 0.0
                             ? fabs(out.at_half_step - out.analytical) / half_scale
                             : 0.0;
    return out;
}

int main(void) {
    uint32_t tokens[SEQ_LEN] = {1, 5, 3, 9, 2};
    /* targets[i] is the token that should follow position i: the input shifted
     * left by one, with a token after the window as the final target. */
    uint32_t targets[SEQ_LEN] = {5, 3, 9, 2, 7};

    neural_model_t model = {0};
    if (model_new_seeded(&model, VOCAB, EMB, HEADS, LAYERS, MAX_SEQ, SEED) !=
        MODEL_SUCCESS) {
        fprintf(stderr, "model_new_seeded failed\n");
        return 1;
    }
    /* Dropout off: a stochastic forward pass has no gradient a finite
     * difference can check, because the two perturbed evaluations would sample
     * different masks. tests/core/test_dropout_gradient_check.c covers that
     * path separately, holding the mask fixed. */
    model.dropout_rate = 0.0f;

    /* One training step first, so the check runs at a point with a non-trivial
     * gradient rather than at initialization, where several tensors sit at
     * values symmetric enough to hide a sign error. */
    if (model_train_step(&model, tokens, targets[SEQ_LEN - 1], SEQ_LEN) !=
        MODEL_SUCCESS) {
        fprintf(stderr, "warm-up training step failed\n");
        model_free(&model);
        return 1;
    }

    /* The gradient under test: one backward pass, no optimizer, nothing else
     * touched. Everything below reads model->grads. */
    model_zero_gradients(&model);
    float reported_loss = 0.0f;
    size_t supervised = 0;
    if (model_accumulate_gradients_all(&model, tokens, targets, SEQ_LEN,
                                       &reported_loss, &supervised) !=
        MODEL_SUCCESS) {
        fprintf(stderr, "model_accumulate_gradients_all failed\n");
        model_free(&model);
        return 1;
    }
    if (supervised != SEQ_LEN) {
        fprintf(stderr, "expected %d supervised positions, got %zu\n",
                SEQ_LEN, supervised);
        model_free(&model);
        return 1;
    }

    /* Is the loss being differentiated the loss the backward pass used? If these
     * disagree, every comparison below is against the wrong function and would
     * be reported as a gradient error. A float loss against a double
     * recomputation, so the gap allowed is float rounding on a value near 3. */
    double independent_loss = forward_loss(&model, tokens, targets, SEQ_LEN, NULL);
    double loss_gap = fabs(independent_loss - (double)reported_loss);
    printf("loss cross-check: model reports %.9f, recomputed %.9f (gap %.2e)\n",
           (double)reported_loss, independent_loss, loss_gap);
    if (loss_gap > 1e-5) {
        fprintf(stderr, "FAIL: this test is not differentiating the loss the "
                        "backward pass used\n");
        model_free(&model);
        return 1;
    }

    size_t tensor_count = model_param_tensor_count(&model);
    param_tensor_t *tensors = malloc(tensor_count * sizeof(*tensors));
    if (!tensors ||
        model_param_tensors(&model, tensors, tensor_count) != tensor_count) {
        fprintf(stderr, "could not enumerate the parameter inventory\n");
        free(tensors);
        model_free(&model);
        return 1;
    }

    /* One fixed direction stream for the whole run, so a failure reproduces and
     * a rerun does not roll different directions. */
    uint64_t rng = dranzer_rng_stream(SEED, DRANZER_RNG_STREAM_TESTING);

    printf("\n%-24s %6s %14s %14s %11s %11s %9s %6s\n", "tensor", "n",
           "analytical", "numerical", "rel_err", "at h/2", "step", "kinks");

    size_t failures = 0;
    size_t zero_gradient_tensors = 0;
    double worst_rel_error = 0.0;
    const char *worst_tensor = "none";
    size_t improved_at_half_step = 0, measured_at_half_step = 0;
    size_t total_kinks = 0, total_halvings = 0;

    for (size_t t = 0; t < tensor_count; t++) {
        param_tensor_t *tensor = &tensors[t];
        size_t count = tensor->rows * tensor->cols;
        /* grads has exactly the layout params has (see model_types.h), so a
         * tensor's gradient sits at the same offset in the other buffer. */
        const float *grad = model.grads + (tensor->values - model.params);

        float *backup = malloc(count * sizeof(float));
        float *direction = malloc(count * sizeof(float));
        if (!backup || !direction) {
            fprintf(stderr, "allocation failed for %s\n", tensor->name);
            free(backup);
            free(direction);
            free(tensors);
            model_free(&model);
            return 1;
        }
        memcpy(backup, tensor->values, count * sizeof(float));

        for (size_t i = 0; i < count; i++) {
            direction[i] = (dranzer_rng_unit(&rng) < 0.5) ? -1.0f : 1.0f;
        }

        comparison_t c = compare_tensor(&model, tensor, grad, direction, backup,
                                        count, tokens, targets);
        double analytical = c.analytical, numerical = c.numerical;
        double rel_error = c.rel_error, half_rel = c.half_rel_error;
        double step = c.step;
        size_t kinks = c.kinks, halvings = c.halvings;

        /* A tensor whose analytical and numerical directional derivatives are
         * both exactly zero is not a pass. Either it receives no gradient - in
         * which case it is untrained, and that belongs in the model's
         * documentation rather than silently in a test - or the direction came
         * out orthogonal to the gradient, which a random +-1 vector over a
         * non-trivial gradient does not do. Counted, and failed. */
        int no_signal = (analytical == 0.0 && numerical == 0.0);
        if (no_signal) zero_gradient_tensors++;

        /* A tensor still carrying a switched unit after the halving loop has not
         * been measured, whatever its relative error says, so it is not allowed
         * to pass on a number that does not mean what the column header says. */
        int pass = !no_signal && kinks == 0 && rel_error < TOLERANCE;
        if (!pass) failures++;
        if (rel_error > worst_rel_error) {
            worst_rel_error = rel_error;
            worst_tensor = tensor->name;
        }
        if (!no_signal) {
            measured_at_half_step++;
            if (half_rel <= rel_error) improved_at_half_step++;
        }

        const char *verdict = pass ? "PASS"
                            : no_signal ? "FAIL (no gradient signal)"
                            : kinks > 0 ? "FAIL (could not reach a kink-free step)"
                                        : "FAIL";
        printf("%-24s %6zu %14.8f %14.8f %11.2e %11.2e %9.1e %6zu  %s\n",
               tensor->name, count, analytical, numerical, rel_error, half_rel,
               step, kinks, verdict);
        total_kinks += kinks;
        total_halvings += halvings;

        free(backup);
        free(direction);
    }

    printf("\n%zu tensors checked; each tensor's direction covers every element "
           "of it\n", tensor_count);
    printf("worst relative disagreement: %.2e on %s (tolerance %.0e)\n",
           worst_rel_error, worst_tensor, TOLERANCE);
    /* The regime report. If most tensors improve when the step halves, the
     * residual disagreement is the finite-difference method's truncation error,
     * and the gradients are as verified as this method can verify them. If most
     * got worse, the method has hit float cancellation and the tolerance above
     * is measuring the machine rather than the code. */
    printf("%zu step halvings were needed to reach a kink-free evaluation; "
           "%zu switched ReLU units remain\n", total_halvings, total_kinks);
    printf("agreement improved at h/2 for %zu of %zu tensors: %s\n",
           improved_at_half_step, measured_at_half_step,
           improved_at_half_step * 2 >= measured_at_half_step
               ? "limited by the finite-difference method, as intended"
               : "limited by float cancellation in the forward pass");

    /* ---- Power check: does the tolerance above still catch a real error? ----
     *
     * A tolerance loose enough to pass on six toolchains might be loose enough
     * to pass on a wrong gradient, and nothing above would reveal that. So
     * corrupt one gradient by a known factor and require the identical
     * comparison to reject it. Without this, the number in TOLERANCE is an
     * assertion about sensitivity that nobody checked.
     *
     * The corruption is a scale error rather than a sign flip because a scale
     * error is the harder case: a flipped sign produces a relative error near 2
     * and any tolerance catches it. */
    printf("\npower check: corrupting one gradient by %.0f%% must be rejected\n",
           (double)(INJECTED_ERROR_FACTOR - 1.0f) * 100.0);
    int power_failed = 0;
    for (size_t t = 0; t < tensor_count && !power_failed; t++) {
        /* One projection and one norm tensor: the two kinds with the most and
         * least gradient signal, so the check speaks for both ends. */
        if (strcmp(tensors[t].name, "output_projection") != 0 &&
            strcmp(tensors[t].name, "layer1.ln_gamma_ffn") != 0) {
            continue;
        }
        param_tensor_t *tensor = &tensors[t];
        size_t count = tensor->rows * tensor->cols;
        float *grad = model.grads + (tensor->values - model.params);

        float *backup = malloc(count * sizeof(float));
        float *direction = malloc(count * sizeof(float));
        float *corrupted = malloc(count * sizeof(float));
        if (!backup || !direction || !corrupted) {
            fprintf(stderr, "allocation failed during the power check\n");
            free(backup); free(direction); free(corrupted);
            power_failed = 1;
            break;
        }
        memcpy(backup, tensor->values, count * sizeof(float));
        for (size_t i = 0; i < count; i++) {
            direction[i] = (dranzer_rng_unit(&rng) < 0.5) ? -1.0f : 1.0f;
            corrupted[i] = grad[i] * INJECTED_ERROR_FACTOR;
        }

        comparison_t corrupt = compare_tensor(&model, tensor, corrupted, direction,
                                              backup, count, tokens, targets);
        int rejected = corrupt.rel_error >= TOLERANCE;
        printf("  %-22s corrupted rel_err %.2e  %s\n", tensor->name,
               corrupt.rel_error,
               rejected ? "rejected, as it must be"
                        : "ACCEPTED - the tolerance is too loose to be useful");
        if (!rejected) power_failed = 1;

        free(backup);
        free(direction);
        free(corrupted);
    }
    if (power_failed) failures++;

    free(tensors);
    model_free(&model);

    if (failures == 0) {
        printf("\nALL GRADIENT CHECKS PASSED\n");
        return 0;
    }
    printf("\n%zu of %zu tensors FAILED", failures, tensor_count);
    if (zero_gradient_tensors > 0) {
        printf(" (%zu of them received no gradient at all)", zero_gradient_tensors);
    }
    printf("\nSOME GRADIENT CHECKS FAILED\n");
    return 1;
}
