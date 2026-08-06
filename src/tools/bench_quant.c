/*
 * What weight-only quantization costs, measured at the three places it can be
 * measured, because they answer different questions and are routinely reported
 * as if they were one.
 *
 *   1. Weight space. How far the reconstructed tensor is from the original.
 *      Exact, cheap, and the only level independent of how well the model was
 *      trained or what corpus it saw - so it is the only one whose numbers mean
 *      anything outside this benchmark.
 *   2. Logit space. How far the model's outputs move on fixed inputs. Still
 *      deterministic, and the first level where the *structure* of the error
 *      matters rather than its size: errors that cancel across a reduction do
 *      not reach here.
 *   3. Held-out cross-entropy. What a user would notice, and the level where a
 *      difference is easiest to claim and hardest to justify.
 *
 * Level 3 uses a paired design. For every seed one model is trained, evaluated,
 * then quantized and evaluated again, and the *difference* is the sample; the
 * spread of those differences across seeds is the uncertainty. Comparing an
 * unpaired quantized run against an unpaired baseline instead would drag the
 * whole seed-to-seed variance of training into the comparison, which at these
 * model sizes is far larger than the effect being measured. That is the
 * cheapest methodological difference between a quantization number that means
 * something and one that does not, and the baseline's own spread is printed
 * beside the deltas so the size of what pairing removes is visible.
 *
 * See docs/quantization.md.
 */

#include "tools/bench_support.h"
#include "core/cpu_features.h"
#include "core/model.h"
#include "core/model_quantize.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_CSV_PATH "quant_results_v1.csv"
#define MAX_TENSORS 512

/* ---------------------------------------------------------------- corpus ---
 *
 * A first-order Markov process: with probability STAY_PROBABILITY the next
 * token is a fixed successor of the current one, otherwise it is uniform. Three
 * properties earn it its place over anything simpler.
 *
 *   - It is *learnable and generalizable*. Training and held-out windows are
 *     disjoint stretches of the same process, so a model that learns the rule
 *     transfers and a model that memorizes does not. An earlier version of this
 *     tool drew tokens from a function of absolute position; the held-out
 *     stretch then landed on pattern phases training never covered, the model
 *     memorized to a training loss of 0.02, and held-out cross-entropy came out
 *     *worse than uniform*. Every quantization delta measured against that was
 *     noise on a model that predicted nothing.
 *   - Its optimum is *computable*. corpus_entropy_floor() is the cross-entropy
 *     no model can beat, so "did this model learn anything" is a comparison
 *     rather than an impression - and the tool refuses to interpret deltas from
 *     a model that failed it.
 *   - Trained weights are not initialization. Quantization is easy on the
 *     Gaussian weights model_new() draws and harder on the heavier tails
 *     training produces, so measuring on an untrained model would flatter every
 *     scheme equally and rank nothing.
 *
 * It is still synthetic, and docs/quantization.md says where that matters. */

#define CORPUS_VOCAB 32
#define CORPUS_TOTAL 40000
#define CORPUS_TRAIN 32000
#define CORPUS_GAP 64          /* untouched tokens between train and held-out */
#define STAY_PROBABILITY 0.9

/* xorshift64, seeded fixed: the corpus is a constant of this benchmark, not a
 * per-seed variable. Only the model's initialization varies with the seed,
 * which is what makes the paired comparison isolate quantization. */
static uint32_t corpus[CORPUS_TOTAL];

static void corpus_build(void) {
    uint64_t state = 0x243F6A8885A308D3ULL;
    uint32_t current = 3;
    for (size_t i = 0; i < CORPUS_TOTAL; i++) {
        corpus[i] = current;
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        double uniform = (double)((uint32_t)(state >> 32) >> 8) / (double)(1u << 24);
        if (uniform < STAY_PROBABILITY) {
            current = (current * 37u + 11u) % CORPUS_VOCAB;
        } else {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            current = (uint32_t)((state >> 32) % CORPUS_VOCAB);
        }
    }
}

/* Cross-entropy of the process against itself: the best any model can do. */
static double corpus_entropy_floor(void) {
    const double stay = STAY_PROBABILITY + (1.0 - STAY_PROBABILITY) / CORPUS_VOCAB;
    const double other = (1.0 - STAY_PROBABILITY) / CORPUS_VOCAB;
    return -(stay * log(stay) + (double)(CORPUS_VOCAB - 1) * other * log(other));
}

/* ----------------------------------------------------------------- model --- */

typedef struct {
    size_t embedding_dim, num_heads, num_layers, seq_len;
    size_t train_steps;
    size_t eval_windows;
    float learning_rate;
} bench_quant_config_t;

/* Deterministic given the seed: model_new() draws its initialization from
 * rand(), so seeding once here fixes the whole trained model.
 *
 * All four learning-rate fields are set, not just model->learning_rate. With no
 * schedule configured, model_lr_schedule_step() falls back to plateau decay and
 * assigns model->learning_rate = model->metrics.learning_rate, so setting only
 * the first has the rate silently replaced after the first plateau - which is
 * how an earlier version of this tool ran an entire learning-rate sweep that
 * barely depended on the learning rate. tests/cli/test_evaluation_overfit.c
 * sets all four for the same reason. */
static int train_model(neural_model_t *model, const bench_quant_config_t *config,
                       unsigned int seed) {
    srand(seed);
    if (model_new(model, CORPUS_VOCAB, config->embedding_dim, config->num_heads,
                  config->num_layers, config->seq_len) != MODEL_SUCCESS) {
        return -1;
    }
    model_seed_rng(model, seed);
    model->learning_rate = config->learning_rate;
    model->metrics.learning_rate = config->learning_rate;
    model->metrics.initial_learning_rate = config->learning_rate;
    model->base_lr = config->learning_rate;
    model->weight_decay = 0.0f;
    model->dropout_rate = 0.0f;  /* nothing here needs the regularizer, and it
                                  * would put an RNG between the seed and the
                                  * weights for no benefit */

    const size_t span = CORPUS_TRAIN - config->seq_len - 1;
    for (size_t step = 0; step < config->train_steps; step++) {
        size_t offset = (step * 7) % span;
        if (model_train_step(model, &corpus[offset], corpus[offset + config->seq_len],
                             config->seq_len) != MODEL_SUCCESS) {
            model_free(model);
            return -1;
        }
    }
    return 0;
}

/* Held-out windows start past the training span plus a gap and step by a stride
 * coprime with the corpus's structure, so consecutive windows do not overlap
 * into near-duplicates. */
static size_t eval_offset(size_t window) {
    return CORPUS_TRAIN + CORPUS_GAP + window * 37;
}

static int evaluate(neural_model_t *model, const bench_quant_config_t *config,
                    double *out_mean_loss) {
    double total = 0.0;
    for (size_t w = 0; w < config->eval_windows; w++) {
        size_t offset = eval_offset(w);
        double loss = 0.0;
        if (model_evaluate_step(model, &corpus[offset],
                                corpus[offset + config->seq_len],
                                config->seq_len, &loss) != MODEL_SUCCESS) {
            return -1;
        }
        total += loss;
    }
    *out_mean_loss = total / (double)config->eval_windows;
    return 0;
}

/* Logits for every held-out window, laid out windows x vocab. */
static int collect_logits(neural_model_t *model, const bench_quant_config_t *config,
                          float *out) {
    for (size_t w = 0; w < config->eval_windows; w++) {
        size_t offset = eval_offset(w);
        if (model_forward(model, &corpus[offset], config->seq_len,
                          out + w * CORPUS_VOCAB) != MODEL_SUCCESS) {
            return -1;
        }
    }
    return 0;
}

typedef struct {
    double rms_absolute;
    double rms_relative;
    double max_absolute;
    double top1_agreement;
} logit_divergence_t;

static void compare_logits(const float *reference, const float *candidate,
                           size_t windows, logit_divergence_t *out) {
    double sum_squared_difference = 0.0, sum_squared_reference = 0.0;
    double max_absolute = 0.0;
    size_t agreements = 0;

    for (size_t w = 0; w < windows; w++) {
        const float *a = reference + w * CORPUS_VOCAB;
        const float *b = candidate + w * CORPUS_VOCAB;
        size_t argmax_a = 0, argmax_b = 0;
        for (size_t i = 0; i < CORPUS_VOCAB; i++) {
            if (a[i] > a[argmax_a]) argmax_a = i;
            if (b[i] > b[argmax_b]) argmax_b = i;
            double difference = (double)a[i] - (double)b[i];
            sum_squared_difference += difference * difference;
            sum_squared_reference += (double)a[i] * (double)a[i];
            double absolute = fabs(difference);
            if (absolute > max_absolute) max_absolute = absolute;
        }
        if (argmax_a == argmax_b) agreements++;
    }

    const double count = (double)(windows * CORPUS_VOCAB);
    out->rms_absolute = sqrt(sum_squared_difference / count);
    double reference_rms = sqrt(sum_squared_reference / count);
    out->rms_relative = (reference_rms > 0.0) ? out->rms_absolute / reference_rms : 0.0;
    out->max_absolute = max_absolute;
    out->top1_agreement = (windows > 0) ? (double)agreements / (double)windows : 0.0;
}

/* ---------------------------------------------------------------- schemes --- */

typedef struct {
    const char *label;
    int bits;
    quant_granularity_t granularity;
    int include_embeddings;
    int include_biases_and_norms;
} scheme_t;

static const scheme_t schemes[] = {
    { "int8/tensor",     8, QUANT_GRANULARITY_TENSOR, 0, 0 },
    { "int8/row",        8, QUANT_GRANULARITY_ROW,    0, 0 },
    { "int8/column",     8, QUANT_GRANULARITY_COLUMN, 0, 0 },
    { "int8/col+emb",    8, QUANT_GRANULARITY_COLUMN, 1, 0 },
    { "int8/col+all",    8, QUANT_GRANULARITY_COLUMN, 1, 1 },
    { "int4/tensor",     4, QUANT_GRANULARITY_TENSOR, 0, 0 },
    { "int4/row",        4, QUANT_GRANULARITY_ROW,    0, 0 },
    { "int4/column",     4, QUANT_GRANULARITY_COLUMN, 0, 0 },
    { "int4/col+emb",    4, QUANT_GRANULARITY_COLUMN, 1, 0 },
    { "int4/col+all",    4, QUANT_GRANULARITY_COLUMN, 1, 1 },
};
#define SCHEME_COUNT (sizeof(schemes) / sizeof(schemes[0]))

/* --------------------------------------------------------------- samples --- */

typedef struct {
    double values[64];
    size_t count;
} samples_t;

static void sample_add(samples_t *samples, double value) {
    if (samples->count < sizeof(samples->values) / sizeof(samples->values[0])) {
        samples->values[samples->count++] = value;
    }
}

static double sample_mean(const samples_t *samples) {
    if (samples->count == 0) return 0.0;
    double total = 0.0;
    for (size_t i = 0; i < samples->count; i++) total += samples->values[i];
    return total / (double)samples->count;
}

/* Sample standard deviation (n-1). With the handful of seeds this runs it is a
 * coarse estimate, reported as a spread rather than a confidence interval,
 * which would imply more than the sample size supports. */
static double sample_stddev(const samples_t *samples) {
    if (samples->count < 2) return 0.0;
    double mean = sample_mean(samples);
    double total = 0.0;
    for (size_t i = 0; i < samples->count; i++) {
        double difference = samples->values[i] - mean;
        total += difference * difference;
    }
    return sqrt(total / (double)(samples->count - 1));
}

static double sample_min(const samples_t *samples) {
    double best = samples->count ? samples->values[0] : 0.0;
    for (size_t i = 1; i < samples->count; i++) {
        if (samples->values[i] < best) best = samples->values[i];
    }
    return best;
}

static double sample_max(const samples_t *samples) {
    double best = samples->count ? samples->values[0] : 0.0;
    for (size_t i = 1; i < samples->count; i++) {
        if (samples->values[i] > best) best = samples->values[i];
    }
    return best;
}

/* ------------------------------------------------------------------ main --- */

typedef struct {
    samples_t delta_ce, weight_rms, logit_rms, top1;
    double effective_bits, weight_max, logit_max, levels_used;
} scheme_result_t;

static void usage(const char *program) {
    printf("Usage: %s [--seeds N] [--steps N] [--per-tensor] [--csv-path FILE]\n\n"
           "  --seeds N     independently trained models to pair over (default 5)\n"
           "  --steps N     training steps per model (default 20000)\n"
           "  --per-tensor  also print the per-tensor weight-error breakdown\n"
           "  --csv-path F  results file to append to (default %s)\n",
           program, DEFAULT_CSV_PATH);
}

int main(int argc, char **argv) {
    size_t seeds = 5;
    size_t steps = 20000;
    int per_tensor = 0;
    const char *csv_path = DEFAULT_CSV_PATH;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        }
        if (strcmp(argv[i], "--seeds") == 0 && i + 1 < argc) {
            long value = strtol(argv[++i], NULL, 10);
            if (value < 1 || value > 64) {
                fprintf(stderr, "Error: --seeds must be between 1 and 64\n");
                return 1;
            }
            seeds = (size_t)value;
            continue;
        }
        if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            long value = strtol(argv[++i], NULL, 10);
            if (value < 1 || value > 1000000) {
                fprintf(stderr, "Error: --steps must be between 1 and 1000000\n");
                return 1;
            }
            steps = (size_t)value;
            continue;
        }
        if (strcmp(argv[i], "--per-tensor") == 0) { per_tensor = 1; continue; }
        if (strcmp(argv[i], "--csv-path") == 0 && i + 1 < argc) {
            csv_path = argv[++i];
            continue;
        }
        fprintf(stderr, "Error: unknown option '%s'\n", argv[i]);
        usage(argv[0]);
        return 1;
    }

    cpu_features_detect();
    bench_metadata_t metadata;
    bench_collect_metadata(&metadata);
    bench_print_metadata(&metadata);

    bench_quant_config_t config = {
        .embedding_dim = 64, .num_heads = 4, .num_layers = 2, .seq_len = 4,
        .train_steps = steps, .eval_windows = 128, .learning_rate = 0.003f,
    };

    corpus_build();
    const double floor_ce = corpus_entropy_floor();
    const double uniform_ce = log((double)CORPUS_VOCAB);

    printf("corpus: first-order Markov, vocab %d, P(successor) %.2f\n"
           "        %d tokens, %d for training, held-out beyond %d\n"
           "        entropy floor %.4f nats, uniform %.4f nats\n",
           CORPUS_VOCAB, STAY_PROBABILITY, CORPUS_TOTAL, CORPUS_TRAIN,
           CORPUS_TRAIN + CORPUS_GAP, floor_ce, uniform_ce);
    printf("model:  emb %zu, heads %zu, layers %zu, seq %zu, %zu steps, "
           "%zu held-out windows, %zu seeds\n\n",
           config.embedding_dim, config.num_heads, config.num_layers,
           config.seq_len, config.train_steps, config.eval_windows, seeds);

    float *reference_logits = malloc(config.eval_windows * CORPUS_VOCAB * sizeof(float));
    float *candidate_logits = malloc(config.eval_windows * CORPUS_VOCAB * sizeof(float));
    scheme_result_t *results = calloc(SCHEME_COUNT, sizeof(*results));
    static model_quant_tensor_t tensor_slots[MAX_TENSORS];
    if (!reference_logits || !candidate_logits || !results) {
        fprintf(stderr, "Error: out of memory\n");
        free(reference_logits); free(candidate_logits); free(results);
        return 1;
    }

    samples_t baseline_ce = {0};
    int printed_tensors = 0;

    for (size_t s = 0; s < seeds; s++) {
        neural_model_t model = {0};
        if (train_model(&model, &config, (unsigned int)(1000 + s)) != 0) {
            fprintf(stderr, "Error: seed %zu failed to train\n", s);
            free(reference_logits); free(candidate_logits); free(results);
            return 1;
        }

        double reference_ce = 0.0;
        if (evaluate(&model, &config, &reference_ce) != 0 ||
            collect_logits(&model, &config, reference_logits) != 0) {
            fprintf(stderr, "Error: seed %zu failed to evaluate\n", s);
            model_free(&model);
            free(reference_logits); free(candidate_logits); free(results);
            return 1;
        }
        sample_add(&baseline_ce, reference_ce);

        /* One trained model serves every scheme: the weights are snapshotted,
         * and each scheme quantizes a restored copy. Retraining per scheme
         * would cost 10x the time to reproduce a model that is bit-identical
         * anyway. */
        const size_t param_bytes = model.total_param_count * sizeof(float);
        float *pristine = malloc(param_bytes);
        if (!pristine) {
            fprintf(stderr, "Error: out of memory snapshotting parameters\n");
            model_free(&model);
            free(reference_logits); free(candidate_logits); free(results);
            return 1;
        }
        memcpy(pristine, model.params, param_bytes);

        for (size_t sc = 0; sc < SCHEME_COUNT; sc++) {
            const scheme_t *scheme = &schemes[sc];
            memcpy(model.params, pristine, param_bytes);

            model_quant_config_t quant_config;
            model_quantize_default_config(&quant_config);
            quant_config.bits = scheme->bits;
            quant_config.granularity = scheme->granularity;
            quant_config.include_embeddings = scheme->include_embeddings;
            quant_config.include_biases_and_norms = scheme->include_biases_and_norms;

            model_quant_report_t report = {
                .tensors = tensor_slots, .tensor_capacity = MAX_TENSORS
            };
            double candidate_ce = 0.0;
            logit_divergence_t divergence = {0};

            if (model_quantize_weights(&model, &quant_config, &report) != 0 ||
                evaluate(&model, &config, &candidate_ce) != 0 ||
                collect_logits(&model, &config, candidate_logits) != 0) {
                fprintf(stderr, "Error: scheme %s failed\n", scheme->label);
                free(pristine); model_free(&model);
                free(reference_logits); free(candidate_logits); free(results);
                return 1;
            }
            compare_logits(reference_logits, candidate_logits,
                           config.eval_windows, &divergence);

            scheme_result_t *result = &results[sc];
            sample_add(&result->delta_ce, candidate_ce - reference_ce);
            sample_add(&result->weight_rms, report.combined.rms_relative);
            sample_add(&result->logit_rms, divergence.rms_relative);
            sample_add(&result->top1, divergence.top1_agreement);
            result->effective_bits = report.effective_bits_per_value;
            result->levels_used = (double)report.combined.levels_used;
            if (report.combined.max_abs_error > result->weight_max) {
                result->weight_max = report.combined.max_abs_error;
            }
            if (divergence.max_absolute > result->logit_max) {
                result->logit_max = divergence.max_absolute;
            }

            if (per_tensor && !printed_tensors && sc + 1 == SCHEME_COUNT) {
                printf("per-tensor weight error, %s, seed %u:\n",
                       scheme->label, (unsigned int)(1000 + s));
                size_t shown = report.tensors_total < MAX_TENSORS
                             ? report.tensors_total : MAX_TENSORS;
                for (size_t t = 0; t < shown; t++) {
                    const model_quant_tensor_t *entry = &tensor_slots[t];
                    if (!entry->quantized) continue;
                    printf("  %-22s %-11s %5zux%-5zu rms.rel %.5f  max %.3e  "
                           "levels %3zu  clipped %zu\n",
                           entry->name, param_kind_name(entry->kind),
                           entry->rows, entry->cols, entry->error.rms_relative,
                           (double)entry->error.max_abs_error,
                           entry->error.levels_used, entry->error.clipped);
                }
                printf("\n");
                printed_tensors = 1;
            }
        }

        memcpy(model.params, pristine, param_bytes);
        free(pristine);
        model_free(&model);
    }

    /* A model that did not learn cannot report a meaningful accuracy cost:
     * quantizing noise produces noise, and the delta would be a measurement of
     * nothing. This is a hard gate rather than a footnote because that is
     * precisely the failure this tool shipped with once already. */
    const double baseline_mean = sample_mean(&baseline_ce);
    const int learned = baseline_mean < uniform_ce - 0.2;

    printf("Baseline held-out cross-entropy over %zu seeds: %.4f mean, "
           "%.4f stddev, %.4f to %.4f\n",
           seeds, baseline_mean, sample_stddev(&baseline_ce),
           sample_min(&baseline_ce), sample_max(&baseline_ce));
    printf("  against an entropy floor of %.4f and a uniform model's %.4f: %s\n",
           floor_ce, uniform_ce,
           learned ? "the model learned the process" : "*** THE MODEL DID NOT LEARN ***");
    if (!learned) {
        printf("\n  Every delta below is a measurement of noise. Fix the training\n"
               "  configuration before reading anything into this table.\n");
    }
    printf("\nThat baseline spread is what an unpaired comparison would have to\n"
           "beat. The paired deltas below do not carry it.\n\n");

    printf("%-14s %10s %9s %7s %11s %10s %7s %10s %10s %10s\n",
           "scheme", "w.rms.rel", "eff.bits", "levels", "lg.rms.rel", "lg sd",
           "top1", "dCE mean", "dCE sd", "seeds req");

    FILE *csv = fopen(csv_path, "a");
    if (csv && ftell(csv) == 0) {
        fprintf(csv, "scheme,bits,granularity,embeddings,biases_norms,seeds,"
                     "learned,entropy_floor,uniform_ce,"
                     "weight_rms_relative,weight_max_abs,levels_used,"
                     "effective_bits,logit_rms_relative,logit_rms_stddev,logit_max_abs,"
                     "top1_agreement,baseline_ce_mean,baseline_ce_stddev,"
                     "delta_ce_mean,delta_ce_stddev,delta_ce_min,delta_ce_max,"
                     BENCH_METADATA_CSV_HEADER "\n");
    }

    for (size_t sc = 0; sc < SCHEME_COUNT; sc++) {
        const scheme_t *scheme = &schemes[sc];
        scheme_result_t *result = &results[sc];
        const double delta_mean = sample_mean(&result->delta_ce);
        const double delta_stddev = sample_stddev(&result->delta_ce);
        /* "Resolvable" means the mean paired delta is at least twice its own
         * spread. With a handful of seeds this is a rule of thumb, not a test,
         * and the column is named for what it is. */
        const int resolvable = (delta_stddev > 0.0) &&
                               (fabs(delta_mean) > 2.0 * delta_stddev);

        /* Seeds this delta would need before |mean| exceeded twice the
         * standard error, at the effect size and spread just measured. Turning
         * "not resolvable" into a number is what makes it actionable: it says
         * whether the answer is one more afternoon of compute or out of reach
         * at this model size, rather than leaving the reader to guess. */
        char requirement[16];
        if (resolvable) {
            snprintf(requirement, sizeof(requirement), "resolved");
        } else if (delta_stddev > 0.0 && fabs(delta_mean) > 0.0) {
            double ratio = 2.0 * delta_stddev / fabs(delta_mean);
            double needed = ratio * ratio;
            if (needed > 99999.0) {
                snprintf(requirement, sizeof(requirement), ">1e5");
            } else {
                snprintf(requirement, sizeof(requirement), "~%.0f", needed);
            }
        } else {
            snprintf(requirement, sizeof(requirement), "-");
        }

        printf("%-14s %10.5f %9.2f %7.0f %11.5f %10.5f %7.3f %10.5f %10.5f %10s\n",
               scheme->label, sample_mean(&result->weight_rms),
               result->effective_bits, result->levels_used,
               sample_mean(&result->logit_rms), sample_stddev(&result->logit_rms),
               sample_mean(&result->top1),
               delta_mean, delta_stddev, requirement);

        if (csv) {
            bench_csv_field(csv, scheme->label);
            fprintf(csv, ",%d,", scheme->bits);
            bench_csv_field(csv, quant_granularity_name(scheme->granularity));
            fprintf(csv, ",%d,%d,%zu,%d,%.9f,%.9f,%.9f,%.9f,%.0f,%.4f,"
                         "%.9f,%.9f,%.9f,%.6f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f",
                    scheme->include_embeddings, scheme->include_biases_and_norms,
                    seeds, learned, floor_ce, uniform_ce,
                    sample_mean(&result->weight_rms), result->weight_max,
                    result->levels_used, result->effective_bits,
                    sample_mean(&result->logit_rms),
                    sample_stddev(&result->logit_rms), result->logit_max,
                    sample_mean(&result->top1), baseline_mean,
                    sample_stddev(&baseline_ce), delta_mean, delta_stddev,
                    sample_min(&result->delta_ce), sample_max(&result->delta_ce));
            bench_csv_metadata(csv, &metadata);
        }
    }

    /* Granularity comparisons, paired seed by seed.
     *
     * The table above prints each scheme's mean and spread, and read as
     * independent means those spreads overlap heavily - the row and column
     * schemes look indistinguishable, and even tensor's disadvantage sits near
     * the noise. But every scheme ran on the *same* trained models, so the
     * comparison between two of them is paired exactly as the accuracy delta
     * is, and pairing removes the seed-to-seed variation common to both. This
     * section is the same argument the header makes about level 3, applied one
     * level down, and it is what turns "these look similar" into an answer. */
    printf("\nGranularity, compared pairwise on the same seeds (logit rms.rel):\n\n");
    printf("%-34s %11s %11s %11s\n",
           "comparison", "mean diff", "spread", "resolvable");

    static const struct { size_t left, right; const char *label; } pairs[] = {
        { 0, 2, "int8: tensor minus column" },
        { 1, 2, "int8: row minus column" },
        { 5, 7, "int4: tensor minus column" },
        { 6, 7, "int4: row minus column" },
    };
    for (size_t p = 0; p < sizeof(pairs) / sizeof(pairs[0]); p++) {
        const samples_t *left = &results[pairs[p].left].logit_rms;
        const samples_t *right = &results[pairs[p].right].logit_rms;
        samples_t differences = {0};
        size_t n = (left->count < right->count) ? left->count : right->count;
        for (size_t i = 0; i < n; i++) {
            sample_add(&differences, left->values[i] - right->values[i]);
        }
        double mean = sample_mean(&differences);
        double spread = sample_stddev(&differences);
        int resolvable = (spread > 0.0) && (fabs(mean) > 2.0 * spread);
        printf("%-34s %11.6f %11.6f %11s\n",
               pairs[p].label, mean, spread, resolvable ? "yes" : "no");
    }
    printf("\nA positive mean means the first scheme moved the logits further,\n"
           "i.e. was worse. \"Resolvable\" is again |mean| > 2x its own spread.\n");

    printf("\nw.rms.rel  weight error, relative to the weights' own magnitude\n"
           "eff.bits   bits per parameter the policy would actually store,\n"
           "           counting float32 scales and the tensors left alone\n"
           "levels     distinct grid levels the worst tensor used\n"
           "lg.rms.rel logit movement, relative to the logits' magnitude,\n"
           "           with its own spread across seeds beside it - the row and\n"
           "           column schemes are close enough that the spread decides\n"
           "top1       fraction of windows whose argmax is unchanged\n"
           "dCE        paired held-out cross-entropy delta, quantized minus not\n"
           "seeds req  seeds this delta would need to clear twice its standard\n"
           "           error, at the effect size and spread measured here\n");

    if (csv) {
        printf("\nResults appended to %s\n", csv_path);
        fclose(csv);
    }
    free(reference_logits);
    free(candidate_logits);
    free(results);
    return learned ? 0 : 1;
}
