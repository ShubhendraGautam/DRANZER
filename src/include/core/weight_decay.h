#ifndef WEIGHT_DECAY_H
#define WEIGHT_DECAY_H

#include "core/model_params.h"
#include "core/model_types.h"

/*
 * Which tensors decoupled weight decay applies to.
 *
 * Why this is a policy and not a loop
 * -----------------------------------
 * The decay used to be applied inside adam_update(), over the whole flat
 * parameter vector - every float the model owns, including biases and every
 * LayerNorm gain and offset. That is not the AdamW every reference
 * implementation and every baseline this project could be compared against
 * uses, and the difference is not a matter of regularization strength.
 *
 * A LayerNorm gain starts at 1.0 and multiplies a normalized activation.
 * Decaying it pulls it towards 0, which shrinks the layer's output scale toward
 * nothing - a change to what the model computes, applied every step, with no
 * gradient signal behind it. A bias starts at 0, where decay does nothing at
 * first and then fights whatever offset the data asks for. Neither is
 * regularization in the sense the hyperparameter names; the weight in "weight
 * decay" means the multiplicative weights.
 *
 * Keeping it as a decay *policy over the tensor inventory* rather than a range
 * of floats means the rule is stated once, in terms a reader can check against
 * a reference implementation, and a tensor added to model_new() inherits the
 * rule for its kind instead of silently inheriting whatever the flat loop did.
 *
 * The convention this matches
 * ---------------------------
 * Decay projections and embeddings; leave biases and normalization parameters
 * alone. That is what PyTorch's transformer examples, nanoGPT, and the original
 * AdamW paper's image and language setups all do.
 *
 * Embeddings are the one genuinely contested entry - some implementations
 * exclude them, and with tied embeddings the question is entangled with the
 * output head. They are decayed here because the output projection is a matmul
 * operand that has to be decayed for the "matches the reference" claim to hold,
 * and while embeddings are untied in this model, treating the two differently
 * would make the tied-embedding ablation (docs/design-checklist.md, v0.5) change
 * two things at once.
 */

/* Nonzero if `kind` is decayed under this project's policy. */
int weight_decay_applies_to(param_kind_t kind);

/* Apply one step of decoupled weight decay - param -= lr * weight_decay * param
 * - to the eligible tensors only.
 *
 * Decoupled means the decay does not pass through the adaptive step, so it is a
 * separate pass here rather than a term inside adam_update(). No-op, and
 * MODEL_SUCCESS, when weight_decay is zero or negative. */
model_errors_t model_apply_weight_decay(neural_model_t *model, float lr,
                                        float weight_decay);

#endif /* WEIGHT_DECAY_H */
