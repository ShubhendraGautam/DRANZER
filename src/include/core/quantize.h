#ifndef QUANTIZE_H
#define QUANTIZE_H

#include <stddef.h>
#include <stdint.h>

/* Symmetric weight-only quantization, and the error it introduces.
 *
 * This header is deliberately about *simulated* quantization: values are
 * quantized to an integer grid and immediately mapped back to float, in place,
 * so nothing downstream changes representation. That isolates the only question
 * this stage is trying to answer - what does the integer grid cost in accuracy -
 * from the two engineering questions that follow it, storage and bandwidth.
 * Measuring them together is how a quantization result becomes unfalsifiable:
 * a speedup and an accuracy loss arrive in the same commit and neither can be
 * attributed. See docs/quantization.md.
 *
 * Symmetric, not affine: the grid is centred on zero with no zero-point, so a
 * quantized value is exactly `scale * q`. Trained weight tensors are close
 * enough to zero-centred that the asymmetric variant's extra parameter buys
 * little, and this project would rather have one scheme it has measured than
 * two it has not. */

/* Which values share a scale.
 *
 * The axis is not a detail, and it is easy to get backwards. This project
 * stores a weight matrix as `k x n` and computes `C = A @ W`, so **a column of
 * W is one output channel and a row is one input channel.** PyTorch stores
 * `nn.Linear` the other way round, as `out x in`, and quantizes along dim 0 -
 * which is per-output-channel, and corresponds to this project's *columns*.
 * Anyone porting a per-row recipe across that boundary silently gets the other
 * axis.
 *
 * There is a mechanism to prefer columns, beyond convention. An output element
 * sums over the reduction axis, which is the rows. Give every row its own
 * scale and a single output element accumulates products carrying `k` different
 * quantization scales, so the errors do not share a common factor and cannot
 * cancel. Give every column its own scale and each output element's products
 * all carry the same one. That predicts columns should win; both are
 * implemented so the prediction can be checked rather than assumed. */
typedef enum {
    /* Every value in the tensor shares one scale. Cheapest to store - one
     * float per tensor - and the most exposed to a single large outlier,
     * which is the failure mode the granularity comparison exists to
     * quantify. */
    QUANT_GRANULARITY_TENSOR = 0,
    /* One scale per row: per *input* channel for this project's layout. Costs
     * `rows` floats. Included to be measured against the column scheme, not
     * because it is expected to be the right choice. */
    QUANT_GRANULARITY_ROW,
    /* One scale per column: per *output* channel, the scheme the literature
     * means. Costs `cols` floats and reads the tensor with a stride, which
     * matters not at all here - this runs once per model, not per token. */
    QUANT_GRANULARITY_COLUMN,
    QUANT_GRANULARITY_COUNT
} quant_granularity_t;

/* Error introduced in one tensor, in the units the tensor is stored in.
 *
 * Both an absolute and a relative measure are kept because neither is
 * sufficient alone: absolute error is what propagates through a matmul, while
 * relative error is what compares tensors of different magnitudes. */
typedef struct {
    size_t count;             /* values quantized */
    float max_abs_error;      /* worst single value */
    double rms_error;         /* root mean square over the tensor */
    double rms_relative;      /* rms_error / rms of the original values */
    double original_rms;      /* magnitude the above is relative to */
    /* Values whose magnitude hit the end of the representable range. With
     * symmetric scales derived from the maximum, this is only ever the maximum
     * itself, so a large count means many values share the extreme - useful for
     * telling a genuine outlier apart from a saturated tensor. */
    size_t clipped;
    /* Distinct integer levels actually used, out of the 2^bits available. A
     * tensor using far fewer levels than it was given is one where a single
     * outlier has stretched the scale and wasted the grid - the per-tensor
     * granularity's characteristic failure, visible here directly rather than
     * inferred from the error. */
    size_t levels_used;
} quant_error_t;

/* Valid bit widths. 4 and 8 are what the roadmap calls for; the code is
 * written for any width in this range so an intermediate one can be measured
 * without new paths. */
#define QUANT_MIN_BITS 2
#define QUANT_MAX_BITS 16

/* Quantize `values` to a `bits`-wide symmetric grid and map straight back,
 * in place. `rows` x `cols` row-major; QUANT_GRANULARITY_TENSOR ignores the
 * split and treats the buffer as one row.
 *
 * Returns 0, or -1 for a null pointer, an empty tensor, an out-of-range bit
 * width, or an unknown granularity - leaving the values untouched in every
 * failing case. `error_out` may be NULL.
 *
 * A row that is entirely zero has no scale to derive and is left exactly as it
 * is rather than being given an arbitrary one; it contributes zero error, which
 * is both true and what the reconstruction would produce anyway.
 *
 * A group holding a single value is exactly lossless at any bit width, since
 * the scale is that value and the one level reproduces it. That is a real
 * property rather than a rounding accident, and it is why per-column scales on
 * a one-row bias vector store a 32-bit scale per value to achieve nothing -
 * the degenerate corner model_quantize.h's default policy avoids. */
int quantize_dequantize(float *values, size_t rows, size_t cols,
                        int bits, quant_granularity_t granularity,
                        quant_error_t *error_out);

/* The same grid applied to a copy: `dst` receives the reconstruction and
 * `src` is not modified. Same return contract. Used by the tests to compare
 * the two forms, and by callers that need the original kept. */
int quantize_dequantize_into(const float *src, float *dst,
                             size_t rows, size_t cols,
                             int bits, quant_granularity_t granularity,
                             quant_error_t *error_out);

/* Storage form of the same symmetric grid.
 *
 * Codes are biased by qmax and packed least-significant bit first, with no
 * alignment between values. Scales are float32 in group order (tensor, rows,
 * or columns). The helpers expose their exact buffer sizes so bundle code can
 * bounds-check an artifact before allocating or decoding it.
 *
 * quantize_pack() rejects non-finite inputs; quantize_unpack() additionally
 * rejects non-finite/negative scales, the unused all-ones code, and non-zero
 * padding bits. This makes a packed payload canonical rather than merely
 * decodable. All functions return 0 on success and -1 on invalid arguments or
 * an overflowing size calculation. */
int quantized_scale_count(size_t rows, size_t cols,
                          quant_granularity_t granularity,
                          size_t *out_count);
int quantized_packed_size(size_t value_count, int bits, size_t *out_size);
int quantize_pack(const float *src, size_t rows, size_t cols,
                  int bits, quant_granularity_t granularity,
                  float *scales, size_t scales_count,
                  uint8_t *packed, size_t packed_size);
int quantize_unpack(const float *scales, size_t scales_count,
                    const uint8_t *packed, size_t packed_size,
                    size_t rows, size_t cols, int bits,
                    quant_granularity_t granularity, float *dst);

/* Combine per-tensor errors into one figure for a whole model. RMS values
 * combine as a count-weighted quadratic mean, not an average of averages,
 * which would weight a bias vector the same as an embedding matrix. */
void quant_error_accumulate(quant_error_t *total, const quant_error_t *tensor);

/* Stable lowercase names ("tensor", "row", "column") for CLI flags, CSV rows, and
 * documentation. quant_granularity_from_name() returns -1 for an unknown
 * name. */
const char *quant_granularity_name(quant_granularity_t granularity);
int quant_granularity_from_name(const char *name, quant_granularity_t *out);

#endif /* QUANTIZE_H */
