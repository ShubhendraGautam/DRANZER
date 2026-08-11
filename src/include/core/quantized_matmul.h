#ifndef QUANTIZED_MATMUL_H
#define QUANTIZED_MATMUL_H

#include "core/quantize.h"
#include <stddef.h>
#include <stdint.h>

/* A row-major weight matrix kept in the same packed symmetric representation
 * used by bundle version 2. Only INT8 and INT4 are runtime formats: the
 * generic 2..16-bit packer exists for artifact compatibility and accuracy
 * experiments, but formats without a measured kernel do not masquerade as a
 * runtime optimization. */
typedef struct {
    size_t rows;
    size_t cols;
    int bits;
    quant_granularity_t granularity;
    float *scales;
    size_t scale_count;
    uint8_t *packed;
    size_t packed_size;
} quantized_weight_matrix_t;

/* Encode a float matrix without modifying it. `out` must be zero-initialized.
 * Returns 0 on success and -1 on invalid shape/configuration, non-finite input,
 * overflow, or allocation failure. */
int quantized_weight_matrix_encode(quantized_weight_matrix_t *out,
                                   const float *values,
                                   size_t rows, size_t cols,
                                   int bits,
                                   quant_granularity_t granularity);

void quantized_weight_matrix_free(quantized_weight_matrix_t *matrix);

/* C (m x n) = A (m x k) @ B (k x n), with B retained as INT8 or packed INT4
 * and widened per tile. Products and accumulation remain float32. `weights`
 * must have rows == k; C is fully overwritten. tile==0 uses the ordinary
 * matmul tile size. Returns 0 or -1 for an invalid matrix/shape. */
int matmul_quantized_weight(const float *restrict A,
                            const quantized_weight_matrix_t *weights,
                            float *restrict C,
                            size_t m, size_t k, size_t tile);

#endif /* QUANTIZED_MATMUL_H */
