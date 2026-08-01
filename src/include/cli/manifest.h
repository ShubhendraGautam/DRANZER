#ifndef MANIFEST_H
#define MANIFEST_H

#include "byte_pair_encoding.h"
#include "cli/checkpoint.h"
#include "cli/cli.h"
#include "core/model.h"
#include <stddef.h>

/* Create a new read-only run manifest with O_EXCL semantics. It records
 * every resolved training value plus the names explicitly supplied on the
 * command line. Existing manifests are never overwritten. */
int run_manifest_write(const cli_args_t *args,
                       const checkpoint_run_state_t *run_state,
                       const neural_model_t *model,
                       const bpe_encoder_t *encoder,
                       const char *resume_source,
                       char *out_path, size_t out_path_size);

#endif /* MANIFEST_H */
