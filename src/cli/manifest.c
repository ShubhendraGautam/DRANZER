#include "cli/manifest.h"
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int write_escaped(FILE *file, const char *key, const char *value) {
    if (fprintf(file, "%s = \"", key) < 0) return 0;
    for (const unsigned char *cursor = (const unsigned char *)value; *cursor; cursor++) {
        switch (*cursor) {
            case '\\': if (fputs("\\\\", file) == EOF) return 0; break;
            case '"': if (fputs("\\\"", file) == EOF) return 0; break;
            case '\n': if (fputs("\\n", file) == EOF) return 0; break;
            case '\r': if (fputs("\\r", file) == EOF) return 0; break;
            case '\t': if (fputs("\\t", file) == EOF) return 0; break;
            default:
                if (*cursor < 0x20 || *cursor == 0x7f) {
                    if (fprintf(file, "\\x%02x", *cursor) < 0) return 0;
                } else if (fputc(*cursor, file) == EOF) return 0;
        }
    }
    return fputs("\"\n", file) != EOF;
}

int run_manifest_write(const cli_args_t *args,
                       const checkpoint_run_state_t *run_state,
                       const neural_model_t *model,
                       const bpe_encoder_t *encoder,
                       const char *resume_source,
                       char *out_path, size_t out_path_size) {
    if (!args || !run_state || !model || !encoder || !out_path || out_path_size == 0 ||
        !args->checkpoint_dir[0]) return -1;
    if (mkdir(args->checkpoint_dir, 0755) != 0 && errno != EEXIST) return -1;

    char path[2048];
    int descriptor = -1;
    for (unsigned int suffix = 0; suffix < 1000; suffix++) {
        int written = snprintf(
            path, sizeof(path), "%s/run_%016" PRIx64 "_seed_%u_pid_%ld_%u.manifest",
            args->checkpoint_dir, run_state->input_fingerprint, args->seed,
            (long)getpid(), suffix);
        if (written < 0 || (size_t)written >= sizeof(path)) return -1;
        descriptor = open(path, O_WRONLY | O_CREAT | O_EXCL, 0644);
        if (descriptor >= 0) break;
        if (errno != EEXIST) return -1;
    }
    if (descriptor < 0) return -1;

    FILE *file = fdopen(descriptor, "w");
    if (!file) {
        close(descriptor);
        remove(path);
        return -1;
    }

    size_t effective_batch = (size_t)args->batch_size *
                             (size_t)args->gradient_accumulation_steps;
    int ok = fprintf(file,
                     "manifest_version = 1\n"
                     "mode = train\n"
                     "resumed = %d\n",
                     resume_source && resume_source[0] ? 1 : 0) >= 0 &&
             write_escaped(file, "resume_source", resume_source ? resume_source : "") &&
             write_escaped(file, "explicit_options", args->explicit_options) &&
             write_escaped(file, "input_file", run_state->input_file) &&
             write_escaped(file, "validation_file", run_state->validation_file) &&
             write_escaped(file, "model_path", run_state->model_path) &&
             write_escaped(file, "tokenizer_path", run_state->tokenizer_path) &&
             write_escaped(file, "checkpoint_dir", args->checkpoint_dir) &&
             fprintf(file,
                     "input_fingerprint_fnv1a = %016" PRIx64 "\n"
                     "input_bytes = %" PRIu64 "\n"
                     "vocab_size = %zu\n"
                     "tokenizer_vocab_size = %zu\n"
                     "tokenizer_mode = %s\n"
                     "pad_token_id = %" PRIu32 "\n"
                     "unk_token_id = %" PRIu32 "\n"
                     "bos_token_id = %" PRIu32 "\n"
                     "eos_token_id = %" PRIu32 "\n"
                     "embedding_dim = %zu\n"
                     "num_heads = %zu\n"
                     "num_layers = %zu\n"
                     "max_seq_len = %zu\n"
                     "train_window = %zu\n"
                     "epochs = %d\n"
                     "batch_size = %d\n"
                     "gradient_accumulation_steps = %d\n"
                     "effective_batch_size = %zu\n"
                     "shuffle = %d\n"
                     "optimizer = %s\n"
                     "learning_rate = %.9g\n"
                     "dropout = %.9g\n"
                     "grad_clip = %.9g\n"
                     "weight_decay = %.9g\n"
                     "warmup_steps = %u\n"
                     "total_steps = %u\n"
                     "seed = %u\n"
                     "checkpoint_interval = %d\n"
                     "keep_checkpoints = %d\n"
                     "use_gpu = %d\n",
                     run_state->input_fingerprint, run_state->input_bytes,
                     model->vocab_size, encoder->vocab_size,
                     bpe_encoder_has_special_tokens(encoder) ? "special-v1" : "legacy",
                     bpe_encoder_special_token_id(encoder, BPE_SPECIAL_PAD),
                     bpe_encoder_special_token_id(encoder, BPE_SPECIAL_UNK),
                     bpe_encoder_special_token_id(encoder, BPE_SPECIAL_BOS),
                     bpe_encoder_special_token_id(encoder, BPE_SPECIAL_EOS),
                     model->embedding_dim,
                     model->num_heads, model->num_layers, model->max_seq_len,
                     run_state->train_window, args->epochs, args->batch_size,
                     args->gradient_accumulation_steps, effective_batch, args->shuffle,
                     model->optimizer_type == OPTIMIZER_SGD ? "sgd" : "adam",
                     model->metrics.initial_learning_rate, model->dropout_rate,
                     model->grad_clip_norm, model->weight_decay, model->warmup_steps,
                     model->total_steps, args->seed, args->checkpoint_interval,
                     args->keep_checkpoints, args->use_gpu) >= 0;

    if (ok) ok = fflush(file) == 0;
    if (ok) ok = fsync(descriptor) == 0;
    if (fclose(file) != 0) ok = 0;
    if (ok) ok = chmod(path, 0444) == 0;
    if (!ok) {
        remove(path);
        return -1;
    }
    int copied = snprintf(out_path, out_path_size, "%s", path);
    return copied >= 0 && (size_t)copied < out_path_size ? 0 : -1;
}
