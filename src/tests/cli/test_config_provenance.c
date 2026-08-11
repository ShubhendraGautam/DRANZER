#include "cli/config.h"
#include <stdio.h>
#include <string.h>
#include <unistd.h>

int main(void) {
    char path[128];
    snprintf(path, sizeof(path), "/tmp/dranzer_config_provenance_%ld.cfg", (long)getpid());

    config_t before;
    config_t after;
    config_get_defaults(&before);
    before.seed = 42;
    before.batch_size = 8;
    before.gradient_accumulation_steps = 4;
    before.shuffle = 1;
    before.architecture_flags = UINT32_C(1);
    before.tokenizer_vocab_size = 273;
    before.tokenizer_has_special_tokens = 1;
    before.pad_token_id = 256;
    before.unk_token_id = 257;
    before.bos_token_id = 258;
    before.eos_token_id = 259;
    before.input_fingerprint = UINT64_C(0x0123456789abcdef);
    before.input_bytes = 987654;
    before.validation_fingerprint = UINT64_C(0xfedcba9876543210);
    before.validation_bytes = 12345;
    before.validation_tokens = 4321;
    before.validation_cross_entropy = 2.5;
    before.validation_perplexity = 10.25;
    strcpy(before.model_path, "models/reference.pth");
    strcpy(before.tokenizer_path, "models/reference.tokenizer");
    strcpy(before.input_path, "data/reference.txt");
    strcpy(before.validation_path, "data/held-out.txt");
    strcpy(before.checkpoint_dir, "runs/reference");

    if (config_save(path, &before) != 0 || config_load(path, &after) != 0) {
        remove(path);
        fprintf(stderr, "configuration save/load failed\n");
        return 1;
    }
    remove(path);

    if (after.seed != before.seed ||
        after.batch_size != before.batch_size ||
        after.gradient_accumulation_steps != before.gradient_accumulation_steps ||
        after.shuffle != before.shuffle ||
        after.architecture_flags != before.architecture_flags ||
        after.tokenizer_vocab_size != before.tokenizer_vocab_size ||
        after.tokenizer_has_special_tokens != before.tokenizer_has_special_tokens ||
        after.pad_token_id != before.pad_token_id ||
        after.unk_token_id != before.unk_token_id ||
        after.bos_token_id != before.bos_token_id ||
        after.eos_token_id != before.eos_token_id ||
        after.input_fingerprint != before.input_fingerprint ||
        after.input_bytes != before.input_bytes ||
        after.validation_fingerprint != before.validation_fingerprint ||
        after.validation_bytes != before.validation_bytes ||
        after.validation_tokens != before.validation_tokens ||
        after.validation_cross_entropy != before.validation_cross_entropy ||
        after.validation_perplexity != before.validation_perplexity ||
        strcmp(after.model_path, before.model_path) != 0 ||
        strcmp(after.tokenizer_path, before.tokenizer_path) != 0 ||
        strcmp(after.input_path, before.input_path) != 0 ||
        strcmp(after.validation_path, before.validation_path) != 0 ||
        strcmp(after.checkpoint_dir, before.checkpoint_dir) != 0) {
        fprintf(stderr, "configuration provenance changed across roundtrip\n");
        return 1;
    }

    printf("\nCONFIG PROVENANCE CHECK PASSED\n");
    return 0;
}
