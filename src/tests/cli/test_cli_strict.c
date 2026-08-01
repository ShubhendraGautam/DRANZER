#include "cli/cli.h"
#include <stdio.h>
#include <string.h>

static int must_reject(int argc, char **argv) {
    cli_args_t args;
    return cli_parse(argc, argv, &args) != 0;
}

int main(void) {
    char *unknown[] = {"app", "train", "--mystery"};
    char *bad_mode[] = {"app", "mystery"};
    char *missing[] = {"app", "train", "--epochs"};
    char *partial[] = {"app", "train", "--epochs", "12x"};
    char *overflow[] = {"app", "train", "--batch-size",
                        "999999999999999999999999999999"};
    char *not_finite[] = {"app", "train", "--learning-rate", "nan"};
    char *bad_optimizer[] = {"app", "train", "--optimizer", "rmsprop"};
    char *zero_batch[] = {"app", "train", "--batch-size", "0"};
    char *full_dropout[] = {"app", "train", "--dropout", "1"};
    char *negative_seed[] = {"app", "train", "--seed", "-1"};
    char *bad_penalty[] = {"app", "generate", "--repetition-penalty", "0.5"};
    char *empty_stop[] = {"app", "generate", "--stop", ""};
    char *misplaced_stop[] = {"app", "infer", "--stop", "x"};
    char *too_many_stops[] = {
        "app", "generate",
        "--stop", "1", "--stop", "2", "--stop", "3",
        "--stop", "4", "--stop", "5", "--stop", "6",
        "--stop", "7", "--stop", "8", "--stop", "9",
    };
    char *valid[] = {"app", "train", "--batch-size", "4",
                     "--gradient-accumulation", "3", "--shuffle",
                     "--checkpoint-interval", "0", "--dropout", "0.2"};
    char *valid_generation[] = {
        "app", "generate", "--prompt", "p", "--stop", "END",
        "--stop", "DONE", "--min-length", "3",
        "--repetition-penalty", "1.2",
    };
    cli_args_t parsed;

    int failed = !must_reject(3, unknown) || !must_reject(2, bad_mode) ||
                 !must_reject(3, missing) || !must_reject(4, partial) ||
                 !must_reject(4, overflow) || !must_reject(4, not_finite) ||
                 !must_reject(4, bad_optimizer) || !must_reject(4, zero_batch) ||
                 !must_reject(4, full_dropout) || !must_reject(4, negative_seed) ||
                 !must_reject(4, bad_penalty) || !must_reject(4, empty_stop) ||
                 !must_reject(4, misplaced_stop) ||
                 !must_reject(20, too_many_stops) ||
                 cli_parse(11, valid, &parsed) != 0 || parsed.batch_size != 4 ||
                 parsed.gradient_accumulation_steps != 3 || !parsed.shuffle ||
                 parsed.checkpoint_interval != 0 || parsed.dropout_rate != 0.2f ||
                 strstr(parsed.explicit_options, "--gradient-accumulation") == NULL ||
                 cli_parse(12, valid_generation, &parsed) != 0 ||
                 parsed.stop_sequence_count != 2 ||
                 strcmp(parsed.stop_sequences[0], "END") != 0 ||
                 strcmp(parsed.stop_sequences[1], "DONE") != 0 ||
                 parsed.minimum_generation_length != 3 ||
                 parsed.repetition_penalty != 1.2f;

    if (failed) fprintf(stderr, "strict CLI parser regression\n");
    else printf("\nSTRICT CLI PARSING CHECK PASSED\n");
    return failed;
}
