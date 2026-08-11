/* Deterministic fuzzing for the two paths that take untrusted bytes: the BPE
 * tokenizer and the corpus reader.
 *
 * Bundle loading is already fuzzed by a mutation sweep in test_model_bundle.c,
 * which is the right shape for a format with a header to validate. These two are
 * different: they accept arbitrary bytes by design - any file a user points
 * --input at is valid input - so there is no "reject it" answer to check. What
 * has to hold instead is that no input, however hostile, causes a crash, an
 * out-of-bounds access, an unbounded allocation, or a round trip that silently
 * loses data.
 *
 * Deterministic rather than random. A fuzz run whose inputs depend on the clock
 * finds a crash once and then cannot reproduce it, and a nightly job that fails
 * one time in ten with no way to re-run the failing case is worse than no job.
 * Every input here comes from the project's own generator (core/rng.h) seeded
 * from a fixed constant, so the case number in a failure message is enough to
 * reproduce it, and a case that ever fails can be promoted into the fixture list
 * at the top as a permanent regression test.
 *
 * This finds memory errors only when run under a sanitizer, which is where it
 * earns its place in the nightly matrix (ASan and UBSan both). Run without one
 * it still checks the invariants below, which are worth something on their own.
 */
/* mkstemp(), write(), and close() are POSIX rather than C, and this file is also
 * compiled in the -std=c11 -Wpedantic configuration. Requested before any
 * include, which is where a feature-test macro has to go. */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "byte_pair_encoding.h"
#include "cli/stream.h"
#include "core/rng.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MAX_INPUT 4096
#define CASES 400
#define VOCAB 300

static int failures;

static void fail(const char *what, size_t case_index) {
    fprintf(stderr, "FAIL [case %zu]: %s\n", case_index, what);
    failures++;
}

/* Inputs that have historically broken text handling, checked before the random
 * cases so a regression in one of them is reported plainly rather than as a case
 * number. Anything the random sweep ever finds belongs here too. */
static const struct { const char *bytes; size_t length; const char *what; } fixtures[] = {
    { "", 0, "empty input" },
    { "\0", 1, "a single NUL" },
    { "a\0b", 3, "an embedded NUL" },
    { "\xff\xfe\xfd", 3, "bytes that are not valid UTF-8" },
    { "\xc3", 1, "a truncated two-byte UTF-8 sequence" },
    { "\xe2\x82", 2, "a truncated three-byte UTF-8 sequence" },
    { "\xf0\x9f\x98\x80", 4, "a four-byte emoji" },
    { "\n\n\n\n", 4, "newlines only" },
    { "          ", 10, "whitespace only" },
    { "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", 32, "one byte repeated, maximal merges" },
    { "\x80\x80\x80\x80", 4, "continuation bytes with no lead byte" },
    { "\xed\xa0\x80", 3, "a UTF-16 surrogate encoded as UTF-8" },
};

/* Train, encode, decode. The invariants:
 *   - every call returns a defined status rather than crashing;
 *   - every produced token id is inside the vocabulary the encoder reports;
 *   - decode(encode(x)) == x, byte for byte, for every byte value.
 *
 * The round trip is the load-bearing one. A tokenizer that drops a byte produces
 * a corpus that differs from the file on disk, and every loss number measured
 * afterwards is measured on something other than the data the manifest hashed.
 *
 * This sweep originally found that 0x00 alone vanished because tokens were C
 * strings. Tokens are now length-delimited bytes, so NUL has no exception here:
 * the manifest bytes and the bytes presented to the model must be identical. */
static void exercise(const char *input, size_t length, size_t case_index,
                     const char *label) {
    bpe_encoder_t encoder = {0};
    if (bpe_encoder_new(&encoder, VOCAB) != BPE_SUCCESS) {
        fail("bpe_encoder_new failed", case_index);
        return;
    }

    /* Training on hostile bytes must not crash and must leave the encoder in a
     * usable state whether it succeeded or not. A failure is a legitimate answer
     * here - there is nothing to learn from an empty input. */
    (void)bpe_train(&encoder, input, length);

    bpe_tokens_t tokens = {0};
    bpe_errors_t rc = bpe_encode(&encoder, input, length, &tokens);
    if (rc == BPE_SUCCESS) {
        for (size_t i = 0; i < tokens.token_count; i++) {
            if (tokens.token_ids[i] >= encoder.max_vocab_size) {
                fail("encode produced a token id outside the vocabulary", case_index);
                break;
            }
        }

        char *decoded = NULL;
        size_t decoded_length = 0;
        if (bpe_decode(&encoder, tokens.token_ids, tokens.token_count, &decoded,
                       &decoded_length) == BPE_SUCCESS) {
            if (decoded_length != length) {
                fprintf(stderr,
                        "FAIL [case %zu, %s]: round trip returned %zu bytes, "
                        "expected %zu\n",
                        case_index, label, decoded_length, length);
                failures++;
            } else if (length > 0 && memcmp(decoded, input, length) != 0) {
                fprintf(stderr,
                        "FAIL [case %zu, %s]: round trip preserved the length "
                        "but changed the bytes\n", case_index, label);
                failures++;
            }
            free(decoded);
        }
        bpe_tokens_free(&tokens);
    }

    bpe_encoder_free(&encoder);
}

/* The corpus reader, against the same bytes written to a file. Chunk sizes are
 * varied deliberately, including 1 and sizes that do not divide the input, since
 * a reader that assumes a chunk boundary falls on a token or a line boundary
 * fails exactly there. */
static void exercise_reader(const char *input, size_t length, size_t case_index,
                            size_t chunk_size) {
    char path[] = "/tmp/dranzer-fuzz-corpus.XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) {
        fail("could not create a temporary corpus", case_index);
        return;
    }
    if (length > 0 && write(fd, input, length) != (ssize_t)length) {
        close(fd);
        remove(path);
        fail("could not write the temporary corpus", case_index);
        return;
    }
    close(fd);

    stream_reader_t *reader = stream_reader_create(path, chunk_size);
    if (reader) {
        char buffer[MAX_INPUT + 1];
        size_t total = 0;
        /* Bounded: a reader that never reports EOF would otherwise hang the
         * nightly job instead of failing it. */
        for (size_t iterations = 0; iterations < MAX_INPUT + 16; iterations++) {
            size_t got = stream_read_chunk(reader, buffer, sizeof(buffer) - 1);
            if (got == 0) break;
            if (got > sizeof(buffer) - 1) {
                fail("stream_read_chunk reported more bytes than it was given "
                     "room for", case_index);
                break;
            }
            total += got;
            if (total > length) {
                fail("stream_read_chunk returned more bytes than the file holds",
                     case_index);
                break;
            }
        }
        if (total != length) {
            fprintf(stderr, "FAIL [case %zu]: reader returned %zu of %zu bytes "
                            "at chunk size %zu\n",
                    case_index, total, length, chunk_size);
            failures++;
        }
        if (!stream_is_eof(reader) && length > 0) {
            fail("reader did not report EOF after consuming the whole file",
                 case_index);
        }
        stream_reader_free(reader);
    }

    remove(path);
}

/* Every one of the 256 byte values, one at a time, wrapped in printable bytes so
 * the encoder has something to merge around it.
 *
 * This check originally turned "the fuzz sweep loses bytes sometimes" into
 * "exactly 0x00 is lost." It now pins the repaired boundary directly: all 256
 * values must survive, and a regression reports the first lost value. */
static void check_every_byte_value(void) {
    size_t lost = 0;
    int first_lost = -1;

    for (int b = 0; b < 256; b++) {
        char input[3];
        input[0] = 'A';
        input[1] = (char)b;
        input[2] = 'Z';

        bpe_encoder_t encoder = {0};
        if (bpe_encoder_new(&encoder, VOCAB) != BPE_SUCCESS) continue;
        (void)bpe_train(&encoder, input, sizeof(input));

        bpe_tokens_t tokens = {0};
        int survived = 0;
        if (bpe_encode(&encoder, input, sizeof(input), &tokens) == BPE_SUCCESS) {
            char *decoded = NULL;
            size_t decoded_length = 0;
            if (bpe_decode(&encoder, tokens.token_ids, tokens.token_count,
                           &decoded, &decoded_length) == BPE_SUCCESS) {
                survived = decoded_length == sizeof(input) &&
                           memcmp(decoded, input, sizeof(input)) == 0;
                free(decoded);
            }
            bpe_tokens_free(&tokens);
        }
        bpe_encoder_free(&encoder);

        if (!survived) {
            lost++;
            if (first_lost < 0) first_lost = b;
        }
    }

    printf("byte values surviving a round trip: %zu of 256\n", 256 - lost);
    if (lost != 0) {
        fprintf(stderr, "FAIL: %zu byte values are lost; first is 0x%02X\n",
                lost, first_lost);
        failures++;
    }
}

int main(void) {
    check_every_byte_value();

    printf("fixtures: %zu inputs known to break text handling\n",
           sizeof(fixtures) / sizeof(fixtures[0]));
    for (size_t i = 0; i < sizeof(fixtures) / sizeof(fixtures[0]); i++) {
        exercise(fixtures[i].bytes, fixtures[i].length, i, fixtures[i].what);
        exercise_reader(fixtures[i].bytes, fixtures[i].length, i, 7);
    }

    /* The random sweep. Four byte distributions, because a uniform-random buffer
     * is not a hard case for a tokenizer - it has no repeated substrings, so it
     * exercises none of the merge logic. The structured distributions are where
     * the merges happen. */
    uint64_t rng = dranzer_rng_stream(20260808, DRANZER_RNG_STREAM_TESTING);
    static char input[MAX_INPUT];
    size_t total_bytes = 0;

    for (size_t c = 0; c < CASES; c++) {
        size_t length = (size_t)(dranzer_rng_unit(&rng) * MAX_INPUT);
        int distribution = (int)(dranzer_rng_unit(&rng) * 4.0);

        for (size_t i = 0; i < length; i++) {
            double u = dranzer_rng_unit(&rng);
            switch (distribution) {
                case 0: /* uniform over all 256 byte values, NULs included */
                    input[i] = (char)(int)(u * 256.0);
                    break;
                case 1: /* a tiny alphabet: maximal merge pressure */
                    input[i] = (char)('a' + (int)(u * 3.0));
                    break;
                case 2: /* printable ASCII with structure */
                    input[i] = (char)(0x20 + (int)(u * 95.0));
                    break;
                default: /* high bytes only: invalid UTF-8 throughout */
                    input[i] = (char)(0x80 + (int)(u * 128.0));
                    break;
            }
        }

        exercise(input, length, c, "random");
        /* Chunk sizes 1 and 3 straddle every boundary; a large one exercises the
         * whole-file path. */
        size_t chunk = (c % 3 == 0) ? 1 : (c % 3 == 1) ? 3 : 8192;
        exercise_reader(input, length, c, chunk);
        total_bytes += length;

        if (failures > 8) {
            fprintf(stderr, "stopping after %d failures\n", failures);
            break;
        }
    }

    printf("random sweep: %d cases, %zu bytes, four byte distributions\n",
           CASES, total_bytes);
    printf("%s\n", failures == 0 ? "TOKENIZER AND CORPUS FUZZ CHECK PASSED"
                                 : "TOKENIZER AND CORPUS FUZZ CHECK FAILED");
    return failures == 0 ? 0 : 1;
}
