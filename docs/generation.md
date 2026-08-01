# Generation runtime

[← Back to README](../README.md)

DRANZER has one incremental decode loop for CLI and C callers. It primes a per-layer KV cache from
the prepared prompt, selects one token at a time, and can deliver each decoded token piece through
a callback without building a complete decoded continuation.

## CLI controls

```bash
./app.out generate \
  --model dranzer.pth \
  --prompt "In the beginning" \
  --length 80 \
  --sampling topp --top-p 0.9 --temperature 0.8 \
  --repetition-penalty 1.15 \
  --min-length 12 \
  --stop "\n\n" --stop "END"
```

- `--stop TEXT` may be repeated eight times. Each value is encoded with the model tokenizer and
  matched against generated tokens only, not the prompt. The matching marker is withheld from
  output. If several registered sequences match the same suffix, registration order wins.
- `--min-length N` requires at least `N` generated content tokens before EOS or a stop sequence can
  terminate decoding. A stop-like sequence produced before that boundary is ordinary output.
- `--repetition-penalty P` applies once per token ID already present in the prompt or continuation.
  Positive logits are divided by `P`; negative logits are multiplied by it. `1.0` disables the
  penalty.

The CLI writes the retained prompt and each safe generated piece immediately and flushes stdout.
It holds only a possible stop-sequence prefix. Stop matching is token-based, so callers that require
arbitrary decoded-byte substring matching should implement that policy in the callback.

## Callback interface

Include `cli/generation.h`, initialize `generation_options_t`, and provide `on_token`:

```c
static int on_token(uint32_t id, const char *text, size_t length, void *context) {
    (void)id;
    FILE *output = context;
    return fwrite(text, 1, length, output) == length ? 0 : 1;
}

generation_options_t options;
generation_options_init(&options);
options.on_token = on_token;
options.callback_data = stdout;

generation_result_t result;
generation_errors_t rc = generation_decode_with_options(
    model, tokenizer, sequence, prompt_count, maximum_new_tokens,
    &options, &result);
```

The callback receives an encoder-owned text pointer valid for that call. Returning nonzero accepts
the current token and stops cleanly; `result.stopped_by_callback` records that outcome. EOS and
matched stop-sequence tokens are retained in the model-visible `sequence` and counted by
`new_count`, but they are not sent to the callback. `emitted_count` is the delivered continuation
length. Callback cancellation truncates the result to the accepted callback prefix.

`generation_stop_sequence_t` values contain token-ID arrays owned by the caller for the duration of
the decode call. Invalid or control-token stop sequences, non-finite controls, repetition penalties
below 1, and invalid sampling filters are rejected before model state is changed.

## Ordering of each decode step

1. Apply repetition penalty to tokens already visible to the model.
2. Mask PAD, UNK, BOS, and tokenizer vocabulary slots with no assigned token.
3. Mask EOS until the minimum-length requirement is met.
4. Apply temperature and greedy, top-k, or top-p selection.
5. Stop on EOS, a registered token sequence, or callback cancellation; otherwise stream every token
   that cannot still be a stop-sequence prefix.

`max_seq_len` is the retained attention window, not a total output limit. After the window fills,
the KV cache evicts its oldest row and reuses that physical slot. Generated token IDs remain in the
caller-provided `sequence` when its `sequence_capacity` permits, while decoded text continues to
stream. The CLI allocates enough token storage for its requested `--length`.

Positions are absolute: the bundled sinusoidal table is used inside the trained window, and the
same formula is evaluated for later positions. This avoids rebasing retained keys and values at
each eviction, but later positions are an extrapolation beyond positions observed in training.
The cache retains contextualized higher-layer keys and values as-is; it does not recompute the
newest window after an eviction.
