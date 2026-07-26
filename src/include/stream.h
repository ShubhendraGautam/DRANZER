/*
 * Streaming file processing for handling large files efficiently
 * Enables processing of files larger than available RAM
 */

#ifndef STREAM_H
#define STREAM_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/* Chunk size for streaming (256KB chunks - good balance) */
#define STREAM_CHUNK_SIZE (256 * 1024)

/* Streaming file reader state */
typedef struct {
    FILE *file;
    char *buffer;
    size_t buffer_size;
    size_t buffer_pos;
    size_t total_read;
    int eof;
} stream_reader_t;

/* Token stream processor */
typedef struct {
    uint32_t *token_buffer;
    size_t capacity;
    size_t size;
    size_t flush_threshold;  /* Flush batch when this size reached */
} token_stream_t;

/**
 * Create a streaming file reader
 * @param filepath: Path to file to read
 * @param chunk_size: Size of chunks to read (use STREAM_CHUNK_SIZE)
 * @return Stream reader or NULL on error
 */
stream_reader_t *stream_reader_create(const char *filepath, size_t chunk_size);

/**
 * Read next chunk from stream
 * @param reader: Stream reader
 * @param output: Buffer to read into
 * @param max_bytes: Maximum bytes to read
 * @return Bytes actually read (0 if EOF)
 */
size_t stream_read_chunk(stream_reader_t *reader, char *output, size_t max_bytes);

/**
 * Check if stream has reached EOF
 */
int stream_is_eof(const stream_reader_t *reader);

/**
 * Get total bytes read so far
 */
size_t stream_get_total_read(const stream_reader_t *reader);

/**
 * Free stream reader
 */
void stream_reader_free(stream_reader_t *reader);

/**
 * Create token stream processor
 * @param initial_capacity: Initial buffer capacity
 * @param flush_threshold: Number of tokens to accumulate before processing
 */
token_stream_t *token_stream_create(size_t initial_capacity, size_t flush_threshold);

/**
 * Add tokens to stream
 * @param stream: Token stream
 * @param tokens: Token IDs to add
 * @param count: Number of tokens
 * @return 0 on success
 */
int token_stream_add(token_stream_t *stream, uint32_t *tokens, size_t count);

/**
 * Check if stream is ready to flush
 */
int token_stream_ready_to_flush(const token_stream_t *stream);

/**
 * Get current token count
 */
size_t token_stream_get_size(const token_stream_t *stream);

/**
 * Reset stream for next batch
 */
void token_stream_reset(token_stream_t *stream);

/**
 * Free token stream
 */
void token_stream_free(token_stream_t *stream);

#endif /* STREAM_H */
