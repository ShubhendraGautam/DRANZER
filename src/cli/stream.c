/*
 * Streaming file processing implementation
 * Handles large files without loading entire content into RAM
 */

#include "cli/stream.h"
#include "common/debug.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* Create a streaming file reader */
stream_reader_t *stream_reader_create(const char *filepath, size_t chunk_size) {
    if (!filepath || chunk_size == 0) {
        return NULL;
    }
    
    FILE *file = fopen(filepath, "r");
    if (!file) {
        fprintf(stderr, "Error: Cannot open file %s\n", filepath);
        return NULL;
    }
    
    stream_reader_t *reader = malloc(sizeof(stream_reader_t));
    if (!reader) {
        fclose(file);
        return NULL;
    }
    
    reader->buffer = malloc(chunk_size);
    if (!reader->buffer) {
        free(reader);
        fclose(file);
        return NULL;
    }
    
    reader->file = file;
    reader->buffer_size = chunk_size;
    reader->buffer_pos = 0;
    reader->total_read = 0;
    reader->eof = 0;
    
    DEBUG_PRINT("Stream reader created with %zu byte chunks\n", chunk_size);
    return reader;
}

/* Read next chunk from stream */
size_t stream_read_chunk(stream_reader_t *reader, char *output, size_t max_bytes) {
    if (!reader || !output || max_bytes == 0) {
        return 0;
    }
    
    if (reader->eof) {
        return 0;
    }
    
    size_t bytes_to_read = max_bytes < reader->buffer_size ? max_bytes : reader->buffer_size;
    size_t bytes_read = fread(reader->buffer, 1, bytes_to_read, reader->file);
    
    if (bytes_read == 0) {
        reader->eof = 1;
        return 0;
    }
    
    memcpy(output, reader->buffer, bytes_read);
    reader->total_read += bytes_read;
    
    if (bytes_read < bytes_to_read || feof(reader->file)) {
        reader->eof = 1;
    }
    
    DEBUG_PRINT("Stream read %zu bytes (total: %zu)\n", bytes_read, reader->total_read);
    return bytes_read;
}

/* Check if stream reached EOF */
int stream_is_eof(const stream_reader_t *reader) {
    if (!reader) return 1;
    return reader->eof;
}

/* Get total bytes read */
size_t stream_get_total_read(const stream_reader_t *reader) {
    if (!reader) return 0;
    return reader->total_read;
}

/* Free stream reader */
void stream_reader_free(stream_reader_t *reader) {
    if (!reader) return;
    
    if (reader->file) {
        fclose(reader->file);
    }
    free(reader->buffer);
    free(reader);
}

/* Create token stream processor */
token_stream_t *token_stream_create(size_t initial_capacity, size_t flush_threshold) {
    token_stream_t *stream = malloc(sizeof(token_stream_t));
    if (!stream) return NULL;
    
    stream->token_buffer = malloc(initial_capacity * sizeof(uint32_t));
    if (!stream->token_buffer) {
        free(stream);
        return NULL;
    }
    
    stream->capacity = initial_capacity;
    stream->size = 0;
    stream->flush_threshold = flush_threshold;
    
    DEBUG_PRINT("Token stream created: capacity=%zu, flush_threshold=%zu\n", 
                initial_capacity, flush_threshold);
    return stream;
}

/* Add tokens to stream */
int token_stream_add(token_stream_t *stream, uint32_t *tokens, size_t count) {
    if (!stream || !tokens || count == 0) {
        return -1;
    }
    
    /* Resize if needed */
    if (stream->size + count > stream->capacity) {
        size_t new_capacity = stream->capacity * 2;
        while (new_capacity < stream->size + count) {
            new_capacity *= 2;
        }
        
        uint32_t *new_buffer = realloc(stream->token_buffer, new_capacity * sizeof(uint32_t));
        if (!new_buffer) {
            return -1;
        }
        
        stream->token_buffer = new_buffer;
        stream->capacity = new_capacity;
        DEBUG_PRINT("Token stream expanded to capacity %zu\n", new_capacity);
    }
    
    memcpy(&stream->token_buffer[stream->size], tokens, count * sizeof(uint32_t));
    stream->size += count;
    
    return 0;
}

/* Check if stream is ready to flush */
int token_stream_ready_to_flush(const token_stream_t *stream) {
    if (!stream) return 0;
    return stream->size >= stream->flush_threshold;
}

/* Get current token count */
size_t token_stream_get_size(const token_stream_t *stream) {
    if (!stream) return 0;
    return stream->size;
}

/* Reset stream for next batch */
void token_stream_reset(token_stream_t *stream) {
    if (!stream) return;
    stream->size = 0;
}

void token_stream_retain_tail(token_stream_t *stream, size_t count) {
    if (!stream || count >= stream->size) return;
    if (count > 0) {
        memmove(stream->token_buffer,
                stream->token_buffer + stream->size - count,
                count * sizeof(*stream->token_buffer));
    }
    stream->size = count;
}

/* Free token stream */
void token_stream_free(token_stream_t *stream) {
    if (!stream) return;
    free(stream->token_buffer);
    free(stream);
}
