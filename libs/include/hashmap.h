#ifndef HASHMAP_H
#define HASHMAP_H

#include <stddef.h>
#include <string.h>
#include <stdlib.h>

typedef enum {
    HASHMAP_SUCCESS = 0 /* Operation successful */,
    HASHMAP_KEY_EXISTS,
    HASHMAP_NULL_MAP,
    HASHMAP_NULL_KEY,
    HASHMAP_NULL_VALUE,
    HASHMAP_ALLOCATION_FAILURE,
    HASHMAP_INVALID_INPUT_ERROR,
} hashmap_errors_t;

typedef enum {
    HASHMAP_VALUE_START = 0,
    HASHMAP_VALUE_TYPE_INT,
    HASHMAP_VALUE_TYPE_FLOAT,
    HASHMAP_VALUE_TYPE_STRING,
    HASHMAP_VALUE_TYPE_FLOAT_VECTOR,
    HASHMAP_VALUE_END,
} hashmap_value_type_t;

typedef struct hashmap_entry {
    char *key;
    void* value;
    hashmap_value_type_t value_type;
    struct hashmap_entry *next;
} hashmap_entry_t;

typedef struct hashmap {
    hashmap_entry_t **buckets;
    size_t bucket_count;
} hashmap_t;

typedef struct {
    float *data;
    size_t size;
} vector_t;

/* Function declarations */
unsigned long hash(const char *str);

int hashmap_new(hashmap_t *map, int bucket_count);
int hashmap_free(hashmap_t *map);
int hashmap_insert(hashmap_t *map, const char *key, const void *value, hashmap_value_type_t value_type);
int hashmap_get(hashmap_t *map, const char *key, void *value, hashmap_value_type_t *value_type);
int hashmap_remove(hashmap_t *map, const char *key);
int hashmap_contains(hashmap_t *map, const char *key);

#endif /* HASHMAP_H */