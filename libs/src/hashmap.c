/*
    Very Basic Hashmap implementation in C to support tokenization.
    Accepts both string keys and length-delimited binary keys.

*/


#include "hashmap.h"
#include "debug.h"

unsigned long hash_bytes(const void *data, size_t length) {
    const unsigned char *bytes = data;
    unsigned long h = 5381;
    for (size_t i = 0; i < length; i++) {
        h = ((h << 5) + h) + bytes[i]; // h * 33 + byte
    }

    return h;
}

unsigned long hash(const char *str) {
    return hash_bytes(str, strlen(str));
}

/* Forward declaration */
static void hashmap_internal_free_value(void *value, hashmap_value_type_t value_type);

int hashmap_new(hashmap_t *map, int bucket_count) {
    DEBUG_PRINT("Creating hashmap with %d buckets\n", bucket_count);
    if ( map == NULL || bucket_count <= 0 ) {
        return HASHMAP_INVALID_INPUT_ERROR; // Invalid input
    }

    map->bucket_count = bucket_count;
    map->buckets = calloc(bucket_count, sizeof(hashmap_entry_t *));
    if ( map->buckets == NULL ) {
        return HASHMAP_ALLOCATION_FAILURE; // Memory allocation failed
    }

    return HASHMAP_SUCCESS;
}

int hashmap_free(hashmap_t *map) {
    if ( map == NULL ) {
        return HASHMAP_NULL_MAP; // Invalid input
    }

    for ( int i = 0; i < map->bucket_count; i++ ) {
        hashmap_entry_t *current = map->buckets[i];
        while ( current != NULL ) {
            hashmap_entry_t *next = current->next;
            free(current->key);
            hashmap_internal_free_value(current->value, current->value_type);
            free(current);
            current = next;
        }
    }
    free(map->buckets);
    map->buckets = NULL;
    map->bucket_count = 0;

    return HASHMAP_SUCCESS;
}

int hashmap_internal_duplicate_value(const void *src, void **dest, hashmap_value_type_t value_type) {
    // Implementation for duplicating value based on its type
    switch ( value_type ) {
        case HASHMAP_VALUE_TYPE_INT:
            *dest = malloc(sizeof(int));
            if ( *dest == NULL ) {
                return HASHMAP_ALLOCATION_FAILURE;
            }
            *(int *)(*dest) = *(const int *)src;
            break;

        case HASHMAP_VALUE_TYPE_UINT32:
            *dest = malloc(sizeof(uint32_t));
            if ( *dest == NULL ) {
                return HASHMAP_ALLOCATION_FAILURE;
            }
            *(uint32_t *)(*dest) = *(const uint32_t *)src;
            break;

        case HASHMAP_VALUE_TYPE_FLOAT:
            *dest = malloc(sizeof(float));
            if ( *dest == NULL ) {
                return HASHMAP_ALLOCATION_FAILURE;
            }
            *(float *)(*dest) = *(const float *)src;
            break;

        case HASHMAP_VALUE_TYPE_STRING:
            *dest = strdup((const char *)src);
            if ( *dest == NULL ) {
                return HASHMAP_ALLOCATION_FAILURE;
            }
            break;

        case HASHMAP_VALUE_TYPE_FLOAT_VECTOR: {
            const vector_t *vec_src = (const vector_t *)src;
            vector_t *vec_dest = malloc(sizeof(vector_t));
            if ( vec_dest == NULL ) {
                return HASHMAP_ALLOCATION_FAILURE;
            }
            vec_dest->size = vec_src->size;
            vec_dest->data = malloc(vec_src->size * sizeof(float));
            if ( vec_dest->data == NULL ) {
                free(vec_dest);
                return HASHMAP_ALLOCATION_FAILURE;
            }
            memcpy(vec_dest->data, vec_src->data, vec_src->size * sizeof(float));
            *dest = vec_dest;
            break;
        }

        default:
            return HASHMAP_INVALID_INPUT_ERROR; // Invalid value type
    }
    return HASHMAP_SUCCESS;
}

/* Helper function to free a value based on its type */
static void hashmap_internal_free_value(void *value, hashmap_value_type_t value_type) {
    if (value == NULL) {
        return;
    }

    switch ( value_type ) {
        case HASHMAP_VALUE_TYPE_INT:
        case HASHMAP_VALUE_TYPE_UINT32:
        case HASHMAP_VALUE_TYPE_FLOAT:
        case HASHMAP_VALUE_TYPE_STRING:
            free(value);
            break;

        case HASHMAP_VALUE_TYPE_FLOAT_VECTOR: {
            vector_t *vec = (vector_t *)value;
            if (vec->data != NULL) {
                free(vec->data);
            }
            free(vec);
            break;
        }

        default:
            // Unknown type, just free the pointer
            free(value);
            break;
    }
}


// Inserts a length-delimited key-value pair into the hashmap.
int hashmap_insert_bytes(hashmap_t *map, const void *key, size_t key_length,
                         const void *value, hashmap_value_type_t value_type) {
    DEBUG_PRINT("Inserting key of %zu bytes with type=%d\n", key_length, value_type);
    hashmap_errors_t rc = HASHMAP_SUCCESS; 

    // Input validation
    if ( map == NULL ) {
        return HASHMAP_NULL_MAP; 
    }

    if ( key == NULL ) {
        return HASHMAP_NULL_KEY; 
    }

    if ( value == NULL ) {
        return HASHMAP_NULL_VALUE; 
    }

    if ( value_type <= HASHMAP_VALUE_START || value_type >= HASHMAP_VALUE_END ) {
        return HASHMAP_INVALID_INPUT_ERROR; // Invalid value type
    }

    // Create a new entry
    hashmap_entry_t *new_entry = calloc(1, sizeof(hashmap_entry_t));
    if ( new_entry == NULL ) {
        return HASHMAP_ALLOCATION_FAILURE;
    }

    // Duplicate the key. The sentinel is not part of its identity.
    if (key_length == SIZE_MAX) {
        free(new_entry);
        return HASHMAP_INVALID_INPUT_ERROR;
    }
    new_entry->key = malloc(key_length + 1);
    if ( new_entry->key == NULL ) {
        free(new_entry);
        return HASHMAP_ALLOCATION_FAILURE;
    }
    if (key_length > 0) memcpy(new_entry->key, key, key_length);
    new_entry->key[key_length] = '\0';
    new_entry->key_length = key_length;

    // Duplicate the value
    rc = hashmap_internal_duplicate_value(value, &new_entry->value, value_type);
    if ( rc != HASHMAP_SUCCESS ) {
        free(new_entry->key);
        free(new_entry);
        return rc; // Propagate the allocation failure
    }

    // Store the value type
    new_entry->value_type = value_type;

    // Explicitly set next to NULL for clarity, though calloc already does this
    new_entry->next = NULL;

    // Find the appropriate bucket
    size_t bucket_index = hash_bytes(key, key_length) % map->bucket_count;

    // If the bucket is empty, insert the new entry directly
    if ( map->buckets[bucket_index] == NULL ) {
        map->buckets[bucket_index] = new_entry;
    } 
    // If the bucket is not empty, we need to check for existing keys and handle collisions by chaining
    else {
        hashmap_entry_t *current = map->buckets[bucket_index];
        while ( current != NULL ) {
            if (current->key_length == key_length &&
                memcmp(current->key, key, key_length) == 0) {
                // Key already exists, update the value
                void *temp_value;
                rc = hashmap_internal_duplicate_value(value, &temp_value, value_type);
                if ( rc != HASHMAP_SUCCESS ) {
                    free(new_entry->key);
                    hashmap_internal_free_value(new_entry->value,
                                                new_entry->value_type);
                    free(new_entry);
                    return rc; // Propagate the allocation failure, old value remains unchanged
                }

                hashmap_internal_free_value(current->value, current->value_type); // Free the old value properly
                current->value = temp_value; // Update to the new value
                current->value_type = value_type; // Update the type

                // Clean up the new entry since we won't be using it
                free(new_entry->key);
                hashmap_internal_free_value(new_entry->value, value_type);
                free(new_entry);
                return HASHMAP_KEY_EXISTS; // Indicate that the key was updated
            }
            if ( current->next == NULL ) {
                break;
            }
            current = current->next;
        }
        current->next = new_entry;
    }

    return HASHMAP_SUCCESS;
}

int hashmap_insert(hashmap_t *map, const char *key, const void *value,
                   hashmap_value_type_t value_type) {
    if (key == NULL) return HASHMAP_NULL_KEY;
    return hashmap_insert_bytes(map, key, strlen(key), value, value_type);
}

// Retrieves the value associated with a length-delimited key from the hashmap.
int hashmap_get_bytes(hashmap_t *map, const void *key, size_t key_length,
                      void *value, hashmap_value_type_t *value_type) {
    // Input validation
    if ( map == NULL ) {
        return HASHMAP_NULL_MAP;
    }

    if ( key == NULL ) {
        return HASHMAP_NULL_KEY;
    }

    if ( value == NULL ) {
        return HASHMAP_NULL_VALUE;
    }

    // Find the appropriate bucket
    size_t bucket_index = hash_bytes(key, key_length) % map->bucket_count;
    hashmap_entry_t *current = map->buckets[bucket_index];

    // Search for the key in the bucket chain
    while ( current != NULL ) {
        if (current->key_length == key_length &&
            memcmp(current->key, key, key_length) == 0) {
            // Key found, copy the value based on its type
            switch ( current->value_type ) {
                case HASHMAP_VALUE_TYPE_INT:
                    *(int *)value = *(const int *)current->value;
                    break;

                case HASHMAP_VALUE_TYPE_UINT32:
                    *(uint32_t *)value = *(const uint32_t *)current->value;
                    break;

                case HASHMAP_VALUE_TYPE_FLOAT:
                    *(float *)value = *(const float *)current->value;
                    break;

                case HASHMAP_VALUE_TYPE_STRING:
                    *(char **)value = (char *)current->value;
                    break;

                case HASHMAP_VALUE_TYPE_FLOAT_VECTOR: {
                    const vector_t *src = (const vector_t *)current->value;
                    *(vector_t *)value = *src;
                    break;
                }

                default:
                    return HASHMAP_INVALID_INPUT_ERROR;
            }

            if ( value_type != NULL ) {
                *value_type = current->value_type;
            }

            return HASHMAP_SUCCESS;
        }
        current = current->next;
    }

    // Key not found
    return HASHMAP_INVALID_INPUT_ERROR;
}

int hashmap_get(hashmap_t *map, const char *key, void *value,
                hashmap_value_type_t *value_type) {
    if (key == NULL) return HASHMAP_NULL_KEY;
    return hashmap_get_bytes(map, key, strlen(key), value, value_type);
}

// Removes a length-delimited key-value pair from the hashmap.
int hashmap_remove_bytes(hashmap_t *map, const void *key, size_t key_length) {
    // Input validation
    if ( map == NULL ) {
        return HASHMAP_NULL_MAP;
    }

    if ( key == NULL ) {
        return HASHMAP_NULL_KEY;
    }

    // Find the appropriate bucket
    size_t bucket_index = hash_bytes(key, key_length) % map->bucket_count;
    hashmap_entry_t *current = map->buckets[bucket_index];
    hashmap_entry_t *prev = NULL;

    // Search for the key in the bucket chain
    while ( current != NULL ) {
        if (current->key_length == key_length &&
            memcmp(current->key, key, key_length) == 0) {
            // Key found, remove it
            if ( prev == NULL ) {
                // Removing the first entry in the bucket
                map->buckets[bucket_index] = current->next;
            } else {
                // Removing an entry in the middle or end of the chain
                prev->next = current->next;
            }

            // Free the entry's resources
            free(current->key);
            hashmap_internal_free_value(current->value, current->value_type);
            free(current);

            return HASHMAP_SUCCESS;
        }
        prev = current;
        current = current->next;
    }

    // Key not found
    return HASHMAP_INVALID_INPUT_ERROR;
}

int hashmap_remove(hashmap_t *map, const char *key) {
    if (key == NULL) return HASHMAP_NULL_KEY;
    return hashmap_remove_bytes(map, key, strlen(key));
}

// Checks if a length-delimited key exists in the hashmap.
int hashmap_contains_bytes(hashmap_t *map, const void *key, size_t key_length) {
    // Input validation
    if ( map == NULL ) {
        return HASHMAP_NULL_MAP;
    }

    if ( key == NULL ) {
        return HASHMAP_NULL_KEY;
    }

    // Find the appropriate bucket
    size_t bucket_index = hash_bytes(key, key_length) % map->bucket_count;
    hashmap_entry_t *current = map->buckets[bucket_index];

    // Search for the key in the bucket chain
    while ( current != NULL ) {
        if (current->key_length == key_length &&
            memcmp(current->key, key, key_length) == 0) {
            // Key found
            return HASHMAP_SUCCESS;
        }
        current = current->next;
    }

    // Key not found
    return HASHMAP_INVALID_INPUT_ERROR;
}

int hashmap_contains(hashmap_t *map, const char *key) {
    if (key == NULL) return HASHMAP_NULL_KEY;
    return hashmap_contains_bytes(map, key, strlen(key));
}
