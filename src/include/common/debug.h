#ifndef DEBUG_H
#define DEBUG_H

#include <stdio.h>

/**
 * Debug logging macro
 * Usage: DEBUG_PRINT("message: %d\n", value);
 * 
 * Compile with DEBUG=1 to enable debug output:
 *   make DEBUG=1
 */

#ifdef DEBUG
    #define DEBUG_PRINT(fmt, ...) \
        do { \
            fprintf(stderr, "[DEBUG] %s:%d in %s(): " fmt, \
                    __FILE__, __LINE__, __func__, ##__VA_ARGS__); \
            fflush(stderr); \
        } while (0)
#else
    #define DEBUG_PRINT(fmt, ...) ((void)0)
#endif

#endif /* DEBUG_H */
