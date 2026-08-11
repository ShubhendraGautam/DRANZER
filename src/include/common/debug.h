#ifndef DEBUG_H
#define DEBUG_H

/**
 * Debug logging macro
 * Usage: DEBUG_PRINT("message: %d\n", value).
 * 
 * Library code never chooses a terminal. A DEBUG build routes messages only
 * when the embedding application defines DRANZER_DEBUG_SINK(fmt, ...)
 * (typically to its own callback/logger) in the compile flags. Without an
 * application-provided sink, DEBUG_PRINT remains silent.
 */

#if defined(DEBUG) && defined(DRANZER_DEBUG_SINK)
    #define DEBUG_PRINT(fmt, ...) \
        DRANZER_DEBUG_SINK("[DEBUG] %s:%d in %s(): " fmt, \
                           __FILE__, __LINE__, __func__, ##__VA_ARGS__)
#else
    #define DEBUG_PRINT(fmt, ...) ((void)0)
#endif

#endif /* DEBUG_H */
