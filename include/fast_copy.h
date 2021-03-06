#ifndef FAST_COPY_H_INCLUDED
#define FAST_COPY_H_INCLUDED

#include <stdlib.h>

#ifndef FAST_COPY_MEMCPY_THRESHOLD
#  define FAST_COPY_MEMCPY_THRESHOLD ((size_t)1ull << 20)
#endif

#ifdef __cplusplus
extern "C" {
#endif
#ifdef FHT_HEADER_ONLY
#include "fast_copy.c"
#else
void *fast_copy(void *out, void *in, size_t m);
#endif
#ifdef __cplusplus
} // extern "C"
#endif

#endif // FAST_COPY_H_INCLUDED
