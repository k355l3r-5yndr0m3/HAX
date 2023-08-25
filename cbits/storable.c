#include "pjrt_c_api.h"

#include <stdalign.h>
#include <stddef.h>

#define SIZE_ALIGN_GETTER(name) \
    size_t hs__sizeof__ ## name () { return sizeof(name); } \
    size_t hs__alignof__ ## name () { return alignof(name); }

