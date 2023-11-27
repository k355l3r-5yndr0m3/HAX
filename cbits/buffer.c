#include "pjrt_c_api.h"
#include "macros.h"

#include <assert.h>
#include <stddef.h>
#include <stdlib.h>

void buffer_destroy(PJRT_Api *api, PJRT_Buffer *buffer) { API_CALL_CATCHER(Buffer_Destroy, fatal_error, .buffer = buffer); }

PJRT_Event *buffer_to_host_buffer(PJRT_Api *api, PJRT_Buffer *src, PJRT_Buffer_MemoryLayout *host_layout, void *dst, size_t *dst_size) { return API_CALL_CATCHER(Buffer_ToHostBuffer, fatal_error, .src = src, .host_layout = host_layout, .dst = dst, .dst_size = *dst_size).event; }

PJRT_Event *buffer_to_host_buffer__automalloc(PJRT_Api *api, PJRT_Buffer *src, PJRT_Buffer_MemoryLayout *host_layout, void **dst, size_t *dst_size) {
    *dst_size = API_CALL_CATCHER(Buffer_ToHostBuffer, fatal_error, .src = src, .host_layout = host_layout, .dst = NULL).dst_size;
    *dst      = malloc(*dst_size); assert(*dst != NULL);
    return API_CALL_CATCHER(Buffer_ToHostBuffer, fatal_error, .src = src, .host_layout = host_layout, .dst = *dst, .dst_size = *dst_size).event;
}

void *buffer_to_host_buffer__automalloc_waiting(PJRT_Api *api, PJRT_Buffer *src, PJRT_Buffer_MemoryLayout *host_layout, size_t *dst_size) {
    *dst_size = API_CALL_CATCHER(Buffer_ToHostBuffer, fatal_error, .src = src, .host_layout = host_layout, .dst = NULL).dst_size;
    void *dst = malloc(*dst_size); assert(dst != NULL);
    PJRT_Event *event = API_CALL_CATCHER(Buffer_ToHostBuffer, fatal_error, .src = src, .host_layout = host_layout, .dst = dst, .dst_size = *dst_size).event;
    API_CALL_CATCHER(Event_Await, fatal_error, .event = event); API_CALL_CATCHER(Event_Destroy, fatal_error, .event = event);
    return dst;
}


size_t buffer_to_host_buffer__get_dst_size(PJRT_Api *api, PJRT_Buffer *src, PJRT_Buffer_MemoryLayout *host_layout) {
    return API_CALL_CATCHER(Buffer_ToHostBuffer, fatal_error, .src = src, .host_layout = host_layout, .dst = NULL).dst_size;
}

void buffer_to_host_buffer__event_await(PJRT_Api *api, PJRT_Buffer *src, PJRT_Buffer_MemoryLayout *host_layout, void *dst, size_t dst_size) {
    PJRT_Event *event = API_CALL_CATCHER(Buffer_ToHostBuffer, fatal_error, .src = src, .host_layout = host_layout, .dst = dst, .dst_size = dst_size).event;
    API_CALL_CATCHER(Event_Await, fatal_error, .event = event);
    API_CALL_CATCHER(Event_Destroy, fatal_error, .event = event);
}

const int64_t *buffer_dimensions(PJRT_Api *api, PJRT_Buffer *buffer, size_t *rank_out) {
    auto result = API_CALL_CATCHER(Buffer_Dimensions, fatal_error, .buffer = buffer);
    *rank_out = result.num_dims;
    return result.dims;
}

const int32_t buffer_element_type(PJRT_Api *api, PJRT_Buffer *buffer) {
    return API_CALL_CATCHER(Buffer_ElementType, fatal_error, .buffer = buffer).type;
}

