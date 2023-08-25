#pragma once
#include "pjrt_c_api.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define API_ARG(fname, ...) ((PJRT_ ## fname ## _Args){ .struct_size = PJRT_ ## fname ## _Args_STRUCT_SIZE, .priv = NULL __VA_OPT__(,) __VA_ARGS__ }) 

#define API_CALL(fname, eout, ...) ({ PJRT_ ## fname ## _Args args = API_ARG(fname __VA_OPT__(,) __VA_ARGS__); eout = api->PJRT_ ## fname (&args); args; })
#define API_CALL_CATCHER(fname, catcher, ...) ({ PJRT_ ## fname ## _Args args = API_ARG(fname __VA_OPT__(,) __VA_ARGS__); PJRT_Error *error = api->PJRT_ ## fname (&args); catcher(api, error); args; })
#define API_CALL_NOERROR(fname, ...) ({ PJRT_ ## fname ## _Args args = API_ARG(fname __VA_OPT__(,) __VA_ARGS__); api->PJRT_ ## fname (&args); args; })
#define API_CALL_RETERR(fname, ...) ({ PJRT_ ## fname ## _Args args = API_ARG(fname __VA_OPT__(,) __VA_ARGS__); PJRT_Error *error = api->PJRT_ ## fname (&args); error; })

#define auto __auto_type

#define ASSERT_ENUM_EQUIVALENT(e, t) static_assert(sizeof(e) == sizeof(t), #e " is umcompatable with " #t)


static inline void 
fatal_error(PJRT_Api *api, PJRT_Error *error) {
    if (error != NULL) {
        auto msg = API_CALL_NOERROR(Error_Message, .error = error);
        fprintf(stderr, "Error %.*s\n", (int)msg.message_size, msg.message);
        API_CALL_NOERROR(Error_Destroy, .error = error);
        exit(-1);
    }
}
