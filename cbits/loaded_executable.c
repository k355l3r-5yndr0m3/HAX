#include "pjrt_c_api.h"


#include "macros.h"
#include <assert.h>

// loaded executable
void 
loaded_executable_destroy(PJRT_Api *api, PJRT_LoadedExecutable *executable) {
    API_CALL_CATCHER(LoadedExecutable_Destroy, fatal_error, .executable = executable);
}


void
loaded_executable_execute(PJRT_Api *api, PJRT_LoadedExecutable *executable,
                          PJRT_ExecuteOptions *options,
                          PJRT_Buffer ***argument_lists,
                          size_t num_devices, size_t num_args,
                          PJRT_Buffer*** output_lists,
                          PJRT_Event** device_complete_events,
                          PJRT_Device* execute_device) {
    assert(options == NULL); // TODO: Implement this
    API_CALL_CATCHER(LoadedExecutable_Execute, fatal_error,
            .executable = executable,
            .options = options, 
            .argument_lists = argument_lists,
            .num_devices = num_devices,
            .num_args = num_args,
            .output_lists = output_lists,
            .device_complete_events = device_complete_events,
            .execute_device = execute_device);
}


void 
loaded_executable_execute__1_await(PJRT_Api *api, PJRT_LoadedExecutable *executable, 
                                   PJRT_ExecuteOptions *options, 
                                   PJRT_Buffer **argument_list, 
                                   size_t num_args, 
                                   PJRT_Buffer **output_list, 
                                   PJRT_Device *execute_device) {
    assert(options == NULL); 
    PJRT_ExecuteOptions ops = {
        .struct_size = PJRT_ExecuteOptions_STRUCT_SIZE,
        .priv = NULL,
    };
    PJRT_Event *event;
    API_CALL_CATCHER(LoadedExecutable_Execute, fatal_error, 
            .executable = executable, 
            .options = &ops,
            .argument_lists = &argument_list,
            .num_devices = 1, .num_args = num_args,
            .output_lists = &output_list,
            .device_complete_events = &event,
            .execute_device = execute_device);
    API_CALL_CATCHER(Event_Await, fatal_error, .event = event);
    API_CALL_CATCHER(Event_Destroy, fatal_error, .event = event);
}
