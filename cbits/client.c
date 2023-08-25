#include "pjrt_c_api.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "macros.h"

// Client
PJRT_Client *client_create(PJRT_Api *api) {
    return API_CALL_CATCHER(Client_Create, fatal_error, 
                            .create_options = NULL,
                            .num_options = 0,
                            .kv_get_callback = NULL, .kv_get_user_arg = NULL,
                            .kv_put_callback = NULL, .kv_put_user_arg = NULL).client;
}
void client_destroy(PJRT_Api *api, PJRT_Client *client) { API_CALL_CATCHER(Client_Destroy, fatal_error, .client = client); }

const char *client_platform_name(PJRT_Api *api, PJRT_Client *client, size_t *platform_name_size_out) {
    auto result = API_CALL_CATCHER(Client_PlatformName, fatal_error, .client = client);
    *platform_name_size_out = result.platform_name_size;
    return result.platform_name;
}

// Client devices
PJRT_Device **client_addressable_devices(PJRT_Api *api, PJRT_Client *client, size_t *num_addressable_devices_out) {
    auto result = API_CALL_CATCHER(Client_AddressableDevices, fatal_error, .client = client);
    *num_addressable_devices_out = result.num_addressable_devices;
    return result.addressable_devices;
}
PJRT_Device *client_lookup_addressable_device(PJRT_Api *api, PJRT_Client *client, int id) {
    return API_CALL_CATCHER(Client_LookupAddressableDevice, fatal_error, .client = client, .local_hardware_id = id).addressable_device;
}


PJRT_LoadedExecutable *client_compile(PJRT_Api *api, PJRT_Client *client, char *code, size_t code_size, const char *compile_options, size_t compile_options_size) {
    PJRT_Program program = {
        .struct_size = PJRT_Program_STRUCT_SIZE, .priv = NULL, 
        .code = code, .code_size = code_size,
        .format = "mlir", .format_size = 4,
    };
    return API_CALL_CATCHER(Client_Compile, fatal_error, .client = client, .program = &program, .compile_options = compile_options, .compile_options_size = compile_options_size).executable;
}


PJRT_Buffer *client_buffer_from_host_buffer(PJRT_Api *api, PJRT_Client *client,
                                            const void *data, int32_t type,
                                            const int64_t *dims, size_t num_dims,
                                            const int64_t* byte_strides, size_t num_byte_strides,
                                            int32_t host_buffer_semantics, PJRT_Device *device,
                                            PJRT_Buffer_MemoryLayout *device_layout,
                                            PJRT_Event **done_with_host_buffer_out) {
    assert(device_layout == NULL); // TODO: Support device layout, this is tricky because I don't quite understand tiled
    ASSERT_ENUM_EQUIVALENT(PJRT_Buffer_Type, int32_t);
    ASSERT_ENUM_EQUIVALENT(PJRT_HostBufferSemantics, int32_t);
    auto result = API_CALL_CATCHER(Client_BufferFromHostBuffer, fatal_error,
                                   .client = client, .data = data,
                                   .type = type, .dims = dims,
                                   .num_dims = num_dims, .byte_strides = byte_strides,
                                   .num_byte_strides = num_byte_strides,
                                   .host_buffer_semantics = host_buffer_semantics,
                                   .device = device, .device_layout = device_layout);

    *done_with_host_buffer_out = result.done_with_host_buffer;
    return result.buffer;
}


void client_default_device_assignment(PJRT_Api *api, PJRT_Client *client, int num_replicas, int num_partitions, size_t default_assignment_size, int* default_assignment) {
    API_CALL_CATCHER(Client_DefaultDeviceAssignment, fatal_error,
            .client = client,
            .num_replicas = num_replicas, .num_partitions = num_partitions,
            .default_assignment_size = default_assignment_size, .default_assignment = default_assignment);
}

























// Special buffer creation function
typedef struct {
    PJRT_Api *api;
    PJRT_Event *readied;
    void *data;
} free_host_buffer_callback_arg;
void free_host_buffer_callback(PJRT_Error *error, void *user_arg) {
    free_host_buffer_callback_arg *arg = user_arg;
    PJRT_Api *api = arg->api;
    fatal_error(api, error);
    API_CALL_CATCHER(Event_Destroy, fatal_error, .event = arg->readied);
    free(arg->data);
    free(arg);
}
PJRT_Buffer *client_buffer_from_host_buffer__autofree(PJRT_Api *api, PJRT_Client *client, 
                                                      void *data, int32_t type,
                                                      const int64_t *dims, size_t num_dims,
                                                      const int64_t* byte_strides, size_t num_byte_strides,
                                                      int32_t host_buffer_semantics, PJRT_Device *device,
                                                      PJRT_Buffer_MemoryLayout *device_layout) {
    assert(device_layout == NULL); // TODO: Support device layout, this is tricky because I don't quite understand tiled
    ASSERT_ENUM_EQUIVALENT(PJRT_Buffer_Type, int32_t);
    ASSERT_ENUM_EQUIVALENT(PJRT_HostBufferSemantics, int32_t);
    auto result = API_CALL_CATCHER(Client_BufferFromHostBuffer, fatal_error,
                                   .client = client, .data = data,
                                   .type = type, .dims = dims,
                                   .num_dims = num_dims, .byte_strides = byte_strides,
                                   .num_byte_strides = num_byte_strides,
                                   .host_buffer_semantics = host_buffer_semantics,
                                   .device = device, .device_layout = device_layout);
    free_host_buffer_callback_arg *arg = malloc(sizeof(free_host_buffer_callback_arg));
    *arg = (free_host_buffer_callback_arg){ .api = api, .readied = result.done_with_host_buffer, .data = data };
    API_CALL_CATCHER(Event_OnReady, fatal_error, .event = result.done_with_host_buffer, free_host_buffer_callback, .user_arg = arg);
    return result.buffer;
}
