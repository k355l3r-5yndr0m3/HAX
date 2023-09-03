#include "pjrt_c_api.h"

#include <dlfcn.h>
#include <stddef.h>
#include <stdio.h>

#include "macros.h"


const PJRT_Api *
load_pjrt_plugin(const char *plugin_path) {
    void *plugin = dlopen(plugin_path, RTLD_NOW | RTLD_LOCAL);
    if (plugin == NULL) {
        fprintf(stderr, "Cannot load PjRt plugin %s\n", plugin_path);
        fprintf(stderr, "%s\n", dlerror());
        fflush(stderr);
        return NULL;
    } else {
        const PJRT_Api* (*GetPjrtApi)(void) = dlsym(plugin, "GetPjrtApi");
        if (GetPjrtApi == NULL) {
            fprintf(stderr, "Cannot find the symbol \"GetPjrtApi\" in %s\n", plugin_path);
            fflush(stderr);
            dlclose(plugin);
            return NULL;
        } else {
            printf("Loaded plugin %s\n", plugin_path);
            const PJRT_Api *api = GetPjrtApi();
            if (api->pjrt_api_version.major_version != PJRT_API_MAJOR || 
                api->pjrt_api_version.minor_version != PJRT_API_MINOR) {
                printf("Warning, plugin api version differs from version used in compilation\n"
                       "  Expected: %d.%d\n"
                       "  Actual  : %d.%d\n"
                       "Errors might occur as a result\n", PJRT_API_MAJOR, PJRT_API_MINOR, api->pjrt_api_version.major_version, api->pjrt_api_version.minor_version);
            }
            printf("Initializing plugin\n");
            auto error = API_CALL_RETERR(Plugin_Initialize);
            if (error != NULL) {
                fprintf(stderr, "Error in initializing plugin\n");
                auto error_msg = API_CALL_NOERROR(Error_Message, .error = error);
                fprintf(stderr, "%.*s\n", (int)error_msg.message_size, error_msg.message);
                API_CALL_NOERROR(Error_Destroy, .error = error);
                dlclose(plugin);
                fflush(stderr);
                return NULL;
            } else {
                auto plugin_attrs = API_CALL(Plugin_Attributes, error);
                if (error) {
                    fprintf(stderr, "Trouble querying plugin attributes\n");
                    API_CALL_NOERROR(Error_Destroy, .error = error);
                } else if (plugin_attrs.num_attributes > 0) {
                    printf("Plugin attributes:\n");
                    for (size_t i = 0; i < plugin_attrs.num_attributes; i++) {
                        PJRT_NamedValue *attr = &plugin_attrs.attributes[i];
                        printf("  %.*s: ", (int)attr->name_size, attr->name);
                        switch (attr->type) {
                            case PJRT_NamedValue_kString:
                                printf("%.*s\n", (int)attr->value_size, attr->string_value);
                                break;
                            case PJRT_NamedValue_kInt64:
                                printf("%ld\n", attr->int64_value);
                                break;
                            case PJRT_NamedValue_kInt64List:
                                for (size_t j = 0; j < attr->value_size; j++) {
                                    printf("%c%ld%c", j == 0 ? '[' : ' ', 
                                                      attr->int64_array_value[j],
                                                      j == attr->value_size - 1 ? ']' : ',');
                                }
                                putchar('\n');
                                break;
                            case PJRT_NamedValue_kFloat:
                                printf("%.0f\n", attr->float_value);
                                break;
                        }
                    }
                }
                fflush(stderr);
                fflush(stdout);
                return api;
            }
        }
    }
}




