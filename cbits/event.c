#include "pjrt_c_api.h"

#include "macros.h"

void event_destroy(PJRT_Api *api, PJRT_Event *event) { API_CALL_CATCHER(Event_Destroy, fatal_error, .event = event); }
void event_await  (PJRT_Api *api, PJRT_Event *event) { API_CALL_CATCHER(Event_Await, fatal_error, .event = event); }

void event_onready(PJRT_Api *api, PJRT_Event *event, PJRT_Event_OnReadyCallback callback, void *user_arg) { API_CALL_CATCHER(Event_OnReady, fatal_error, .event = event, .callback = callback, .user_arg = user_arg); }

void event_wait_then_destroy(PJRT_Api *api, PJRT_Event *event) { API_CALL_CATCHER(Event_Await, fatal_error, .event = event); API_CALL_CATCHER(Event_Destroy, fatal_error, .event = event); }
// 
