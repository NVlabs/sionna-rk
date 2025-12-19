#ifndef _NR_RECEIVER_EXTERN_H__
#define _NR_RECEIVER_EXTERN_H__

#include "nr_receiver_defs.h"

typedef struct receiver_interface_s {
    receiver_initfunc_t        *init;
    receiver_initfunc_t        *init_thread;
    receiver_shutdownfunc_t    *shutdown;
    receiver_compute_llrfunc_t *compute_llr;
    receiver_symbols_requestedfunc_t *symbols_requested;
} receiver_interface_t;

// global access point for the plugin interface
extern receiver_interface_t receiver_interface;

int load_receiver_lib( char *version, receiver_interface_t * );
int free_receiver_lib( receiver_interface_t *receiver_interface );

#endif // _NR_RECEIVER_EXTERN_H__
