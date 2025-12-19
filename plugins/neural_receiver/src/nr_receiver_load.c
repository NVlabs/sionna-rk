#include "common/config/config_userapi.h"
#include "common/utils/LOG/log.h"
#include "common/utils/load_module_shlib.h"
#include "nr_receiver_defs.h"
#include "nr_receiver_extern.h"

// TODO: Q: can this be inside the loader function?
/* receiver_arg is used to initialize the config module so that the loader works as expected */
static char *receiver_arg[64]={"receivertest",NULL};

static int32_t receiver_no_thread_init() {
    return 0;
}

static int32_t receiver_all_symbols_requested(NR_DL_FRAME_PARMS *frame_parms) {
    return -1;
}

int load_receiver_lib( char *version, receiver_interface_t *interface )
{
    char *ptr = (char*)config_get_if();
    char libname[64] = "receiver";

    if (ptr == NULL) {  // config module possibly not loaded
        uniqCfg = load_configmodule( 1, receiver_arg, CONFIG_ENABLECMDLINEONLY );
        logInit();
    }

    // function description array for the shlib loader
    loader_shlibfunc_t shlib_fdesc[] = { {.fname = "receiver_init" },
                                         {.fname = "receiver_init_thread", .fptr = &receiver_no_thread_init },
                                         {.fname = "receiver_shutdown" },
                                         {.fname = "receiver_compute_llr" },
                                         {.fname = "receiver_symbols_requested", .fptr = (int(*)()) &receiver_all_symbols_requested }};

    int ret;
    ret = load_module_version_shlib( libname, version, shlib_fdesc, sizeofArray(shlib_fdesc), NULL );
    if (ret)
        printf("Error loading %s_%s, ignoring receiver plugin", libname, version);
    fflush(stdout);
    if (ret && strcmp(libname, "receiver") == 0 && (!version || !version[0] || strcmp(version, "orig") == 0))
        return ret; // receiver lib is optional
    else
        AssertFatal((ret >= 0), "Error loading receiver library");

    // assign loaded functions to the interface
    receiver_interface_t module_if = { };
    module_if.init = (receiver_initfunc_t *)shlib_fdesc[0].fptr;
    module_if.init_thread = (receiver_initfunc_t *)shlib_fdesc[1].fptr;
    module_if.shutdown = (receiver_shutdownfunc_t *)shlib_fdesc[2].fptr;
    module_if.compute_llr = (receiver_compute_llrfunc_t *)shlib_fdesc[3].fptr;
    module_if.symbols_requested = (receiver_symbols_requestedfunc_t*)shlib_fdesc[4].fptr; // may be filled with predicate function by external control entities

    AssertFatal( module_if.init() == 0, "Error starting receiver library %s %s\n", libname, version );

    memcpy(interface, &module_if, sizeof(*interface));
    return 0;
}

int free_receiver_lib( receiver_interface_t *receiver_interface )
{
    if (receiver_interface->shutdown)
        return receiver_interface->shutdown();
    else
        return 0;
}
