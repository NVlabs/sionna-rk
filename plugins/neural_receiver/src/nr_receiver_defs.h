
#ifndef __NR_RECEIVER_DEFS_H__
#define __NR_RECEIVER_DEFS_H__

#include <stdint.h>
#include "openair1/PHY/defs_nr_common.h"
#include "openair1/PHY/defs_gNB.h"

typedef int32_t(receiver_initfunc_t)(void);
typedef int32_t(receiver_shutdownfunc_t)(void);

typedef int(receiver_compute_llrfunc_t)( PHY_VARS_gNB *gNB,
                     int ulsch_id,
                     int slot,
                     frame_t frame,
                     NR_DL_FRAME_PARMS *frame_parms,
                     NR_gNB_PUSCH *pusch_vars,
                     nfapi_nr_pusch_pdu_t *rel15_ul,
                     c16_t **rxF,
                     c16_t **ul_ch,
                     int16_t *llr,
                     int soffset,
                     int16_t const* lengths,
                     int start_symbol,
                     int num_symbols,
                     int output_shift,
                     uint32_t nvar );

typedef int(receiver_symbols_requestedfunc_t)(
                     NR_DL_FRAME_PARMS *frame_parms );

receiver_initfunc_t              receiver_init;
receiver_shutdownfunc_t          receiver_shutdown;
receiver_compute_llrfunc_t       receiver_compute_llr;
receiver_symbols_requestedfunc_t receiver_symbols_requested;

#endif // __NR_RECEIVER_DEFS_H__
