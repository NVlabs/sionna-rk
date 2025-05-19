#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import pytest

import numpy as np
import sionna as sn
from scipy.interpolate import interp1d
import ldpc_decoder

################################################################################
# Test against Sionna decoder for different code parameters
# We test that the resulting BLER is similar to the one from Sionna
################################################################################

def decode_cuda(enc, llr, num_iter):
    bs = llr.shape[0]
    Zc = enc._z
    k = enc.k
    n = enc.n
    BG = 1 if enc._bg == "bg1" else 2
    num_vn = 68*Zc if BG == 1 else 52*Zc
    parity_start = 22*Zc if BG == 1 else 10*Zc
    llr_np = llr.numpy()

    llr_ch = np.clip(llr_np/16*127, -127, 127).astype(np.int8)

    llr_input = np.zeros((bs, num_vn),dtype=np.int8) # takes care of the punctured bits (initialized to 0 LLR)
    llr_input[:,2*Zc:k] =  llr_ch[:,:k-2*Zc] # unpunctured message bits
    llr_input[:,k:parity_start] = 127      # shortened bits
    llr_input[:,parity_start:parity_start+n-k+2*Zc] = llr_ch[:,k-2*Zc:] # parity bits
    uhats = np.zeros((bs, k), dtype=np.uint8)
    for i, y in enumerate(llr_input):
        u_packed = ldpc_decoder.decode(BG, Zc, y, k, num_iter)
        uhats[i] = np.unpackbits(u_packed.astype(np.uint8))[:k]
    return uhats

all_test_vectors = [
    (120, 600, 1, 5),
    (864, 982, 3, 8),
    (1920, 2560, 2.5, 5),
    (3520, 5334, 2, 4.5),
    (8448, 24848, 1.5, 3),
]

@pytest.mark.parametrize("test_vector", all_test_vectors)
def test_bler(test_vector, pytestconfig, num_iter=None,
              trials=None, quiet=None):
    """Check the CUDA 5G LDPC Decoder against the sionna decoder."""

    if pytestconfig is not None:
        if num_iter is None:
            num_iter = pytestconfig.getoption('iters')
        if trials is None:
            trials = pytestconfig.getoption('trials')
        quiet = not pytestconfig.getoption('verbose')

    test_vectors = [test_vector]

    if not quiet: print(f'Start testing: {num_iter} iterations')

    num_bits_per_symbol = 2
    constellation = sn.phy.mapping.Constellation("qam", num_bits_per_symbol)
    mapper = sn.phy.mapping.Mapper(constellation=constellation)
    demapper = sn.phy.mapping.Demapper("maxlog", constellation=constellation)
    awgn_channel = sn.phy.channel.AWGN()
    binary_source = sn.utils.BinarySource()

    for k, n, start, end in test_vectors:

        enc = sn.phy.fec.ldpc.encoding.LDPC5GEncoder(k, n)
        dec = sn.phy.fec.ldpc.decoding.LDPC5GDecoder(enc, hard_out=True, cn_update="minsum", num_iter=num_iter)
        if not quiet: print(f'K = {k:4}, N = {n:5}, BG = {enc._bg:1}, Z = {enc._z:3}... ', end='', flush=True)

        coderate= k/n
        ebno_dbs = np.linspace(start, end, 10)
        blers_cuda = np.zeros_like(ebno_dbs, dtype=np.float32)
        blers_sionna = np.zeros_like(ebno_dbs, dtype=np.float32)

        sn.config.seed = 42
        u = binary_source([trials, k])

        for i, ebno_db in enumerate(ebno_dbs):
            sn.config.seed = 42
            no = sn.utils.ebnodb2no(ebno_db, num_bits_per_symbol=num_bits_per_symbol,coderate=coderate)
            c = enc(u)
            x = mapper(c)
            y = awgn_channel([x, no])
            llr = -demapper([y,no]) # sionna defines LLRs the wrong way around

            u_hat_sionna = dec(-llr)
            bler_sionna = sn.utils.compute_bler(u, u_hat_sionna)

            u_hat_cuda = decode_cuda(enc, llr, num_iter)
            bler_cuda = sn.utils.compute_bler(u, u_hat_cuda)

            blers_sionna[i] = bler_sionna
            blers_cuda[i] = bler_cuda
            if bler_sionna < 1e-5 and bler_cuda < 1e-5:
                break


        x1 = interp1d(blers_cuda, ebno_dbs, kind='linear', fill_value="extrapolate")
        x2 = interp1d(blers_sionna, ebno_dbs, kind='linear', fill_value="extrapolate")
        delta_snr = x1(0.01).item() - x2(0.01).item()
        if not quiet: print(f'Delta SNR: {abs(delta_snr):.2f} dB {"better" if delta_snr < 0 else "worse"}')
        if delta_snr > 0.2:
            raise AssertionError(f'Performance Deviation strong for K = {k}, N = {n}, BG = {enc._bg}, Z = {enc._z}')

    if not quiet: print('done')
