import pytest
import numpy as np
import sionna as sn
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
import tensorflow as tf

def decode_cuda(compiled_decoder, enc, llr, num_iter):
    """
    Wrapper to call the CUDA decoder.
    Adapts Sionna/Input LLRs to the format expected by the CUDA kernel.
    """
    bs = llr.shape[0]
    Zc = enc._z
    k = enc.k
    n = enc.n
    BG = 1 if enc._bg == "bg1" else 2

    # Dimensions based on BaseGraph
    num_vn = 68*Zc if BG == 1 else 52*Zc
    parity_start = 22*Zc if BG == 1 else 10*Zc

    # Convert LLRs to numpy and int8
    # Note: Sionna LLRs are usually logits (floats).
    # The notebook scales them: llr_np/32*127.
    # We'll follow the notebook's scaling.
    llr_np = llr.numpy()
    llr_ch = np.clip(llr_np/32*127, -127, 127).astype(np.int8)

    # Prepare input buffer including punctured/shortened bits
    llr_input = np.zeros((bs, num_vn), dtype=np.int8)

    # Map channel LLRs to Variable Nodes
    # 2*Zc are punctured columns in 5G LDPC
    llr_input[:, 2*Zc:k] = llr_ch[:, :k-2*Zc] # unpunctured message bits

    # Shortened bits are set to max LLR (certainty)
    llr_input[:, k:parity_start] = 127

    # Parity bits
    llr_input[:, parity_start:parity_start+n-k+2*Zc] = llr_ch[:, k-2*Zc:]

    uhats = np.zeros((bs, k), dtype=np.uint8)

    for i in range(bs):
        # Call the CUDA decoder
        # Expected signature: decode(BG, Z, llrs, block_length, num_iter)
        # llr_input[i] is shape (num_vn,)

        u_packed = compiled_decoder.decode(BG, Zc, llr_input[i], k, num_iter)

        # Unpack bits
        # u_packed is uint8 array of packed bits
        unpacked = np.unpackbits(u_packed.view(np.uint8))

        # Take the first k bits
        uhats[i] = unpacked[:k]

    return uhats

# Test configurations covering different lifting sets (ils 0-7)
# Each (k, n) pair is chosen to exercise a different lifting factor Z
# BG selection per 3GPP 38.212: BG2 if k<=292 OR (k<=3824 AND rate<=0.67) OR rate<=0.25
TEST_CONFIGS = [
    # BG1 configs - different lifting sets (rate > 0.67 for k > 292)
    ("bg1", 440, 550),    # Z from lifting set 0 (powers of 2), rate=0.8
    ("bg1", 660, 825),    # Z from lifting set 1 (multiples of 3), rate=0.8
    ("bg1", 800, 1000),   # Z from lifting set 0, rate=0.8
    # BG2 configs - different lifting sets (k<=292 OR rate<=0.67)
    ("bg2", 292, 584),    # Z from lifting set 1, rate=0.5
    ("bg2", 200, 400),    # Z from lifting set 0, rate=0.5
    ("bg2", 100, 200),    # Z from lifting set 0, rate=0.5
]

@pytest.mark.parametrize("bg,k,n", TEST_CONFIGS)
def test_decoder_identity(compiled_decoder, bg, k, n):
    """
    Test that the decoder works with high SNR (effectively clean channel).
    Tests multiple (k, n) configurations to exercise different lifting factors.
    """
    encoder = LDPC5GEncoder(k, n)

    # Verify we're testing the expected base graph
    expected_bg = "bg1" if bg == "bg1" else "bg2"
    assert encoder._bg == expected_bg, f"Expected {expected_bg} but got {encoder._bg}"

    # Random source
    bs = 10 # Batch size
    u = np.random.randint(0, 2, size=(bs, k)).astype(np.float32)

    # Encode
    c = encoder(u)

    # Modulate (BPSK for simplicity in LLR generation)
    # 0 -> +1, 1 -> -1
    x = 1.0 - 2.0 * c

    # No Noise (High SNR)
    y = x

    # Demodulate to LLRs
    # LLR = log(P(x=0)/P(x=1))
    # For BPSK with x \in {+1, -1} and y = x + n:
    # LLR = 4*y/N0.
    # We can just use y as LLRs scaled up.
    # The decoder expects int8 [-127, 127].
    # c=0 -> x=+1 -> LLR large positive -> 127
    # c=1 -> x=-1 -> LLR large negative -> -127

    llr_sim = y * 30.0 # Scale to be confidently inside int8 range but not overflow heavily before clipping if noisy

    # decode_cuda expects tensorflow tensor or compatible for .numpy() call
    llr_tf = tf.convert_to_tensor(llr_sim, dtype=tf.float32)

    # Decode
    num_iter = 50 # Increased iterations for convergence check

    print(f"Testing: BG={1 if encoder._bg == 'bg1' else 2}, Z={encoder._z}, k={k}, n={n}")

    u_hat = decode_cuda(compiled_decoder, encoder, llr_tf, num_iter)

    # Compare
    # u is float32 0/1, u_hat is uint8 0/1
    u_int = u.astype(np.uint8)

    # Check systematic bits (transmitted part)
    start_sys = 2 * encoder._z
    sys_errors = np.sum(u_int[:, start_sys:k] != u_hat[:, start_sys:k])

    print(f"Systematic Bit Errors: {sys_errors}/{bs * (k - start_sys)}")

    if sys_errors > 0:
        print(f"Found {sys_errors} systematic bit errors")

    # Check all bits
    total_errors = np.sum(u_int != u_hat)
    print(f"Total Bit Errors: {total_errors}/{bs * k}")

    # Assert exact decoding for clean channel (no noise)
    assert sys_errors == 0, f"Found {sys_errors} systematic bit errors in clean channel test for {bg} (k={k}, n={n}, Z={encoder._z}) - expected 0"
    assert total_errors == 0, f"Found {total_errors} total bit errors in clean channel test for {bg} (k={k}, n={n}, Z={encoder._z}) - expected 0"

def test_decoder_bler_perf(compiled_decoder):
    """
    Run a small MC simulation to check BLER is reasonable at a specific SNR point.
    Using parameters from the notebook.
    """
    # Notebook params
    k = 800
    n = 1000
    num_iter = 8

    enc = LDPC5GEncoder(k, n)

    # Simulation params
    num_bits_per_symbol = 2 # QPSK

    import tensorflow as tf
    dtype = tf.complex64

    constellation = sn.phy.mapping.Constellation("qam", num_bits_per_symbol, dtype=dtype)
    # Explicitly set dtype for Mapper and Demapper to avoid inference mismatch
    mapper = sn.phy.mapping.Mapper(constellation=constellation, dtype=dtype)
    demapper = sn.phy.mapping.Demapper("maxlog", constellation=constellation, dtype=dtype)
    awgn_channel = sn.phy.channel.AWGN(dtype=dtype)
    binary_source = sn.phy.mapping.BinarySource()

    # Choose an SNR where BLER should be low but measurable or 0
    # From notebook plot, at 4dB BLER is around 10^-2 for BG1 (approx)
    # Let's pick 6dB where it should be very low/zero
    ebno_db = 6.0
    coderate = k/n
    no = sn.phy.utils.ebnodb2no(ebno_db, num_bits_per_symbol=num_bits_per_symbol, coderate=coderate)

    bs = 100
    u = binary_source([bs, k])
    c = enc(u)
    x = mapper(c)
    y = awgn_channel(x, no)
    llr = demapper(y, no) # returns LLRs (logits)
    # Sionna Demapper: "Output: LLRs for each bit."
    # Note: Notebook says "llr = -demapper(y,no) # sionna defines LLRs the wrong way around"
    # This might be due to 0 vs 1 definition.
    # If Sionna LLR > 0 implies bit 0?
    # Usually LLR = log(P(b=1)/P(b=0)) or log(P(b=0)/P(b=1)).
    # Sionna: "positive LLR indicates that the bit is likely 1".
    # 5G LDPC usually expects LLR = log(P(0)/P(1)).
    # Let's check notebook: "llr = -demapper(y,no)"
    # So we invert it.

    llr = -llr

    u_hat = decode_cuda(compiled_decoder, enc, llr, num_iter)

    bler = sn.phy.utils.compute_bler(u, u_hat)

    assert bler <= .1, f"BLER {bler} at 6dB"
