import numpy as np
from sionna.fec.ldpc import LDPC5GEncoder

# Damping of 1 for debugging (=minsum from Sionna)
DAMPING_FACTOR = 1. # heuristic value 0.75 is typically a good choice
MAX_LLR_VALUE = 127 # depends on dtype

def get_bg(ils, bg, verbose=False):
    """Pregenerates basegraph description for the given lifting set index and basegraph number.

    This can be precomputed and stored before the decoding process.

    Parameters
    ----------
    ils : int | 0,...,7
        lifting set index of bg as defined in 38.212 Tab 5.3.2-1.
    bg : int | 1,2
        Basegraph number
    verbose : bool
        If True, additional information is printed

    Returns
    -------
    bg_vn : list of tuples
        Each tuple contains the variable node index, the check node index, the cyclic shift and the offset for the memory access of the message.
    bg_cn : list of tuples
        Each tuple contains the check node index, the variable node index, the cyclic shift and the offset for the memory access of the message.
    bg_vn_degree : list
        List of variable node degrees.
    bg_cn_degree : list
        List of check node degrees.
    """

    if bg==1:
        bg = "bg1"
    else:
        bg = "bg2"
    # Use sionna to load the basegraph
    enc = LDPC5GEncoder(12, 24) # dummy encoder
    mat_ref = enc._load_basegraph(ils, bg)

    #########################################################
    # Generate bg compact description
    #########################################################

    # From VN perspective
    bg_vn = []
    msg_offset = 0 # a counter how many messages blocks have been passed already
    for idx_vn in range(mat_ref.shape[1]):
        t = []
        for idx_cn in range(mat_ref.shape[0]):
            if mat_ref[idx_cn, idx_vn] != -1:
                t.append((idx_vn, idx_cn, int(mat_ref[idx_cn, idx_vn]),
                          msg_offset))
                msg_offset += 1
        bg_vn.append(t)

    if verbose:
        print(bg_vn)

    # From CN perspective
    bg_cn = []
    for idx_cn in range(mat_ref.shape[0]):
        t = []
        for idx_vn in range(mat_ref.shape[1]):
            if mat_ref[idx_cn, idx_vn] != -1:
                # find message offset from VN perspective
                # Find matching entry in bg_vn to get message offset
                for vn_entry in bg_vn[idx_vn]:
                    if vn_entry[1] == idx_cn:
                        msg_offset = vn_entry[3]
                        break
                t.append((idx_cn, idx_vn, int(mat_ref[idx_cn, idx_vn]),
                          msg_offset))
        bg_cn.append(t)
    if verbose:
        print(bg_cn)

    bg_vn_degree = [len(bg_vn[i]) for i in range(len(bg_vn))]
    bg_cn_degree = [len(bg_cn[i]) for i in range(len(bg_cn))]

    if verbose:
        print(bg_vn_degree)
        print(bg_cn_degree)

    return bg_vn, bg_cn, bg_vn_degree, bg_cn_degree

#########################################################
# CUDA relevant functions below
#########################################################

def init_basegraph(bg, z):
    """Initializes the basegraph, its dimensions and number of edges/messages.

    Parameters
    ----------
    bg : int | 1,2
        Basegraph number
    z : int | 2,...,384
        Lifting factor

    Returns
    -------
    bg_vn : list of tuples
        Each tuple contains the variable node index, the check node index, the cyclic shift and the offset for the memory access of the message.
    bg_cn : list of tuples
        Each tuple contains the check node index, the variable node index, the cyclic shift and the offset for the memory access of the message.
    bg_vn_degree : list
        List of variable node degrees.
    bg_cn_degree : list
        List of check node degrees.
    num_cols : int
        Number of variable nodes.
    num_rows : int
        Number of check nodes.
    num_edges : int
        Number of edges/messages in the graph.
    """

    # select base graph dimensions
    if bg == 1:
        num_rows = 46
        num_cols = 68
        num_nnz = 316 # num non-zero elements in bg
    else: # bg2
        num_rows = 42
        num_cols = 52
        num_nnz = 197 # num non-zero elements in bg

    # number of variable nodes
    num_vns = num_cols * z
    # number of check nodes
    num_cns = num_rows * z

    # number of edges/messages in the graph
    num_edges = num_nnz * z

    # lifting set according to 38.212 Tab 5.3.2-1
    s_val = [[2, 4, 8, 16, 32, 64, 128, 256],
             [3, 6, 12, 24, 48, 96, 192, 384],
             [5, 10, 20, 40, 80, 160, 320],
             [7, 14, 28, 56, 112, 224],
             [9, 18, 36, 72, 144, 288],
             [11, 22, 44, 88, 176, 352],
             [13, 26, 52, 104, 208],
             [15, 30, 60, 120, 240]]

    # find lifting set index
    ils = -1
    for i in range(len(s_val)):
        for j in range(len(s_val[i])):
            if z == s_val[i][j]:
                ils = i
                break
    # this case should not happen
    assert ils != -1, "Lifting factor not found in lifting set"

    # load base graph
    # This will become a lookup table in CUDA
    bg_vn, bg_cn, bg_vn_degree, bg_cn_degree = get_bg(ils, bg)

    return bg_vn, bg_cn, bg_vn_degree, bg_cn_degree, num_cols, num_rows, num_edges


def update_cn(llr_msg, llr_total, z, bg_cn, bg_cn_degree, num_rows, first_iter):
    """
    Inputs
    ------
    llr_msg: np.array [num_edges]
        Incoming LLRs from variable nodes
    llr_total: np.array [num_vns]
        Accumulated LLRs from variable nodes
    z: int | 2,...,384
        Lifting factor
    bg_cn: list of tuples
        Check node configuration
    bg_cn_degree: list
        Check node degree
    num_rows: int
        Number of rows in the base graph
    first_iter: bool
        Whether this is the first iteration
    Returns
    -------
    llr_msg: np.array [num_edges]
        Updated LLRs from check nodes
    """
    # Check node update function

    # this can be parallelized in CUDA
    for i in range(z): # between 2 and 384
        for idx_row in range(num_rows): # either 46 or 42

            cn_degree = bg_cn_degree[idx_row] # check node degree

            # list of tuples (idx_row, idx_col, s, msg_offset)
            cn = bg_cn[idx_row] # len(cn) = cn_degree

            # search the "extrinsic" min of all incoming LLRs
            # this means we need to find the min and the second min of all incoming LLRs
            min_1 = MAX_LLR_VALUE
            min_2 = MAX_LLR_VALUE
            idx_min = -1
            node_sign = 1

            # temp buffer for signs
            msg_sign = np.ones((19,)) # max CN degree is 19

            for ii in range(cn_degree):

                # calculate the index of the message in the LLR array
                msg_offset = cn[ii][3]
                s = cn[ii][2]
                idx_col = cn[ii][1]
                msg_idx = msg_offset * z + i

                # total VN message
                t = llr_total[idx_col*z + (i+s)%z]

                # make extrinsic by subtracting the previous msg
                if not first_iter: # ignore in first iteration
                    t -= llr_msg[msg_idx]

                # store sign for 2nd recursion
                sign = 1 if np.abs(t) == 0 else np.sign(t)

                # could be also used for syndrome-based check or early termination
                node_sign *= sign
                msg_sign[ii] = sign # for later sign calculation

                # find min and second min
                t_abs = np.abs(t)

                if t_abs < min_1:
                    min_2 = min_1
                    min_1 = t_abs
                    idx_min = msg_idx
                elif t_abs < min_2:
                    min_2 = t_abs

            # apply damping factor
            min_1 *= DAMPING_FACTOR
            min_2 *= DAMPING_FACTOR

            # clip min_val to MAX_LLR_VALUE
            min_1 = np.clip(min_1, -MAX_LLR_VALUE, MAX_LLR_VALUE)
            min_2 = np.clip(min_2, -MAX_LLR_VALUE, MAX_LLR_VALUE)

            # apply min and second min to the outgoing LLR
            for ii in range(cn_degree):
                msg_offset = cn[ii][3]

                msg_idx = msg_offset * z + i
                if msg_idx == idx_min:
                    min_val = min_2
                else:
                    min_val = min_1

                # and update outgoing msg including sign
                llr_msg[msg_idx] = min_val * node_sign * msg_sign[ii]

    return llr_msg

def update_vn(llr_msg, llr_ch, llr_total, z, bg_vn, bg_vn_degree, num_cols):
    """
    Inputs
    ------
    llr_msg: np.array [num_edges]
        Incoming LLRs from check nodes
    llr_ch: np.array [num_vns]
        Channel LLRs
    llr_total: np.array [num_vns]
        Accumulated LLRs from variable nodes
    z: int | 2,...,384
        Lifting factor
    bg_vn: list of tuples
        Variable node configuration
    bg_vn_degree: list
        Variable node degree
    num_cols: int
        Number of variable nodes

    Returns
    -------
    llr_total: np.array [num_vns]
        Updated LLRs from variable nodes

    """

    # this can be parallelized in CUDA
    for i in range(z): # between 2 and 384
        for idx_col in range(num_cols): # either 52 or 68

            vn_degree = bg_vn_degree[idx_col] # variable node degree

            # list of tuples (idx_col, idx_row, s, msg_offset)
            vn = bg_vn[idx_col] # len(vn) = vn_degree

            # accumulate all incoming LLRs
            msg_sum = 0 # should be int16
            for j in range(vn_degree):

                msg_offset = vn[j][3]
                s = vn[j][2]

                # index of the msg in the LLR array
                # it is the idx_col-th variable node, and the j-th message from the idx_row-th check node
                msg_idx = msg_offset * z + (i-s)%z
                # accumulate all incoming LLRs
                msg_sum += llr_msg[msg_idx].astype(np.int16)

            # add the channel LLRs
            msg_sum += llr_ch[idx_col*z + i].astype(np.int16)

            llr_total[idx_col*z + i] = msg_sum

    return llr_total


def pack_bits(llr_total, block_length, pack_bits=False):
    """
    Inputs
    ------
    llr_total: np.array [num_vns]
        LLRs from variable nodes
    block_length: int
        Number of payload bits that are returned after decoding
    pack_bits: bool
        If True, the bits are packed into a byte array
    Returns
    -------
    bits: np.array [block_length]
        Decoded bits
    """

    # OAI wants the bits in a byte array
    if pack_bits:
        # round length to the nearest multiple of 8
        block_length = int(np.ceil(block_length/8)*8)
        bits = (llr_total[:block_length]<0).astype(np.uint8)
        bits = np.packbits(bits)
    else:
        bits = (llr_total[:block_length]<0).astype(np.uint8)

    return bits

def decode_ldpc(bg, z, block_length, num_iter, llr_ch):
    """
    Inputs
    ------
    bg: int | 1,2
        Basegraph used for decoding

    z: int | 2,...,384
        Lifting factor

    block_length: int
        Number of payload bits that are returned after decoding

    num_iter: int
        Max number of decoding iterations

    llr_ch: np.array [68+z] or [52*z] for bg1 and bg2, respectively
        Received channel LLRs
    """


    ############################
    # Initialize Variables
    ############################

    bg_vn, bg_cn, bg_vn_degree, bg_cn_degree, num_cols, num_rows, num_edges = init_basegraph(bg, z)

    # temporary message buffer
    # max size is 316*384
    llr_msg = np.zeros((num_edges,), dtype=np.int8) # no need to initialize to 0

    # VN accumulator
    # # most likely we should always init the max size of the LLR array
    # max size is 68*384
    # The accumulator needs higher precision than the message buffer
    #llr_total = np.zeros((num_vns,), dtype=np.int16) # no need to initialize to 0
    llr_ch = llr_ch.astype(np.int8)

    ############################
    # Main Decoding Loop
    ############################
    llr_total = np.copy(llr_ch).astype(np.int16) # llr_total will be updated in the VN update

    for i in range(num_iter):

        # CN update
        # llr_msg not read, only written to in first iteration; will be filled with outputs of this function
        llr_msg = update_cn(llr_msg, llr_total, z, bg_cn, bg_cn_degree, num_rows, i==0)

        # VN update
        llr_total = update_vn(llr_msg, llr_ch, llr_total, z, bg_vn, bg_vn_degree, num_cols)

    # pack bits
    bits = pack_bits(llr_total, block_length, pack_bits=True)

    # apply syndrome check
    # if not successful; num_iter+=1

    return bits, llr_total

if __name__ == '__main__':

    import sys

    fd = sys.stdout
    if len(sys.argv) > 1:
        fd = open(sys.argv[1], 'w')
    bg = 1
    if len(sys.argv) > 2:
        bg = int(sys.argv[2])

    def array_to_blockstring(a):
        return '     ' + np.array2string(a,
            separator=', ', prefix='    ',
            max_line_width=80, threshold=2000000000)[1:-1].replace('[', '{').replace(']', '}')

    def range_check(a, min, max):
        assert np.all(a >= min)
        assert np.all(a <= max)
        return a

    print(f'#define bg{bg}_CN_DEGREE_TABLE(op) ' + ', '.join([ f'op(bg{bg}_cn_degree_{i})' for i in range(8) ]), file=fd)
    print(f'#define bg{bg}_VN_DEGREE_TABLE(op) ' + ', '.join([ f'op(bg{bg}_vn_degree_{i})' for i in range(8) ]), file=fd)
    print(f'#define bg{bg}_CN_TABLE(op) ' + ', '.join([ f'op(bg{bg}_cn_{i})' for i in range(8) ]), file=fd)
    print(f'#define bg{bg}_VN_TABLE(op) ' + ', '.join([ f'op(bg{bg}_vn_{i})' for i in range(8) ]), file=fd)

    for ils in range(8):
        bg_vn, bg_cn, bg_vn_degree, bg_cn_degree = get_bg(ils, bg)

        print('static uint32_t const bg%d_cn_degree_%d[] = {' % (bg, ils), file=fd)
        print(array_to_blockstring(np.array(bg_cn_degree, dtype=np.uint32)), file=fd)
        print('};', file=fd)

        print('static uint32_t const bg%d_vn_degree_%d[] = {' % (bg, ils), file=fd)
        print(array_to_blockstring(np.array(bg_vn_degree, dtype=np.uint32)), file=fd)
        print('};', file=fd)

        max_degree = np.max(np.array(bg_cn_degree, dtype=np.uint32))
        bg_cn_a = np.zeros((len(bg_cn), max_degree, 4), dtype=np.uint16)
        for i, ibg in enumerate(bg_cn):
            bg_cn_a[i,:len(ibg)] = range_check(np.array(ibg), min=0, max=2**16-1)

        print('static struct { uint16_t vn; uint16_t s; } const bg%d_cn_%d[%d][%d] = {' % (bg, ils, *bg_cn_a.shape[-2::-1]), file=fd)
        print(array_to_blockstring(bg_cn_a[:,:,1:3].transpose(1, 0, 2)), file=fd)
        print('};', file=fd)

        max_degree = np.max(np.array(bg_vn_degree, dtype=np.uint32))
        bg_vn_a = np.zeros((len(bg_vn), max_degree, 4), dtype=np.uint16)
        for i, ibg in enumerate(bg_vn):
            bg_vn_a[i,:len(ibg)] = range_check(np.array(ibg), 0, 2**16-1)

        print('static struct { uint16_t cn; uint16_t s; } const bg%d_vn_%d[%d][%d] = {' % (bg, ils, *bg_vn_a.shape[-2::-1]), file=fd)
        print(array_to_blockstring(bg_vn_a[:,:,1:3].transpose(1, 0, 2)), file=fd)
        print('};', file=fd)

    fd.close()
