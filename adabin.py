

import numpy as np


def _bin_map(matrix, factor):
    mat = np.where(~np.isnan(matrix), matrix, 0.)
    height, width = mat.shape
    res_h, res_w = int(height / factor), int(width / factor)
    tail_h, tail_w = height % factor, width % factor
    in_mat = mat[:(height - tail_h), :(width - tail_w)]
    in_h, in_w = in_mat.shape
    out_mat = in_mat.reshape(res_h, factor, res_w, factor).mean(1).mean(2)
    res = np.kron(out_mat, np.ones((factor, factor)))
    if width % factor != 0:
        res = np.concatenate([res, np.tile(res[:, -1], (tail_w, 1)).T], axis=1)
    if height % factor != 0:
        res = np.concatenate([res, np.tile(res[-1], (tail_h, 1))], axis=0)
    return res


def adap_bin(signal, noise, min_SN=3):
    '''
    adap_bin(signal, noise, min_SN=3)

    Adaptive binning algorithm for 2D maps.
        
    Parameters
    ----------
    signal, noise: two-dimensional np.array
        The original signal and noise.

    min_SN: float
        The target S/N.
    
    Returns
    -------
    res_signal, res_noise: two-dimensinal np.array
        The binned signal and noise.

    maps: two-dimensinal np.array
        In which map number each pixel is binned (with integer data type).
        maps is one-to-one matched to signal.
        Call recon_maps(signal, noise, maps) to get res_signal, res_noise.
    '''
    shape = signal.shape
    
    res_signal = signal.copy()
    res_noise = noise.copy()
    maps = np.zeros(shape)
    sn = signal / noise

    for i in range(1, int(np.log(min(shape)) / np.log(2)) + 1):
        k = 2 ** i
        s = _bin_map(signal, k)
        n = np.sqrt(_bin_map(noise**2, k)) / k
        res_signal = np.where(sn > min_SN, res_signal, s)
        res_noise = np.where(sn > min_SN, res_noise, n)
        maps[res_signal == s] = i
        sn = s / n

    return res_signal, res_noise, maps


def recon_maps(signal, noise, maps):
    s_list, n_list = [signal], [noise]
    s, n = np.zeros((signal.shape)), np.zeros((noise.shape))
    for i in range(1, int(np.max(maps) + 1)):
        k = 2 ** i
        s_list.append(_bin_map(signal, k))
        n_list.append(np.sqrt(_bin_map(noise**2, k)) / k)
    for i in range(int(np.max(maps) + 1)):
        s = np.where(maps == i, s_list[i], s)
        n = np.where(maps == i, n_list[i], n)
    return s, n

#def hist_2d(data, n, min_count = 10):
#    hist, x_edges, y_edges = np.histogram2d(data, bins=2**n)
#    bins_list = []

    # loop over the matrix, tiling 2**n elements together, until we reach the size of the matrix
#    for i in range(0, n):
        # get the indices and sum of each 2**n "tile" of the matrix

        # find out which tiles satisfy the count requirement and add the indices of those tiles to the nth bins_list
        # the indices should not be flattened; we need to retrieve the tile edges later
        # we should only add the tile if at least one of its indices is not present in bins_list

        # update the previous n-1 bins_list entries so they don't contain tiles that contain the indices
        # that are contained in the n bins_list
        # (each index should only appear once in bins_list)

def get_block_sums_and_indices(matrix, block_size):
    """
    Divide the input matrix into non-overlapping blocks of shape (block_size, block_size),
    and return a list of:
      - The indices (i, j) of the original matrix elements in each block.
      - The sum of elements in each block.

    Parameters
    ----------
    matrix : 2D np.array
        The input matrix to be divided.
    block_size : int
        The size of each square block.

    Returns
    -------
    block_indices_list : list of lists of tuples
        Each inner list contains (i, j) indices for one block.
    block_sums_list : list of floats
        Each value is the sum of one block.
    """
    h, w = matrix.shape
    block_indices_list = []
    block_sums_list = []

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            if i + block_size <= h and j + block_size <= w:
                indices = [(i + di, j + dj) for di in range(block_size) for dj in range(block_size)]
                block_sum = sum(matrix[idx] for idx in indices)
                block_indices_list.append(indices)
                block_sums_list.append(block_sum)

    return block_indices_list, block_sums_list

def adaptive_binning(matrix, threshold):
    """
    Perform adaptive binning on a matrix. At each level, larger tiles (2^n x 2^n)
    are tested, and if their sum exceeds the threshold, they're added to the binning.
    Any overlapping smaller tiles are removed.

    Parameters
    ----------
    matrix : 2D np.array
        The matrix to bin.
    threshold : float
        The minimum sum required for a tile to be accepted.

    Returns
    -------
    bins_list : list of lists
        Each list contains the accepted tile indices at a binning level.
    """
    bins_list = []
    assigned_indices = set()
    max_power = int(np.log2(matrix.shape[0]))

    for n in range(max_power + 1):
        block_size = 2 ** n
        block_indices_list, block_sums_list = get_block_sums_and_indices(matrix, block_size)
        current_bins = []
        current_indices = set()

        for block_indices, block_sum in zip(block_indices_list, block_sums_list):
            if all(idx in assigned_indices for idx in block_indices):
                continue
            if block_sum >= threshold:
                current_bins.append(block_indices)
                current_indices.update(block_indices)
                assigned_indices.update(block_indices)

        # Remove overlapping tiles from previous levels
        for prev_level in range(n):
            bins_list[prev_level] = [
                tile for tile in bins_list[prev_level]
                if all(idx not in current_indices for idx in tile)
            ]

        bins_list.append(current_bins)

    return bins_list

import numpy as np

def get_block_info(matrix, row, col, block_size, used):
    """
    Extract indices and sum of a block from a matrix, excluding used indices.

    Parameters
    ----------
    matrix : np.ndarray
        Input 2D matrix.
    row, col : int
        Starting row and column for the block.
    block_size : int
        Size of the block.
    used : set of tuple
        Set of already used indices to exclude.

    Returns
    -------
    indices : list of tuple
        List of unbinned (row, col) indices in the block.
    val_sum : float
        Sum of the values at the selected indices.
    """
    shape = matrix.shape
    indices = []
    val_sum = 0
    for i in range(row, min(row + block_size, shape[0])):
        for j in range(col, min(col + block_size, shape[1])):
            idx = (i, j)
            if idx not in used:
                val_sum += matrix[i, j]
                indices.append(idx)
    return indices, val_sum

def adaptive_binning_resolution_preserving(matrix, threshold=1.5, block_sizes=[1, 2, 4]):
    """
    Adaptive binning on a 2D matrix, forming tiles of increasing size
    and binning them if their unbinned sum exceeds the threshold or
    if it's the final binning level.

    Parameters
    ----------
    matrix : np.ndarray
        Input 2D matrix.
    threshold : float
        Minimum sum required for a tile to be accepted.
    block_sizes : list of int
        Block sizes to try (in pixels).

    Returns
    -------
    bins_by_level : list of list of list of tuple
        List for each level containing accepted tile index lists.
    """
    shape = matrix.shape
    bins_by_level = []
    used_indices = set()

    for level, block_size in enumerate(block_sizes):
        new_bin = []
        for i in range(0, shape[0], block_size):
            for j in range(0, shape[1], block_size):
                indices, val_sum = get_block_info(matrix, i, j, block_size, used_indices)
                if not indices:
                    continue
                if val_sum >= threshold or level == len(block_sizes) - 1:
                    new_bin.append(indices)
                    used_indices.update(indices)
        bins_by_level.append(new_bin)

    return bins_by_level


