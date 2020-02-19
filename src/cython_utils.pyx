import cython
import numpy as np


@cython.boundscheck(False)
cpdef int find_best_index(int[:] values) nogil:
    cdef size_t index
    cdef size_t best_index = 0
    cdef int best_value = values[0]
    for index in range(values.shape[0]):
        if values[index] < best_value:
            best_value = values[index]
            best_index = index
    return best_index


@cython.boundscheck(False)
cpdef int[:] get_tile_order(int[:,:] mat, int[:] tile_order):
    cdef size_t index
    for index in range(mat.shape[0]):
        tile_order[index] = find_best_index(mat[index])
    return tile_order


@cython.boundscheck(False)
cpdef int euclid_2d(int[:, :] mat_1, int[:,:] mat_2) nogil:
    cdef size_t row, col
    cdef int total = 0
    cdef int dx
    for row in range(mat_1.shape[0]):
        for col in range(mat_1.shape[1]):
            dx = mat_1[row, col] - mat_2[row, col]
            total += dx * dx
    return total


@cython.boundscheck(False)
cpdef int euclid_3d(int[:, :, :] mat_1, int[:, :, :] mat_2) nogil:
    cdef size_t row, col
    cdef int total = 0
    cdef int dx, dy, dz
    for row in range(mat_1.shape[0]):
        for col in range(mat_1.shape[1]):
            dx = mat_1[row, col, 0] - mat_2[row, col,0]
            dy = mat_1[row, col, 1] - mat_2[row, col,1]
            dz = mat_1[row, col, 2] - mat_2[row, col,2]
            total += dx * dx
            total += dy * dy
            total += dz * dz
    return total


@cython.boundscheck(False)
cpdef int[:, :] batch_euclid_2d(int[:, :, :] img_tiles, int[:, :, :] sample_tiles, int[:, :] buffer):
    cdef size_t img_index, tile_index
    for img_index in range(img_tiles.shape[0]):
        for tile_index in range(sample_tiles.shape[0]):
            buffer[img_index, tile_index] = euclid_2d(img_tiles[img_index], sample_tiles[tile_index])
    return buffer


@cython.boundscheck(False)
cpdef int[:, :] batch_euclid_3d(int[:, :, :, :] img_tiles, int[:, :, :, :] sample_tiles, int[:, :] buffer):
    cdef size_t img_index, tile_index
    for img_index in range(img_tiles.shape[0]):
        for tile_index in range(sample_tiles.shape[0]):
            buffer[img_index, tile_index] = euclid_3d(img_tiles[img_index], sample_tiles[tile_index])
    return buffer
