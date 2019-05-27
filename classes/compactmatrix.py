import numpy as np
from .base import Helper


class CompactMatrix(object):
    """
    Sparse way of storing the weights

    Parameters
    ----------
    mat: np.ndarray.
        Dense matrix containing the weights an a lot of zeros

    Attributes
    ----------
    matrix: list or np.ndarray
         Compact matrix
    shape : (int, int)
        shape of the matrix
    sparse: boolean
        Is the matrix sparse or dense
    """

    def __init__(self, mat):
        self.shape = mat.shape

        # when matrix is not sparse enough: classical sparse matrix better
        nb_non_zero = len(mat.nonzero()[0])
        if nb_non_zero / mat.size > 0.5:
            self.sparse = False
            self.matrix = np.ndarray(self.shape, dtype=tuple)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    self.matrix[i, j] = (i, j, mat[i, j])
        else:
            self.sparse = True
            self.matrix = []
            for i in range(self.shape[0]):
                row = []
                for j in range(self.shape[1]):
                    if mat[i, j] != 0:
                        row.append((i, j, mat[i, j]))
                self.matrix.append(row)

        self.size = nb_non_zero

    def to_dense(self):
        mat = np.zeros(self.shape)
        if self.sparse:
            for row in self.matrix:
                for data in row:
                    mat[data[0], data[1]] = data[2]
        else:
            for row in range(self.shape[0]):
                for col in range(self.shape[1]):
                    mat[row, col] = self.matrix[row, col][2]
        return mat

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.matrix[item]
        elif isinstance(item, tuple) and len(item) == 2:
            if self.sparse:
                row = self.matrix[item[0]]
                for data in row:
                    if data[1] == item[1]:
                        return data[2]
            else:
                return self.matrix[item][2]
        return 0

    def __setitem__(self, key, value):
        if isinstance(key, int):
            self.matrix = value
        elif isinstance(key, tuple) and len(key) == 2:
            if self.sparse:
                row = self.matrix[key[0]]
                for j, data in enumerate(row):
                    if data[1] == key[1]:
                        if value == 0:
                            del row[j]
                            self.size -= 1
                        else:
                            row[j] = (key[0], key[1], value)
                        return
                    elif data[1] > key[1]:
                        if data != 0:
                            row.insert(j, (key[0], key[1], value))
                            self.size += 1
                        return
            else:
                self.matrix[key] = value

    def get_kernel(self, index, length, kernel_size):
        """
        rebuild the kernel weights
        :param index: int or (int, int). Center neuron index
        :param length: Length of the layer (dimension 0)
        :param kernel_size: (int, int).
        :return: np.ndarray containing the extracted weights around the index
        """
        kernel = np.zeros(kernel_size)
        if isinstance(index, int):
            source_index_1d = index
            source_index_2d = Helper.get_index_2d(index, length)
        elif isinstance(index, tuple) and len(index) == 2:
            source_index_1d = Helper.get_index_1d(index, length)
            source_index_2d = index
        else:
            return kernel
        if not (0 <= source_index_1d < self.shape[0]):
            return kernel
        for row in range(kernel_size[0]):
            for col in range(kernel_size[1]):
                x = source_index_2d[0] + row - (kernel_size[0] - 1) // 2
                y = source_index_2d[1] + col - (kernel_size[1] - 1) // 2
                dest_index_1d = Helper.get_index_1d((x, y), length)
                if 0 <= dest_index_1d < self.shape[1]:
                    kernel[row, col] = self[source_index_1d, dest_index_1d]
        return kernel