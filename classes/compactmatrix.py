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
    matrix: list[list[(int, int, float)]]
         Compact representation of the matrix
         redundancy is used to speed up the spike propagation
    shape : (int, int)
        shape of the matrix: (dim source ensemble, dim dest ensemble)
    size: int
        the number of non zero coefficient in the matrix
    sparse: boolean
        Is the matrix sparse or dense
    """

    def __init__(self, mat):
        self.shape = mat.shape
        self.size = len(mat.nonzero()[0])
        self.sparse = True
        self.matrix = []

        for i in range(self.shape[0]):
            row = []
            for j in range(self.shape[1]):
                if mat[i, j] != 0:
                    row.append((i, j, mat[i, j]))
            self.matrix.append(row)

    def to_dense(self):
        mat = np.zeros(self.shape)
        for row in self.matrix:
            for data in row:
                mat[data[0], data[1]] = data[2]
        return mat

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.matrix[item]
        elif isinstance(item, tuple) and len(item) == 2:
            row = self.matrix[item[0]]
            for data in row:
                # check the correct dest index
                if data[1] == item[1]:
                    return data[2]
        return 0

    def __setitem__(self, key, value):
        if isinstance(key, int):
            self.matrix = value
        elif isinstance(key, tuple) and len(key) == 2:
            row = self.matrix[key[0]]
            for j, data in enumerate(row):
                # if dest index weight already exist
                if data[1] == key[1]:
                    row[j] = (key[0], key[1], value)
                    return
                # if non existent
                elif data[1] > key[1]:
                    # add the weight
                    row.insert(j, (key[0], key[1], value))
                    self.size += 1
                    return

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
            return kernel, self.matrix[0]
        if not (0 <= source_index_1d < self.shape[0]):
            return kernel
        for row in range(kernel_size[0]):
            for col in range(kernel_size[1]):
                # TODO: check if kernel_size[1] for both ?
                x = source_index_2d[0] + row - (kernel_size[0] - 1) // 2
                y = source_index_2d[1] + col - (kernel_size[1] - 1) // 2
                dest_index_1d = Helper.get_index_1d((x, y), length)
                if 0 <= dest_index_1d < self.shape[1]:
                    kernel[row, col] = self[source_index_1d, dest_index_1d]
        return kernel

    def add(self, other, min_weight, max_weight):
        # only works for scalar
        if isinstance(other, (float, int)):

            for row, whole_row in enumerate(self.matrix):
                for col, data in enumerate(whole_row):

                    # clamping
                    new_weight = np.clip(data[2] + other, min_weight, max_weight)
                    self.matrix[row][col] = (data[0], data[1], new_weight)
        else:
            raise Exception("CompactMatrix add bad operand")

    def get_all_weights(self):
        mat = []
        for row in self.matrix:
            for data in row:
                mat.append(data[2])
        return mat


class SharedCompactMatrix(CompactMatrix):
    """
        Sparse way of storing the weights for a convolutional connection
        the matrix stores the index of the weight in the kernel instead of the weight itself

        Parameters
        ----------
        mat: np.ndarray.
            Dense matrix containing the weights an a lot of zeros

        Attributes
        ----------
        super.matrix: list[list[(int, int, (int, int))]]
             Compact representation of the matrix
             redundancy is used to speed up the spike propagation

        kernel: np.ndarray
            kernel shared by the whole connection
        """
    def __init__(self, mat, kernel):
        super(SharedCompactMatrix, self).__init__(mat)
        self.kernel = kernel

    def get_kernel(self, index=None, length=None, kernel_size=None):
        return self.kernel

    def __getitem__(self, item):
        if isinstance(item, int):
            return [(d[0], d[1], self.kernel[d[2]]) for d in self.matrix[item]]
        elif isinstance(item, tuple) and len(item) == 2:
            row = self.matrix[item[0]]
            for data in row:
                if data[1] == item[1]:
                    return self.kernel[data[2]]
        return 0

    def __setitem__(self, key, value):
        if isinstance(key, tuple) and len(key) == 2:
            row = self.matrix[key[0]]
            for j, data in enumerate(row):
                if data[1] == key[1]:
                    self.kernel[data[2]] = value
                    return

    def add(self, other, min_weight, max_weight):
        # only works for scalar
        if isinstance(other, (float, int)):

            for row in self.kernel.shape[0]:
                for col in self.kernel.shape[1]:
                    data = self.kernel[row, col]
                    # clamping
                    new_weight = np.clip(data[2] + other, min_weight, max_weight)
                    self.kernel[row][col] = (data[0], data[1], new_weight)
        else:
            raise Exception("CompactMatrix add bad operand")

    def get_all_weights(self):
        mat = []
        for row in self.kernel:
            for data in row:
                mat.append(data)
        return mat


class DenseCompactMatrix(CompactMatrix):
    """
        Sparse way of storing the weights for a convolutional connection
        the matrix stores the index of the weight in the kernel instead of the weight itself

        Parameters
        ----------
        mat: np.ndarray.
            Dense matrix containing the weights an a lot of zeros

        Attributes
        ----------
        super.matrix: np.ndarray[int, int, float]
             dense representation of the matrix
             redundancy is used to speed up the spike propagation

        """
    def __init__(self, mat):
        super(DenseCompactMatrix, self).__init__(mat)
        self.size = np.prod(self.shape)
        self.sparse = False

        # change matrix to dense
        self.matrix = np.ndarray(self.shape, dtype=tuple)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self.matrix[i, j] = (i, j, mat[i, j])

    def to_dense(self):
        mat = np.zeros(self.shape)
        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                mat[row, col] = self.matrix[row, col][2]
        return mat

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.matrix[item]
        elif isinstance(item, tuple) and len(item) == 2:
            return self.matrix[item][2]
        return 0

    def __setitem__(self, key, value):
        if isinstance(key, int):
            self.matrix[key] = value
        elif isinstance(key, tuple) and len(key) == 2:
            self.matrix[key] = (key[0], key[1], value)

    def add(self, other, min_weight, max_weight):
        # only works for scalar
        if isinstance(other, (float, int)):

            for row in self.shape[0]:
                for col in self.shape[1]:

                    data = self.matrix[row, col]
                    # clamping
                    new_weight = np.clip(data[2] + other, min_weight, max_weight)
                    self.matrix[row][col] = (data[0], data[1], new_weight)
        else:
            raise Exception("CompactMatrix add bad operand")
