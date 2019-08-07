import numpy as np
from .base import Helper


class CompactMatrix(object):
    """
    Sparse way of storing the weights

    Parameters
    ----------
    :param mat: Dense matrix containing the weights and a lot of zeros
    :type mat: object np.ndarray

    Attributes
    ----------
    :ivar matrix: Compact representation of the matrix, redundancy is used to speed up the spike propagation
    :type matrix: list of list of (int, int, float)
    :ivar shape: shape of the matrix: (dim source ensemble, dim dest ensemble)
    :type shape: (int, int)
    :ivar size: the number of non zero coefficient in the matrix
    :type size: int
    :ivar sparse: is the matrix sparse or dense
    :type sparse: boolean
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
        """
        transform the sparse matrix in a dense matrix
        :return: the dense matrix
        """
        mat = np.zeros(self.shape)
        for row in self.matrix:
            for data in row:
                mat[data[0], data[1]] = data[2]
        return mat

    def __getitem__(self, item):
        """
        Getter for the matrix
        the whole line return is used when a neuron spike and all the connected neurons needs to be notified
        :param item: the line or position of the weight
        :type item: int or (int, int)
        :return: the whole line of (source, dest, weight) if index is int
            a single (source, dest, weight) if index is tuple
        """
        if isinstance(item, int):
            return self.matrix[item]
        elif isinstance(item, tuple) and len(item) == 2:
            row = self.matrix[item[0]]
            for data in row:
                # check the correct dest index
                if data[1] == item[1]:
                    return data[2]
        return 0

    def __setitem__(self, index, value):
        """
        Setter for the matrix
        If the weight does not exist at this index, it will create a new link and set its weight
        :param index: index
        :type index: int
        :param value: new value
        :type value: float
        :return:
        """
        if isinstance(index, tuple) and len(index) == 2:
            row = self.matrix[index[0]]
            for j, data in enumerate(row):
                # if dest index weight already exist
                if data[1] == index[1]:
                    row[j] = (index[0], index[1], value)
                    return
                # if non existent
                elif data[1] > index[1]:
                    # add the weight
                    row.insert(j, (index[0], index[1], value))
                    self.size += 1
                    return

    def get_kernel(self, index, length, kernel_size):
        """
        rebuild the kernel weights
        :param index: index of the neuron a the center of the kernel
        :type index: int or (int, int)
        :param length: Length of the layer (dimension 0)
        :type length: int
        :param kernel_size: dimension of kernel
        :type kernel_size: (int, int)
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
                x = source_index_2d[0] + row - (kernel_size[0] - 1) // 2
                y = source_index_2d[1] + col - (kernel_size[1] - 1) // 2
                dest_index_1d = Helper.get_index_1d((x, y), length)
                if 0 <= dest_index_1d < self.shape[1]:
                    kernel[row, col] = self[source_index_1d, dest_index_1d]
        return kernel

    def add(self, other, min_weight, max_weight):
        """
        Add a scalar to all the weights of the matrix, then clip them between a min and a max
        :param other: the number to add to the matrix
        :type other: float or int
        :param min_weight: minimum value of the weight
        :type min_weight: float
        :param max_weight: maximum value of the weight
        :type max_weight: float
        """
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
        """
        Extract all the weight of the matrix in a list
        :return: weight list
        :rtype: list of float
        """
        weight_list = []
        for row in self.matrix:
            for data in row:
                weight_list.append(data[2])
        return weight_list

    def saturate_weights(self, wmin, wmax, threshold):
        """
        Makes the weights of the matrix binary
        :param wmin: value of category 1ow
        :type wmin: float
        :param wmax: value of category high
        :type wmax: float
        :param threshold: threshold which separates the categories
        :type threshold: float
        """
        for row in self.matrix:
            for col, target in enumerate(row):
                row[col] = target[:-1] + (wmin if target[-1] < threshold else wmax,)


class SharedCompactMatrix(CompactMatrix):
    """
    Sparse way of storing the weights for a convolutional connection
    the matrix stores the index of the weight in the kernel instead of the weight itself

    Parameters
    ----------
    :param mat: Dense matrix containing the index of the weights and a lot of zeros
    :type mat: object np.ndarray
    :param kernel: shared kernel for the connection
    :type kernel: object np.ndarray

    Attributes
    ----------
    """
    def __init__(self, mat, kernel):
        super(SharedCompactMatrix, self).__init__(mat)
        self.kernel = kernel

    def get_kernel(self, index=None, length=None, kernel_size=None):
        return self.kernel

    def __getitem__(self, item):
        """
        Getter for the matrix, use the index stored in the matrix with the wight in the kernel
        the whole line return is used when a neuron spike and all the connected neurons needs to be notified
        :param item: the line or position of the weight
        :type item: int or (int, int)
        :return: the whole line of (source, dest, kernel weight) if index is int
            a single (source, dest, kernel weight) if index is tuple
        """
        if isinstance(item, int):
            return [(d[0], d[1], self.kernel[d[2]]) for d in self.matrix[item]]
        elif isinstance(item, tuple) and len(item) == 2:
            row = self.matrix[item[0]]
            for data in row:
                if data[1] == item[1]:
                    return self.kernel[data[2]]
        return 0

    def __setitem__(self, index, value):
        """
        Setter for the kernel matrix
        :param index: index
        :type index: tuple
        :param value: new value
        :type value: float
        :return:
        """
        if isinstance(index, tuple) and len(index) == 2:
            row = self.matrix[index[0]]
            for j, data in enumerate(row):
                if data[1] == index[1]:
                    self.kernel[data[2]] = value
                    return

    def add(self, other, min_weight, max_weight):
        """
        Add a scalar to all the weights of the matrix, then clip them between a min and a max
        :param other: the number to add to the matrix
        :type other: float or int
        :param min_weight: minimum value of the weight
        :type min_weight: float
        :param max_weight: maximum value of the weight
        :type max_weight: float
        """
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
        """
        Extract all the weight of the matrix in a list
        :return: weight list
        :rtype: list of float
        """
        weight_list = []
        for row in self.kernel:
            for data in row:
                weight_list.append(data)
        return weight_list

    def saturate_weights(self, wmin, wmax, threshold):
        """
        Makes the weights of the matrix binary
        :param wmin: value of category 1ow
        :type wmin: float
        :param wmax: value of category high
        :type wmax: float
        :param threshold: threshold which separates the categories
        :type threshold: float
        """
        for row in self.kernel.shape[0]:
            for col in self.kernel.shape[1]:
                self.kernel[row, col] = wmin if self.kernel[row, col] < threshold else wmax


class DenseCompactMatrix(CompactMatrix):
    """
    Classical way of storing the weights for a dense connection

    Parameters
    ----------
    :param mat: Dense matrix containing the weights and no zeros
    :type mat: object np.ndarray

    Attributes
    ----------
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
        """
        transform the sparse matrix in a dense matrix
        :return: the dense matrix
        """
        mat = np.zeros(self.shape)
        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                mat[row, col] = self.matrix[row, col][2]
        return mat

    def __getitem__(self, item):
        """
        Getter for the matrix
        the whole line return is used when a neuron spike and all the connected neurons needs to be notified
        :param item: the line or position of the weight
        :type item: int or (int, int)
        :return: the whole line of (source, dest, weight) if index is int
            a single (source, dest, weight) if index is tuple
        """
        if isinstance(item, int):
            return self.matrix[item]
        elif isinstance(item, tuple) and len(item) == 2:
            return self.matrix[item][2]
        return 0

    def __setitem__(self, index, value):
        """
        Setter for the matrix
        :param index: index
        :type index: int
        :param value: new value
        :type value: float
        :return:
        """
        if isinstance(index, tuple) and len(index) == 2:
            self.matrix[index] = (index[0], index[1], value)

    def add(self, other, min_weight, max_weight):
        """
        Add a scalar to all the weights of the matrix, then clip them between a min and a max
        :param other: the number to add to the matrix
        :type other: float or int
        :param min_weight: minimum value of the weight
        :type min_weight: float
        :param max_weight: maximum value of the weight
        :type max_weight: float
        """
        # only works for scalar
        if isinstance(other, (float, int)):

            for row in range(self.shape[0]):
                for col in range(self.shape[1]):

                    data = self.matrix[row, col]
                    # clamping
                    new_weight = np.clip(data[2] + other, min_weight, max_weight)
                    self.matrix[row][col] = (data[0], data[1], new_weight)
        else:
            raise Exception("CompactMatrix add bad operand")
