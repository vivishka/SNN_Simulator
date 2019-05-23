import numpy as np


class Sparse(object):
    """
    Sparse way of storing the weights

    :param mat: np.ndarray. Dense matrix containg the weights an a lot of zeros

    """
    def __init__(self, mat):
        self.shape = mat.shape

        self.mat = []

        nb_non_zero = 0  # nonzero() 	Return the indices of the elements that are non-zero. ndarray doc
        for i in range(self.shape[0]):
            row = []
            for j in range(self.shape[1]):
                if mat[i, j] != 0:
                    row.append((i, j, mat[i, j]))
                    nb_non_zero += 1
            self.mat.append(row)

        self.size = nb_non_zero

    def todense(self):
        mat = np.zeros(self.shape)
        for row in self.mat:
            for data in row:
                mat[data[0], data[1]] = data[2]
        return mat

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.mat[item]
        elif isinstance(item, tuple) and len(item) == 2:
            row = self.mat[item[0]]
            # todo: optimisation dichotomy
            for data in row:
                if data[1] == item[1]:
                    return data[2]
        return 0

    def __setitem__(self, key, value):
        if isinstance(key, int):
            self.mat = value
        elif isinstance(key, tuple) and len(key) == 2:
            row = self.mat[key[0]]
            # todo: optimisation dichotomy
            for j, data in enumerate(row):
                if data[1] == key[1]:
                    row[j] = (key[0], key[1], value)

    def get_kernel(self, index, length, kernel_size):
        """
        rebuild the kernel weights
        :param index: int or (int, int). Center neuron index
        :param length: Length of the layer (dimension 0)
        :param kernel_size: (int, int).
        :return: np.ndarray containing the extracted weights around the index
        """
        if isinstance(index, int):
            index_1d = index
            index_2d = Sparse.get_index_2d(index, length)
        elif isinstance(index, tuple) and len(index) == 2:
            index_1d = Sparse.get_index_1d(index, length)
            index_2d = index
        else:
            return
        kernel = np.zeros(kernel_size)
        for row in range(kernel_size[1]):
            for col in range(kernel_size[1]):
                x = index_2d[0] + row - kernel_size[0] // 2
                y = index_2d[1] + col - kernel_size[1] // 2
                try:
                    kernel[row, col] = self[(index_1d, Sparse.get_index_1d((x, y), length))]
                except IndexError:
                    pass
        return kernel

