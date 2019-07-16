from .compactmatrix import DenseCompactMatrix, CompactMatrix, SharedCompactMatrix
from .base import Helper, MeasureTiming
import copy
import numpy as np
import logging as log
import matplotlib.pyplot as plt


class Weights(object):
    """
    array of Weights
    the 1st dimension is the index of the layer
    the 2nd or 2nd and 3rd are for the index of the neuron
    """

    def __init__(
            self, source_dim, dest_dim,
            kernel_size=None, mode=None, real=False,
            wmin=0, wmax=0.6, mu=0.8, sigma=0.05,
            **kwargs):

        super(Weights, self).__init__()
        self.ensemble_index_dict = {}
        self.ensemble_number = 0
        self.mode = mode  # shared, pooling
        self.real = real
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, tuple) and len(kernel_size) == 2:
            self.kernel_size = kernel_size
        elif kernel_size is None:
            self.kernel_size = None
        else:
            Helper.log('Connection', log.ERROR, 'Wrong kernel size')
            raise Exception("wrong kernel size")
        self.source_dim = source_dim  # (x,y)
        self.dest_dim = dest_dim  # (x,y)
        self.wmin = wmin
        self.wmax = wmax
        self.mu = mu
        self.sigma = sigma

        # TODO: if source_dim != dest dim
        #  padding
        #  stride
        #  ALL 2 ALL connection
        self.matrix = None
        if kernel_size is None:
            self.init_weights_dense()
        else:
            if mode == 'shared':
                if 'first' in kwargs and not kwargs['first']:
                    # First: optimisation: copy the index matrix across all connections from the same blocks
                    self.init_weight_shared(model=kwargs['connection'].connection_list[0].weights.matrix)
                else:
                    self.init_weight_shared()
            elif mode == 'pooling':
                if 'first' in kwargs and not kwargs['first']:
                    # First: optimisation: copy the index matrix across all connections from the same blocks
                    self.init_weight_pooling(model=kwargs['connection'].connection_list[0].weights.matrix)
                else:
                    self.init_weight_pooling()
            else:
                self.init_weight_kernel()

    def generate_random_matrix(self, dim = None):
        if dim is None:
            mat = np.random.rand()
        elif isinstance(dim, int):
            mat = np.random.randn(dim)
        else:
            mat = np.random.randn(*dim)
        delta = self.wmax - self.wmin
        mat = mat * (self.sigma * delta) + (self.mu * delta)
        # prevents weights being stuck in saturation from the start
        mat = mat.clip(self.wmin + 0.01 * delta, self.wmax - 0.01 * delta)
        return mat

    def init_weights_dense(self):
        tmp_matrix = self.generate_random_matrix((np.prod(self.source_dim), np.prod(self.dest_dim)))
        self.matrix = DenseCompactMatrix(tmp_matrix)

    def init_weight_kernel(self):
        tmp_matrix = np.zeros((np.prod(self.source_dim), np.prod(self.dest_dim)))
        self.mu *= 2. / np.prod(self.kernel_size)
        # for every source neuron
        for source_row in range(self.source_dim[0]):
            for source_col in range(self.source_dim[1]):

                # for every square in the kernel:
                for kern_row in range(-self.kernel_size[0] // 2 + 1, self.kernel_size[0] // 2 + 1):
                    for kern_col in range(-self.kernel_size[1] // 2 + 1, self.kernel_size[1] // 2 + 1):

                        # test if the kernel is square is inside the matrix
                        if (0 <= source_row + kern_row < self.source_dim[0] and
                                0 <= source_col + kern_col < self.source_dim[1]):

                            # computes the source (row) and dest (col) indexes
                            index_x = source_row * self.source_dim[0] + source_col
                            index_y = (source_row + kern_row) * self.source_dim[0] + (source_col + kern_col)

                            # check if fuckuped
                            if 0 <= index_x < tmp_matrix.shape[0] and 0 <= index_y < tmp_matrix.shape[1]:
                                weight = self.generate_random_matrix()
                                tmp_matrix[(index_x, index_y)] = weight
                            else:
                                Helper.log('Connection',
                                           log.WARNING,
                                           'index ({}, {})out of range in weight matrix'.format(index_x, index_y))

        self.matrix = CompactMatrix(tmp_matrix)

    def init_weight_shared(self, model=None):
        kernel = self.generate_random_matrix(self.kernel_size)
        # kernel = np.random.randn(*self.kernel_size) * (self.wmax - self.wmin) / 10 + (self.wmax - self.wmin) * 0.8
        if self.real:
            for i in range(len(kernel[0])):
                for j in range(len(kernel[1])):
                    kernel[i, j] = int(kernel[i, j])

        if model is not None:
            # no deep copy: can share the index matrix
            self.matrix = copy.copy(model)
            self.matrix.kernel = kernel
            return

        tmp_matrix = np.zeros((np.prod(self.source_dim), np.prod(self.dest_dim)), dtype=object)
        # TODO: normalization

        # for every source neuron
        for source_row in range(self.source_dim[0]):
            for source_col in range(self.source_dim[1]):

                # for every square in the kernel:
                for kern_row in range(-self.kernel_size[0] // 2 + 1, self.kernel_size[0] // 2 + 1):
                    for kern_col in range(-self.kernel_size[1] // 2 + 1, self.kernel_size[1] // 2 + 1):

                        # test if the kernel is square is inside the matrix
                        if (0 <= source_row + kern_row < self.source_dim[0] and
                                0 <= source_col + kern_col < self.source_dim[1]):

                            # computes the source (row) and dest (col) indexes
                            index_x = source_row * self.source_dim[0] + source_col
                            index_y = (source_row + kern_row) * self.source_dim[0] + (source_col + kern_col)

                            # check if fuckuped
                            if 0 <= index_x < tmp_matrix.shape[0] and 0 <= index_y < tmp_matrix.shape[1]:
                                tmp_matrix[(index_x, index_y)] = \
                                    (kern_row + self.kernel_size[0] // 2, kern_col + self.kernel_size[1] // 2)
                            else:
                                Helper.log('Connection',
                                           log.WARNING,
                                           'index ({}, {})out of range in weight matrix'.format(index_x, index_y))
        self.matrix = SharedCompactMatrix(mat=tmp_matrix, kernel=kernel)

    def init_weight_pooling(self, model=None):
        if model is not None:
            # no deep copy: can share the index matrix
            self.matrix = copy.copy(model)
            return

        tmp_matrix = np.zeros((np.prod(self.source_dim), np.prod(self.dest_dim)))
        # for every source neuron
        for dest_row in range(self.dest_dim[0]):
            for dest_col in range(self.dest_dim[1]):
                for kern_row in range(self.kernel_size[0]):
                    for kern_col in range(self.kernel_size[1]):

                        # computes the source (row) and dest (col) indexes
                        index_y = dest_row * self.dest_dim[1] + dest_col
                        index_x = index_y * self.kernel_size[0] + kern_col + kern_row * self.source_dim[1] + dest_row * self.source_dim[1]
                        #     # handles incoherent sizes
                        #     if 0 <= index_x < tmp_matrix.shape[0] and 0 <= index_y < tmp_matrix.shape[1]:
                        tmp_matrix[(index_x, index_y)] = 1

        self.matrix = CompactMatrix(mat=tmp_matrix)

    # @MeasureTiming('get_weight')
    def get_target_weights(self, index):
        """
        read weights of connections to neurons receiving a spike from neuron index 'index'
        """
        return self.matrix[index]
        # need real implementation later depending on matrix format

    def restore(self):
        # if self.kernel_size is None:
        #     self.init_weights_dense()
        # else:
        #     if self.shared:
        #         self.init_weight_shared()
        #     else:
        #         self.init_weight_kernel()
        pass

    def plot(self):
        plt.figure()
        kernel = self.matrix.get_kernel()
        plt.imshow(np.array(kernel), cmap="gray")

    def __getitem__(self, index):
        return self.matrix[index]

    def __setitem__(self, index, value):
        self.matrix[index] = value
