from .compactmatrix import DenseCompactMatrix, CompactMatrix, SharedCompactMatrix
from .base import Helper
import copy
import numpy as np
import logging as log
import matplotlib.pyplot as plt


class Weights(object):
    """
    array of Weights

    Parameters
    ----------
    :param source_dim: dimension of the source index
    :type source_dim: (int, int)
    :param dest_dim: dimension of the destination index
    :type dest_dim: (int, int)
    :param kernel_size: if specified, the kernel linking the 2 layers; if not, layers will be fully connected
    :type kernel_size: int or (int, int) or None
    :param mode: dense (None), shared, split or pooling. how are the neurons connected between the ensembles
    :type mode: str
    :param integer_weight:
    :type integer_weight: bool
    :param wmin: minimum weight value allowed
    :type wmin: float
    :param wmax: maximum weight value allowed
    :type wmax: float
    :param mu: mean value for weight distribution
    :type mu: float
    :param sigma: standard deviation for weight distribution
    :type sigma: float
    :param **kwargs: dict of arguments to configure the connection

    Attributes
    ----------
    :ivar ensemble_index_dict:
    :type ensemble_index_dict: dict of Ensemble: int
    :ivar ensemble_number:
    :type ensemble_number:
    :ivar matrix:
    :type matrix: CompactMatrix or None

    """
    def __init__(
            self, source_dim, dest_dim,
            kernel_size=None, mode=None, integer_weight=False,
            wmin=0, wmax=0.6, mu=0.8, sigma=0.05,
            **kwargs):

        super(Weights, self).__init__()
        self.ensemble_index_dict = {}
        self.ensemble_number = 0
        self.mode = mode  # shared, pooling, split
        self.integer_weight = integer_weight
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
            elif mode == 'split':
                if 'first' in kwargs and not kwargs['first']:
                    # First: optimisation: copy the index matrix across all connections from the same blocks
                    self.init_weight_split(model=kwargs['connection'].connection_list[0].weights.matrix)
                else:
                    self.init_weight_split()
            else:
                self.init_weight_kernel()

    def generate_random_matrix(self, dim=None):
        """
        generates a random matrix of given dimension using a normal distribution of parameters mu and sigma
        :param dim: matrix dimension
        :type dim: tuple of int or object
        :return: random matrix
        :rtype: object np.ndarray
        """
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
        """
        initializer for creating a dense weight matrix (all to all connection) between 2 ensembles
        """
        tmp_matrix = self.generate_random_matrix((np.prod(self.source_dim), np.prod(self.dest_dim)))
        self.matrix = DenseCompactMatrix(tmp_matrix)

    def init_weight_kernel(self):
        """
        initializer for creating a weight matrix (N to N connection) between 2 ensembles
        """
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
        """
        initializer for creating a shared kernel weight matrix (convolutional connection) between 2 ensembles
        :param model: index structure matrix shared among all the weights. Optimisation to avoid identical computation
        :type model: object np.ndarray
        """
        kernel = self.generate_random_matrix(self.kernel_size)
        # kernel = np.random.randn(*self.kernel_size) * (self.wmax - self.wmin) / 10 + (self.wmax - self.wmin) * 0.8
        if self.integer_weight:
            for i in range(len(kernel[0])):
                for j in range(len(kernel[1])):
                    kernel[i, j] = int(kernel[i, j])

        if model is not None:
            # no deep copy: can share the index matrix
            self.matrix = copy.copy(model)
            self.matrix.kernel = kernel
            return

        tmp_matrix = np.zeros((np.prod(self.source_dim), np.prod(self.dest_dim)), dtype=object)

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
        """
        initializer for creating a pooling weight matrix (convolutional pooling connection) between 2 ensembles
        :param model: index structure matrix shared among all the weights. Optimisation to avoid identical computation
        :type model: object np.ndarray
        """
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
                        index_x = index_y * self.kernel_size[0] + kern_col \
                            + kern_row * self.source_dim[1] \
                            + dest_row * self.source_dim[1]
                        #     # handles incoherent sizes
                        #     if 0 <= index_x < tmp_matrix.shape[0] and 0 <= index_y < tmp_matrix.shape[1]:
                        tmp_matrix[(index_x, index_y)] = 1

        self.matrix = CompactMatrix(mat=tmp_matrix)

    def init_weight_split(self, model=None):
        """
        initializer for creating a dense weight matrix (all to all connection) between 2 ensembles
        :param model: index structure matrix shared among all the weights. Optimisation to avoid identical computation
        :type model: object np.ndarray
        """
        if model is not None:
            # no deep copy: can share the index matrix
            self.matrix = copy.copy(model)
            return

        tmp_matrix = np.zeros((np.prod(self.source_dim), np.prod(self.dest_dim)))
        offset = int(self.source_dim[1]//self.dest_dim[1])
        for dest in range(self.dest_dim[1]):
            tmp_matrix[dest * offset:(dest + 1) * offset, dest] = 1
        self.matrix = CompactMatrix(mat=tmp_matrix)

    def restore(self):
        pass

    def plot(self):
        """
        only for weights with kernel
        display a visual representation of the kernel
        """
        plt.figure()
        kernel = self.matrix.get_kernel(None, None, None)
        plt.imshow(np.array(kernel), cmap="gray")

    def __getitem__(self, index):
        return self.matrix[index]

    def __setitem__(self, index, value):
        self.matrix[index] = value
