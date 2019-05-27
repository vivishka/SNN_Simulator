from .compactmatrix import CompactMatrix
from .base import Helper
import numpy as np


class Weights(object):
    """
    array of Weights
    the 1st dimension is the index of the layer
    the 2nd or 2nd and 3rd are for the index of the neuron
    """

    def __init__(self, source_dim, dest_dim, kernel_size=None, shared=False):
        super(Weights, self).__init__()
        self.ensemble_index_dict = {}
        self.ensemble_number = 0
        self.shared = shared  # ?
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        self.source_dim = source_dim  # (x,y)
        self.dest_dim = dest_dim  # (x,y)

        # TODO: if source_dim != dest dim
        #  padding
        #  stride
        #  ALL 2 ALL connection
        tmp_matrix = np.zeros((np.prod(source_dim), np.prod(dest_dim)))
        if kernel_size is None:
            tmp_matrix = np.random.rand(np.prod(source_dim), np.prod(dest_dim)) * 2. / np.sqrt(np.prod(dest_dim))

        else:
            # for every source neuron
            for source_row in range(source_dim[0]):
                for source_col in range(source_dim[1]):
                    # for every square in the kernel:
                    for kern_row in range(-self.kernel_size[0] // 2 + 1, self.kernel_size[0] // 2 + 1):
                        for kern_col in range(-self.kernel_size[1] // 2 + 1, self.kernel_size[1] // 2 + 1):
                            # test if the kernel is square is inside the matrix
                            if (0 <= source_row + kern_row < source_dim[0] and
                                    0 <= source_col + kern_col < source_dim[1]):
                                index_x = source_row * source_dim[0] + source_col
                                index_y = (source_row + kern_row) * source_dim[0] + (source_col + kern_col)
                                tmp_matrix[(index_x, index_y)] = Helper.init_weight() * 2. / np.prod(self.kernel_size)

        self.matrix = CompactMatrix(tmp_matrix)

    def get_target_weights(self, index):
        """
        read weights of connections to neurons receiving a spike from neuron index 'index'
        """
        return self.matrix[index]
        # need real implementation later depending on matrix format

    def __getitem__(self, index):
        return self.matrix[index]

    def __setitem__(self, index, value):
        self.matrix[index] = value
