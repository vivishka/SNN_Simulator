from .sparse import Sparse
from .base import Helper
import numpy as np


class Weights(object):
    """
    array of Weights
    the 1st dimension is the index of the layer
    the 2nd or 2nd and 3rd are for the index of the neuron
    """

    def __init__(self, source_dim, dest_dim, kernel_size, shared=False, sparse=False):
        super(Weights, self).__init__()
        self.ensemble_index_dict = {}
        self.ensemble_number = 0
        self.shared = shared  # ?
        self.kernel_size = kernel_size
        self.source_dim = source_dim  # (x,y)
        self.dest_dim = dest_dim  # (x,y)
        self.sparse = sparse

        tmp_matrix = np.zeros((np.prod(source_dim), np.prod(dest_dim)))
        for source_row in range(source_dim[0]):
            for source_col in range(source_dim[1]):
                for kern_row in range(-kernel_size[0] // 2 + 1, kernel_size[0] // 2 + 1):
                    for kern_col in range(-kernel_size[1] // 2 + 1, kernel_size[1] // 2 + 1):
                        if 0 <= source_row + kern_row < source_dim[0] and 0 <= source_col + kern_col < source_dim[1]:
                            index_x = source_row * source_dim[0] + source_col
                            index_y = (source_row + kern_row) * source_dim[0] + (source_col + kern_col)
                            print("img: ({}, {}), kern: ({}, {}), index: ({}, {})".format(
                                source_row, source_col,
                                kern_row, kern_col,
                                index_x, index_y
                            ))
                            tmp_matrix[(index_x, index_y)] = Helper.init_weight()

        self.matrix = Sparse(tmp_matrix)

    def get_target_weights(self, index):
        """
        read weights of connections to neurons receiving a spike from neuron index 'index'
        """
        if self.sparse:
            return self.matrix[index]
        else:
            return self.matrix[index]
        # need real implementation later depending on matrix format

    def __getitem__(self, index):
        return self.matrix[index[0]][index[1:]]

    def __setitem__(self, index, value):
        self.matrix[index[0]][index[1:]] = value
