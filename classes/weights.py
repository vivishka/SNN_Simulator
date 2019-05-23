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

        # for row in range(n):
        #     for col in range(n):
        #         for i in kern:
        #             for j in kern:
        #                 if 0 <= row + i < n and 0 <= col + j < n:
        #                     A[row * n + col, (row + i) * n + (col + j)] = np.random.rand()

        tmp_matrix = np.zeros(np.do)
        for row in range(dim[0]):

            for kernel_row in range(kernel_size[0]):

                row_offset = int((kernel_row - kernel_size[0] // 2) * np.sqrt(dim[1]))

                for kernel_col in range(kernel_size[1]):

                    col_offset = (kernel_col - kernel_size[1] // 2)

                    index = row + row_offset + col_offset

                    if 0 <= index < dim[1]:
                        tmp_matrix[row, index] = Helper.init_weight()

        self.matrix = Sparse(tmp_matrix)

    def check_ensemble_index(self, source_e):
        # deptecated
        if source_e not in self.ensemble_index_dict:
            self.ensemble_index_dict[source_e] = self.ensemble_number
            self.ensemble_number += 1
            self.matrix.append(None)
        return self.ensemble_index_dict[source_e]

    def set_weights(self, source_e, weight_array):
        """ sets the weights of the axons from the specified ensemble """
        ens_number = self.check_ensemble_index(source_e)
        self.matrix[ens_number] = weight_array
        return ens_number

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
