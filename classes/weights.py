from .sparse import Sparse
from .base import Helper
import numpy as np


class Weights(object):
    """
    array of Weights
    the 1st dimension is the index of the layer
    the 2nd or 2nd and 3rd are for the index of the neuron
    """

    def __init__(self, dim, sparse=False, shared=False, kernel_size=(1, 1)):
        super(Weights, self).__init__()
        self.ensemble_index_dict = {}
        self.ensemble_number = 0
        self.shared = shared  # ?
        self.kernel_size = kernel_size
        self.dim = dim  # (x,y)
        self.sparse = sparse

        tmp_matrix = np.ndarray(dim)
        tmp_matrix.fill(0)
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
