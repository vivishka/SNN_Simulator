class Weights(object):
    """
    array of Weights
    the 1st dimension is the index of the layer
    the 2nd or 2nd and 3rd are for the index of the neuron
    """

    def __init__(self, shared=False):
        super(Weights, self).__init__()
        self.matrix = []
        self.ensemble_index_dict = {}
        self.ensemble_number = 0
        self.shared = shared

    def check_ensemble_index(self, source_e):
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
        return self.matrix[index]
        # need real implementation later depending on matrix format

    def __getitem__(self, index):
        return self.matrix[index[0]][index[1:]]

    def __setitem__(self, index, value):
        self.matrix[index[0]][index[1:]] = value
