
import numpy as np
from .base import SimulationObject
from .layer import Bloc
from .weights import Weights
import sys
sys.dont_write_bytecode = True


class Connection(SimulationObject):
    """
    A connection is a list of axons connected between 2 ensembles

    Parameters
    ----------
    source: Ensemble or Bloc
         The emitting ensemble
    dest : Ensemble or Bloc
        The receiving ensemble
    pattern: ConnectionPattern
        the way neurons should be connected together
    *args, **kwargs
        The list of arguments to configure the connection

    Attributes
    ----------
    source: Ensemble
         Stores the emitting ensemble
    dest : Ensemble
        Stores the receiving ensemble
    axon_list: [Axon]
        The list of axons
    """

    objects = []

    def __init__(self, source_l, dest_l, kernel=[1,1], *args, **kwargs):
        super(Connection, self).__init__("Connect_{0}".format(id(self)))
        Connection.objects.append(self)
        self.stride = kwargs['stride'] if 'stride' in kwargs else 1
        self.padding = kwargs['padding'] if 'padding' in kwargs else 0  # TODO: perhaps ?
        self.shared = True if 'shared' in args else False
        self.all2all = True if 'all2all' in args else False
        self.weights = kwargs['weights'] if 'weights' in kwargs else None
        self.sparse = True if 'sparse' in args else False
        self.active = False
        self.in_neurons_spiking = []
        self.in_ensemble = None
        self.out_ensemble = None

        # the destination object is turned into a list of ensembles
        self.dest_l_list = dest_l.ensemble_list

        dest_e_dim = len(dest_l.ensemble_list)
        dest_n_dim = self.dest_l_list[0].neuron_array.shape

        # Default behaviour when connecting to a dense network: all to all
        # TODO: re organize default behaviour
        # if kernel is None:
        #     if dest_n_dim[0] == 1:
        #         self.all2all = True
        #     else:
        #         self.kernel = (1, 1)
        # else:
        #     self.kernel = (kernel, kernel) if isinstance(kernel, int) else kernel
        #     self.shared = True
        # check if connection is from ensemble to ensemble, generate sub-connections if needed recursively
        if len(source_l.ensemble_list) + len(dest_l.ensemble_list) > 2:
            for l_in in source_l.ensemble_list:
                for l_out in dest_l.ensemble_list:
                    Connection(l_in, l_out, kernel, *args, **kwargs)

        else:
            # TODO incomplete
            source_l.out_connections.append(self)
            dest_l.in_connections.append(self)
            self.out_ensemble = dest_l
            self.in_ensemble = source_l
            self.weights = Weights((len(self.out_ensemble.neuron_list),len(self.in_ensemble.neuron_list)), sparse=True, kernel_size=kernel)
            self.active = True

    def connect_layers(self, source_e, dest_e):
        """
        Create a connection between neurons of the source and destination
        Initializes the weights of the destination neurons
        one2one: kernel = 1; conv: kernel = (n x m) & weights shared

        Parameters
        ----------
        source_e : Ensemble
            Source Ensemble
        dest_e : Ensemble
            destination Ensemble
        """

        # TODO: better init of the weight depending on dimension
        # weights setting
        # TODO: load and save from variable / file
        # dictionary: (source obj ID, dest obj ID) : np array
        if self.shared:
            weights = self.weights[source_e.index, dest_e.index]
            # changing the weight matrix of one neuron will change the matrix
            # for all the neurons of the ensemble as they share the same object
            dest_e.neuron_list[0].weights.set_weights(source_e, weights)
        else:
            for i, dest_n in enumerate(dest_e.neuron_list):
                weights = self.weights[source_e.index, dest_e.index, i]
                dest_n.weights.set_weights(source_e, weights)

        # # creation of axons for each neuron of the source ensemble
        # for row in range(source_e.size[0]):
        #     for col in range(source_e.size[1]):
        #         axon = Axon(source_e=source_e, source_n=source_e[(row, col)], dest_e=dest_e)
        #         self.axon_list.append(axon)

        # connection of each neuron of the destination to the source axons
        for row in range(dest_e.size[0]):
            for col in range(dest_e.size[1]):
                self.connect_kernel(source_e=source_e, dest_e=dest_e, dest_n_index=(row, col))

    def connect_kernel(self, source_e, dest_e, dest_n_index):
        """
        Create a connection between neurons of the source and one neuron of the destination
        all2all: flag set, one2one: kernel = 1; conv: kernel = (n x m) & weights shared

        Parameters
        ----------
        source_e : Ensemble
            Source Ensemble
        dest_e : Ensemble
            destination Ensemble
        dest_n_index: (int, int)
            destination neuron index
        """

        if self.all2all:
            for row in range(source_e.size[0]):
                for col in range(source_e.size[1]):
                    source_axon = source_e.neuron_array[row, col].outputs[-1]
                    source_axon.add_synapse(dest_n=dest_e[dest_n_index], index_n=(row, col))
        else:
            # kernel size 1: 0 to 0, 2: 0 to 1, 3: -1 to 1, ...
            for row in range(-((self.kernel[0] - 1) // 2), self.kernel[0] // 2 + 1):
                source_row = dest_n_index[0] * self.stride + row
                for col in range(-((self.kernel[1] - 1) // 2), self.kernel[1] // 2 + 1):
                    source_col = dest_n_index[1] * self.stride + col

                    # lazy range test but eh #1
                    # the stride may cause issues when the destination ensemble size does not match
                    #  perhaps add warning
                    try:
                        # add the synapse to the source axon (-1: last added axon)
                        source_axon = source_e.neuron_array[source_row, source_col].outputs[-1]
                        source_axon.add_synapse(dest_n=dest_e[dest_n_index], index_n=(row, col))
                    except IndexError:
                        # zero padding
                        pass



    def register_neuron(self, index):
        self.in_neurons_spiking.append(index[0]*len(self.out_ensemble.ensemble_list)+index[1])

    def step(self):
        for index in self.in_neurons_spiking:
            targets = self.weights.get_target_weights(index)
            # for target in targets:
            self.out_ensemble.input_spike_buffer += targets
            # TODO here
