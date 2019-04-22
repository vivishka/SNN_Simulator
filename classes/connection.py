
import numpy as np
from .base import SimulationObject
from .ensemble import Bloc
from .neuron import Weights
import sys
sys.dont_write_bytecode = True


class ConnectionPattern(object):
    """
    Distribution of axons between ensembles

    Parameters
    ---------
    type: NeuronType
         The emitting neuron
    dest : [NeuronType]
        The list of connected neurons
    weights : [float]
        The list of weights associated to each destination neuron

    Attributes
    ----------
    source: NeuronType
        The emitting neuron
    dest : [NeuronType]
        The list of connected neurons
    spike_notifier: SpikeNotifier
        The object to notify when this axon has a spike to propagate
    index: int
        the index of the source neuron in its ensemble

    """

    # TODO: add more functions for different pattern
    def __init__(self, pattern='all2all', **kwargs):
        super(ConnectionPattern, self).__init__()
        self.pattern = pattern
        self.param = kwargs
        # self.functions = {
        #     all2all: self.all2all,
        #     convolution: self.convolution,
        #     one2one: self.one2one
        # }
        # self.function = self.functions[type]

    def create(self, dest):
        # self.function(index, dest)
        return self.all2all(dest)

    @staticmethod
    def all2all(dest):
        """ from the index of the  axon, returns the connection pattern
            returns list of tuple (neuron, weight)
        """
        neuron_list = []
        weight_list = []
        for neuron in dest.get_neuron_list():
            neuron_list.append(neuron)
            weight_list.append(np.random.uniform(-0.05, 0.5))
        return neuron_list, weight_list

    @staticmethod
    def convolution(index, dest, kernel):
        # TODO: test dimensions
        # TODO: padding in parameter
        neuron_list = []
        weight_list = []
        if isinstance(kernel, int):
            k_height = k_width = (kernel - 1) // 2
        else:
            k_height = (kernel[0] - 1) // 2
            k_width = (kernel[1] - 1) // 2
        e_height = dest.size[0]
        e_width = dest.size[1]
        valid_list = []
        # test the edges
        for line in range(- k_height, k_height + 1):
            y = index[0] + line
            if 0 < y < e_height:
                for col in range(- k_width, k_width + 1):
                    x = index[1] + col
                    if 0 < x < e_width:
                        valid_list.append((x, y))
        # nb = len(valid_list)
        for x, y in valid_list:
            neuron_list.append(dest[x][y])
            weight_list.append(np.random.uniform(-0.05, 0.5))

        return neuron_list, weight_list


class Axon(SimulationObject):
    """
    Represents the connection between a neuron and a list of neurons

    Parameters
    ---------
    source_n: NeuronType
         The emitting neuron
    dest_n_list : [NeuronType]
        The list of connected neurons
    index_list : [int] or [(int, int)]
        The list of index of the source neuron in its ensemble

    Attributes
    ----------
    source_n: NeuronType
        The emitting neuron
    dest_n_list : [NeuronType]
        The list of connected neurons
    spike_notifier: SpikeNotifier
        The object to notify when this axon has a spike to propagate
    index_list: [int] or [(int, int)]
        The list of index of the source neuron in its ensemble

    """

    def __init__(self, source_n, dest_n_list, index_list):
        super(Axon, self).__init__("Axon {0}".format(id(self)))

        self.source_n = source_n
        # list of tuple (neuron, index)
        self.dest_n_index_list = list(zip(dest_n_list, index_list))
        self.spike_notifier = None
        self.ensemble_index = dest_n_list[0].weights.ensemble_index_dict[source_n.ensemble]
        # TODO: possibility to add delay / FIR filter

        # associate the source, destination and weight
        source_n.add_output(self)
        for i, neuron in enumerate(dest_n_list):
            neuron.add_input(self)

    def set_notifier(self, spike_notifier):
        """ Used to simulate axons only when they received a spike """
        self.spike_notifier = spike_notifier

    def create_spike(self):
        """ Register a spike emitted by the source neuron"""
        self.spike_notifier(self)

    def propagate_spike(self):
        """ Called by the notifier, propagates spike to connected neurons """
        for dest_n, synapse_index in self.dest_n_index_list:
            dest_n.receive_spike((self.ensemble_index,) + synapse_index)


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
    pattern: ConnectionPattern
        the way neurons should be connected together, default is all to all
    """

    objects = []
    # TODO: give pattern / distribution of weights
    # - one to one, all to all, all to %
    # - fixed weight, uniformly random, other
    # TODO: padding and stride

    def __init__(self, source_o, dest_o, **kwargs):
        super(Connection, self).__init__("Connect_{0}".format(id(self)))
        Connection.objects.append(self)
        self.source_o = source_o
        self.dest_o = dest_o
        self.axon_list = []
        self.kernel = kwargs['kernel'] if 'kernel' in kwargs else (3, 3)
        self.shared = False

        # depending on the source type, a specific function is called
        if isinstance(source_o, Bloc):
            connect_function = self.connect_bloc_ensemble
        else:
            connect_function = self.connect_ensemble_ensemble

        # the destination object is turned into a list of ensembles
        if isinstance(dest_o, Bloc):
            dest_e_list = dest_o.ensemble_list
        else:
            dest_e_list = [dest_o]

        # connect the source object to the list of ensemble
        for dest_e in dest_e_list:
            connect_function(source_b=source_o, dest_e=dest_e)

    def connect_bloc_ensemble(self, source_b, dest_e):
        """
        Always used for convolution. If one by one needed: use connection on each sub ensembles
        source_b is Block, dest_e is Ensemble
        """
        block_weights = Weights(shared=True)
        for source_e in source_b.ensemble_list:
            # TODO: better init of the weight depending on dimension
            ens_weight = np.random.rand(*self.kernel) * 1.5 - 0.5
            block_weights.set_weights(source_e, ens_weight)

        # shares the weight matrix to all destination neuron
        i = 0
        for neuron in dest_e.neuron_list:
            i += 1
            neuron.set_weights(block_weights)

        # connect each ensemble of the block
        for source_e in source_b.ensemble_list:
            # self.connect_ensemble(ens, dest_e)
            self.connect_ensemble_ensemble(source_e, dest_e, True)

    def connect_ensemble_ensemble(self, source_e, dest_e, conv=False):
        """
        dim = 1:
            ens-ens only
        dim = 2:
            all 2 all: kernel = size
            one 2 one: kernel = 1
            conv: kernel = (n x m) & weights shared
        """
        if not conv:
            dest_n_list = []
            index_list = []
            # TODO: weights before
            # for each source neuron, creates an axon
            for index, source_n in enumerate(source_e.neuron_list):
                for dest_n in dest_e.neuron_list:
                    dest_n_list.append(dest_n)
                    index_list.append(index)
                self.axon_list.append(Axon(source_n, dest_n_list, index_list))
        else:
            size = source_e.size
            # for each neuron of the source ensemble
            for line in range(size[0]):
                for col in range(size[1]):
                    source_n = source_e[(line, col)]
                    self.connect_kernel(source_n, dest_e, (line, col))

    def connect_kernel(self, source_n, dest_e, index):
        """ connect a neuron from a bloc to another ensemble """
        k_height = (self.kernel[0] - 1) // 2
        k_width = (self.kernel[1] - 1) // 2
        # TODO: check for even size
        dest_n_list = []
        index_list = []
        for i in range(self.kernel[0]):
            y = index[0] + i - k_height
            for j in range(self.kernel[1]):
                x = index[1] + j - k_width

                # test if in range
                try:
                    dest_n = dest_e[(y, x)]
                    dest_n_list.append(dest_n)
                    index_list.append((j, i))
                except IndexError:
                    # zero padding
                    pass
        self.axon_list.append(Axon(source_n, dest_n_list, index_list))

    def set_notifier(self, spike_notifier):
        """ Used to simulate axons only when they received a spike """
        for axon in self.axon_list:
            axon.set_notifier(spike_notifier)

# TODO: test dimensions
# TODO: padding in parameter
