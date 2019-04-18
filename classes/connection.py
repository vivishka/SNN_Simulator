
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
        The object to notifiy when this axon has a spike to propagate
    index: int
        the index of the source neuron in its ensemble

    """

    # TODO: add more functions for different pattern
    def __init__(self, type='all2all', **kwargs):
        super(ConnectionPattern, self).__init__()
        self.type = type
        self.param = kwargs
        # self.functions = {
        #     all2all: self.all2all,
        #     convolution: self.convolution,
        #     one2one: self.one2one
        # }
        # self.function = self.functions[type]

    def create(self, index, dest):
        # self.function(index, dest)
        return self.all2all(index, dest)

    @staticmethod
    def all2all(index, dest):
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
    def convolution(index, dest, kernel=3):
        # TODO: test dimensions
        # TODO: padding in parameter
        neuron_list = []
        weight_list = []
        if isinstance(kernel, int):
            k_height = k_width = (kernel - 1) / 2
        else:
            k_height = (kernel[0] - 1) / 2
            k_width = (kernel[1] - 1) / 2
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


# class Connection(object):
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
        The object to notifiy when this axon has a spike to propagate
    index_list: [int] or [(int, int)]
        The list of index of the source neuron in its ensemble

    """

    def __init__(self, source_n, dest_n_list, index_list):
        super(Axon, self).__init__("Axon {0}".format(id(self)))

        self.source_n = source_n
        # list of tuple (neuron, index)
        self.dest_n_index_list = list(zip(dest_n_list, index_list))
        self.spike_notifier = None
        # TODO: possibility to add delay / FIR filter

        # associciate the source, destination and weight
        source_n.add_output(self)
        # no need anymore
        # for i, neuron in enumerate(dest_n_list):
        #     neuron.add_input(self)

    def set_notifier(self, spike_notifier):
        """ Used to simulate axons only when they received a spike """
        self.spike_notifier = spike_notifier

    def create_spike(self):
        """ Register a spike emitted by the source neuron"""
        self.spike_notifier.register_spike(self)

    def propagate_spike(self):
        """ Called by the notifier, propagates spike to connected neurons """
        for dest_n, index in self.dest_n_index_list:
            dest_n.receive_spike(index)


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

    def __init__(self, source_o, dest_o, *args, **kwargs):
        super(Connection, self).__init__("Connect_{0}".format(id(self)))
        Connection.objects.append(self)
        self.source_o = source_o
        self.dest_o = dest_o
        self.axon_list = []
        self.kernel = kwargs['kernel'] if 'kernel' in kwargs else (3, 3)
        # self.pattern = ConnectionPattern() if pattern is None else pattern

        if issubclass(source_o, Bloc):
            if issubclass(dest_o, Bloc):
                dest_e_list = dest_o.ensemble_list
            else:
                dest_e_list = [dest_o]
            for dest_e in dest_e_list:
                self.connect_bloc(source_b=source_o, dest_e=dest_e)
        else:
            pass
            # TODO: connect ensemble to bloc

    def connect_bloc(self, source_b, dest_e):
        """ for now: it is always for convolution
        source is block, dest is ensemble
        """
        print(
            "connecting bloc {} with ensemble {}"
            .format(source_b.index, dest_e.index))
        # creation of the shared weight matrix for each ensemble of the source
        block_weights = Weights(shared=True)
        for source_e in source_b.ensemble_list:
            # TODO: better init of the weiht depending on dimension
            ens_weight = np.random.rand(*self.kernel)
            block_weights.set_weights(source_e, ens_weight)

        # shares the weight matrix to all destination neuron
        i = 0
        for neuron in dest_e.neuron_list:
            i += 1
            neuron.set_weights(block_weights)
        print(block_weights.weights)
        print("shared with {} neurons".format(i))

        # connect each ensenble of the block
        for source_e in source_b.ensemble_list:
            # self.connect_ensemble(ens, dest_e)
            self.connect_ensemble(source_e, dest_e, True)

    def connect_ensemble(self, source_e, dest_e, conv=False):
        print(
            " -connecting ensemble {} with ensemble {}"
            .format(source_e.index, dest_e.index))
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
            # for each neuron of the source ensenble
            for line in range(size[0]):
                for col in range(size[1]):
                    source_n = source_e[(line, col)]
                    self.connect_kernel(source_n, dest_e, (line, col))

    def connect_kernel(self, source_n, dest_e, index):
        """ connect a neuron from a bloc to another ensemble """
        k_height = int((self.kernel[0] - 1) / 2)
        k_width = int((self.kernel[1] - 1) / 2)
        # TODO: check for even size
        dest_n_list = []
        index_list = []
        for i in range(self.kernel[0]):
            y = index[0] + i - k_height
            for j in range(self.kernel[1]):
                x = index[1] + j - k_width

                # test if in range
                try:
                    # print("adding ({},{}) as ({},{})".format(y, x, j, i))
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

# # TODO: test dimensions
# # TODO: padding in parameter
# neuron_list = []
# weight_list = []
# if isinstance(kernel, int):
#     k_height = k_width = (kernel - 1) / 2
# else:
#     k_height = (kernel[0] - 1) / 2
#     k_width = (kernel[1] - 1) / 2
# e_height = dest.size[0]
# e_width = dest.size[1]
# valid_list = []
# # test the edges
# for line in range(- k_height, k_height + 1):
#     y = index[0] + line
#     if y > 0 and y < e_height:
#         for col in range(- k_width, k_width + 1):
#             x = index[1] + col
#             if x > 0 and x < e_width:
#                 valid_list.append((x, y))
# # nb = len(valid_list)
# for x, y in valid_list:
#     neuron_list.append(dest[x][y])
#     weight_list.append(np.random.uniform(-0.05, 0.5))


