
import numpy as np
from .base import SimulationObject
import sys
sys.dont_write_bytecode = True


class ConnectionPattern(object):
    '''
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

    '''

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
        self.all2all(index, dest)

    def all2all(self, index, dest):
        """ from the index of the  axon, returns the connection pattern
            returns list of tuple (neuron, weight)
        """
        neuron_list = []
        weight_list = []
        for neuron in dest.get_neuron_list():
            neuron_list.append(neuron)
            weight_list.append(np.random.uniform(-0.05, 0.5))
        return neuron_list, weight_list

    def convolution(self, index, dest, kernel=3):
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
            if y > 0 and y < e_height:
                for col in range(- k_width, k_width + 1):
                    x = index[1] + col
                    if x > 0 and x < e_width:
                        valid_list.append((x, y))
        # nb = len(valid_list)
        for x, y in valid_list:
            neuron_list.append(dest[x][y])
            weight_list.append(np.random.uniform(-0.05, 0.5))

        return neuron_list, weight_list


# class Connection(object):
class Axon(SimulationObject):
    '''
    Represents the connection between a neuron and a list of neurons

    Parameters
    ---------
    source: NeuronType
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

    '''

    def __init__(self, source, dest, weights):
        super(Axon, self).__init__("Axon {0}".format(id(self)))

        self.source = source
        self.dest = dest
        self.spike_notifier = None
        self.index = source.index
        # TODO: possibility to add delay / FIR filter here

        if len(dest) != len(weights):
            raise ValueError(
                'destination neuron and weights should be the same dimension')

        # associciate the source, destination and weight
        source.add_output(self)
        for i, neuron in enumerate(dest):
            neuron.add_input(self, weights[i])

    def set_notifier(self, spike_notifier):
        """ Used to simulate axons only when they received a spike """
        self.spike_notifier = spike_notifier

    def create_spike(self):
        """ Register a spike emitted by the source neuron"""
        self.spike_notifier.register_spike(self)

    def propagate_spike(self):
        """ Called by the notifier, propagates spike to connected neurons """
        for neuron in self.dest:
            neuron.receive_spike(self)


class Connection(SimulationObject):
    '''
    A connection is a list of axons connected between 2 ensembles

    Parameters
    ----------
    source: Ensemble
         The emitting ensemble
    dest : Ensemble
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
    '''

    objects = []
    # TODO: give pattern / distribution of weights
    # - one to one, all to all, all to %
    # - fixed weight, uniformly random, other

    def __init__(self, source, dest, pattern=None, *args, **kwargs):
        super(Connection, self).__init__("Connect_{0}".format(id(self)))
        Connection.objects.append(self)
        self.source = source
        self.dest = dest
        self.axon_list = []
        self.pattern = ConnectionPattern() if pattern is None else pattern
        source_list = source.get_neuron_list()
        # dest_list = dest.get_neuron_list()
        # TODO: specify dimension ?
        # TODO: convolutional
        # TODO: weight distribution
        #

        for neuron in source_list:
            # computes the list of neurons and the weight of each synapse
            d, w = self.pattern.create(0, self.dest)
            self.axon_list.append(Axon(neuron, d, w))

    def set_notifier(self, spike_notifier):
        """ Used to simulate axons only when they received a spike """
        for axon in self.axon_list:
            axon.set_notifier(spike_notifier)
