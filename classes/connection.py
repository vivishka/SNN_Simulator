
import numpy as np
from base import SimulationObject
import sys
sys.dont_write_bytecode = True


class ConnectionPattern(object):
    """docstring for ConnectionPattern."""

    # TODO: add more functions for different pattern
    def __init__(self, type='all2all', **kwargs):
        super(ConnectionPattern, self).__init__()
        self.type = type
        self.param = kwargs

    def create(self, index, dest):
        """ from the index of the  axon, returns the connection pattern
            returns list of tuple (neuron, weight)
        """
        neuron_list = []
        weight_list = []
        for neuron in dest.get_neuron_list():
            neuron_list.append(neuron)
            weight_list.append(np.random.rand())
        return neuron_list, weight_list


# class Connection(object):
class Axon(SimulationObject):
    '''
    Represents the connection between a neuron and an ensemble of neurons

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


    '''
    objects = []

    def __init__(self, source, dest, weights):
        super(Axon, self).__init__("Axon {0}".format(id(self)))
        Axon.objects.append(self)

        self.source = source
        self.dest = dest
        self.spike_notifier = None

        if len(dest) != len(weights):
            raise ValueError(
                'destination neuron and weights should be the same dimension')

        # associciate the source, destination and weight
        source.add_output(self)
        for i, neuron in enumerate(dest):
            neuron.add_input(self, weights[i])

    def set_notifier(self, spike_notifier):
        """ Used to simulate axons only when they received a spike """
        print('connect', self.source, self.dest)
        self.spike_notifier = spike_notifier

    def create_spike(self):
        """ Register a spike emitted by the source neuron"""
        self.spike_notifier.register_spike(self)

    def propagate_spike(self):
        """ Called by the notifier, propagates spike to connected neurons """
        print("propagating spike")
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
    *args
        The list of arguments to configure the connection

    Attributes
    ----------
    source: Ensemble
         Stores the emitting ensemble
    dest : Ensemble
        Stores the receiving ensemble
    axon_list: [Axon]
        The list of axons
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
            d, w = self.pattern.create(0, self.dest)
            ax = Axon(neuron, d, w)
            self.axon_list.append(ax)

    def set_notifier(self, spike_notifier):
        """ Used to simulate axons only when they received a spike """
        for axon in self.axon_list:
            axon.set_notifier(spike_notifier)
