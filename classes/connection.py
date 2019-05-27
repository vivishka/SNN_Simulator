import logging as log
import copy
from .base import SimulationObject, Helper
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
    con_count = 0

    def __init__(self, source_l, dest_l, kernel=None, *args, **kwargs):

        super(Connection, self).__init__("Connect_{0}".format(id(self)))
        Connection.objects.append(self)
        self.id = Connection.con_count
        Connection.con_count += 1
        self.connection_list = []
        self.stride = kwargs['stride'] if 'stride' in kwargs else 1
        self.shared = True if 'shared' in args else False
        self.all2all = True if 'all2all' in args else False
        self.weights = kwargs['weights'] if 'weights' in kwargs else None
        self.sparse = True if 'sparse' in args else False
        self.active = False
        self.in_neurons_spiking = []
        self.source_e = None
        self.dest_e = None
        self.is_probed = False
        self.probed_values = []

        Helper.log('Connection', log.INFO, 'new connection {0} created between layers {1} and {2}'
                   .format(self.id, source_l.id, dest_l.id))

        # check if connection is from ensemble to ensemble, generate sub-connections if needed recursively
        # TODO: idea: try to change __new()__ to return the list of sub connections
        if isinstance(source_l, Bloc) or isinstance(dest_l, Bloc):
            Helper.log('Connection', log.INFO, 'meta-connection detected, creating sub-connections')
            for l_in in source_l.ensemble_list:
                for l_out in dest_l.ensemble_list:
                    self.connection_list.append(Connection(l_in, l_out, kernel, *args, **kwargs))

        else:
            source_l.out_connections.append(self)
            dest_l.in_connections.append(self)
            self.dest_e = dest_l
            self.source_e = source_l
            self.weights = Weights(
                source_dim=self.source_e.size,
                dest_dim=self.dest_e.size,
                kernel_size=kernel,
                sparse=True)
            self.active = True
            self.connection_list = [self]

    def register_neuron(self, index):
        """ Registers the index of the source neurons that spiked"""
        index_1d = Helper.get_index_1d(index_2d=index, length=self.source_e.size[0])
        self.in_neurons_spiking.append(index_1d)
        Helper.log('Connection', log.DEBUG, ' neuron {}/{} registered for receiving spike'.format(index, index_1d))

    def step(self):
        for index in self.in_neurons_spiking:

            targets = self.weights.get_target_weights(index)
            # for target in targets:
            self.dest_e.receive_spike(targets)
            Helper.log('Connection', log.DEBUG, 'spike propagated from layer {0} to {1}'
                       .format(self.source_e.id, self.dest_e.id))
        self.in_neurons_spiking = []

    def get_weights_copy(self):
        return copy.deepcopy(self.weights.matrix)

    def probe(self):
        if self.is_probed:
            self.probed_values.append(self.get_weights_copy())

    def add_probe(self):
        self.is_probed = True
        self.probed_values = [self.get_weights_copy()]

    def __getitem__(self, item):
        return self.connection_list[item]