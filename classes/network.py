
# from neuron import Neuron
from connection import Connection
from ensemble import Ensemble
from node import Node
from probe import Probe
# from base import SimationObject


import sys
sys.dont_write_bytecode = True

# TODO: with, synapse log


class Network(object):
    """docstring for model."""

    objects = []

    def __init__(self):
        super(Network, self).__init__()
        self.objects = {
            Ensemble: [],
            Node: [],
            Connection: [],
            # Network: [],
            Probe: [],
            # Neuron: []
        }
        self.__ensembles = self.objects[Ensemble]
        self.__nodes = self.objects[Node]
        self.__connections = self.objects[Connection]
        # self.__networks = self.objects[Network]
        self.__probes = self.objects[Probe]
        # self.__neurons = self.objects[Neuron]

    # def neuron(self, o):
    #     # check type, dim
    #     self.__neurons.append(o)
    #
    # def node(self, o):
    #     # check type, dim
    #     self.__nodes.append(o)
    #
    # def synapse(self, o):
    #     # check type, dim
    #     self.__synapses.append(o)

    def build(self):
        for attr, value in self.objects.items():
            self.objects[attr] = attr.get_objects()
        print(self.objects)

    def get_all_objects(self):
        return self.objects

    # @property
    # def neurons(self):
    #     return self.__neurons

    # @property
    # def connections(self):
    #     return self.__connections
