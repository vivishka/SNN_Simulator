
from .connection import Connection
from .ensemble import Ensemble
from .node import Node, Reset
from .probe import Probe

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
            Probe: [],
            Reset: []

        }
        self.__ensembles = self.objects[Ensemble]
        self.__nodes = self.objects[Node]
        self.__connections = self.objects[Connection]
        self.__probes = self.objects[Probe]
        self.__reset = self.objects[Reset]

    def build(self):
        for attr, value in self.objects.items():
            self.objects[attr] = attr.get_objects()

    def get_all_objects(self):
        return self.objects
