import logging as log
from .base import Helper
from .connection import Connection
from .layer import Ensemble
from .encoder import Node, Reset
from .probe import Probe

import sys
sys.dont_write_bytecode = True


class Network(object):
    """docstring for model."""

    objects = []

    def __init__(self):
        super(Network, self).__init__()
        self.objects = {
            Ensemble: [],
            Node: [],
            Connection: [],

        }
        self.__ensembles = self.objects[Ensemble]
        self.__nodes = self.objects[Node]
        self.__connections = self.objects[Connection]
        Helper.log('Network', log.INFO, 'new network created')

    def build(self):
        for attr, value in self.objects.items():
            self.objects[attr] = attr.get_objects()
        Helper.log('Network', log.INFO, 'network built')

    def get_all_objects(self):
        return self.objects
