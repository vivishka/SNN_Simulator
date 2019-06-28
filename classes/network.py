import logging as log
from .base import Helper
from .connection import Connection
from .layer import Ensemble, Bloc
from .encoder import Node, Encoder
from .decoder import Decoder
import sys
sys.dont_write_bytecode = True


class Network(object):
    """docstring for model."""

    objects = []

    def __init__(self):
        super(Network, self).__init__()
        self.objects = {
            Ensemble: [],
            Bloc: [],
            Node: [],
            Connection: [],
            Encoder: [],
            Decoder: []
        }
        # self.__ensembles = self.objects[Ensemble]
        # self.__nodes = self.objects[Node]
        # self.__connections = self.objects[Connection]
        self.n_learners = 0
        Helper.log('Network', log.INFO, 'new network created')

    def build(self):

        for attr, value in self.objects.items():
            self.objects[attr] = list(set(self.objects[attr] + attr.get_objects()))
            attr.flush()
        for ens in self.objects[Ensemble]:
            if ens.learner:
                self.n_learners += 1

        Helper.log('Network', log.INFO, 'network built')

    def set_sim(self, sim):
        for attr, value in self.objects.items():
            for obj in value:
                obj.sim = sim

    def get_all_objects(self):
        return self.objects

    def restore(self):
        for attr, value in self.objects.items():
            for obj in value:
                obj.restore()
