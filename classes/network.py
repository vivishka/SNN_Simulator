import logging as log
from .base import Helper
from .connection import Connection
from .layer import Ensemble, Block
from .encoder import Encoder
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
            Block: [],
            Connection: [],
            Encoder: [],
            Decoder: []
        }
        Helper.log('Network', log.INFO, 'new network created')

    def build(self):
        for object_class, stored_object_list in self.objects.items():
            new_object_list = object_class.get_objects()
            for new_object in new_object_list:
                if new_object not in self.objects[object_class]:
                    self.objects[object_class].append(new_object)
            object_class.flush()

        Helper.log('Network', log.INFO, 'network built')

    def set_sim(self, sim):
        for attr, value in self.objects.items():
            for obj in value:
                obj.sim = sim

    def restore(self):
        for attr, value in self.objects.items():
            for obj in value:
                obj.restore()
