
from .base import Helper
# from .neuron import NeuronType
# from .ensemble import Ensemble
# import numpy as np


class Learner(object):
    """"""

    def __init__(self, layer):
        self.layer = layer
        layer.learner = self
        self.shared = layer.weights.shared
        self.in_spikes = []
        self.out_spikes = []

    def in_spike(self, index):
        self.in_spikes.append((index, Helper.time))

    def out_spike(self):
        self.out_spikes.append(Helper.time)
        # TODO: magic here
