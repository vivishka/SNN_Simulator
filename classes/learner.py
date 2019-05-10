
from .base import Helper
# from .neuron import NeuronType
# from .ensemble import Ensemble
import numpy as np


class Learner(object):
    """"""

    def __init__(self, layer, eta_up=0.1, eta_down=0.1, tau_up=0.1, tau_down=0.1, min_weight=0, max_weight=1):
        self.layer = layer
        layer.learner = self
        self.shared = layer.ensemble_list.shared
        self.in_spikes = []
        self.out_spikes = []
        self.eta_up = eta_up
        self.eta_down = eta_down
        self.tau_up = tau_up
        self.tau_down = tau_down
        self.min_weight = min_weight
        self.max_weight = max_weight

    def in_spike(self, e_index, n_index, other_index):
        self.in_spikes.append((e_index, n_index, other_index, Helper.time))

    def out_spike(self, e_index, n_index):
        self.out_spikes.append((e_index, n_index, Helper.time))

    def learn(self):
        dw = np.zeros([len(self.layer.ensemble_list), len(self.layer.ensemble_list[0].neuron_list), ])
        for out_s in self.out_spikes:
            for in_s in self.in_spikes:
                if(in_s[0], in_s[1]) == (out_s[0], out_s[1]):
                    if out_s[3] - in_s[4] > 0: # increase synapse
                        pass
                    else: # decrease synapse
                        pass
        self.layer.set_weights(dw)



