
from .base import Helper
# from .neuron import NeuronType
# from .ensemble import Ensemble
import numpy as np
import logging as log


class Learner(object):
    """"""

    def __init__(self, layer, eta_up=0.1, eta_down=0.1, tau_up=0.1, tau_down=0.1, min_weight=0, max_weight=1):
        self.layer = layer
        self.in_spikes = []
        self.out_spikes = []
        self.eta_up = eta_up
        self.eta_down = eta_down
        self.tau_up = tau_up
        self.tau_down = tau_down
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.buffer_in = []  # [time, source_n, dest_n  , weight, source_c, batch_index]
        self.buffer_out = []  # [time, source_n, batch_id]

        Helper.log('Learner', log.INFO, 'Learner initialized on ensemble {0}'.format(self.layer.id))

    def in_spike(self, source_n, dest_n, weight, source_c):
        self.buffer_in.append([Helper.time, source_n, dest_n, weight, source_c, Helper.input_index])

    def out_spike(self, source_n):
        self.buffer_out.append([Helper.time, source_n, Helper.input_index])

    def reset_input(self):  # call every input cycle
        self.in_spikes.append(self.buffer_in)
        self.out_spikes.append(self.buffer_out)
        self.buffer_in = []
        self.buffer_out = []
        Helper.log('Learner', log.DEBUG, 'Learner of ensemble {0} reset for next input'.format(self.layer.id))

    def process(self):  # call every batch
        Helper.log('Learner', log.DEBUG, 'Processing learning ensemble {0}'.format(self.layer.id))
        for experiment in range(Helper.input_index):  # for each experiment in the batch that ends
            Helper.log('Learner', log.DEBUG, 'Processing batch'.format(experiment))
            for out_s in self.out_spikes[experiment]:
                for in_s in self.in_spikes[experiment]:
                    dt = out_s[0] - in_s[0]
                    if dt >= 0:
                        dw = self.eta_up * self.max_weight / 2 * np.exp(- dt * in_s[3] / self.tau_up)
                    else:
                        dw = - self.eta_down * self.max_weight / 2 * np.exp(dt * in_s[3] / self.tau_down)
                    # in_s[4].update_weight(in_s[3] + dw, in_s[1], in_s[2])
                    in_s[4].weights[(in_s[1], in_s[2])] = in_s[3] + dw  # update weights in source connection
        self.out_spikes = []
        self.in_spikes = []
        Helper.log('Learner', log.INFO, 'Processing learning ensemble {0} complete'.format(self.layer.id))





