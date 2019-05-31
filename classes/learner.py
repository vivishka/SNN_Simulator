
from .base import Helper
# from .neuron import NeuronType
# from .ensemble import Ensemble
import numpy as np
import logging as log


class Learner(object):
    """"""

    def __init__(self, eta_up=0.1, eta_down=0.1, tau_up=1, tau_down=1):
        self.layer = None
        self.in_spikes = []
        self.out_spikes = []
        self.eta_up = eta_up
        self.eta_down = eta_down
        self.tau_up = tau_up
        self.tau_down = tau_down
        self.buffer_in = []  # [time, source_n, dest_n  , weight, source_c, input_index]
        self.buffer_out = []  # [time, source_n, batch_id]
        self.active = True

        # Helper.log('Learner', log.INFO, 'Learner initialized on ensemble {0}'.format(self.layer.id))
        Helper.log('Learner', log.INFO, 'Learner initialized TODO: change log'.format())

    def in_spike(self, source_n, dest_n, weight, source_c):
        self.buffer_in.append([Helper.time, source_n, dest_n, weight, source_c, Helper.input_index])
        Helper.log('Learner', log.DEBUG, 'Learner of ensemble {0} registered input spike {1}'.format(self.layer.id, [Helper.time, source_n, dest_n, weight, source_c, Helper.input_index]))

    def out_spike(self, source_n):
        if self.active:
            self.buffer_out.append([Helper.time, source_n, Helper.input_index])
            Helper.log('Learner', log.DEBUG, 'Learner of ensemble {0} registered output spike from {1}'.format(self.layer.id, source_n))
            Helper.log('Learner', log.DEBUG, 'Appended {} to buffer'.format(self.buffer_out[-1]))

    def reset_input(self):  # call every input cycle
        Helper.log('Learner', log.DEBUG, 'Learner reset')
        Helper.log('Learner', log.DEBUG, 'Appended {} to memory'.format(self.buffer_out))
        self.in_spikes.append(self.buffer_in)
        self.out_spikes.append(self.buffer_out)
        if self.buffer_out:
            Helper.log('Learner', log.DEBUG, 'Appended {} to buffer'.format(self.buffer_out[-1]))
        else:
            Helper.log('Learner', log.DEBUG, 'Appended empty buffer')
        self.buffer_in = []
        self.buffer_out = []
        Helper.log('Learner', log.DEBUG, 'Learner of ensemble {0} reset for next input'.format(self.layer.id))

    def process(self):  # call every batch
        Helper.log('Learner', log.DEBUG, 'Processing learning ensemble {0}'.format(self.layer.id))
        # for each experiment in the batch that ends
        for experiment_index in range(Helper.input_index):
            Helper.log('Learner', log.DEBUG, 'Processing input cycle {}'.format(experiment_index))

            # for each spike emitted by the Ensemble during this experiment
            for out_s in self.out_spikes[experiment_index]:
                Helper.log('Learner', log.DEBUG, "Processing output spike of neuron {}".format(out_s[1]))

                for in_s in self.in_spikes[experiment_index]:

                    # if the emitted out_s came from the same neuron which received in_s
                    if out_s[1] == in_s[2]:
                        weight = in_s[3]
                        dt = out_s[0] - in_s[0]
                        if dt >= 0:
                            dw = self.eta_up * in_s[4].wmax / 2 * np.exp(- dt * in_s[3] / self.tau_up)
                        else:
                            dw = - self.eta_down * in_s[4].wmax / 2 * np.exp(dt * in_s[3] / self.tau_down)
                        # in_s[4].update_weight(in_s[3] + dw, in_s[1], in_s[2])
                        Helper.log('Learner', log.DEBUG, 'Connection {} Weight {} {} updated dw = {}'.format(in_s[4].id,
                                                                                                             in_s[1],
                                                                                                             in_s[2],
                                                                                                             dw))
                        new_w = in_s[3] + dw
                        if new_w > in_s[4].wmax:
                            new_w = in_s[4].wmax
                        elif new_w < in_s[4].wmin:
                            new_w = in_s[4].wmin
                        in_s[4].weights[(in_s[1], in_s[2])] = new_w  # update weights in source connection
        self.out_spikes = []
        self.in_spikes = []
        for connection in self.layer.in_connections:
            connection.probe()
        Helper.log('Learner', log.INFO, 'Processing learning ensemble {0} complete'.format(self.layer.id))

    def restore(self):
        self.in_spikes = []
        self.out_spikes = []
        self.buffer_in = []
        self.buffer_out = []


class LearnerClassifier(Learner):

    def __init__(self, eta_up=0.1, eta_down=0.1, tau_up=0.1, tau_down=0.1, feedback_gain=0.001):
        super(LearnerClassifier, self).__init__(eta_up, eta_down, tau_up, tau_down)
        self.feedback_gain = feedback_gain

    def process(self):
        # remove multiple output spikes from buffer TODO: optimize : high complexity

        for index, experiment in enumerate(self.out_spikes):
            if experiment:
                duplicates = []
                Helper.log('Learner', log.INFO, 'Classifier learner processing input experiment {}'.format(index))
                for out in experiment:
                    Helper.log('Learner', log.DEBUG, 'Classifier learner processing spike {}'.format(out))
                    for other in experiment:
                        Helper.log('Learner', log.DEBUG, 'Classifier learner comparing with spike {}'.format(other))
                        if out != other and out[0] == other[0]:
                            duplicates.append(out)
                            Helper.log('Learner', log.INFO,
                                       'Classifier learner ignored simultaneous output spikes, reducing weights')
                for duplicate in duplicates:
                    for con in self.layer.in_connections:
                        con.weights.matrix.add(-self.feedback_gain, con.wmin, con.wmax)
                    self.out_spikes[index].remove(duplicate)
            else:
                Helper.log('Learner', log.INFO,
                           'Classifier learner found no output spikes, increasing weights')
                for con in self.layer.in_connections:
                    con.weights.matrix.add(self.feedback_gain, con.wmin, con.wmax)
            if not experiment:
                # TODO: git gud
                Helper.log('Learner', log.DEBUG, 'No valid spike during input cycle {}: increasing all weights'
                           .format(index))
        super(LearnerClassifier, self).process()

