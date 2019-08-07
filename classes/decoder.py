import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import logging as log
from .base import Helper
from .layer import Ensemble, Bloc
from .neuron import NeuronType


class NeuronLog(NeuronType):
    """"
    This neuron is only used to store all the received spikes and times for later uses.
    This type of neuron is used in decoders

    Parameters
    ----------
    **kwargs
        Same as NeuronType, not used

    Attributes
    ----------
    spike_times: [(int, int)]
        list of tuple (index, time) of every received spike

    """
    def __init__(self, *args, **kwargs):
        super(NeuronLog, self).__init__(*args, **kwargs)
        self.spike_times = []

    def receive_spike(self, index_1d, weight):
        """ logs the ensemble index and the time of the received spike in a tuple
        The weight is not important
        """
        if weight > 1e-6:
            super(NeuronLog, self).receive_spike(index_1d, weight)
            for spike in self.received:
                Helper.log('Decoder', log.DEBUG, ' neuron {} received spike {}'.format(self.index_2d, spike))
                self.spike_times.append((spike[0], self.ensemble.sim.curr_time))
            self.received = []

    def step(self):
        """" non stepping neuron"""
        pass

    def reset(self):
        super(NeuronLog, self).reset()
        # Helper.log('Decoder', log.DEBUG, ' spikes for input: {}'.format(self.spike_times))
        self.spike_times = []

    def restore(self):
        super(NeuronLog, self).restore()
        self.spike_times = []


class Decoder(Ensemble):
    # TODO: change decoder achi to be a module of layer, not
    """
    Ensemble used to store and decode the train of spikes from other neurons
    Uses NeuronLog type for logging
    Can use different decoding functions.

    Parameters
    ----------
    size: int or (int, int)
         Size of the ensemble

    Attributes
    ----------


    """

    objects = []

    def __init__(self, size, absolute_time = False):
        super(Decoder, self).__init__(
            size=size,
            neuron_type=NeuronLog(),
            learner=None
        )
        self.decoded_wta = []
        self.decoded_image = []
        self.absolute_time = absolute_time
        Decoder.objects.append(self)

    def get_first_spike(self):
        """
        Decoding algorithm based on WTA: only uses the time of first spike

        :return: an array the same size as this ensemble containing the time
        of the first spike received by each neuron
        """
        image = np.zeros(self.size)

        # for every neuron, extracts the time of the first spike
        first_spike_list = [n.spike_times[0][1] for n in self.neuron_list if n.spike_times]
        if first_spike_list:
            min_val = min(first_spike_list)
            if self.absolute_time:
                min_val = (min_val // self.sim.input_period) * self.sim.input_period
        else:
            min_val = None
            Helper.log("Decoder", log.DEBUG, "get first spike: No spike received")

        Helper.log("Decoder", log.DEBUG, "get first spike: First spike logged at time {}".format(min_val))
        for row in range(self.size[0]):
            for col in range(self.size[1]):
                if self.neuron_array[row, col].spike_times:
                    # extracts the time of the first arrived spike
                    value = self.neuron_array[row, col].spike_times[0][1]
                    image[row, col] = value - min_val
                else:
                    # image[row, col] = None
                    image[row, col] = self.sim.input_period + 1 * self.sim.dt
        return image

    def decoded_image(self):
        # DEPRECATED
        """
        Decoding algorithm to directly decode encoded input
        Mainly used for debug

        :return: an array the same size as this ensemble representing the decoded value
        """
        image = np.zeros(self.size, )

        # find the maximum and minimum values
        spike_list = [[t for (i, t) in n.spike_times] for n in self.neuron_list if n.spike_times]
        min_val = min([min(l) for l in spike_list])
        max_val = max([max(l) for l in spike_list])

        for row in range(self.size[0]):
            for col in range(self.size[1]):
                decoded_sum = 0
                for index, time in self.neuron_array[row, col].spike_times:
                    decoded_sum += (1 - (time - min_val) / (max_val - min_val)) * (index + 1)
                image[row, col] = decoded_sum

        max_val = image.max()
        min_val = image.min()
        image = (image - min_val)/(max_val - min_val) * 255
        return image.astype(np.uint8)

    def plot(self, index=-1, title=None):

        table = self.decoded_wta
        if table[0].shape[0] == 1:
            graph = np.ndarray((len(table), table[0].shape[1]))
            for row, value in enumerate(table):
                graph[row] = value
        else:
            graph = self.decoded_wta[index]
        t_min = np.min(graph)
        t_max = self.sim.input_period + 1 * self.sim.dt
        if t_min == t_max:
            t_min = 0
        norm = colors.Normalize(vmin=t_min, vmax=t_max)
        plt.figure()
        plt.imshow(graph, cmap='gray_r', norm=norm)
        plt.xlabel('column number')
        plt.ylabel('row number')
        if title is not None:
            plt.title(title)

    def step(self):
        pass

    def reset(self):
        self.decoded_wta.append(self.get_first_spike())
        super(Decoder, self).reset()

    def restore(self):
        super(Decoder, self).restore()
        self.decoded_wta = []


class DecoderClassifier(Decoder):

    def __init__(self, size):
        super(DecoderClassifier, self).__init__(size)

    def get_correlation_matrix(self):
        cor_mat = np.zeros((self.sim.dataset.n_cats, self.size[1]))
        # nb_exp = len(self.decoded_wta)
        for index, result in enumerate(self.decoded_wta):
            #  gets all the neurons index that spiked first
            dec_cats = [i for i, v in enumerate(result.tolist()[0]) if v == 0]
            for cat in dec_cats:
                cor_mat[self.sim.dataset.labels[index % len(self.sim.dataset.labels)], cat] += 1
                # /self.dataset.pop_cats[self.dataset.labels[index%len(self.dataset.labels)]]
        return cor_mat

    def get_accuracy(self):
        correct = 0
        for index, result in enumerate(self.decoded_wta):
            dec_cats = [i for i, v in enumerate(result.tolist()[0]) if v == 0]
            label = self.sim.dataset.labels[index % len(self.sim.dataset.labels)]
            if len(dec_cats) == 1 and dec_cats[0] == label:
                correct += 1
        return correct / len(self.decoded_wta)



class DigitSpykeTorch(Decoder):

    def __init__(self, size):
        super(DigitSpykeTorch, self).__init__(size)
        self.first_time = None
        self.voltage_list = []
        self.highest_voltage = 0.

    # from .connection import Connection
    def receive_spike(self, targets, source_c):
        if self.first_time is None or self.first_time == self.sim.curr_time:
            voltage = source_c.source_e.first_voltage
            self.voltage_list.append(voltage)
            if voltage > self.highest_voltage:
                self.first_time = self.sim.curr_time
                self.highest_voltage = voltage


class DecoderSpykeTorch(Bloc):

    def __init__(self, size, n_cat, mode='unit', k=None):
        super(DecoderSpykeTorch, self).__init__(depth=n_cat, size=size, neuron_type=NeuronType())
        for i in range(n_cat):
            ens = DigitSpykeTorch(size=size)
            ens.bloc = self
            self.ensemble_list[i] = ens
            self.mode = mode
            self.k = k
            # TODO:  overwrite reset to store previous
            # TODO: fix None bug

    def get_value(self):
        digit = None
        first_time = float('inf')
        highest_voltage = 0
        if self.mode == 'unit':
            for i, digit_ens in enumerate(self.ensemble_list):
                if digit_ens.first_time is not None and \
                        digit_ens.first_time <= first_time and \
                        digit_ens.highest_voltage > highest_voltage:
                    digit = i
                    highest_voltage = digit_ens.highest_voltage
                    first_time = digit_ens.first_time
        elif self.mode == 'mean':
            for i, digit_ens in enumerate(self.ensemble_list):
                if digit_ens.first_time is not None and \
                        digit_ens.first_time <= first_time and \
                        np.mean(digit_ens.voltage_list) > highest_voltage:
                    digit = i
                    highest_voltage = np.mean(digit_ens.voltage_list)
                    first_time = digit_ens.first_time
        elif self.mode == 'k_highest':
            digit_occurence = [0 for _ in range(10)]
            digit_voltage_list = []
            for i, digit_ens in enumerate(self.ensemble_list):
                if digit_ens.voltage_list:
                    for voltage in digit_ens.voltage_list:
                        digit_voltage_list.append((voltage, i))

            digit_voltage_list.sort(key=lambda t: t[0], reverse=True)

            for i in range(min(self.k, len(digit_voltage_list))):
                digit_occurence[digit_voltage_list[i][1]] += 1
            return digit_occurence

        return digit




