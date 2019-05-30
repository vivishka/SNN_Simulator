import numpy as np
import matplotlib.pyplot as plt
import logging as log
from .base import Helper
from .layer import Ensemble
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

    def receive_spike(self, index, weight):
        """ logs the ensemble index and the time of the received spike in a tuple
        The weight is not important
        """
        if weight != 0.:
            super(NeuronLog, self).receive_spike(index, weight)
            for spike in self.received:
                Helper.log('Decoder', log.DEBUG, ' neuron {} received spike {}'.format(self.index, spike))
                self.spike_times.append((spike[0], Helper.time))
            self.received = []

    def step(self):
        """" non stepping neuron"""
        pass

    def reset(self):
        super(NeuronLog, self).reset()
        # Helper.log('Decoder', log.DEBUG, ' spikes for input: {}'.format(self.spike_times))
        self.spike_times = []


class Decoder(Ensemble):
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

    def __init__(self, size):
        super(Decoder, self).__init__(
            size=size,
            neuron_type=NeuronLog(),
            learner=None
        )
        self.decoded_wta = []
        self.decoded_image = []

    def get_first_spike(self):
        """
        Decoding algorithm based on WTA: only uses the time of first spike

        :return: an array the same size as this ensemble containing the time
        of the first spike received by each neuron
        """
        image = np.zeros(self.size)

        # for every neuron, extracts the time of the first spike
        first_spike_list = [n.spike_times[0][1] for n in self.neuron_list if n.spike_times]
        if not first_spike_list:
            Helper.log("Decoder", log.DEBUG, "get first spike: No spike received")
            return image

        min_val = min(first_spike_list)
        Helper.log("Decoder", log.DEBUG, "get first spike: First spike logged at time {}".format(min_val))
        for row in range(self.size[0]):
            for col in range(self.size[1]):
                if self.neuron_array[row, col].spike_times:
                    # extracts the time of the first arrived spike
                    value = self.neuron_array[row, col].spike_times[0][1]
                    image[row, col] = value - min_val
                else:
                    # image[row, col] = None
                    image[row, col] = Helper.input_period +1*Helper.dt
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
        plt.figure()
        plt.imshow(graph, cmap='gray_r')
        plt.xlabel('column number')
        plt.ylabel('row number')
        if title is not None:
            plt.title(title)
        # for row in range(self.size[0]):
        #     for col in range(self.size[1]):
        #         if table[row, col] == np.nan:
        #             plt.plot(row, col, 'bo')

    def step(self):
        pass

    def reset(self):
        self.decoded_wta.append(self.get_first_spike())
        super(Decoder, self).reset()


class DecoderClassifier(Decoder):

    def __init__(self, size, dataset):
        super(DecoderClassifier, self).__init__(size)
        self.dataset = dataset

    def get_correlation_matrix(self):
        cor_mat = np.zeros((self.dataset.n_cats, self.size[1]))
        nb_exp = len(self.decoded_wta)
        for index, result in enumerate(self.decoded_wta):
            #  gets all the neurons index that spiked first
            dec_cat = [i for i, v in enumerate(result.tolist()[0]) if v == 0]
            for cat in dec_cat:
                cor_mat[self.dataset.labels[index], cat] += 1/self.dataset.pop_cats[self.dataset.labels[index]]
        return cor_mat
