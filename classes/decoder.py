import numpy as np
from .base import Helper
from .layer import Ensemble
from .neuron import NeuronType


class NeuronLog(NeuronType):
    """"
    This neuron is only used to store all the received spikes and times for later uses.
    This type of neuron is used in decoders

    Parameters
    ----------
    ensemble: Ensemble
        Same as NeuronType
    index: (int, int)
        Same as NeuronType
    **kwargs
        Same as NeuronType

    Attributes
    ----------
    spike_times: [(int, int)]
        list of tuple (ensemble index, time) of every received spike

    """
    def __init__(self, ensemble, index, **kwargs):
        super(NeuronLog, self).__init__(ensemble, index, **kwargs)
        self.spike_times = []

    def receive_spike(self, index):
        """ logs the ensemble index and the time of the received spike in a tuple"""
        super(NeuronLog, self).receive_spike(index)
        for spike in self.received:
            self.spike_times.append((spike[0], Helper.time))
        self.received = []

    def step(self):
        """" non stepping neuron"""
        pass


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
        super(Decoder, self).__init__(size, NeuronLog)

    def get_first_spike(self):
        """
        Decoding algorithm based on WTA: only uses the time of first spike

        :return: an array the same size as this ensemble containing the time
        of the first spike received by each neuron
        # TODO: change from 255 to 1.0
        """
        image = np.zeros(self.size, dtype=np.uint8)

        first_spike_list = [n.spike_times[0][1] for n in self.neuron_list if n.spike_times]
        if not first_spike_list:
            return image

        min_val = min(first_spike_list)
        max_val = max(first_spike_list)

        for row in range(self.size[0]):
            for col in range(self.size[1]):
                if self.neuron_array[row, col].spike_times:
                    value = self.neuron_array[row, col].spike_times[0][1]
                    if min_val == max_val:
                        value = 0
                    else:
                        value = ((value - min_val) / (max_val - min_val))
                    image[row, col] = np.uint8((1 - value) * 255)

                else:
                    image[row, col] = 0
        return image

    def decoded_image(self):
        """
        Decoding algorithm to directly decode encoded input
        Mainly used for debug

        :return: an array the same size as this ensemble representing the decoded value
        # TODO: change from 255 to 1.0
        # TODO: edge cases (empty list, min == max, ...)
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

    def step(self):
        pass
