import numpy as np
import matplotlib.image as mpimg
from .base import Helper
from .ensemble import Ensemble
from .neuron import NeuronType


class NeuronLog(NeuronType):
    """"
    # for now: WTA
    """
    def __init__(self, ensemble, index, **kwargs):
        super(NeuronLog, self).__init__(ensemble, index, **kwargs)
        self.spike_times = []

    def step(self):
        if self.received:
            for index in self.received:
                self.spike_times.append((index[0], Helper.time))
            self.received = []


class Decoder(Ensemble):

    def __init__(self, size):
        super(Decoder, self).__init__(size, NeuronLog)
        self.size = size
        self.dim = 1 if isinstance(size, int) else len(size)

    def get_first_spike(self):
        image = np.zeros(self.size, dtype=np.uint8)

        first_spike_list = [n.spike_times[0][1] for n in self.neuron_list if n.spike_times]
        min_val = min(first_spike_list)
        max_val = max(first_spike_list)

        for line in range(self.size[0]):
            for col in range(self.size[1]):
                if self.neuron_array[line, col]:
                    value = self.neuron_array[line, col].spike_times[0][1]
                    value = ((value - min_val) / (max_val - min_val))
                    image[line, col] = np.uint8((1 - value) * 255)

                else:
                    image[line, col] = 0
        return image

    def decoded_image(self):
        image = np.zeros(self.size, )

        # index_list = [[i for (i, t) in n.spike_times] for n in self.neuron_list if n.spike_times]
        spike_list = [[t for (i, t) in n.spike_times] for n in self.neuron_list if n.spike_times]

        min_val = min([min(l) for l in spike_list])
        max_val = max([max(l) for l in spike_list])

        # spike_list = [[1 - (t - min_val)/(max_val - min_val) for t in n] for n in spike_list]

        for line in range(self.size[0]):
            for col in range(self.size[1]):
                sum = 0
                for index, time in self.neuron_array[line, col].spike_times:
                    sum += (1 - (time - min_val) / (max_val - min_val)) * (index + 1)
                image[line, col] = sum

        max_val = image.max()
        min_val = image.min()
        image = (image - min_val)/(max_val - min_val) * 255
        return image.astype(np.uint8)