import numpy as np
import matplotlib.image as mpimg
from .ensemble import Ensemble
from .neuron import NeuronType


class NeuronLog(NeuronType):
    """"
    # for now: WTA
    """
    def __init__(self):
        super(NeuronLog, self).__init__()
        self.spike_times = []

    def step(self, dt, time):
        if self.received:
            self.spike_times.append(time)


class Decoder(Ensemble):

    def __init__(self, size):
        super(Decoder, self).__init__(size, NeuronLog)
        self.size = size
        self.dim = 1 if isinstance(size, int) else len(size)

    def get_first_spike(self):
        image = np.zeros(self.size, dtype=np.uint8)

        first_spike_list = [n.spike_times[0] for n in self.neuron_list if n.spike_times]
        min_val = min(first_spike_list)
        max_val = max(first_spike_list)

        for line in range(self.size[0]):
            for col in range(self.size[1]):
                if self.neuron_array[line, col]:
                    value = self.neuron_array[line, col][0]
                    value = ((value - min_val) / (max_val - min_val))
                    image[line, col] = ((1 - value) * 255).astype(np.uint8)

                else:
                    image[line, col] = 0
        return image
