from .base import SimulationObject
from .neuron import NeuronType
from .layer import Bloc
from .dataset import *
import numpy as np
import sys
sys.dont_write_bytecode = True


class DelayedNeuron(NeuronType):
    """
    Neuron used as a in a node
    Fires after the specified set delay

    Parameters
    ---------

    Attributes
    ----------
    delay : float
        the next time the neuron should fire
    active: bool
        Defines if the neuron is ready to fire
        deactivates after firing once, activates when new value is set
    """

    def __init__(self,):
        super(DelayedNeuron, self).__init__()
        self.delay = float('inf')
        self.active = False
        Helper.log('Neuron', log.INFO, 'neuron type : delayed')

    def step(self):
        # Helper.log('Neuron', log.DEBUG, 'neuron delay step')
        if self.active and Helper.time >= self.delay:
            Helper.log('Neuron', log.DEBUG, 'neuron delay expired : firing')
            self.active = False
            self.send_spike()

    def set_value(self, delay, active=True):
        self.delay = delay + Helper.time
        self.active = active
        Helper.log('Neuron', log.DEBUG, 'neuron delay set to {}'.format(delay))


class Encoder(Bloc):
    
    objects = []
    
    def __init__(self, depth, size, in_min, in_max, neuron_type=None):
        super(Encoder, self).__init__(depth, size, neuron_type=neuron_type)
        self.in_min = in_min
        self.in_max = in_max
        self.record = []
        Encoder.objects.append(self)

    def encode(self, data):
        return []

    def restore(self):
        self.record = []


class EncoderGFR(Encoder):
    """
    Creates a list of array to encode values into spikes
    Needs a Node that will provide values

    Parameters
    ---------
    size: int or (int, int)
        The dimension of the value or image
    depth : int
        The number of neuron used to encode a single value. Resolution
    in_min : float
        The minimum value of the gaussian firing field
    in_max : float
        The maximum value of the gaussian firing field
    delay_max : float
        The maximum delay created by the gaussian field
    threshold: float [0. - 1.]
        the ratio of the delay_max over which the neuron is allowed to fire
    gamma : float
        Parameter that influences the width of gaussian field

    Attributes
    ----------
    size: int or (int, int)
        The dimension of the value or image
    depth : int
        The number of neuron used to encode a single value. Resolution

    """

    def __init__(self, depth, size, in_min, in_max, delay_max=1, threshold=0.9, gamma=1.5):
        super(EncoderGFR, self).__init__(
            depth=depth,
            size=size,
            in_min=in_min,
            in_max=in_max,
            neuron_type=DelayedNeuron()
        )
        self.delay_max = delay_max
        self.threshold = threshold
        self.gamma = gamma
        Helper.log('Encoder', log.INFO, 'new encoder GFR, layer {0}'.format(self.id))

    def encode(self, values):
        sigma = (self.in_max - self.in_min) / (self.depth - 2.0) / self.gamma
        sequence = []
        for ens_index, ens in enumerate(self.ensemble_list):

            mu = self.in_min + (ens_index + 1 - 1.5) * ((self.in_max - self.in_min) / (self.depth - 2.0))
            for index, neuron in enumerate(ens.neuron_list):

                if isinstance(values, (int, float)):
                    value = values
                elif isinstance(values, (list, tuple)):
                    value = values[index]
                elif isinstance(values, np.ndarray):
                    value = values[index // self.size[1], index % self.size[0]]
                else:
                    raise Exception("unsupported input format")

                delay = (1 - np.exp(-0.5 * ((value - mu) / sigma) ** 2)) * self.delay_max

                if delay < self.threshold:
                    neuron.set_value(delay)
                    neuron.step()
                    sequence.append(delay)
                else:
                    sequence.append(self.delay_max)

        self.record.append(sequence)

    def step(self):
        pass

    def plot(self):
        plt.figure()
        plt.imshow(self.record, cmap='gray_r')
        plt.xlabel('input number')
        plt.ylabel('Neuron delay after first')
        plt.title('Encoder sequence')


class EncoderDoG(Encoder):
    def __init__(self, size, in_min, in_max, sigma, kernel_sizes, delay_max=1, threshold=None, double_filter=True):
        depth = len(sigma) * (2 if double_filter else 1)
        super(EncoderDoG, self).__init__(
            depth=depth,
            size=size,
            in_min=in_min,
            in_max=in_max,
            neuron_type=DelayedNeuron()
        )
        self.sigma = sigma  # [(s1,s2), (s3,s4) ...]
        self.kernel_sizes = kernel_sizes  # [5,7...]
        self.delay_max = delay_max
        self.threshold = threshold
        self.double_filter = double_filter

    @MeasureTiming('enc_dog')
    def encode(self, data):
        delays = np.zeros((self.size[0], self.size[1], self.depth))
        nb_per_value = 2 if self.double_filter else 1
        for index, sigmas in enumerate(self.sigma):

            # fact = self.sigma[index][1]/self.sigma[index][0]
            # data = np.reshape(data, self.size)
            # data_p = flt.gaussian(data, self.sigma[index][0])
            # plt.figure()
            # plt.imshow(data_p, cmap='gray')
            # plt.title('data_p layer ' + str(index)')
            # data_n = flt.gaussian(data, self.sigma[index][1])
            # plt.figure()
            # plt.imshow(data_n, cmap='gray')
            # plt.title('data_n layer ' + str(index))
            if self.double_filter:
                data_f = [self.filter(data, sigmas[0], sigmas[1], self.kernel_sizes[index]),
                          self.filter(data, sigmas[1], sigmas[0], self.kernel_sizes[index])]
            else:
                data_f = [self.filter(data, sigmas[0], sigmas[1], self.kernel_sizes[index])]
            # data_f = (data_p-data_n, data_n-data_p)
            # plt.figure()
            # plt.imshow(data_f, cmap='gray')
            # plt.title('data_f layer ' + str(index))

            for k, data in enumerate(data_f):
                i_min = data.min()
                i_max = data.max()
                data_t = (data - i_min) / (i_max - i_min)
                # plt.figure()
                # plt.imshow(data_t, cmap='gray')
                # plt.title('data_t layer ' + str(2 * index + k))
                threshold = np.mean(data_t) * 1.1 if self.threshold is None else self.threshold
                for row in range(self.size[0]):
                    for col in range(self.size[1]):
                        if data_t[row, col] >= threshold:
                            # delay = self.delay_max - (1 - self.threshold) * data_t[row, col]
<<<<<<< Updated upstream
                            delay = self.delay_max * (2 - data_t[row, col] / threshold)
=======
                            delay = self.delay_max * (2 - data_t[row, col] / self.threshold)
                            self.ensemble_list[nb_per_value * index + k].neuron_array[row, col].set_value(delay)
>>>>>>> Stashed changes
                        else:
                            delay = self.delay_max

                        delays[row, col, nb_per_value * index + k] = delay
        self.record.append(delays)

    def plot(self, index=-1, layer=0):
        plt.figure()
        plt.imshow(self.record[index][:, :, layer], cmap='gray_r')
        plt.title('Encoder sequence for input {} layer {}'.format(index, layer))

    @staticmethod
    def filter(image, sigma1, sigma2, size=7):
        # create kernel (code from spyketorch)
        w = size // 2
        x, y = np.mgrid[-w:w + 1:1, -w:w + 1:1]
        a = 1.0 / (2 * np.pi)
        prod = x * x + y * y
        f1 = (1 / (sigma1 * sigma1)) * np.exp(-0.5 * (1 / (sigma1 * sigma1)) * prod)
        f2 = (1 / (sigma2 * sigma2)) * np.exp(-0.5 * (1 / (sigma2 * sigma2)) * prod)
        dog = a * (f1 - f2)
        dog_mean = np.mean(dog)
        dog = dog - dog_mean
        dog_max = np.max(dog)
        dog = dog / dog_max

        # plt.figure()
        # plt.imshow(dog, cmap='gray')

        # Apply kernel to image
        img_padded = np.zeros((image.shape[0] + 2 * w, image.shape[1] + 2 * w))
        img_padded[w:image.shape[0]+w, w:image.shape[1]+w] = image
        img_filtered = np.zeros(image.shape)
        for row in range(image.shape[0]):
            for col in range(image.shape[1]):
                px = 0
                for k_row in range(size):
                    for k_col in range(size):
                        px += img_padded[row + k_row, col + k_col] * dog[k_row, k_col]
                img_filtered[row, col] = px
        return img_filtered


class Node(SimulationObject):
    """
        input source of the system, feeds the value an encoder

        can use several sub nodes to code for a single value

        Parameters
        ---------
        decoder: Encoder
            The number of neuron used to encode one value
        input : float or callable function
            The value to convert to spikes. can change with time
        period : float
            The time between 2 input changes
        *args, **kwargs
            The list and dict of arguments passed to the input function

        Attributes
        ----------
        size: NeuronType
            The number of neuron used to code one value
        input : float or callable function
            The value to convert to spikes
        period : float
            Time between 2 input changes
        next_period : float
            Time when of next input change
        active: bool
            will it change the values when the time reaches the threshold

        """
    objects = []

    def __init__(self, encoder, dataset, *args, **kwargs):
        super(Node, self).__init__()
        Node.objects.append(self)
        self.encoder = encoder
        self.dataset = dataset
        self.args = args
        self.kwargs = kwargs
        Helper.log('Encoder', log.INFO, 'new node created')

    def step(self):
        if isinstance(self.dataset, Dataset):
            value = self.dataset.next()
        elif callable(self.dataset):
            value = self.dataset(*self.args, **self.kwargs)
        else:
            value = self.dataset
        Helper.log('Encoder', log.INFO, 'Node sending next data')
        self.encoder.encode(value)

    def restore(self):
        self.dataset.index = 0
