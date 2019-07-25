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
        if self.active and self.ensemble.sim.curr_time >= self.delay:
            Helper.log('Neuron', log.DEBUG, 'neuron delay expired : firing')
            self.active = False
            self.send_spike()

    def set_value(self, delay, active=True):
        self.delay = delay + self.ensemble.sim.curr_time
        self.active = active
        Helper.log('Neuron', log.DEBUG, 'neuron delay set to {}'.format(delay))

    def reset(self):
        super(DelayedNeuron, self).reset()
        self.active = False


def create_dog_filters(kernel_size_list, sigma_list, double_filter):
    filters = []
    for index, sigmas in enumerate(sigma_list):

        size = kernel_size_list[index]
        # create kernel (code from spyketorch)
        w = size // 2
        x, y = np.mgrid[-w:w + 1:1, -w:w + 1:1]
        a = 1.0 / (2 * np.pi)
        prod = x * x + y * y
        f1 = (1 / (sigmas[0] * sigmas[0])) * np.exp(-0.5 * (1 / (sigmas[0] * sigmas[0])) * prod)
        f2 = (1 / (sigmas[1] * sigmas[1])) * np.exp(-0.5 * (1 / (sigmas[1] * sigmas[1])) * prod)
        dog = a * (f1 - f2)
        dog_mean = np.mean(dog)
        dog = dog - dog_mean
        dog_max = np.max(dog)
        dog = dog / dog_max
        filters.append(dog)

        if double_filter:
            filters.append(dog * -1.)

    return filters


def create_gabor_filters(kernel_size, orientation_list, div):
    filters = []
    # Code from Spyketorch
    for orientation in orientation_list:
        w = kernel_size//2
        x, y = np.mgrid[-w:w+1:1, -w:w+1:1]
        lamda = kernel_size * 2 / div
        sigma = lamda * 0.8
        sigma_sq = sigma * sigma
        g = 0.3
        theta = (orientation * np.pi) / 180
        Y = y*np.cos(theta) - x*np.sin(theta)
        X = y*np.sin(theta) + x*np.cos(theta)
        gabor = np.exp(-(X * X + g * g * Y * Y) / (2 * sigma_sq)) * np.cos(2 * np.pi * X / lamda)
        gabor_mean = np.mean(gabor)
        gabor = gabor - gabor_mean
        gabor_max = np.max(gabor)
        gabor = gabor / gabor_max
        filters.append(gabor)

    return filters


class Encoder(Bloc):
    
    objects = []
    
    def __init__(self, depth, size, in_min, in_max, delay_max=1, neuron_type=None, spike_all_last=False):
        super(Encoder, self).__init__(depth, size, neuron_type=neuron_type)
        self.in_min = in_min
        self.in_max = in_max
        self.delay_max = delay_max
        self.record = []
        self.spike_all_last = spike_all_last
        Encoder.objects.append(self)

    def encode(self, data):
        return []

    def restore(self):
        self.record = []

    def set_delay(self, data, index, delays):
        """ Delays is being modified """
        for row in range(self.size[0]):
            for col in range(self.size[1]):
                delay = data[row, col]
                if self.spike_all_last or delay < self.delay_max:
                    self.ensemble_list[index].neuron_array[row, col].set_value(delay)
                delays[row, col, index] = delay


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
    in_min : float or int
        The minimum value of the gaussian firing field
    in_max : float or int
        The maximum value of the gaussian firing field
    delay_max : float or int
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

    def __init__(self, depth, size, in_min, in_max, delay_max=1., threshold=0.9, gamma=1.5, spike_all_last=False):
        super(EncoderGFR, self).__init__(
            depth=depth,
            size=size,
            in_min=in_min,
            in_max=in_max,
            delay_max=delay_max,
            neuron_type=DelayedNeuron(),
            spike_all_last=spike_all_last

        )
        self.threshold = threshold
        self.gamma = gamma
        Helper.log('Encoder', log.INFO, 'new encoder GFR, layer {0}'.format(self.id))

    def encode(self, values):
        sigma = (self.in_max - self.in_min) / (self.depth - 2.0) / self.gamma
        sequence = np.zeros((self.size[1], self.depth))
        for ens_index, ens in enumerate(self.ensemble_list):

            mu = self.in_min + (ens_index + 1 - 1.5) * ((self.in_max - self.in_min) / (self.depth - 2.0))
            for index, neuron in enumerate(ens.neuron_list):

                if isinstance(values, (int, float)):
                    value = values
                elif isinstance(values, (list, tuple)):
                    value = values[index]
                elif isinstance(values, np.ndarray):
                    value = values[index // self.size[1], index % self.size[1]]
                else:
                    raise Exception("unsupported input format")

                delay = (1 - np.exp(-0.5 * ((value - mu) / sigma) ** 2)) * self.delay_max

                if delay < self.threshold:
                    neuron.set_value(delay)
                    neuron.step()
                    sequence[index, ens_index] = delay
                else:
                    if self.spike_all_last:
                        neuron.set_value(self.threshold)
                        neuron.step()
                    sequence[index, ens_index] = self.delay_max

        self.record.append(sequence)

    def step(self):
        pass

    def plot(self, index=None):
        if index:
            plt.figure()
            plt.imshow([seq[index] for seq in self.record], cmap='gray_r')
            plt.xlabel('Neuron index')
            plt.ylabel('Input number')
            plt.title('Data {}: Neuron delay after fastest'.format(index))
        else:
            for ens in range(self.size[1]):
                plt.figure()
                plt.imshow([seq[ens] for seq in self.record], cmap='gray_r')
                plt.xlabel('Neuron index')
                plt.ylabel('Input number')
                plt.title('Data {}: Neuron delay after fastest'.format(ens))

class EncoderFilter(Encoder):
    def __init__(
            self, depth, size, in_min, in_max, kernel_size, delay_max=1,
            neuron_type=None, spike_all_last=False, threshold=None):
        super(EncoderFilter, self).__init__(depth, size, in_min, in_max, delay_max, neuron_type, spike_all_last)
        self.filters = []
        self.threshold = threshold
        self.kernel_size = kernel_size

    def create_filters(self):
        pass

    def thresh_norm(self, image):
        threshold = np.mean(image) * 1. if self.threshold is None else self.threshold
        temporal_image = np.clip(image-threshold, a_min=0., a_max=None)
        temporal_image = (1 - temporal_image / temporal_image.max()) * self.delay_max
        return temporal_image

    @staticmethod
    def apply_conv(image, kernel):
        size = kernel.shape[0]
        w = size // 2

        # Apply kernel to image
        img_padded = np.zeros((image.shape[0] + 2 * w, image.shape[1] + 2 * w))
        img_padded[w:image.shape[0]+w, w:image.shape[1]+w] = image
        img_filtered = np.zeros(image.shape)
        for row in range(image.shape[0]):
            for col in range(image.shape[1]):
                px = 0
                for k_row in range(size):
                    for k_col in range(size):
                        px += img_padded[row + k_row, col + k_col] * kernel[k_row, k_col]
                img_filtered[row, col] = px
        return img_filtered

    def encode(self, data):
        delays = np.zeros((self.size[0], self.size[1], self.depth))
        for index, kernel in enumerate(self.filters):

            data_f = self.apply_conv(image=data, kernel=kernel)
            data_n = self.thresh_norm(image=data_f)
            self.set_delay(data=data_n, index=index, delays=delays)

        self.record.append(delays)

    def plot(self, index=-1, layer=0):
        plt.figure()
        plt.imshow(self.record[index][:, :, layer], cmap='gray_r')
        plt.title('Encoder sequence for input {} layer {}'.format(index, layer))


class EncoderDoG(EncoderFilter):
    def __init__(
            self, size, in_min, in_max, sigma, kernel_sizes, delay_max=1,
            threshold=None, double_filter=True, spike_all_last=False):
        depth = len(sigma) * (2 if double_filter else 1)
        super(EncoderDoG, self).__init__(
            depth=depth, size=size, in_min=in_min, in_max=in_max, delay_max=delay_max,
            kernel_size=None, neuron_type=DelayedNeuron(),
            spike_all_last=spike_all_last, threshold=threshold
            )
        self.sigma = sigma  # [(s1,s2), (s3,s4) ...]
        self.kernel_sizes = kernel_sizes  # [5,7...]
        self.threshold = threshold
        self.double_filter = double_filter
        self.filters = create_dog_filters(kernel_size_list=kernel_sizes, sigma_list=sigma, double_filter=double_filter)


class EncoderGabor(EncoderFilter):

    def __init__(
            self, size, orientations, kernel_size, in_min=0, in_max=255, delay_max=0.75,
            spike_all_last=False, threshold=None, div=4):
        depth = len(orientations)
        super(EncoderGabor, self).__init__(
            depth=depth, size=size, in_min=in_min, in_max=in_max, delay_max=delay_max,
            neuron_type=DelayedNeuron(), spike_all_last=spike_all_last,
            threshold=threshold, kernel_size=kernel_size)
        self.orientations = orientations
        self.div = div
        self.filters = create_gabor_filters(kernel_size=kernel_size, orientation_list=orientations, div=div)


class EncoderLinear(Encoder):
    """
    Creates a list of array to encode values into spikes
    Needs a Node that will provide values

    Parameters
    ---------
    size: int or (int, int)
        The dimension of the value or image
    depth : int
        The number of neuron used to encode a single value. Resolution
    in_min : float or int
        The minimum value of the gaussian firing field
    in_max : float or int
        The maximum value of the gaussian firing field
    delay_max : float or int
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

    def __init__(self, size, in_min=0, in_max=1, delay_max=1.):
        super(EncoderLinear, self).__init__(
            depth=1,
            size=size,
            in_min=in_min,
            in_max=in_max,
            delay_max=delay_max,
            neuron_type=DelayedNeuron(),
            spike_all_last=False

        )
        Helper.log('Encoder', log.INFO, 'new encoder Linear, layer {0}'.format(self.id))

    def encode(self, values):
        sequence = np.zeros((self.size[1], self.depth))
        ens = self.ensemble_list[0]

        for index, neuron in enumerate(ens.neuron_list):

            if isinstance(values, (int, float)):
                value = values
            elif isinstance(values, (list, tuple)):
                value = values[index]
            elif isinstance(values, np.ndarray):
                value = values[index // self.size[1], index % self.size[1]]
            else:
                raise Exception("unsupported input format")

            delay = (1 - (value - self.in_min) / (self.in_max - self.in_min)) * self.delay_max

            neuron.set_value(delay)
            neuron.step()
            sequence[index, 0] = delay

        self.record.append(sequence)

    def step(self):
        pass

    def plot(self, index=None):
        if index:
            plt.figure()
            plt.imshow([seq[index] for seq in self.record], cmap='gray_r')
            plt.xlabel('Neuron index')
            plt.ylabel('Input number')
            plt.title('Data {}: Neuron delay after fastest'.format(index))
        else:
            for ens in range(self.size[1]):
                plt.figure()
                plt.imshow([seq[ens] for seq in self.record], cmap='gray_r')
                plt.xlabel('Neuron index')
                plt.ylabel('Input number')
                plt.title('Data {}: Neuron delay after fastest'.format(ens))


class Node(SimulationObject):
    """
        input source of the system, feeds the value an encoder

        can use several sub nodes to code for a single value

        Parameters
        ---------
        encoder: Encoder or List
            Encoders fed by the node

        Attributes
        ----------

        """
    objects = []

    def __init__(self, encoder):
        super(Node, self).__init__()
        Node.objects.append(self)
        self.encoder_list = encoder if isinstance(encoder, list) else [encoder]
        assert all([isinstance(enc, Encoder) for enc in self.encoder_list])
        Helper.log('Encoder', log.INFO, 'new node created')

    def step(self):
        if isinstance(self.sim.dataset, Dataset):
            value = self.sim.dataset.next()
        else:
            value = self.sim.dataset
        Helper.log('Encoder', log.INFO, 'Node sending next data')
        for encoder in self.encoder_list:
            encoder.encode(value)

    def restore(self):
        try:
            self.sim.dataset.index = 0
        except AttributeError:
            pass
