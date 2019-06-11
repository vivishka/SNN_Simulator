import logging as log
from .base import SimulationObject, Helper
from .neuron import NeuronType
from .layer import Ensemble, Bloc
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
    ensemble: Node
        The Node this neuron belongs to
    index: int
        The index (in 1D or 2D) of the neuron in the ensemble

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


class GaussianFiringNeuron(NeuronType):
    """
    Neuron used as a in a node
    Fires after the specified set delay

    Parameters
    ---------
    ensemble: Node
        The Node this neuron belongs to
    index: int
        The index (in 1D or 2D) of the neuron in the ensemble

    Attributes
    ----------
    firing_time : float
        the next time the neuron should fire
    active: bool
        Defines if the neuron is ready to fire
        deactivates after firing once, activates when new value is set
    mu: float
        gaussian parameter of the mean
    sigma: float
        gaussian parameter of the std dev
    """

    nb_spikes = 0

    def __init__(self,):
        super(GaussianFiringNeuron, self).__init__()
        self.firing_time = -1
        self.active = False
        self.mu = self.sigma = self.delay_max = self.threshold = None
        Helper.log('Neuron', log.DEBUG, str(self.index_2d) + 'neuron type: gaussian firing encoder')

    def set_params(self, mu, sigma, delay_max, threshold):
        self.mu = mu
        self.sigma = sigma
        self.delay_max = delay_max
        self.threshold = threshold

    def step(self):
        if self.active and Helper.time >= self.firing_time:
            Helper.log('Encoder', log.DEBUG, ' neuron {} from layer {} fired'.format(self.index_2d, self.ensemble.id))
            self.active = False
            self.send_spike()
            GaussianFiringNeuron.nb_spikes += 1
        else:
            pass
            # self.ensemble.active_neuron_list.append(self)

    def set_value(self, value):
        delay = (1-np.exp(-0.5*((value-self.mu)/self.sigma)**2))*self.delay_max
        Helper.log('Encoder', log.DEBUG, "neuron {} will encode value {} spike at {}".format(self.index_2d, value, delay))
        if delay < (self.delay_max * self. threshold):
            self.firing_time = Helper.time + delay
            self.active = True


# class EncoderEnsemble(Ensemble):
#
#     def __init__(self, size, neuron_type, label='', **kwargs):
#         super(EncoderEnsemble, self).__init__(size, neuron_type, label, **kwargs)
#         Helper.log('Encoder', log.INFO, 'new encoder ensemble, layer {0}'.format(self.id))
#
#     def step(self):
#         for neuron in self.neuron_list:
#             neuron.step()


class Encoder(Bloc):
    
    objects = []
    
    def __init__(self, depth, size, in_min, in_max, neuron_type=None):
        super(Encoder, self).__init__(depth, size, neuron_type=neuron_type)
        self.in_min = in_min
        self.in_max = in_max

        Encoder.objects.append(self)

    def encode(self, data):
        return []

    def restore(self):
        pass


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
    ensemble_list: [Ensemble]
        There are nb ensembles. all neuron from the same ensemble have the same curve

    """

    # def __init__(self, depth, size, in_min, in_max, delay_max, threshold=0.9, gamma=1.5):
    #     super(EncoderGFR, self).__init__(
    #         depth=depth,
    #         size=size,
    #         neurontype=GaussianFiringNeuron()
    #     )
    #
    #     sigma = (in_max - in_min) / (depth - 2.0) / gamma
    #
    #     for ens_index, ens in enumerate(self.ensemble_list):
    #
    #         mu = in_min + (ens_index + 1 - 1.5) * ((in_max - in_min) / (depth - 2.0))
    #         for neuron in ens.neuron_list:
    #             neuron.set_params(
    #                 mu=mu,
    #                 sigma=sigma,
    #                 delay_max=delay_max,
    #                 threshold=threshold)
    #
    #             # those neurons needs to always be active
    #             # TODO: optimize this
    #             ens.probed_neuron_set.add(neuron)
    #     EncoderGFR.objects.append(self)
    #     Helper.log('Encoder', log.INFO, 'new encoder bloc, layer {0}'.format(self.id))

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
                    raise Exception("unsuported input format")

                delay = (1 - np.exp(-0.5 * ((value - mu) / sigma) ** 2)) * self.delay_max

                if delay < self.threshold:
                    neuron.set_value(delay)
                    neuron.step()

    def step(self):
        pass


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


class EncoderDoG(Ensemble):
    def __init__(self, size, in_min=0, in_max=255, delay_max=1, sigma1=1, sigma2=3):
        super(EncoderDoG, self).__init__(size, NeuronType=DelayedNeuron())
