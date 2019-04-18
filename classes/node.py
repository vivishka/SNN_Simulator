
from .base import SimulationObject
from .neuron import NeuronType
from .ensemble import Ensemble
import numpy as np
import sys
sys.dont_write_bytecode = True


def gauss_sequence(value, dimension, x_min, x_max, y_max, gamma=1.5):
    firing_sequence = np.zeros(dimension)
    sigma = (x_max - x_min) / (dimension - 2.0) / gamma
    for i in range(dimension):
        mu = x_min + (i - 1.5) * ((x_max - x_min) / (dimension - 2.0))
        y = np.exp(-0.5*((value-mu)/sigma)**2)
        firing_sequence[i] = (1 - y) * y_max if y > 0.1 else -1
    return firing_sequence


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
        disactivates after firing once, activates when new value is set
    """

    def __init__(self, ensemble, index):
        super(DelayedNeuron, self).__init__(ensemble, index)
        self.delay = -1
        self.active = False

    def step(self, dt, time):
        if self.active and time >= self.delay:
            # print("node neur {} fired".format(self.label))
            self.active = False
            self.send_spike()

    def set_value(self, delay, active=True):
        self.delay = delay
        self.active = active


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
    delay : float
        the next time the neuron should fire
    active: bool
        Defines if the neuron is ready to fire
        disactivates after firing once, activates when new value is set
    """

    def __init__(self, ensemble, index):
        super(GaussianFiringNeuron, self).__init__(ensemble, index)
        self.value = None
        self.firing_time = -1
        self.active = False
        self.mu = self.sigma = self.delay_max = self.threshold = None

    def set_params(self, mu, sigma, delay_max, threshold):
        self.mu = mu
        self.sigma = sigma
        self.delay_max = delay_max
        self.threshold = threshold

    def step(self, dt, time):
        if self.active and time >= self.firing_time:
            # print("node neur {} fired".format(self.label))
            self.active = False
            self.send_spike()

    def set_value(self, value, time):
        delay = (1-np.exp(-0.5*((value-self.mu)/self.sigma)**2))*self.delay_max
        if delay < (self.delay_max * self. threshold):
            self.firing_time = time + delay
            self.active = True


class Encoder(object):
    """
    Creates a list of array to encode an image into spikes

    Parameters
    ---------
    size: int or (int, int)
        The dimension of the value or image
    nb : int
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
    nb : int
        The number of neuron used to encode a single value. Resolution
    ensemble_list: [Ensemble]
        There are nb ensembles. all neuron from the same ensemvble have the same curve
    dim: int
        the dimension of the input

    """

    def __init__(self, size, nb, in_min, in_max, delay_max, threshold=.9, gamma=1.5):
        # super(Encoder, self).__init__()
        self.size = size
        self.nb = nb
        self.ensemble_list = []
        self.dim = 1 if isinstance(size, int) else len(size)

        sigma = (in_max - in_min) / (nb - 2.0) / gamma
        for ens_index in range(nb):
            ens = Ensemble(
                size=size,
                neuron_type=GaussianFiringNeuron,
                index=ens_index)
            self.ensemble_list.append(ens)

            mu = in_min + (ens_index - 1.5) * ((in_max - in_min) / (nb - 2.0))
            for neuron in ens.neuron_list:
                neuron.set_params(
                    mu=mu,
                    sigma=sigma,
                    delay_max=delay_max,
                    threshold=threshold)

    def set_one_value(self, value, time, index):
        for ens in self.ensemble_list:
            ens[index].set_value(value, time)

    def set_values(self, values, time):
        if self.dim == 1:
            for i, val in enumerate(values):
                self.set_one_value(val, time, i)
        else:
            for i, row in enumerate(values):
                for j, val in enumerate(row):
                    self.set_one_value(val, time, (i, j))

    def __getitem__(self, index):
        return self.ensemble_list[index]

    def __setitem__(self, index, value):
        self.ensemble_list[index] = value


class PeriodicInput(SimulationObject):
    """
        input source of the system, feeds the value into the encoder

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
        time: float
            Time ellapsed since the last input change

        """
    objects = []

    def __init__(self, encoder, value, period=-1, delay=0, *args, **kwargs):
        super(PeriodicInput, self).__init__()
        PeriodicInput.objects.append(self)
        self.encoder = encoder
        self.value = value
        self.period = period
        self.next_input = delay
        self.active = True
        self.args = args
        self.kwargs = kwargs

    def step(self, dt, time):
        if self.active and time >= self.next_input:
            if callable(self.value):
                value = self.value(*self.args, **self.kwargs)
            else:
                value = self.value

            self.encoder.set_values(value, time)

            if self.period > 0:
                self.next_input += self.period
            else:
                self.active = False


class Node(Ensemble):
    """
    Converts float input into sequence of spikes
    can use several sub nodes to code for a single value

    Parameters
    ---------
    size: int
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
    time: float
        Time ellapsed since the last input change

    """

    # TODO: consider an ensemble of nodes
    objects = []

    def __init__(self, size, input, period, label='', *args, **kwargs):
        lbl = label if label != '' else id(self)
        super(Node, self).__init__(size, DelayedNeuron, lbl)
        Node.objects.append(self)
        self.size = size
        self.input = input
        self.period = period
        self.next_input = period
        self. time = 0
        self.args = args
        self.kwargs = kwargs
        self.set_value()

    def set_value(self):
        """ activates all neurons and set their new firing times """
        if callable(self.input):
            value = self.input(*self.args, **self.kwargs)
        else:
            value = self.input
        trigger = gauss_sequence(value, self.size, 0, 1, 0.1)
        for i, neuron in enumerate(self.get_neuron_list()):
            if trigger[i] >= 0:
                neuron.set_value(trigger[i] + self.time)

    def step(self, dt, time):
        self.time = time
        for neuron in self.neuron_list:
            neuron.step(dt, time)
        if time >= self.next_input:
            self.set_value()
            self.next_input += self.period

    def reset(self):
        pass


class Reset(SimulationObject):
    """docstring for Reset."""

    objects = []

    def __init__(self, delay, period):
        super(Reset, self).__init__()
        Reset.objects.append(self)
        self. delay = delay
        self.period = period
        self.next_reset = delay
        self.reset_funct = None

    def set_reset_funt(self, function):
        self.reset_funct = function

    def step(self, dt, time):
        if time > self.next_reset:
            self.next_reset += self.period
            # print("resting {}".format(time))
            self.reset_funct()
