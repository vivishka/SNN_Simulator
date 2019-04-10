
# from base import SimationObject
from neuron import NeuronType
from ensemble import Ensemble
import numpy as np
import sys
sys.dont_write_bytecode = True


def compute_gauss_time(value, dimension, x_min, x_max, y_max, gamma=1.5):
    firing_sequence = np.zeros(dimension)
    sigma = (x_max - x_min) / (dimension - 2.0) / gamma
    for i in range(dimension):
        mu = x_min + (i - 1.5) * ((x_max - x_min) / (dimension - 2.0))
        y = np.exp(-0.5*((value-mu)/sigma)**2)
        firing_sequence[i] = (1 - y) * y_max if y > 0.1 else -1
    return firing_sequence


class GaussianFieldUnit(NeuronType):
    """docstring for GaussianFieldUnit."""

    def __init__(self, ensemble, index):
        super(GaussianFieldUnit, self).__init__(ensemble, index)
        self.trigger = -1
        self.active = False
        self.time = 0

    def step(self, dt):
        self.time += dt
        if self.active and self.time >= self.trigger:
            self.trigger = True
            self.send_spike()

    def set_value(self, trigger, active):
        self.trigger = trigger
        self.active = active


class Node(Ensemble):
    '''
    Converts float input into sequence of spikes
    can use several sub nodes to code for a single value

    Parameters
    ---------
    size: int
         The number of neuron used to code one value
    input : float, function, generator
        The value to convert to spikes. can change with time
    period : int
        The number of steps between 2 input changes

    Attributes
    ----------
    size: NeuronType
        Stores the number of neuron used to code one value
    input : float, function, generator
        Stores the value to convert to spikes
    period : int
        Stores the number of steps between 2 input changes
    time: int
        Time ellapsed since the last input change
    '''

    objects = []

    def __init__(self, size, input, period, *args):
        super(Node, self).__init__(size, GaussianFieldUnit)
        Node.objects.append(self)
        self.size = size
        self.input = input
        self.period = period
        self.args = args
        self.time = 0

    def set_value(self, value):
        trigger = compute_gauss_time(value, self.size, 0, 1, 10)
        for i, neuron in enumerate(self.get_neuron_list):
            neuron.new_value(trigger[i], True)

    def step(self, dt):
        self.time += dt
        if (self.time >= self.period):
            # TODO: compute new value
            self.set_value(self.input)
            self.time = 0
