
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
    # TODO: inherit from ensemble
    '''
    Converts float input into sequence of spikes
    can use several sub nodes to code for a single value

    Parameters
    ---------
    size: int
         The number of neuron used to code one value
    input : float, function, generator
        The value to convert to spikes. will change
    period : int
        The number of steps between 2 input changes

    Attributes
    ----------
    source: NeuronType
        The emitting neuron
    dest : [NeuronType]
        The list of connected neurons
    spike_notifier: SpikeNotifier
        The object to notifiy when this axon has a spike to propagate


    '''

    objects = []

    def __init__(self, size, input, period, *args):
        super(Node, self).__init__(size, GaussianFieldUnit)
        Node.objects.append(self)

        # TODO: compute gaussian cruves
        self.period = period
        # TODO: present new input every period
        self.args = args


    def set_value(self):
        trigger = compute_gauss_time(input, size, 0, 1, 10)
        for i, neuron in enumerate(self.get_neuron_list):
            neuron.new_value(trigger[i], True)
