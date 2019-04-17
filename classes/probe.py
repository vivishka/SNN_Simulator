
from .base import SimulationObject
from .ensemble import Ensemble
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.dont_write_bytecode = True


class Probe(SimulationObject):
    """docstring for Probe."""

    objects = []

    def __init__(self, target, variable):
        super(Probe, self).__init__()
        Probe.objects.append(self)
        self.target = target
        self.size = 1
        self.dim = 1
        self.variable_name = variable
        self.is_spike = variable in ('spike_in, spike_out')
        target.add_probe(self, variable)

        # dimension check
        if isinstance(target, Ensemble):
            self.size = target.size
            if isinstance(self.size, tuple):
                self.dim = 2
        self.nb = self.size if self.dim == 1 else self.size[0] * self.size[1]

        # value array initialization
        self.values = np.ndarray(self.size, dtype=list)
        for i, element in enumerate(self.values):
            if self.dim == 1:
                self.values[i] = []
            else:
                for j in range(len(element)):
                    self.values[i][j] = []

    def log_value(self, index, value):
        # TODO: call several methods for spikes or values
        # possibility to have multiple variable for a probe
        # print("probed neuron {0} at {1}".format(index, value))
        self.values[index].append(value)

    def log_spike_out(self, index, time):
        self.values[index].append(time)

    def log_spike_in(self, index, time, weight):
        self.values[index].append((time, weight))

    def plot(self):
        fig = plt.figure()
        plt.title(self.target.label)
        plt.xlabel('time')
        colors = ['k', 'r', 'b', 'g', 'm']
        if self.variable_name == 'spike_in':
            plt.grid(axis='y')
            # plt.scatter(*zip(*self.values))
            for index, graph in enumerate(self.values):
                # plt.subplot(self.size, 1, index)
                # TODO: solve this
                color = colors[index % 5]
                for time in graph:
                    plt.plot(time, index, 'o', color=color)
            plt.ylabel('neuron index')
        elif self.variable_name == 'spike_out':
            plt.grid()
            color = [colors[i % 5] for i in range(self.nb)]
            plt.eventplot(self.values.flatten(), color=color)
            plt.ylabel('neuron index')
        else:
            plt.ylabel(self.variable_name)
            for graph in self.values.flatten():
                plt.plot(graph)
        return fig
