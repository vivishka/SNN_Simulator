
from .base import SimulationObject
from .ensemble import Ensemble
import matplotlib.pyplot as plt
# import numpy as np
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
        self.variable = variable
        self.is_spike = variable in ('spike_in, spike_out')
        if isinstance(target, Ensemble):
            self.size = target.size
            # TODO: handle 2D
        target.add_probe(self, variable)
        self.values = [[] for i in range(self.size)]

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
        if self.variable == 'spike_in':
            plt.grid(axis='y')
            # plt.scatter(*zip(*self.values))
            for index, graph in enumerate(self.values):
                # plt.subplot(self.size, 1, index)
                # TODO: solve this
                color = colors[index % 5]
                for time in graph:
                    plt.plot(time, index, 'o', color=color)
            plt.ylabel('neuron index')
        elif self.variable == 'spike_out':
            plt.grid()
            color = [colors[i % 5] for i in range(self.size)]
            plt.eventplot(self.values, color=color)
            # # plt.scatter(*zip(*self.values))
            # for index, graph in enumerate(self.values):
            #     color = colors[index % 5]
            #     for time in graph:
            #         plt.plot(time, index, 'o', color=color)
            plt.ylabel('neuron index')
        else:
            plt.ylabel(self.variable)
            for graph in self.values:
                plt.plot(graph)
        return fig
