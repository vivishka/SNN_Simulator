
import numpy as np
from .base import SimulationObject, Helper
from .layer import Ensemble
from .neuron import NeuronType
from .connection import Connection
import matplotlib.pyplot as plt
import sys
sys.dont_write_bytecode = True


class Probe(SimulationObject):
    """docstring for Probe."""

    objects = []
    # TODO: work for spikes in and out,

    def __init__(self, target, variables):
        super(Probe, self).__init__()
        Probe.objects.append(self)
        self.target = target
        self.variables = [variables] if isinstance(variables, str) else variables

        if isinstance(target, Ensemble):
            self.__probe_neurons(target.neuron_list)
        elif isinstance(target, NeuronType):
            self.__probe_neurons([target])
        elif isinstance(target, list) and all(isinstance(n, NeuronType) for n in target):
            self.__probe_neurons(target)
        elif isinstance(target, Connection):
            self.__probe_connection()
        else:
            raise Exception("wrong type given to probe target")

        # self.is_spike = variable in 'spike_in, spike_out'

    def __probe_neurons(self, neuron_list):
        self.target = neuron_list
        for neuron in neuron_list:
            for var in self.variables:
                neuron.add_probe(self, var)

    def __probe_connection(self):
        self.target.add_probe(self)

    def get_data(self, variable):
        if variable not in self.variables:
            print("no probe set for {}".format(variable))
            return
        values = []
        for i, neuron in enumerate(self.target):
            values.append(neuron.probed_values[variable])
        return values

    def plot(self, variable):
        if variable not in self.variables:
            print("no probe set for {}".format(variable))
            return
        fig = plt.figure()
        plt.title(self.target.label)
        plt.xlabel('time')
        colors = ['k', 'r', 'b', 'g', 'm']
        values = self.get_data(variable)
        for i, neuron in enumerate(values):
            # if variable == 'spike_in':
            #     plt.grid(axis='y')
            #     for index, graph in enumerate(values):
            #         color = colors[index % 5]
            #         for time in graph:
            #             plt.plot(time, index, 'o', color=color)
            #     plt.ylabel('neuron index')
            # elif variable == 'spike_out':
            #     plt.grid()
            #     color = [colors[i % 5] for i in range(self.nb)]
            #     plt.eventplot(values.flatten(), color=color)
            #     plt.ylabel('neuron index')

            plt.ylabel(variable)
            plt.plot(*zip(*values[i]), color=colors[i % 5])
        return fig
