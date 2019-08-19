
# import logging as log
from .base import MeasureTiming
import numpy as np
from .layer import Ensemble, Bloc
from .neuron import NeuronType
from .connection import Connection
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sys
sys.dont_write_bytecode = True


class Probe(object):
    """
    Notify the target that the specified values are probed
    Mother class, should not be instanced by itself

    Parameters
    ----------
    target: object or list[objects]
        probed object
    var : str or list[str] or None
        probed variables

    Attributes
    ----------
    target: object object or list[objects]
         probed object
    var : list[str]
        The receiving ensemble
    """

    def __init__(self, target, var=None):
        super(Probe, self).__init__()
        self.target = target
        self.var = [var] if isinstance(var, str) else var


class ConnectionProbe(Probe):
    """
    Probe for connections
    if a not active (group of) Connection is given, all the sub Connections are probed

    Parameters
    ----------
    target: Connection
        probed Connection, can be active or not

    Attributes
    ----------

    """
    def __init__(self, target):
        if not isinstance(target, Connection):
            raise Exception("wrong type given to probe target")
        super(ConnectionProbe, self).__init__(target.connection_list)

        for connection in self.target:
            connection.add_probe()

    def get_data(self, connection_index):
        return self.target[connection_index].probed_values

    @MeasureTiming('probe_plot')
    def plot(self, connection_index='all', max_nb_neuron=None):
        plt.figure()
        target_range = []
        if connection_index == 'all':
            target_range = range(len(self.target))
        elif isinstance(connection_index, int):
            target_range = [connection_index]
        elif isinstance(connection_index, list):
            target_range = connection_index

        for connect_index in target_range:

            # range and graph init
            matrix_list = self.get_data(connection_index=connect_index)
            nb_weight = len(matrix_list[0])
            if max_nb_neuron is not None:
                nb_weight = max_nb_neuron
            nb_matrix = len(matrix_list)
            graph = [np.ndarray((nb_matrix,)) for _ in range(nb_weight)]

            # data extraction
            for batch_number, weights in enumerate(matrix_list):
                for weight_index in range(nb_weight):
                    graph[weight_index][batch_number] = weights[weight_index]

            # plotting
            for weight in graph:
                plt.plot(weight)

    def get_best_input(self, dest_index):

        connections = [connect for connect in self.target if connect.dest_e.index == dest_index]
        best_kernel = np.ndarray(self.target[0].weights.kernel_size)

        for row in range(best_kernel.shape[0]):
            for col in range(best_kernel.shape[1]):
                mu_on = []
                for connect in connections:
                    w = connect.weights.matrix.get_kernel()[(row, col)]
                    if w > 0.5:
                        mu_on.append(connect.source_e.neuron_list[0].mu)
                best_kernel[row, col] = np.clip(np.mean(mu_on) if mu_on else 0, 0, 255)
        return best_kernel

    def print_best_input(self, nb_layer):
        for dest_index in range(nb_layer):
            plt.figure()
            # TODO: fix, bug
            mat = self.get_best_input(dest_index)
            norm = colors.Normalize(vmin=0, vmax=255)
            plt.imshow(mat, cmap='gray', norm=norm)
        plt.show()

    def print(self):
        data = []
        for index, con in enumerate(self.target):
            data.append(con.probed_values[-1].to_dense().tolist()[0])
            print(['%.4f' % elem for elem in data[-1]])


class NeuronProbe(Probe):
    
    def __init__(self, target, variables):
        super(NeuronProbe, self).__init__(target, variables)
        if isinstance(target, Bloc):
            self.target = []
            for ens in target.ensemble_list:
                self.target += ens.neuron_list
        elif isinstance(target, Ensemble):
            self.target = target.neuron_list
        elif isinstance(target, NeuronType):
            self.target = [target]
        elif isinstance(target, list) and all(isinstance(n, NeuronType) for n in target):
            self.target = target
        elif isinstance(target, np.ndarray):
            neuron_array = target.flatten()
            if all(isinstance(n, NeuronType) for n in neuron_array):
                self.target = neuron_array
        else:
            raise Exception("wrong type given to probe target")

        for neuron in self.target:
            for var in self.var:
                neuron.add_probe(self, var)

    def get_data(self, variable):
        if variable not in self.var:
            print("no probe set for {}".format(variable))
            return
        values = []
        for neuron in self.target:
            values.append(neuron.probed_values[variable])
        return values

    def plot(self, variables):
        if isinstance(variables, str):
            var_list = [variables]
        else:
            var_list = variables
        for variable in var_list:
            if variable not in self.var:
                print("no probe set for {}".format(variable))
                return
            if variable == 'spike_out':
                self.plot_spike_out()
            else:
                self.plot_variable(variable)

    def plot_variable(self, variable):
        if variable not in self.var:
            print("no probe set for {}".format(variable))
            return

        fig = plt.figure()
        plt.xlabel('time')
        values = self.get_data(variable)
        for i, neuron in enumerate(values):
            plt.ylabel(variable)
            plt.plot(*zip(*values[i]))
        return fig

    def plot_spike_out(self):
        if 'spike_out' not in self.var:
            print("no probe set for spike_out")
            return

        fig = plt.figure()
        plt.xlabel('time')
        plt.ylabel('spike out')
        plt.grid(axis='both')
        values = self.get_data('spike_out')
        for i, spike_times in enumerate(values):
            plt.scatter(spike_times, [i] * len(spike_times), marker='.')
        return fig

