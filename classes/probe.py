
# import logging as log
import numpy as np
from .layer import Ensemble
from .neuron import NeuronType
from .connection import Connection
import matplotlib.pyplot as plt
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

    def plot(self, connection_index='all', neuron_index='all'):
        plt.figure()
        graph = None
        target_range = []
        if connection_index == 'all':
            target_range = range(len(self.target))
        elif isinstance(connection_index, int):
            target_range = [connection_index]
        elif isinstance(connection_index, list):
            target_range = connection_index

        for connect_index in target_range:

            values = self.get_data(connection_index=connect_index)
            if neuron_index == 'all':
                # all weights of the matrix
                graph = np.ndarray((len(values), values[0].size))
                for t, batch_matrix in enumerate(values):
                    i = 0
                    for row in batch_matrix:
                        for neuron_weight in row:
                            w = neuron_weight[2]
                            graph[t, i] = w
                            i += 1

                # row: weight index, col: time
                graph = graph.transpose()

            elif isinstance(neuron_index, int):
                # multiple lines
                # row: time, col: weight index
                graph = np.ndarray((len(values), len(values[0][neuron_index])))
                for t, batch_matrix in enumerate(values):
                    row = batch_matrix[neuron_index]
                    for i, neuron_weight in enumerate(row):
                        w = neuron_weight[2]
                        graph[t, i] = w

                # row: weight index, col: time
                graph = graph.transpose()

            elif isinstance(neuron_index, tuple):
                graph = [[]]
                for batch_matrix in values:
                    graph[0].append(batch_matrix[neuron_index])

            for weight in graph:
                plt.plot(weight)

    def print(self):
        data = []
        for index, con in enumerate(self.target):
            data.append(con.probed_values[-1].to_dense().tolist()[0])
            print(['%.4f' % elem for elem in data[-1]])


class NeuronProbe(Probe):
    
    def __init__(self, target, variables):
        super(NeuronProbe, self).__init__(target, variables)

        if isinstance(target, Ensemble):
            neuron_list = target.neuron_list
        elif isinstance(target, NeuronType):
            neuron_list = [target]
        elif isinstance(target, list) and all(isinstance(n, NeuronType) for n in target):
            neuron_list = target
        else:
            raise Exception("wrong type given to probe target")

        self.target = neuron_list
        for neuron in neuron_list:
            for var in self.var:
                neuron.add_probe(self, var)

    def get_data(self, variable):
        if variable not in self.var:
            print("no probe set for {}".format(variable))
            return
        values = []
        for i, neuron in enumerate(self.target):
            values.append(neuron.probed_values[variable])
        return values

    def plot(self, variable):
        if variable not in self.var:
            print("no probe set for {}".format(variable))
            return
        fig = plt.figure()
        plt.xlabel('time')
        colors = ['k', 'r', 'b', 'g', 'm']
        values = self.get_data(variable)
        for i, neuron in enumerate(values):
            plt.ylabel(variable)
            plt.plot(*zip(*values[i]), color=colors[i % 5])
        return fig
