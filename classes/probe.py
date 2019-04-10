
from base import SimulationObject
from ensemble import Ensemble
import matplotlib.pyplot as plt
# import numpy as np
import sys
sys.dont_write_bytecode = True


class Probe(SimulationObject):
    """docstring for Probe."""

    objects = []

    def __init__(self, obj, variable):
        super(Probe, self).__init__()
        Probe.objects.append(self)
        self.size = 1
        if isinstance(obj, Ensemble):
            self.size = obj.size
            # TODO: handle 2D
        obj.set_probe(self, variable)
        self.values = [[] for i in range(self.size)]

    def send_value(self, index, value):
        self.values[index].append(value)

    def plot(self):
        plt.figure()
        for graph in self.values:
            plt.plot(graph)
        plt.show()
