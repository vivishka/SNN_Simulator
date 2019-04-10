
from base import SimationObject
import matplotlib.pyplot as plt
# import numpy as np
import sys
sys.dont_write_bytecode = True


class Probe(SimationObject):
    """docstring for Probe."""

    objects = []

    def __init__(self, obj):
        super(Probe, self).__init__()
        Probe.objects.append(self)
        self.values = []
        obj.set_probe(self, 'voltage')

    def send_value(self, value):
        self.values.append(value)

    def plot(self):
        plt.figure()
        plt.plot(self.values)
        plt.show()
