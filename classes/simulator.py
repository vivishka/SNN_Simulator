
# from neuron import Neuron
from .connection import Connection
from .ensemble import Ensemble
from .node import Reset, Node
# from node import Node
from .base import Helper
import sys
sys.dont_write_bytecode = True


class Simulator(object):
    """docstring for simulator."""

    def __init__(self, model, dt):
        super(Simulator, self).__init__()
        self.model = model
        self.nb_step = 0
        Helper.dt = dt
        Helper.time = 0
        model.build()
        self.objects = model.get_all_objects()
        self.ensembles = self.objects[Ensemble]
        self.connections = self.objects[Connection]
        self.nodes = self.objects[Node]
        self.input_reset = self.objects[Reset]
        self.spike_list = []

    def run(self, time):
        # shares the spike register with all the axons
        for connect in self.connections:
            connect.set_notifier(self.register_spike)

        # shares the reset function with the reset objects
        for reset in self.input_reset:
            reset.set_reset_funt(self.reset)

        # this or for all axons ?
        self.nb_step = int(time / Helper.dt)
        for i in range(self.nb_step):
            self.step()
            # printProgressBar(i, self.nb_step)

    def step(self):
        """ for every steps, evaluate ensembles, then propagate spikes """
        # TODO:  progres bar
        Helper.step()
        # print("{:.4f}".format(self.time))
        for reset in self.input_reset:
            reset.step()
        for node in self.nodes:
            node.step()
        for ens in self.ensembles:
            ens.step()
        self.propagate_all()

    def reset(self):
        for ens in self.ensembles:
            ens.reset()

    def register_spike(self, axon):
        self.spike_list.append(axon)

    def propagate_all(self):
        for axon in self.spike_list:
            axon.propagate_spike()
        self.spike_list = []
