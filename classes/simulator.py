
from .connection import Connection
from .neuron import NeuronType
from .layer import Ensemble
from .encoder import Reset, Node
from .base import Helper
import multiprocessing as mp
import sys
import time
sys.dont_write_bytecode = True


class Simulator(object):
    """docstring for simulator."""

    def __init__(self, model, dt=0.001, input_period=float('inf')):
        super(Simulator, self).__init__()
        self.model = model
        self.nb_step = 0
        self.input_period = input_period
        self.next_reset = input_period
        Helper.dt = dt
        Helper.time = 0
        model.build()
        self.objects = model.get_all_objects()
        self.ensembles = self.objects[Ensemble]
        self.connections = self.objects[Connection]
        self.nodes = self.objects[Node]
        self.spike_list = []
        self.step_time = 0
        self.prop_time = 0

    def run(self, duration):
        start = time.time()
        # shares the spike register with all the axons
        # for connect in self.connections:
        #     connect.set_notifier(self.register_spike)

        # starts the inputs
        for node in self.nodes:
            node.step()
        # runs for the specified number of steps
        self.nb_step = int(duration / Helper.dt)
        for i in range(self.nb_step):
            if Helper.time >= self.next_reset:
                self.reset()
                print("reset")
                self.next_reset += self.input_period
                for node in self.nodes:
                    node.step()
            self.step()

        end = time.time()
        print(
            "network of {0} neurons"
                .format(NeuronType.nb_neuron))
        print(
            "total time of {0}, step: {1}, synapse: {2}"
                .format(end - start, self.step_time, self.prop_time))

    def step(self):
        """ for every steps, evaluate inputs, then ensembles,
        then propagate spikes """
        # TODO:  progress bar
        Helper.step()
        # print("{:.4f}".format(Helper.time))

        start = time.time()
        for ens in self.ensembles:
            ens.step()
        # self.step_ens()
        mid = time.time()
        self.propagate_all()
        end = time.time()
        self.step_time += mid - start
        self.prop_time += end - mid

    def step_ens(self):
        processes = [mp.Process(target=ens.step) for ens in self.ensembles]
        for p in processes:
            p.start()

        # Exit the completed processes
        for p in processes:
            p.join()

    def reset(self):
        # TODO: reset connection if active
        for ens in self.ensembles:
            ens.reset()

    def register_spike(self, axon):
        # print("reg")
        # print(len(self.spike_list))
        self.spike_list.append(axon)
        # print(len(self.spike_list))

    def propagate_all(self):
        # print(len(self.spike_list))
        for con in self.connections:
            con.step()

