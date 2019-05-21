import logging as log
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
        Helper.log('Simulator', log.INFO, 'new simulator created')

    def run(self, duration):
        Helper.log('Simulator', log.INFO, 'simulation start')
        start = time.time()
        self.nb_step = int(duration / Helper.dt)
        Helper.log('Simulator', log.INFO, 'total steps: {0}'.format(self.nb_step))
        # shares the spike register with all the axons
        # for connect in self.connections:
        #     connect.set_notifier(self.register_spike)
        Helper.log('Simulator', log.INFO, 'nodes init')
        # starts the inputs
        for node in self.nodes:
            node.step()
        # runs for the specified number of steps
        for i in range(self.nb_step):
            Helper.log('Simulator', log.INFO, 'next step {0}'.format(i+1))
            if Helper.time >= self.next_reset:
                Helper.log('Simulator', log.INFO, 'end of input cycle: reset of network and next input')
                self.reset()
                self.next_reset += self.input_period
                for node in self.nodes:
                    node.step()
            self.step()

        end = time.time()

        Helper.log('Simulator', log.INFO, 'simulating ended')
        Helper.log('Simulator', log.INFO, 'network of {0} neurons'.format(NeuronType.nb_neuron))
        Helper.log('Simulator', log.INFO, 'total time of {0}, step: {1}, synapse: {2}'
                   .format(end - start, self.step_time, self.prop_time))

    def step(self):
        """ for every steps, evaluate inputs, then ensembles,
        then propagate spikes """
        # TODO:  progress bar
        Helper.step()
        # print("{:.4f}".format(Helper.time))
        Helper.log('Simulator', log.INFO, 'simulating ensembles')
        start = time.time()
        for ens in self.ensembles:
            ens.step()
        Helper.log('Simulator', log.INFO, 'all ensembles simulated')
        # self.step_ens()
        mid = time.time()
        Helper.log('Simulator', log.INFO, 'simulating connections')
        self.propagate_all()
        Helper.log('Simulator', log.INFO, 'all connections propagated')
        end = time.time()
        Helper.log('Simulator', log.INFO, 'end of step {0}'.format(Helper.step_nb))
        self.step_time += mid - start
        self.prop_time += end - mid

    def step_ens(self): # TODO: obsolete ?
        processes = [mp.Process(target=ens.step) for ens in self.ensembles]
        for p in processes:
            p.start()

        # Exit the completed processes
        for p in processes:
            p.join()

    def reset(self):
        # TODO: reset connection if active
        Helper.log('Simulator', log.INFO, 'resetting all ensembles')
        for ens in self.ensembles:
            ens.reset()
        Helper.log('Simulator', log.INFO, 'all ensembles reset')

    def register_spike(self, axon):
        # print("reg")
        # print(len(self.spike_list))
        self.spike_list.append(axon)
        # print(len(self.spike_list))

    def propagate_all(self):
        # print(len(self.spike_list))
        for con in self.connections:
            if con.active:
                Helper.log('Simulator', log.DEBUG, 'propagating through connection {0}'.format(con.id))
                con.step()

