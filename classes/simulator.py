import logging as log
import pickle
from .connection import Connection
from .neuron import NeuronType
from .layer import Ensemble
from .encoder import Node
from .base import Helper
import sys
import time
sys.dont_write_bytecode = True


class Simulator(object):
    """docstring for simulator."""

    def __init__(self, model, dt=0.001, batch_size=1, input_period=float('inf')):
        super(Simulator, self).__init__()
        self.model = model
        self.nb_step = 0
        self.input_period = input_period
        self.next_reset = input_period
        Helper.dt = dt
        Helper.time = 0
        Helper.input_period = input_period
        model.build()
        self.objects = model.get_all_objects()
        self.ensembles = self.objects[Ensemble]
        self.connections = self.objects[Connection]
        self.nodes = self.objects[Node]
        self.spike_list = []
        self.step_time = 0
        self.prop_time = 0
        self.batch_size = batch_size
        Helper.log('Simulator', log.INFO, 'new simulator created')

    def run(self, duration):
        Helper.log('Simulator', log.INFO, 'simulation start')
        start = time.time()
        self.nb_step = int(duration / Helper.dt)
        Helper.log('Simulator', log.INFO, 'total steps: {0}'.format(self.nb_step))
        # shares the spike register with all the axons
        Helper.log('Simulator', log.INFO, 'nodes init')
        # starts the inputs
        for node in self.nodes:
            node.step()
        # runs for the specified number of steps
        for i in range(self.nb_step):
            Helper.log('Simulator', log.DEBUG, 'next step {0}'.format(i+1))
            if Helper.time >= self.next_reset:
                Helper.log('Simulator', log.DEBUG, 'end of input cycle: reset of network and next input')
                self.reset()
                self.next_reset += self.input_period
                for node in self.nodes:
                    node.step()

            self.step()
        # self.reset()
        end = time.time()

        Helper.log('Simulator', log.INFO, 'simulating ended')
        Helper.log('Simulator', log.INFO, 'network of {0} neurons'.format(NeuronType.nb_neuron))
        Helper.log('Simulator', log.INFO, 'total time of {0}, step: {1}, synapse: {2}'
                   .format(end - start, self.step_time, self.prop_time))

        Connection.flush()
        Node.flush()
        Ensemble.flush()
        Helper.reset()

    def step(self):
        """ for every steps, evaluate inputs, then ensembles,
        then propagate spikes """
        # TODO:  progress bar
        Helper.step()
        # print("{:.4f}".format(Helper.time))
        Helper.log('Simulator', log.DEBUG, 'simulating ensembles')
        start = time.time()
        for ens in self.ensembles:
            ens.step()
        Helper.log('Simulator', log.DEBUG, 'all ensembles simulated')
        mid = time.time()
        Helper.log('Simulator', log.DEBUG, 'simulating connections')
        self.propagate_all()
        Helper.log('Simulator', log.DEBUG, 'all connections propagated')
        end = time.time()
        Helper.log('Simulator', log.DEBUG, 'end of step {0}'.format(Helper.step_nb))
        self.step_time += mid - start
        self.prop_time += end - mid

    def reset(self):
        # TODO: reset connection if active
        Helper.log('Simulator', log.DEBUG, 'resetting all ensembles')
        for ens in self.ensembles:
            ens.reset()
        Helper.log('Simulator', log.DEBUG, 'all ensembles reset')

        if Helper.input_index == self.batch_size:
            Helper.log('Simulator', log.INFO, 'end of batch: updating matrices')
            for ensemble in self.ensembles:
                if ensemble.learner:
                    # ensemble.step()
                    ensemble.learner.process()
            Helper.input_index = 0
        else:
            Helper.log('Simulator', log.INFO, 'next input')
            Helper.input_index += 1

    def register_spike(self, axon):
        # print("reg")
        # print(len(self.spike_list))
        self.spike_list.append(axon)
        # print(len(self.spike_list))

    def propagate_all(self):
        # print(len(self.spike_list))
        for con in self.connections:
            if con.active:
                # Helper.log('Simulator', log.DEBUG, 'propagating through connection {0}'.format(con.id))
                con.step()

    def save(self, file):
        with open(file, 'wb') as savefile:
            data = []
            Helper.log('Simulator', log.INFO, 'saving weights ...')
            for con in self.connections:
                if con.active:
                    data.append((con.id, con.weights.matrix))
            pickle.dump(data, savefile, pickle.HIGHEST_PROTOCOL)
            Helper.log('Simulator', log.INFO, 'done')

    def load(self, file):
        with open(file, 'rb') as savefile:
            Helper.log('Simulator', log.INFO, 'loading weights ...')
            data = pickle.load(savefile)
            for con in data:
                self.connections[con[0]].weights.matrix = con[1]
            Helper.log('Simulator', log.INFO, 'done')
