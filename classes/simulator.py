import logging as log
import pickle
from .connection import *
from .neuron import NeuronType
from .layer import Ensemble
from .encoder import Node
from .base import Helper
import sys
import time
sys.dont_write_bytecode = True


class Simulator(object):
    """
    The heart of the software
    Builds the network with the given parameters
    can then be run for a set number of step
    """

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
        self.step_time = 0
        self.prop_time = 0
        self.batch_size = batch_size
        Helper.log('Simulator', log.INFO, 'new simulator created')

    def run(self, duration):
        Helper.log('Simulator', log.INFO, 'simulation start')
        start = time.time()
        self.nb_step = int(duration / Helper.dt)
        Helper.log('Simulator', log.INFO, 'total steps: {0}'.format(self.nb_step))

        # starts the input nodes
        Helper.log('Simulator', log.INFO, 'nodes init')
        for node in self.nodes:
            node.step()

        # runs for the specified number of steps
        for i in range(self.nb_step):
            Helper.log('Simulator', log.DEBUG, 'next step {0}'.format(i+1))

            # every input period, reset and restart
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

    def step(self):
        """
        for every steps, evaluate inputs, then ensembles, then propagate spikes
        also updates the Helper
        """

        Helper.step()
        # Ensembles
        Helper.log('Simulator', log.DEBUG, 'simulating ensembles')
        start_ens = time.time()
        for ens in self.ensembles:
            ens.step()
        end_ens = time.time()
        Helper.log('Simulator', log.DEBUG, 'all ensembles simulated')

        # Connections
        Helper.log('Simulator', log.DEBUG, 'simulating connections')
        start_con = time.time()
        self.propagate_all()
        end_con = time.time()
        Helper.log('Simulator', log.DEBUG, 'all connections propagated')

        Helper.log('Simulator', log.DEBUG, 'end of step {0}'.format(Helper.step_nb))
        self.step_time += end_ens - start_ens
        self.prop_time += end_con - start_con

    def reset(self):

        # reset all neurons and save the spikes
        Helper.log('Simulator', log.DEBUG, 'resetting all ensembles')
        for ens in self.ensembles:
            ens.reset()
        Helper.log('Simulator', log.DEBUG, 'all ensembles reset')

        # apply learner if present
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

    def propagate_all(self):
        """ Steps all the active connections """
        for con in self.connections:
            if con.active:
                # Helper.log('Simulator', log.DEBUG, 'propagating through connection {0}'.format(con.id))
                con.step()

    def save(self, file):
        with open(file, 'wb') as savefile:
            data = []
            Helper.log('Simulator', log.INFO, 'saving weights ...')
            for index, con in enumerate(self.connections):
                if not isinstance(con, DiagonalConnection) and con.active:
                    Helper.log('Simulator', log.INFO, 'saving weight matrix connection {}'.format(index))
                    Helper.log('Simulator', log.INFO, 'matrix size {}'.format(con.weights.matrix.size))
                    data.append((index, con.weights.matrix))
            pickle.dump(data, savefile, pickle.HIGHEST_PROTOCOL)
            Helper.log('Simulator', log.INFO, 'done')

    def load(self, file):
        with open(file, 'rb') as savefile:
            Helper.log('Simulator', log.INFO, 'loading weights ...')
            data = pickle.load(savefile)
            for con in data:
                if not isinstance(self.connections[con[0]], DiagonalConnection) and self.connections[con[0]].active:
                    Helper.log('Simulator', log.INFO, 'loading weight matrix connection {}'.format(con[0]))
                    Helper.log('Simulator', log.INFO, 'matrix size {}'.format(con[1].size))
                    self.connections[con[0]].weights.matrix = con[1]
            Helper.log('Simulator', log.INFO, 'done')

    def flush(self):

        Helper.reset()
        self.next_reset = self.input_period