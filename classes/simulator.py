import logging as log
# from multiprocessing import Process, Queue
import multiprocessing as mp
import pickle
from .connection import *
from .neuron import NeuronType
from .layer import Ensemble
from .encoder import Node, Encoder
from .base import Helper
from .learner import *
import sys
import time
sys.dont_write_bytecode = True


class Simulator(object):
    """
    The heart of the software
    Builds the network with the given parameters
    can then be run for a set number of step
    """
    @MeasureTiming('sim_init')
    def __init__(self, model, dt=0.001, batch_size=1, input_period=float('inf')):
        super(Simulator, self).__init__()
        self.model = model
        self.nb_step = 0
        self.input_period = input_period
        self.next_reset = input_period
        Helper.dt = dt
        Helper.time = 0
        Helper.input_period = input_period
        Helper.batch_size = batch_size
        model.build()
        self.objects = model.get_all_objects()
        self.ensembles = self.objects[Ensemble]
        self.blocs = self.objects[Bloc]
        self.connections = self.objects[Connection]
        self.nodes = self.objects[Node]
        self.step_time = 0
        self.prop_time = 0
        self.batch_size = batch_size
        self.duration = -1
        self.start = 0
        self.last_time = time.time()
        self.steptimes = []

        Helper.log('Simulator', log.INFO, 'new simulator created')

    @MeasureTiming('sim_run')
    def run(self, duration,  monitor_connection=None, convergence_threshold=0.01):
        self.duration = duration
        Helper.log('Simulator', log.INFO, 'simulation start')
        self.nb_step = int(duration / Helper.dt)
        Helper.log('Simulator', log.INFO, 'total steps: {0}'.format(self.nb_step))

        # starts the input nodes
        Helper.log('Simulator', log.INFO, 'nodes init')
        for node in self.nodes:
            node.step()

        converged = False
        self.start = time.time()
        # runs for the specified number of steps
        while Helper.step_nb < self.nb_step and not converged:
            Helper.log('Simulator', log.DEBUG, 'next step {0}'.format(self.nb_step))
            self.step()
            if monitor_connection:
                conv_coeff = monitor_connection.get_convergence()
                if conv_coeff < convergence_threshold:
                    converged = True
                    Helper.log('Simulator', log.INFO, 'Connection weight converged, ending simulation at step {} '
                               .format(Helper.step_nb))
        if monitor_connection and not converged:
            Helper.log('Simulator', log.WARNING, 'Connection weight did not converged, final convergence {} '
                       .format(conv_coeff))
        end = time.time()

        Helper.log('Simulator', log.INFO, 'simulating ended')
        Helper.log('Simulator', log.INFO, 'network of {0} neurons'.format(NeuronType.nb_neuron))
        Helper.log('Simulator', log.INFO, 'total time of {0}, step: {1}, synapse: {2}'
                   .format(end - self.start, self.step_time, self.prop_time))

    @MeasureTiming('sim_step')
    def step(self):
        """
        for every steps, evaluate inputs, then ensembles, then propagate spikes
        also updates the Helper
        """
        # every input period, reset and restart
        if Helper.time >= self.next_reset:
            Helper.log('Simulator', log.DEBUG, 'end of input cycle: reset of network and next input')
            self.reset()
            self.next_reset += self.input_period
            for node in self.nodes:
                node.step()
            time_left = int((time.time() - self.start) / Helper.time * (self.duration - Helper.time))
            print('Time {} / {}, {}:{}:{} left '
                  .format(int(Helper.time),
                          self.duration,
                          int(time_left // 3600),
                          int((time_left // 60) % 60),
                          int(time_left % 60)))
            self.steptimes.append(time.time()-self.last_time)
            self.last_time = time.time()


        Helper.step()

        # Ensembles
        Helper.log('Simulator', log.DEBUG, 'simulating ensembles')
        start_ens = time.time()
        for ens in self.ensembles:
            ens.step()
            if isinstance(ens.bloc, Encoder):
                for neuron in ens.neuron_list:
                    if neuron.active:
                        ens.active_neuron_set.update(ens.neuron_list)
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

    @MeasureTiming('sim_reset')
    def reset(self):

        # reset all neurons and save the spikes
        Helper.log('Simulator', log.DEBUG, 'resetting all ensembles')
        for ens in self.ensembles:
            ens.reset()
        Helper.log('Simulator', log.DEBUG, 'all ensembles reset')

        # apply learner if present
        if Helper.input_index + 1 >= self.batch_size:
            Helper.log('Simulator', log.INFO, 'end of batch: updating matrices')
            for ensemble in self.ensembles:
                if ensemble.learner and ensemble.learner.active:
                    # ensemble.step()
                    ensemble.learner.process()
            Helper.input_index = 0
        else:
            Helper.log('Simulator', log.INFO, 'next input')
            Helper.input_index += 1

        # apply threshold adaptation on block
        for bloc in self.blocs:
            bloc.apply_threshold_adapt()

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
                if not isinstance(con, DiagonalConnection) \
                        and con.mode != 'pooling' \
                        and con.active:
                    Helper.log('Simulator', log.INFO, 'saving weight matrix connection {}'.format(index))
                    Helper.log('Simulator', log.INFO, 'matrix size {}'.format(con.weights.matrix.size))
                    data.append((con.id, con.weights))

            pickle.dump(data, savefile, pickle.HIGHEST_PROTOCOL)
            Helper.log('Simulator', log.INFO, 'done')

    def load(self, file):
        with open(file, 'rb') as savefile:
            Helper.log('Simulator', log.INFO, 'loading weights ...')
            data = pickle.load(savefile)
            for con in data:
                for receptor in self.connections:
                    Helper.log('Simulator', log.INFO, 'loading weight matrix connection {}'.format(con[0]))
                    # Helper.log('Simulator', log.INFO, 'matrix size {}'.format(con[1].matrix.size))
                    if receptor.id == con[0]:
                        receptor.weights = con[1]
                        break



                # if not isinstance(self.connections[con[0]], DiagonalConnection) \
                #         and self.connections[con[0]].mode != 'pooling' \
                #         and self.connections[con[0]].active:

            Helper.log('Simulator', log.INFO, 'done')

    def flush(self):

        Helper.reset()
        self.next_reset = self.input_period

    def plot_steptimes(self):
        plt.figure()
        plt.title("Input process duration")
        plt.plot(self.steptimes)


class SimulatorMp(Simulator):
    def __init__(self, model, dt=0.01, batch_size=1, input_period=float('inf'), processes=2):
        super(SimulatorMp, self).__init__(model, dt, batch_size, input_period)
        self.processes = processes

        # init multiprocess
        Helper.log('Simulator', log.INFO, 'Init multiprocess')
        mp.set_start_method('spawn')
        self.mpqueues = []
        self.learning_split = []
        for proc in range(self.processes):
            self.mpqueues.append(mp.SimpleQueue())

            # Compute learner repartition
            if self.model.n_learners % self.processes == 0:
                self.learning_split.append(self.model.n_learners / self.processes)
            elif proc == self.processes - 1:
                self.learning_split.append(self.model.n_learners % self.processes)
            else:
                self.learning_split.append(self.model.n_learners // self.processes)

    @MeasureTiming('sim_reset')
    def reset(self):

        # reset all neurons and save the spikes
        Helper.log('Simulator', log.DEBUG, 'resetting all ensembles')
        for ens in self.ensembles:
            ens.reset()
        Helper.log('Simulator', log.DEBUG, 'all ensembles reset')

        # apply learner if present
        if Helper.input_index + 1 >= self.batch_size:
            Helper.log('Simulator', log.DEBUG, 'end of batch: updating matrices')
            Helper.log('Simulator', log.DEBUG, 'Starting workers with repartition {}'.format(self.learning_split))
            # Start workers
            proc_index = 0
            learner_index = 0
            learning_data = []
            workers = []
            learners = []
            for ensemble in self.ensembles:
                if ensemble.learner and ensemble.learner.active:
                    if learner_index < self.learning_split[proc_index]-1:
                        learner_index += 1
                        learning_data.append(ensemble.learner.get_mp_data())

                    else:

                        learning_data.append(ensemble.learner.get_mp_data())
                        workers.append(mp.Process(target=type(ensemble.learner).process_mp,
                                                  args=(self.mpqueues[proc_index], learning_data.copy())))  # maybe copy the data ?
                        Helper.log('Simulator', log.INFO, 'Worker {} full of {} tasks, starting...'.format(proc_index+1, learner_index))
                        workers[-1].start()
                        proc_index += 1
                        learner_index = 0
                        learning_data = []

                    learners.append(ensemble.learner)
            Helper.log('Simulator', log.DEBUG, 'All workers sent, waiting for join')
            # Wait workers
            for worker in workers:
                worker.join()
                Helper.log('Simulator', log.DEBUG, 'Worker joined')
            Helper.log('Simulator', log.DEBUG, 'All workers joined, updating objects')
            # Update objects
            learner_index = 0
            for queue in self.mpqueues:
                data = queue.get()
                for learning_data in data:
                    learners[learner_index].update(learning_data)
                    learner_index += 1
            Helper.log('Simulator', log.DEBUG, 'Learning done')
            Helper.input_index = 0

        else:
            Helper.log('Simulator', log.INFO, 'next input')
            Helper.input_index += 1

        # apply threshold adaptation on block
        for bloc in self.blocs:
            bloc.apply_threshold_adapt()