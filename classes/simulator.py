# import logging as log
# from multiprocessing import Process, Queue
import multiprocessing as mp
import pickle
from .connection import *
from .neuron import NeuronType
from .layer import Ensemble
from .encoder import Node, Encoder
from .learner import *
from .dataset import *
import sys
import platform
import time
import copy
sys.dont_write_bytecode = True

# if __name__ == '__main__':
class Simulator(object):
    """
    The heart of the software
    Builds the network with the given parameters
    can then be run for a set number of step
    """
    @MeasureTiming('sim_init')
    def __init__(self, model, dataset, dt=0.001, batch_size=1, input_period=float('inf')):
        super(Simulator, self).__init__()
        self.model = model
        self.dataset = dataset
        self.nb_step = 0
        self.input_period = input_period
        self.next_reset = input_period
        self.dt = dt
        self.step_nb = 0
        self.curr_time = 0
        model.build()
        model.set_sim(self)
        self.objects = model.get_all_objects()
        self.ensembles = self.objects[Ensemble]
        self.blocs = self.objects[Bloc]
        self.connections = self.objects[Connection]
        self.nodes = self.objects[Node]
        self.step_time = 0
        self.prop_time = 0
        self.batch_size = batch_size
        self.nb_batches = 0
        self.duration = -1
        self.start = 0
        self.last_time = time.time()
        self.steptimes = []
        self.connections.sort(key=lambda con: con.id)
        Helper.log('Simulator', log.INFO, 'new simulator created')

    @MeasureTiming('sim_run')
    def run(self, duration,  monitor_connection=None, convergence_threshold=0.01):
        self.duration = duration
        Helper.log('Simulator', log.INFO, 'simulation start')
        self.nb_step = int(duration / self.dt)
        Helper.log('Simulator', log.INFO, 'total steps: {0}'.format(self.nb_step))
        self.nb_batches = int(duration) // self.batch_size
        # starts the input nodes
        Helper.log('Simulator', log.INFO, 'nodes init')
        for node in self.nodes:
            node.step()

        self.start = time.time()
        # runs for the specified number of steps
        self.curr_time = 0
        for curr_batch in range(self.nb_batches):
            Helper.log('Simulator', log.DEBUG, 'next batch {0}'.format(curr_batch))
            for curr_input in range(self.batch_size):
                Helper.log('Simulator', log.DEBUG, 'next input {0}'.format(curr_input))
                for curr_step in range(int(self.input_period / self.dt)):
                    self.curr_time += self.dt
                    Helper.log('Simulator', log.DEBUG, 'next step {0}'.format(curr_step))
                    self.step()

                Helper.log('Simulator', log.DEBUG, 'end of input cycle: reset of network and next input')
                self.reset()

                self.steptimes.append(time.time() - self.last_time)
                self.last_time = time.time()
            Helper.log('Simulator', log.DEBUG, 'end of batch: applying learning')
            self.learn()
            self.print_time()

            # if monitor_connection:
                # conv_coeff = monitor_connection.get_convergence()
                # if conv_coeff < convergence_threshold:
                #     converged = True
                #     Helper.log('Simulator', log.INFO, 'Connection weight converged, ending simulation at step {} '
                #                .format(self.step_nb))
        # if monitor_connection and not converged:
        #     Helper.log('Simulator', log.WARNING, 'Connection weight did not converged, final convergence {} '
        #                .format(conv_coeff))
        end = time.time()

        Helper.log('Simulator', log.INFO, 'simulating ended')
        Helper.log('Simulator', log.INFO, 'network of {0} neurons'.format(NeuronType.nb_neuron))
        Helper.log('Simulator', log.INFO, 'total time of {0}, step: {1}, synapse: {2}'
                   .format(end - self.start, self.step_time, self.prop_time))

    @MeasureTiming('sim_step')
    def step(self):
        """
        for every steps, evaluate inputs, then ensembles, then propagate spikes
        """


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

        Helper.log('Simulator', log.DEBUG, 'end of step {0}'.format(self.step_nb))
        self.step_time += end_ens - start_ens
        self.prop_time += end_con - start_con

    @MeasureTiming('sim_reset')
    def reset(self):

        # reset all neurons and save the spikes
        Helper.log('Simulator', log.DEBUG, 'resetting all ensembles')
        for ens in self.ensembles:
            ens.reset()
        Helper.log('Simulator', log.DEBUG, 'all ensembles reset')

        # apply threshold adaptation on block
        for bloc in self.blocs:
            bloc.apply_threshold_adapt()

        for node in self.nodes:
            node.step()

    def learn(self):
        for ensemble in self.ensembles:
            if ensemble.learner and ensemble.learner.active:
                ensemble.learner.process()

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

        pass

    def print_time(self):
        if __name__ == '__main__':
            time_left = int((time.time() - self.start) / self.curr_time * (self.duration - self.curr_time))
            print('Time {} / {}, {}:{}:{} left '
                  .format(int(self.curr_time),
                          self.duration,
                          int(time_left // 3600),
                          int((time_left // 60) % 60),
                          int(time_left % 60)))

            # self.memory_estimate()

    def plot_steptimes(self):
        plt.figure()
        plt.title("Input process duration")
        plt.plot(self.steptimes)

    def memory_estimate(self):
        print('Estimated memory size: {}'.format(len(pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL))))


class SimulatorMp(Simulator):
    def __init__(self, model, dt=0.01, batch_size=1, input_period=float('inf'), processes=3, dataset=None):
        super(Simulator, self).__init__()
        self.model = model
        self.dataset = dataset
        self.nb_step = 0
        self.input_period = input_period
        self.next_reset = input_period
        self.dt = dt
        self.step_nb = 0
        self.curr_time = 0
        model.build()
        self.objects = model.get_all_objects()
        self.ensembles = self.objects[Ensemble]
        self.blocs = self.objects[Bloc]
        self.connections = self.objects[Connection]
        self.nodes = self.objects[Node]
        self.step_time = 0
        self.prop_time = 0
        self.batch_size = batch_size
        self.nb_batches = 0
        self.duration = -1
        self.start = 0
        self.last_time = time.time()
        self.steptimes = []

        self.processes = processes
        # init multiprocess
        Helper.log('Simulator', log.INFO, 'Init multiprocess')
        if platform.system() == 'Windows':
            mp.set_start_method('spawn')
        else:
            mp.set_start_method('fork')
        self.workers = []
        self.pipes = []
        self.split = [0 for _ in range(self.processes)]
        self.copies = []
        self.connections.sort(key=lambda con: con.id)
        for exp in range(self.batch_size):
            self.pipes.append(mp.Pipe())
            self.split[exp % self.processes] += 1

    @MeasureTiming('sim_run')
    def run(self, duration,  monitor_connection=None, convergence_threshold=0.01):
        self.duration = duration
        n_batches = int(duration // self.batch_size)

        Helper.log('Simulator', log.INFO, 'simulation start')
        self.nb_step = int(duration / self.dt)
        Helper.log('Simulator', log.INFO, 'total steps: {0}'.format(self.nb_step))

        # starts the input nodes
        Helper.log('Simulator', log.INFO, 'nodes init')
        # for node in self.nodes:
        #    node.step()

        self.start = time.time()
        # runs for the specified number of steps
        self.workers = []
        for worker_id, worker_load in enumerate(self.split):

            # self.copies.append(copy.deepcopy(self.model))
            data = []
            labels = []
            for _ in range(self.split[worker_id]):
                data.append(self.dataset.next())
                # labels.append(self.dataset.labels[self.dataset.index])

            self.workers.append(mp.Process(target=self.mp_run,
                                           args=(self.pipes[worker_id][1],
                                                 self.model,
                                                 data,
                                                 # labels,
                                                 self.dt,
                                                 self.input_period,
                                                 worker_id,
                                                 )
                                           )
                                )
            self.workers[worker_id].start()

        for batch in range(n_batches):
            Helper.log('Simulator', log.DEBUG, 'next batch {0}'.format(batch))
            # print("next batch")
            # self.print_time()
            # Setup workers


            Helper.log('Simulator', log.INFO, 'All workers sent')
            # update when worker finished
            finished = 0
            all_updates = {}
            while finished < self.processes:
                for worker_id, worker in enumerate(self.workers):
                    if self.pipes[worker_id][0].poll():
                        Helper.log('Simulator', log.INFO, 'worker {} finished, gathering data')
                        update = self.pipes[worker_id][0].recv()
                        for attr, value in update.items():
                            self.connections[attr[0]].update_weight(attr[1], attr[2], value)
                        all_updates = {k: all_updates.get(k, 0) + update.get(k, 0) for k in set(all_updates)
                                       | set(update)}  # merge sum dicts
                        finished += 1

            for worker_id, worker_load in enumerate(self.split):

                # self.copies.append(copy.deepcopy(self.model))
                data = []
                labels = []
                for _ in range(self.split[worker_id]):
                    data.append(self.dataset.next())

                self.pipes[worker_id][0].send([all_updates, data])

            for con in self.connections:
                con.probe()

            self.curr_time = (1 + batch) * self.batch_size * self.input_period
            self.print_time()

        for worker in self.workers:
            worker.kill()

    @staticmethod
    def mp_run(pipe, model, data, dt, input_period, id):
        # print("new worker " + str(id))
        Helper.log('Simulator', log.INFO, 'new worker init')
        my_model = copy.deepcopy(model)
        dataset = Dataset()
        sim = Simulator(model=my_model, dataset=dataset, dt=dt, input_period=input_period)
        while True:
            dataset.index = 0
            dataset.data = data
            sim.run(duration=len(data)*input_period)
            Helper.log('Simulator', log.INFO, 'worker done, extracting delta')
            out = {}
            for ens in my_model.objects[Ensemble]:
                if ens.learner:
                    out = {k: out.get(k, 0) + ens.learner.updates.get(k, 0) for k in set(out) | set(ens.learner.updates)}  # merge sum dicts
            # print('worker {} finished simulating, sending updates'.format(id))
            pipe.send(out)
            # print('data sent, waiting updates')
            update = pipe.recv()
            # print('update received')
            for attr, value in update[0].items():
                sim.connections[attr[0]].update_weight(attr[1], attr[2], value)
            data = update[1]
            sim.flush()
        # print("worker done")

    def print_time(self):
        # if __name__ == '__main__':
        time_left = int((time.time() - self.start) / self.curr_time * (self.duration - self.curr_time))
        print('Time {} / {}, {}:{}:{} left '
              .format(int(self.curr_time),
                      self.duration,
                      int(time_left // 3600),
                      int((time_left // 60) % 60),
                      int(time_left % 60)))
