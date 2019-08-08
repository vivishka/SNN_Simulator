# import logging as log
# from multiprocessing import Process, Queue
import multiprocessing as mp
import pickle
from .connection import *
from .neuron import NeuronType
from .layer import Ensemble
from .encoder import Encoder
# from .learner import *
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
        self.ensembles = []
        self.blocs = []
        self.connections = []
        self.encoders = []
        self.nb_step = 0
        self.input_period = input_period
        self.next_reset = input_period
        self.dt = dt
        self.step_nb = 0
        self.curr_time = 0
        self.step_time = 0
        self.prop_time = 0
        self.batch_size = batch_size
        self.nb_batches = 0
        self.curr_batch = 0
        self.duration = -1
        self.start = 0
        self.last_time = time.time()
        self.steptimes = []
        self.time_enabled = False
        self.autosave = None
        Helper.log('Simulator', log.INFO, 'new simulator created')

    def build(self):
        self.model.build()
        self.model.set_sim(self)
        self.ensembles = self.model.objects[Ensemble]
        self.blocs = self.model.objects[Bloc]
        self.connections = self.model.objects[Connection]
        self.encoders = self.model.objects[Encoder]
        self.connections.sort(key=lambda con: con.id)

    @MeasureTiming('sim_run')
    def run(self, duration,  monitor_connection=None, convergence_threshold=0.01):
        self.build()
        self.duration = duration
        Helper.log('Simulator', log.INFO, 'simulation start')
        self.nb_step = int(duration / self.dt)
        Helper.log('Simulator', log.INFO, 'total steps: {0}'.format(self.nb_step))
        self.nb_batches = int(duration) // self.batch_size

        self.start = time.time()
        if self.time_enabled:
            Helper.print_progress(0, self.nb_batches, 'Simulation progress: ', 'complete, 0:0:0 left', bar_length=30)
        # runs for the specified number of steps
        self.curr_time = 0
        for curr_batch in range(self.nb_batches):
            Helper.log('Simulator', log.DEBUG, 'next batch {0}'.format(curr_batch))
            self.curr_batch = curr_batch + 1
            for curr_input in range(self.batch_size):
                self.start_cycle()
                Helper.log('Simuslator', log.DEBUG, 'next input {0}'.format(curr_input))
                for curr_step in range(int(self.input_period / self.dt)):
                    self.curr_time += self.dt
                    Helper.log('Simulator', log.DEBUG, 'next step {0}'.format(curr_step))
                    self.step()

                Helper.log('Simulator', log.DEBUG, 'end of input cycle: reset of network and next input')
                self.reset()

                self.steptimes.append(time.time() - self.last_time)
                self.last_time = time.time()
            if self.autosave:
                self.save(self.autosave)
            Helper.log('Simulator', log.DEBUG, 'end of batch: applying learning')
            self.learn()
            self.print_time()

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

    def start_cycle(self):
        if isinstance(self.dataset, Dataset):
            value = self.dataset.next()
        else:
            value = self.dataset

        for enc in self.encoders:
            enc.encode(value)

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
            data = {
                'connection': [],
                'threshold': {}
            }
            # save connection weights
            Helper.log('Simulator', log.INFO, 'saving weights ...')
            for index, con in enumerate(self.connections):
                if not isinstance(con, DiagonalConnection) \
                        and con.mode != 'pooling' \
                        and con.active:
                    # Helper.log('Simulator', log.INFO, 'saving weight matrix connection {}'.format(index))
                    # Helper.log('Simulator', log.INFO, 'matrix size {}'.format(con.weights.matrix.size))
                    data['connection'].append((con.id, con.weights))

            # save thresholds
            for ens in self.ensembles:
                if ens.neuron_list and hasattr(ens.neuron_list[0], 'threshold'):
                    data['threshold'][ens.id] = ens.neuron_list[0].threshold

            pickle.dump(data, savefile, pickle.HIGHEST_PROTOCOL)
            Helper.log('Simulator', log.INFO, 'done')

    def load(self, file):
        self.build()
        ext = file.split('.')[-1]
        if ext != 'w':
            Helper.log('Simulator', log.ERROR, 'unknown extension for file {}'.format(file))
            raise ValueError('unknown extension')

        with open(file, 'rb') as savefile:
            Helper.log('Simulator', log.INFO, 'loading weights ...')
            data = pickle.load(savefile)

            # load connections
            for con in data['connection']:
                for receptor in self.connections:
                    Helper.log('Simulator', log.INFO, 'loading weight matrix connection {}'.format(con[0]))
                    # Helper.log('Simulator', log.INFO, 'matrix size {}'.format(con[1].matrix.size))
                    if receptor.id == con[0]:
                        receptor.weights = con[1]
                        break

            # load thresholds
            for ens_id, threshold in data['threshold'].items():
                for ens in self.ensembles:
                    if ens.id == ens_id:
                        for neuron in ens.neuron_list:
                            neuron.threshold = threshold

            Helper.log('Simulator', log.INFO, 'loading done')

    def flush(self):
        pass

    def print_time(self):
        if self.time_enabled:
            time_left = int((time.time() - self.start) / self.curr_time * (self.duration - self.curr_time))
            # print('Time {} / {}, {}:{}:{} left '
            #       .format(int(self.curr_time),
            #               self.duration,
            #               int(time_left // 3600),
            #               int((time_left // 60) % 60),
            #               int(time_left % 60)))
            Helper.print_progress(
                self.curr_batch, self.nb_batches,
                'Simulation progress: ', 'complete, {}:{}:{} left'.format(
                    int(time_left // 3600),
                    int((time_left // 60) % 60),
                    int(time_left % 60)), bar_length=30)

            # self.memory_estimate()

    def plot_steptimes(self):
        plt.figure()
        plt.title("Input process duration")
        plt.plot(self.steptimes)

    def memory_estimate(self):
        print('Estimated memory size: {}'.format(len(pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL))))

    def enable_time(self, state):
        self.time_enabled = state


class SimulatorMp(Simulator):
    def __init__(self, model, dt=0.01, batch_size=1, input_period=1, processes=3, dataset=None):
        super(Simulator, self).__init__()
        self.model = model
        self.dataset = dataset
        self.nb_step = 0
        self.input_period = input_period
        self.next_reset = input_period
        self.dt = dt
        self.step_nb = 0
        self.curr_time = 0
        self.step_time = 0
        self.prop_time = 0
        self.batch_size = batch_size
        self.nb_batches = 0
        self.curr_batch = 0
        self.duration = -1
        self.start = 0
        self.last_time = time.time()
        self.steptimes = []
        self.time_enabled = False
        self.autosave = None
        self.processes = processes
        # init multiprocess
        Helper.log('Simulator', log.INFO, 'Init multiprocess')
        try:
            if platform.system() == 'Windows':
                mp.set_start_method('spawn')
            else:
                mp.set_start_method('fork')
        except:
            pass
        self.workers = []
        self.pipes = []
        self.split = [0 for _ in range(self.processes)]
        self.copies = []

        for exp in range(self.batch_size):
            self.pipes.append(mp.Pipe())
            self.split[exp % self.processes] += 1

    def build(self):
        self.model.build()
        # self.model.set_sim(self) do not set sim yet (will be done in worker)
        self.ensembles = self.model.objects[Ensemble]
        self.blocs = self.model.objects[Bloc]
        self.connections = self.model.objects[Connection]
        self.ensembles = self.model.objects[Ensemble]
        self.connections.sort(key=lambda con: con.id)

    @MeasureTiming('sim_run')
    def run(self, duration,  monitor_connection=None, convergence_threshold=0.01):
        self.build()
        self.duration = duration
        self.nb_batches = int(duration // self.batch_size)

        Helper.log('Simulator', log.INFO, 'simulation start')
        self.nb_step = int(duration / self.dt)
        Helper.log('Simulator', log.INFO, 'total steps: {0}'.format(self.nb_step))

        # if self.time_enabled:
        #     Simulator.print_progress(0, self.nb_batches, 'Simulation progress: ', 'complete, 0:0:0 left')
        # runs for the specified number of steps
        self.workers = []
        for worker_id, worker_load in enumerate(self.split):

            data = []
            labels = []
            for _ in range(self.split[worker_id]):
                data.append(self.dataset.next())
                labels.append(self.dataset.labels[self.dataset.index])

            self.workers.append(mp.Process(target=self.mp_run,
                                           args=(self.pipes[worker_id][1],
                                                 self.model,
                                                 data,
                                                 labels,
                                                 self.dt,
                                                 self.input_period,
                                                 worker_id,
                                                 )
                                           )
                                )
            self.workers[worker_id].start()

        self.start = time.time()
        for batch in range(self.nb_batches):
            Helper.log('Simulator', log.DEBUG, 'next batch {0}'.format(batch))
            # print("next batch")
            # self.print_time()
            # Setup workers
            self.curr_batch = batch + 1
            self.curr_time = (1 + batch) * self.batch_size * self.input_period
            # self.print_time()
            self.print_time()
            Helper.log('Simulator', log.INFO, 'All workers sent')
            # update when worker finished
            finished = 0
            all_updates = {}
            while finished < self.processes:
                for worker_id, worker in enumerate(self.workers):
                    if self.pipes[worker_id][0].poll():
                        Helper.log('Simulator', log.INFO, 'worker {} finished, gathering data'.format(id))
                        update = self.pipes[worker_id][0].recv()
                        for attr, value in update.items():
                            self.connections[attr[0]].update_weight(attr[1], attr[2], value)
                        all_updates = {k: all_updates.get(k, 0) + update.get(k, 0) for k in set(all_updates)
                                       | set(update)}  # merge sum dicts
                        finished += 1
                Helper.log('Simulator', log.INFO, 'worker {} finished, gathering data')
                time.sleep(0.1)
            # print('all updates processed: size {}'.format(len(all_updates)))
            for worker_id, worker_load in enumerate(self.split):

                # self.copies.append(copy.deepcopy(self.model))
                data = []
                labels = []
                for _ in range(self.split[worker_id]):
                    data.append(self.dataset.next())
                    if self.dataset.labels:
                        labels.append(self.dataset.labels[self.dataset.index])

                self.pipes[worker_id][0].send([all_updates, (data, labels)])
            if self.autosave:
                self.save(self.autosave)
            for con in self.connections:
                con.probe()

            self.steptimes.append(time.time() - self.last_time)
            self.last_time = time.time()

        for worker in self.workers:
            worker.kill()

    @staticmethod
    def mp_run(pipe, model, data, labels, dt, input_period, id):
        # print("new worker " + str(id))
        Helper.log('Simulator', log.INFO, 'new worker init')
        my_model = copy.deepcopy(model)
        dataset = Dataset()
        sim = Simulator(model=my_model, dataset=dataset, dt=dt, input_period=input_period)
        for con in my_model.objects[Connection]:
            con.is_probed = False
        while True:
            dataset.index = 0
            dataset.data = data
            dataset.labels = labels
            sim.run(duration=len(data)*input_period)
            Helper.log('Simulator', log.INFO, 'worker {} done, extracting updates'.format(id))
            out = {}
            for ens in my_model.objects[Ensemble]:
                if ens.learner:
                    out = {k: out.get(k, 0) + ens.learner.updates.get(k, 0)
                           for k in set(out) | set(ens.learner.updates)}  # merge sum dicts
            sim.flush()
            pipe.send(out)
            while not pipe.poll():
                time.sleep(0.1)
            try:
                update = pipe.recv()
            except:
                print("Updates crashed")
                print(pipe.poll)
                update = {}
            for attr, value in update[0].items():
                sim.connections[attr[0]].update_weight(attr[1], attr[2], value)
            # print(my_model.objects[Connection][1].weights.matrix[0, 0])
            my_model.restore()
            # print(my_model.objects[Connection][1].weights.matrix[0, 0])
            data = update[1][0]
            labels = update[1][1]
            # print('worker {} updates applied'.format(id))

        # print("worker done")

    # def print_time(self):
    #     # if __name__ == '__main__':
    #     if self.time_enabled:
    #         time_left = int((time.time() - self.start) / self.curr_time * (self.duration - self.curr_time))
    #         print('Time {} / {}, {}:{}:{} left '
    #               .format(int(self.curr_time),
    #                       self.duration,
    #                       int(time_left // 3600),
    #                       int((time_left // 60) % 60),
    #                       int(time_left % 60)))
    #
    #
