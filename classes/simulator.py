
# from neuron import Neuron
from .connection import Connection
from .ensemble import Ensemble
# from node import Node
# from probe import Probe
# from base import SimulationObject
import sys
sys.dont_write_bytecode = True


class SpikeNotifier(object):
    """docstring for Spike_buffer."""

    def __init__(self):
        super(SpikeNotifier, self).__init__()
        self.spike_list = []

    def register_spike(self, axon):
        self.spike_list.append(axon)

    def propagate_all(self):
        for axon in self.spike_list:
            axon.propagate_spike()
        self.spike_list = []


# Print iterations progress
def printProgressBar(
    iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(
        100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


class simulator(object):
    """docstring for simulator."""

    def __init__(self, model, dt):
        super(simulator, self).__init__()
        self.model = model
        self.dt = dt
        self.nb_step = 0
        self.time = 0
        self.objects = model.get_all_objects()
        # self.neurons = self.objects[Neuron]
        self.ensembles = self.objects[Ensemble]
        self.connections = self.objects[Connection]
        self.spike_notifier = SpikeNotifier()

    def run(self, time):
        for connect in self.connections:
            connect.set_notifier(self.spike_notifier)
        # this or for all axons ?
        # TODO: send all simulator infos at once: bugger + dt ?
        self.nb_step = int(time / self.dt)
        for i in range(self.nb_step):
            self.step()
            # printProgressBar(i, self.nb_step)

    def step(self):
        """ for every steps, evaluate ensembles, then propagate spikes """
        # TODO:  progres bar
        self.time += self.dt
        print("{:.4f}".format(self.time))
        for ens in self.ensembles:
            ens.step(self.dt, self.time)
        self.spike_notifier.propagate_all()
