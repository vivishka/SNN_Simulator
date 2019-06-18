import numpy as np
import copy
import logging as log
from .base import SimulationObject, Helper, MeasureTiming
from .learner import Learner
import sys
sys.dont_write_bytecode = True


class Layer(SimulationObject):
    """
    Layer cannot be instanced by itself, it can be a Block or an Ensemble

    Parameters
    ---------


    Attributes
    ----------
    ensemble_list: list[Ensemble]
        if Block: the list of Ensembles inside, if Ensemble: itself
    out_connections: list[Connection]
        outbound Connections
    in_connections: list[Connection]
        inbound Connections
    id: int
        global index of the layer
    """
    layer_count = 0

    def __init__(self, lbl=""):
        super(Layer, self).__init__(lbl)
        self.ensemble_list = []
        self.learner = None
        self.out_connections = []
        self.in_connections = []
        self.id = Layer.layer_count
        Layer.layer_count += 1
        Helper.log('Layer', log.INFO, 'new layer created, {0}'.format(self.id))

    def set_weights(self, dw):
        for ens in self.ensemble_list:
            for neuron in ens.neuron_list:
                neuron.set_weights(neuron.weights.weights + dw)


class Ensemble(Layer):
    """
    An ensemble is an array of neurons. It is mostly used to represent a layer
    it can be a 1D or 2D array

    Parameters
    ---------
    size: int or (int, int)
        Size / number of neurons of the ensemble
    neuron_type : NeuronType
        instanced NeuronType that every neuron of the Ensemble will copy
    block: Bloc
        Blocks in which this Ensemble belongs to
    index: int
        index of the Ensemble in the Block
    learner: Learner or None
        instanced Learner
    *args, **kwargs:
        Arguments passed to initialize the neurons

    Attributes
    ----------
    bloc: Bloc
        Blocks in which this Ensemble belongs to
    index: int:
        index of the Ensemble in the Block
    size: (int, int)
        Size of the ensemble, if the given parameter was an int, the size is (1, n)
    neuron_list: [NeuronType]
        List of initialized neurons, accessible by int
    neuron_array: np.ndarray[NeuronType]
        2D array of the neurons, accessible by (int, int)
    active_neuron_set: set(NeuronType)
        set of the neurons which received a spike and should be simulated the next step
    probed_neuron_set: set(NeuronType)
        set of the neurons which are probed and should be simulated every step
    learner: Learner
        every batch, modifies the weights of the inbound connections

    """

    objects = []

    def __init__(self, size, neuron_type, bloc=None, index=0, learner=None, label='', **kwargs):
        lbl = label if label != '' else id(self)
        super(Ensemble, self).__init__("Ens_{}".format(lbl))
        Ensemble.objects.append(self)
        self.bloc = bloc
        self.index = index
        self.size = (1, size) if isinstance(size, int) else size
        self.neuron_list = []
        self.neuron_array = np.ndarray(self.size, dtype=object)
        self.active_neuron_set = set()
        self.probed_neuron_set = set()
        self.ensemble_list.append(self)
        self.wta = False
        self.inhibited = False
        self.threshold_adapt = False
        self.first_voltage = 0
        self.first_neuron = None
        if learner is not None:
            self.learner = learner
            self.learner.set_layer(self)
        Helper.log('Layer', log.DEBUG, 'layer type : ensemble of size {0}'.format(self.size))
        if len(self.size) == 2:
            for row, element in enumerate(self.neuron_array):
                for col in range(len(element)):
                    # Creates copies of the neuron given as argument
                    neuron = copy.deepcopy(neuron_type)
                    neuron.set_ensemble(ensemble=self, index_2d=(row, col))
                    self.neuron_array[(row, col)] = neuron
                    self.neuron_list.append(neuron)
                    # step every neuron once to initialize
                    self.active_neuron_set.add(neuron)
                    Helper.log('Layer', log.DEBUG,
                               'neuron {0} of layer {1} created'.format(neuron.index_2d, neuron.ensemble.id))

        else:
            raise TypeError("Ensemble size should be int or (int, int)")

    @MeasureTiming('ens')
    def step(self):
        """
        simulate all the neurons of the Ensemble that are either probed or have received a spike
        """
        if not self.inhibited:
            # optimisation: merge the smaller set into the larger is faster
            if len(self.active_neuron_set) > len(self.probed_neuron_set):
                simulated_neuron_set = self.active_neuron_set | self.probed_neuron_set
            else:
                simulated_neuron_set = self.probed_neuron_set | self.active_neuron_set

            for neuron in simulated_neuron_set:
                neuron.step()
            self.active_neuron_set.clear()

            # if WTA, only propagates the neuron which spiked first with highest voltage
            if self.wta and self.first_neuron is not None:
                for con in self.out_connections:
                    con.register_neuron(self.first_neuron)
                if self.learner is not None:
                    self.learner.out_spike(self.first_neuron)

                self.inhibited = True
                self.bloc.propagate_inhibition(Helper.get_index_2d(self.first_neuron, self.size[1]))

                # The first spike of each ens will trigger the threshold adaptation
                if self.threshold_adapt:
                    self.bloc.register_first_layer(self.index)

    # <inhibition region>

    def set_inhibition(self):
        self.wta = True

    def inhibit(self, index_2d_n, radius):
        for row in range(index_2d_n[0] - radius[0], index_2d_n[0] + radius[0] + 1):
            for col in range(index_2d_n[1] - radius[1], index_2d_n[1] + radius[1] + 1):

                # lazy range test but eh #2
                try:
                    self.neuron_array[row, col].inhibited = True
                except IndexError:
                    pass

    # </inhibition region>

    # <spike region>

    def create_spike(self, index_1d):
        if self.wta:
            voltage = self.neuron_list[index_1d].voltage
            if voltage > self.first_voltage:
                self.first_voltage = voltage
                self.first_neuron = index_1d
        else:
            for con in self.out_connections:
                con.register_neuron(index_1d)

            if self.learner is not None:
                self.learner.out_spike(index_1d)

    def receive_spike(self, targets, source_c):
        for target in targets:
            # target: (source_index_1d, dest_index_1d, weight)
            dest_n = self.neuron_list[target[1]]
            dest_n.receive_spike(index_1d=target[0], weight=target[2])
            self.active_neuron_set.add(self.neuron_list[target[1]])

            if self.learner is not None:
                self.learner.in_spike(*target, source_c)

    # </spike region>

    def reset(self):
        """
        called every input period,
        notify the learner of the new period, it will save the spikes
        reset the internal states of every neuron
        """
        if self.learner:
            self.learner.reset_input()
        for neuron in self.neuron_list:
            neuron.reset()
        self.inhibited = False
        self.first_voltage = 0
        self.first_neuron = None

    def restore(self):
        if self.learner is not None:
            self.learner.restore()
        self.active_neuron_set = set()
        for neuron in self.neuron_list:
            neuron.restore()
        self.inhibited = False
        self.first_voltage = 0
        self.first_neuron = None
        self.threshold_adapt = False

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.neuron_list[index]
        else:
            return self.neuron_array[index]

    def __setitem__(self, index, value):
        if isinstance(index, int):
            self.neuron_list[index] = value
            self.neuron_array[Helper.get_index_2d(index, self.size[1])] = value
        else:
            self.neuron_array[index] = value
            self.neuron_list[Helper.get_index_1d(index, self.size[1])] = value


class Bloc(Layer):
    """
    a bloc is a group of ensembles of the same dimension
    Ensembles are not connected together but share common previous and next layers
    they are only used to construct easily but are not simulated as such

    Parameters
    ---------
    depth: int
        Number of Ensemble in this bloc
    size: int or (int, int)
        Size / number of neurons of the ensemble
    neuron_type : NeuronType
        instanced NeuronType that every neuron of the Ensemble will copy
    learner: Learner
        instanced Learner
    *args, **args: list, Dict
        Arguments passed to initialize the Ensembles

    Attributes
    ----------
    depth: int
        Number of Ensemble in this bloc
    ensemble_list: [Ensemble]
        The list of initialized Ensembles
    inhibition_radius: int
        Distance of inhibition from the first spiking neuron
    """
    objects = []
    index = 0

    def __init__(self, depth, size, neuron_type, learner=None, *args, **kwargs):
        super(Bloc, self).__init__()
        Bloc.objects.append(self)
        self.index = Bloc.index
        Bloc.index += 1
        self.depth = depth
        self.size = (1, size) if isinstance(size, int) else size
        self.inhibition_radius = (0, 0)
        self.threshold_adapt = False
        self.t_targ = None
        self.th_min = None
        self.n_th1 = None
        self.n_th2 = None
        self.layer_time = None

        # Ensemble creation
        for i in range(depth):
            ens = Ensemble(
                size=size,
                neuron_type=neuron_type,
                bloc=self,
                index=i,
                learner=copy.deepcopy(learner) if learner is not None else None,
                *args, **kwargs)
            self.ensemble_list.append(ens)
        Helper.log('Layer', log.INFO, 'layer type : bloc of size {0}'.format(depth))

    def set_dataset(self, dataset):
        for ens in self.ensemble_list:
            ens.learner.dataset = dataset

    def set_inhibition(self, radius=None):
        if radius is not None:
            self.inhibition_radius = radius
        self.inhibition_radius = (radius, radius) if isinstance(radius, int) else radius
        for ens in self.ensemble_list:
            Helper.log('Layer', log.INFO, 'ensemble {0} inhibited'.format(ens.id))
            ens.set_inhibition()

    def propagate_inhibition(self, index_2d_n):
        if sum(self.inhibition_radius) > 0:
            for ens in self.ensemble_list:
                ens.inhibit(index_2d_n, self.inhibition_radius)
                Helper.log('Layer', log.INFO, 'ensemble {0} inhibited by propagation'.format(ens.id))

    def activate_threshold_adapt(self, t_targ, th_min, n_th1, n_th2):
        self.threshold_adapt = True
        self.t_targ = t_targ
        self.th_min = th_min
        self.n_th1 = n_th1
        self.n_th2 = n_th2
        self.layer_time = np.ndarray((self.depth,), dtype=float)

        for i, ens in enumerate(self.ensemble_list):
            ens.threshold_adapt = True
            self.layer_time[i] = float('inf')

    def register_first_layer(self, ens_index):
        # This function is  only called by the first spike of each ens
        self.layer_time[ens_index] = Helper.time

    def apply_threshold_adapt(self):
        """
        called once every input cycle, will adapt the threshold of the neurons of each layers depending on spiking time
        2 mechanism:
            - spike time target
            - inter layer competition
        """
        if not self.threshold_adapt:
            return
        min_time = min(self.layer_time)
        # number of simultaneous first spikes
        nb_first = len([0 for time in self.layer_time if time == min_time])

        for index, ens in enumerate(self.ensemble_list):
            # get the first spike time
            time = self.layer_time[index]
            self.layer_time[index] = float('inf')
            time = Helper.input_period if time is None else time % Helper.input_period

            # First th adaptation: spike time target
            old_th = ens.neuron_list[0].threshold
            new_th = old_th - self.n_th1 * (time - self.t_targ)

            # second th adaptation: inter layer competition
            if time == min_time:
                new_th += self.n_th2
            else:
                new_th -= self.n_th2 / self.depth

            # clipping
            new_th = max(self.th_min, new_th)
            for neuron in ens.neuron_list:
                neuron.threshold = new_th

    def restore(self):
        if self.learner is not None:
            self.learner.restore()
        self.threshold_adapt = False
        self.t_targ = None
        self.th_min = None
        self.n_th1 = None
        self.n_th2 = None
        self.layer_time = None

    def __getitem__(self, index):
        return self.ensemble_list[index]

    def __setitem__(self, index, value):
        self.ensemble_list[index] = value
