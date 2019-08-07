import numpy as np
import copy
import logging as log
from .base import SimulationObject, Helper, MeasureTiming
from .neuron import NeuronType
from .connection import *
from .learner import Learner
import sys
sys.dont_write_bytecode = True


class Layer(SimulationObject):
    """
    Layer cannot be instanced by itself, it can be a Block or an Ensemble

    :ivar ensemble_list: if Block: the list of Ensembles inside, if Ensemble: itself
    :type ensemble_list: list of Ensemble
    :ivar learner: for training the incoming connections, is unique per layer
    :type learner: Learner
    :ivar out_connections: outbound Connections
    :type out_connections: list of Connection
    :ivar in_connections: inbound Connections
    :type in_connections: list of Connection
    :ivar id: global index of the layer
    :type id: int
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


class Ensemble(Layer):
    """
    An ensemble is an array of neurons. It is mostly used to represent a layer
    it can be a 1D vector or a 2D array

    Parameters
    ---------
    :param size: Size / number of neurons of the ensemble
    :type size: int or (int, int)
    :param neuron_type : instanced NeuronType that every neuron of the Ensemble will copy
    :type neuron_type: NeuronType
    :param bloc: Blocks in which this Ensemble belongs to
    :type bloc: Bloc
    :param index: index of the Ensemble in the Block
    :type index: int
    :param learner: instanced Learner
    :type learner: Learner or None
    :param kwargs: Arguments passed to initialize the neurons
    :type kwargs: dict

    Attributes
    ----------
    :ivar bloc: Blocks in which this Ensemble belongs to
    :type bloc: Bloc
    :ivar index: index of the Ensemble in the Block
    :type index: int
    :ivar size: Size of the ensemble, if the given parameter was an int, the size is (1, n)
    :type size: (int, int)
    :ivar neuron_list: List of initialized neurons, accessible by int
    :type neuron_list: list of NeuronType
    :ivar neuron_array: 2D array of the neurons, accessible by (int, int)
    :type neuron_array: ndarray of NeuronType
    :ivar active_neuron_set: set of the neurons which received a spike and should be simulated the next step
    :type active_neuron_set: set of NeuronType
    :ivar probed_neuron_set: set of the neurons which are probed and should be simulated every step
    :type probed_neuron_set: set of NeuronType
    :ivar inhibited: if True, will not be simulated until the end of the cycle
    :type inhibited: bool
    :ivar inhibition: True if the ensemble triggers inhibition when one of its neuron spikes
    :type inhibition: bool
    :ivar wta: winner takes all inhibition mode, only one neuron can spike per ensemble per input cycle
    :type wta: bool
    :ivar k_wta_level: the k first neurons are allowed to spike
    :type k_wta_level: int
    :ivar first_neur_volt_list: index and voltage of the first neurons to spike this cycle
    :type first_neur_volt_list: list of (int, float)
    :ivar threshold_adapt: adapt the threshold depending on received / emitted spikes
    :type threshold_adapt: bool
    """

    objects = []

    def __init__(self, size, neuron_type, bloc=None, index=0, learner=None, label='', **kwargs):
        lbl = label if label != '' else id(self)
        super(Ensemble, self).__init__("Ens_{}".format(lbl))
        Ensemble.objects.append(self)
        self.bloc = bloc
        self.index = index
        self.ensemble_list.append(self)
        self.size = (1, size) if isinstance(size, int) else size
        self.neuron_list = []
        self.neuron_array = np.ndarray(self.size, dtype=object)
        self.active_neuron_set = set()
        self.probed_neuron_set = set()
        self.inhibited = False
        self.inhibition = False
        self.wta = False
        self.k_wta_level = 1
        self.first_neur_volt_list = []
        self.threshold_adapt = False
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

    @MeasureTiming('ens_step')
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

            # if WTA, only propagates the neurons which spiked first with highest voltage
            if self.wta and self.first_neur_volt_list:
                # if wta and a spike emitted
                self.inhibited = True
                self.first_neur_volt_list.sort(key=lambda t: t[1])
                k_winners = self.first_neur_volt_list[:self.k_wta_level]
                self.first_neur_volt_list = []
                for winner_index, winner_voltage in k_winners:

                    self.bloc.propagate_inhibition(Helper.get_index_2d(winner_index, self.size[1]))

                    # propagates the first neuron to spike
                    for con in self.out_connections:
                        con.register_neuron(winner_index)

                    # learning only on the first spike
                    if self.learner is not None:
                        self.learner.out_spike(winner_index)

                # The first spike of each ens will trigger the threshold adaptation
                if self.threshold_adapt:
                    self.bloc.register_first_layer(self.index)

    # <inhibition region>

    def set_inhibition(self, wta=True, k_wta_level=1):
        """
        Activates the inhibition feature
        :param wta: winner takes all inhibition mode, only one neuron can spike per ensemble per input cycle
        :type wta: bool
        :param k_wta_level: the k first neurons are allowed to spike
        :type k_wta_level: int
        :return:
        """
        self.inhibition = True
        self.wta = wta
        self.k_wta_level = k_wta_level

    def inhibit(self, index_2d_n, radius):
        """
        Inhibit all the neurons in a radius around the one which spiked
        :param index_2d_n: position of the neuron which spiked
        :type index_2d_n: (int, int)
        :param radius: radius to inhibit, it's actually not a radius because the inhibition shape is a square
        :type radius: (int, int)
        """
        if radius[0] == -1:
            return
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
        """
        Registers the neuron spike for the connection and the learner
        Neurons call this method when they spike
        :param index_1d: neuron index in the list
        :type index_1d: int
        """
        if self.wta:
            # stores the first neurons to spike
            voltage = self.neuron_list[index_1d].voltage
            self.first_neur_volt_list.append((index_1d, voltage))

        else:
            # if only lateral inhibition and no global ens inhibition
            if self.inhibition:
                self.bloc.propagate_inhibition(Helper.get_index_2d(index_1d, self.size[1]))

            for con in self.out_connections:
                con.register_neuron(index_1d)

            if self.learner is not None:
                self.learner.out_spike(index_1d)

    def receive_spike(self, targets, source_c):
        """
        Propagates received weighted spike list to the neurons
        :param targets: list of (source neuron index, dest neuron index, weight)
        :type targets: list of (int, int, float)
        :param source_c: connection which transmitted the spikes
        :type source_c: Connection
        :return:
        """
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
        self.first_neur_volt_list = []

    def restore(self):
        """
        Restores the ensemble
        TODO: better explain
        """
        if self.learner is not None:
            self.learner.restore()
        self.active_neuron_set = set()
        for neuron in self.neuron_list:
            neuron.restore()
        self.inhibited = False
        self.threshold_adapt = False
        self.first_neur_volt_list = []

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
    A bloc is a group of ensembles of the same dimension
    Ensembles are not connected together but share common previous and next layers
    they are only used to construct easily but are not simulated as such

    Parameters
    ---------
    :param depth: Number of Ensemble in this bloc
    :type depth: int
    :param size: Size / number of neurons of the ensemble
    :type size: int or (int, int)
    :param neuron_type : instanced NeuronType that every neuron of the Ensemble will copy
    :type neuron type: NeuronType
    :param learner: instanced Learner. A copy will be shared to all the sub Ensembles
    :type learner: Learner or None
    :param kwargs: Arguments passed to initialize the Ensembles
    :type kwargs: dict

    Attributes
    ----------
    :param inhibition_radius: Distance of inhibition from the first spiking neuron
    :type inhibition_radius: (int, int)

    """
    objects = []
    index = 0

    def __init__(self, depth, size, neuron_type, learner=None, **kwargs):
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
                **kwargs)
            self.ensemble_list.append(ens)
        Helper.log('Layer', log.INFO, 'layer type : bloc of size {0}'.format(depth))

    def set_inhibition(self, wta=True, radius=None, k_wta_level=1):
        """
        Activate inhibition
        :param wta: winner take all, if True, the whole ensemble will be inhibited one of its neuron spike
        :type wta: bool
        :param radius: lateral inhibition radius
        :type radius: int or (int, int)
        :param k_wta_level: number of first neuron allowed to spike, can reduce learning time
        :type k_wta_level: int
        """
        if radius is not None:
            self.inhibition_radius = (radius, radius) if isinstance(radius, int) else radius
        else:
            self.inhibition_radius = (-1, -1)

        for ens in self.ensemble_list:
            Helper.log('Layer', log.INFO, 'ensemble {0} inhibited'.format(ens.id))
            ens.set_inhibition(wta=wta, k_wta_level=k_wta_level)

    def propagate_inhibition(self, index_2d_n):
        """
        Propagates inhibition laterally (across ensembles) in a radius around the spiking neuron
        :param index_2d_n: position of the spiking neuron
        :type index_2d_n: (int, int)
        """
        for ens in self.ensemble_list:
            ens.inhibit(index_2d_n, self.inhibition_radius)
            Helper.log('Layer', log.INFO, 'ensemble {0} inhibited by propagation'.format(ens.id))

    def set_threshold_adapt(self, t_targ, th_min, n_th1, n_th2):
        """
        Changes threshold of an ensemble using 2 methods:
         - try to match a given spiking time
         - inter layer competition
        :param t_targ: target time, between 0. and 1.
        :type t_targ: float
        :param th_min: minimum threshold
        :type th_min: float
        :param n_th1: coefficient for the spike time target
        :type n_th1: float
        :param n_th2: coefficient for the inter layer competition
        :type n_th2: float
        """
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
        """
        This function is  only called by the first spike of each ens and is used for threshold adaptation
        :param ens_index: Ensemble index
        :type ens_index: int
        """
        self.layer_time[ens_index] = self.sim.curr_time

    def apply_threshold_adapt(self):
        """
        called once every input cycle, will adapt the threshold of the neurons of each layers depending on spiking time
        """
        if not self.threshold_adapt:
            return
        min_time = min(self.layer_time)
        # number of simultaneous first spikes
        # nb_first = len([0 for time in self.layer_time if time == min_time])

        for index, ens in enumerate(self.ensemble_list):
            # get the first spike time
            time = self.layer_time[index]
            self.layer_time[index] = float('inf')
            time = self.sim.input_period if time == float('inf') else time % self.sim.input_period

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

    def set_threshold(self, new_th):
        """
        Sets the threshold for all all neurons of all the ensembles
        :param new_th: new threshold
        :type new_th: float
        """
        for ens in self.ensemble_list:
            for neuron in ens.neuron_list:
                neuron.threshold = new_th

    def set_learner(self, learner):
        """
        Gives a new learner to all the ensembles of the bloc
        :param learner: instanced learner that will be copied to all the ensembles
        :type learner: Learner
        """
        self.learner = learner
        for ens in self.ensemble_list:
            ens.learner = copy.deepcopy(learner)
            ens.learner.set_layer(ens)

    def stop_inhibition(self):
        """
        Stops all form of inhibition on all the ensembles of the bloc
        """
        for ens in self.ensemble_list:
            ens.inhibition = False
            ens.wta = False

    def stop_learner(self):
        """
        Stops all learners on all the ensembles of the bloc
        """
        self.learner = None
        for ens in self.ensemble_list:
            ens.learner = None

    def stop_threshold_adapt(self):
        """
        Stops all threshold adaptation on all the ensembles of the bloc
        """
        self.threshold_adapt = False
        self.t_targ = None
        self.th_min = None
        self.n_th1 = None
        self.n_th2 = None
        self.layer_time = None

    def restore(self):
        """
        TODO: better explain
        """
        if self.learner is not None:
            self.learner.restore()

    def __getitem__(self, index):
        return self.ensemble_list[index]

    def __setitem__(self, index, value):
        self.ensemble_list[index] = value
