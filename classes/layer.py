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
        # optimisation: merge the smaller set into the larger is faster
        if len(self.active_neuron_set) > len(self.probed_neuron_set):
            simulated_neuron_set = self.active_neuron_set | self.probed_neuron_set
        else:
            simulated_neuron_set = self.probed_neuron_set | self.active_neuron_set
        for neuron in simulated_neuron_set:
            neuron.step()
        self.active_neuron_set.clear()

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

    # <inhibition region>

    def set_inhibition(self):
        for neuron in self.neuron_list:
            neuron.inhibiting = True

    def inhibit(self, index_2d_n=None, radius=None):
        if index_2d_n is None or radius is None:
            for neuron in self.neuron_list:
                neuron.inhibited = True
        else:
            for row in range(index_2d_n[0] - radius[0], index_2d_n[0] + radius[0] + 1):
                for col in range(index_2d_n[1] - radius[1], index_2d_n[1] + radius[1] + 1):

                    # lazy range test but eh #2
                    try:
                        self.neuron_array[row, col].inhibited = True
                    except IndexError:
                        pass

    def propagate_inhibition(self, index_2d_n):
        self.bloc.propagate_inhibition(index_2d_n)
        self.inhibit()

    # </inhibition region>

    # <spike region>

    def create_spike(self, index_1d):
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

    def restore(self):
        if self.learner is not None:
            self.learner.restore()
        self.active_neuron_set = set()
        for neuron in self.neuron_list:
            neuron.restore()

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
    index = 0

    def __init__(self, depth, size, neuron_type, learner=None, *args, **kwargs):
        super(Bloc, self).__init__()
        self.index = Bloc.index
        Bloc.index += 1
        self.depth = depth
        self.size = (1, size) if isinstance(size, int) else size
        self.inhibition_radius = (1, 1)

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

    def set_inhibition(self, radius=None):
        if radius is not None:
            self.inhibition_radius = radius
        self.inhibition_radius = (radius, radius) if isinstance(radius, int) else radius
        for ens in self.ensemble_list:
            Helper.log('Layer', log.INFO, 'ensemble {0} inhibited'.format(ens.id))
            ens.set_inhibition()

    def propagate_inhibition(self, index_2d_n):
        for ens in self.ensemble_list:
            ens.inhibit(index_2d_n, self.inhibition_radius)
            Helper.log('Layer', log.INFO, 'ensemble {0} inhibited by propagation'.format(ens.id))

    def __getitem__(self, index):
        return self.ensemble_list[index]

    def __setitem__(self, index, value):
        self.ensemble_list[index] = value
