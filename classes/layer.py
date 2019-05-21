import numpy as np
from .base import SimulationObject
from .weights import Weights
import sys
sys.dont_write_bytecode = True


class Layer(SimulationObject):

    def __init__(self,lbl=""):
        super(Layer, self).__init__(lbl)
        self.ensemble_list = []
        self.learner = None
        self.out_connections = []
        self.in_connections = []



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
        Size of the ensemble
    neuron_type : NeuronType
        Class of the neurons
    *args, **kwargs:
        Arguments passed to initialize the neurons

    Attributes
    ----------
    size: (int, int)
        Size of the ensemble, if the given parameter was an int, the size is (1, n)
    neuron_type: NeuronType
        Class of the neurons
    neuron_list: [NeuronType]
        List of initialized neurons
    """

    objects = []
    # index = 0

    def __init__(self, size, neuron_type, label='', bloc=None, index=0, **kwargs):
        lbl = label if label != '' else id(self)
        super(Ensemble, self).__init__("Ens_{}".format(lbl))
        Ensemble.objects.append(self)
        self.bloc = bloc
        self.index = index
        self.size = (1, size) if isinstance(size, int) else size
        self.neuron_list = []
        self.active_neuron_list = []
        self.probed_neuron_set = set()
        self.neuron_array = np.ndarray(self.size, dtype=object)
        self.ensemble_list.append(self)
        self.input_spike_buffer = []

        if len(self.size) == 2:
            for row, element in enumerate(self.neuron_array):
                for col in range(len(element)):
                    neuron = neuron_type(self, (row, col), **kwargs)
                    self.neuron_array[(row, col)] = neuron
                    self.neuron_list.append(neuron)
                    self.active_neuron_list.append(neuron)
        else:
            raise TypeError("Ensemble size should be int or (int, int)")

    def step(self):
        for spike in self.input_spike_buffer:
            self.neuron_list[spike[1]].receive_spike(spike[2])
        self.input_spike_buffer = []
        for neuron in set(self.active_neuron_list) | self.probed_neuron_set:
            neuron.step()
        self.active_neuron_list = []



    def reset(self):
        for neuron in self.neuron_list:
            neuron.reset()

    def add_probe(self, index, value):
        for neuron in self.neuron_list:
            neuron.add_probe(index, value)

    def set_inhibition(self):
        for neuron in self.neuron_list:
            neuron.inhibiting = True

    def inhibit(self, index_n=None, radius=None):
        if index_n is None or radius is None:
            for neuron in self.neuron_list:
                neuron.inhibited = True
        else:
            for row in range(index_n[0] - radius[0], index_n[0] + radius[0] + 1):
                for col in range(index_n[1] - radius[1], index_n[1] + radius[1] + 1):

                    # lazy range test but eh #2
                    try:
                        self.neuron_array[row, col].inhibited = True
                    except IndexError:
                        pass

    def propagate_inhibition(self, index_n):
        self.bloc.propagate_inhibition(index_n)
        self.inhibit()

    def create_spike(self, index):
        for con in self.out_connections:
            con.register_neuron(index)

    def receive_spike(self, targets):
        self.input_spike_buffer += targets
        for target in targets:
            self.active_neuron_list.append(self.neuron_list[target[1]])

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.neuron_list[index]
        else:
            return self.neuron_array[index]

    def __setitem__(self, index, value):
        self.neuron_list[index] = value


class Bloc(Layer):
    """
    a bloc is a group of ensembles of the same dimension
    Ensembles are not connected together but share common previous and next layers
    they are only used to construct easily but are not simulated as such

    Parameters
    ---------
    depth: int
        Number of Ensemble in this bloc
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

    def __init__(self, depth, *args, **kwargs):
        super(Bloc, self).__init__()
        self.index = Bloc.index
        Bloc.index += 1
        self.depth = depth
        self.inhibition_radius = (1, 1)
        for i in range(depth):
            if args or kwargs:
                ens = Ensemble(*args, **kwargs, bloc=self, index=i)
                # ens.bloc = self
                # ens.bloc_index = i
                self.ensemble_list.append(ens)
            else:
                self.ensemble_list.append(None)

    def set_inhibition(self, radius=None):
        if radius is not None:
            self.inhibition_radius = radius
        self.inhibition_radius = (radius, radius) if isinstance(radius, int) else radius
        for ens in self.ensemble_list:
            ens.set_inhibition()

    def propagate_inhibition(self, index_n):
        for ens in self.ensemble_list:
            ens.inhibit(index_n, self.inhibition_radius)

    def __getitem__(self, index):
        return self.ensemble_list[index]

    def __setitem__(self, index, value):
        self.ensemble_list[index] = value
