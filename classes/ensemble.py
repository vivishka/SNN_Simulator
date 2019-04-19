import numpy as np
from .base import SimulationObject
import sys
sys.dont_write_bytecode = True


class Ensemble(SimulationObject):
    """
    An ensemble is an array of neurons. It is mostly used to represent a layer
    it can be a 1D or 2D array

    Parameters
    ---------
    size: int or (int, int)
         Size of the ensemble
    neuron_type : NeuronType
        Class of the neurons
    *args
        The list of arguments passed to initialize the neurons

    Attributes
    ----------
    size: int or (int, int)
        size of the ensemble
    neuron_type: NeuronType
        Class of the neurons
    neuron_list: [NeuronType]
        The list of initialized neurons
    """

    objects = []

    # TODO: default cases
    def __init__(self, size, neuron_type, label='', index=0, **kwargs):
        lbl = label if label != '' else id(self)
        super(Ensemble, self).__init__("Ens_{}".format(lbl))
        Ensemble.objects.append(self)
        self.index = index
        self.size = size
        self.neuron_type = neuron_type
        self.neuron_list = []
        self.neuron_array = np.ndarray(self.size, dtype=object)
        # TODO: change arg passing
        if isinstance(size, int):
            self.dim = 1
            self.neuron_list = [
                neuron_type(self, i, **kwargs) for i in range(size)]
        elif isinstance(size, tuple) and len(size) == 2:
            self.dim = 2
            for line, element in enumerate(self.neuron_array):
                for col in range(len(element)):
                    neuron = neuron_type(self, (line, col), **kwargs)
                    self.neuron_array[(line, col)] = neuron
                    self.neuron_list.append(neuron)
        else:
            raise TypeError("Ensemble dimension should be int or (int, int)")

    def step(self):
        for neuron in self.neuron_list:
            neuron.step()

    def reset(self):
        for neuron in self.neuron_list:
            neuron.reset()

    def get_neuron_list(self):
        return self.neuron_list

    def add_probe(self, index, value):
        for neuron in self.neuron_list:
            neuron.add_probe(index, value)

    def __getitem__(self, index):
        if self.dim == 1:
            return self.neuron_list[index]
        else:
            return self.neuron_array[index]

    def __setitem__(self, index, value):
        self.neuron_list[index] = value


class Bloc(object):
    """
    a bloc is a group of ensembles of the same dimension
    they are not connected together but share common previous and next layers
    they are only used to construct easily but are not simulated as such
    """
    index = 0

    def __init__(self, depth, *args, **kwargs):
        super(Bloc, self).__init__()
        self.index = Bloc.index
        Bloc.index += 1
        self.depth = depth
        self.ensemble_list = []
        for i in range(depth):
            if args or kwargs:
                self.ensemble_list.append(Ensemble(index=i, *args, **kwargs))
            else:
                self.ensemble_list.append(None)

    def __getitem__(self, index):
        return self.ensemble_list[index]

    def __setitem__(self, index, value):
        self.ensemble_list[index] = value

# TODO: distribution mecanim for neuron creation
# TODO: inhibition
# TODO: dimension (1/2)
# TODO: acces indexing []
