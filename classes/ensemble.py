import numpy as np
from .base import SimulationObject
import sys
sys.dont_write_bytecode = True


class Ensemble(SimulationObject):
    '''
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
    '''

    objects = []

    # TODO: default cases
    def __init__(self, size, neuron_type, label='', **kwargs):
        lbl = label if label != '' else id(self)
        super(Ensemble, self).__init__("Ens_{}".format(lbl))
        Ensemble.objects.append(self)

        self.size = size
        self.neuron_type = neuron_type
        # TODO: change arg passing
        # args['param'] = {'threshold': np.random.rand}
        # args = {}
        if isinstance(size, int):
            self.dim = 1
            self.neuron_list = [
                neuron_type(self, i, **kwargs) for i in range(size)]
        elif isinstance(size, tuple) and len(size) == 2:
            self.dim = 2
            self.neuron_array = [
                [neuron_type(self, (line, col), **kwargs)
                    for col in range(size[1])] for line in range(size[0])]
            # flatten the arry into a list
            self.neuron_list = [obj for i in self.neuron_array for obj in i]
        else:
            raise TypeError("Ensemble dimension should be int or (int, int)")

    def step(self, dt, time):
        for neuron in self.neuron_list:
            neuron.step(dt, time)

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


# TODO: distribution mecanim for neuron creation
# TODO: inhibition
# TODO: dimension (1/2)
# TODO: acces indexing []
