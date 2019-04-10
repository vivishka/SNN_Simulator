import numpy as np
from base import SimulationObject
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
        args = {}
        # TODO: change arg passing
        # args['param'] = {'threshold': np.random.rand}
        args = {}
        self.neuron_list = [
            neuron_type(self, i, **args) for i in range(size)]

    def step(self, dt):
        for neuron in self.neuron_list:
            neuron.step(dt)

    def get_neuron_list(self):
        return self.neuron_list
        # TODO: flatten if 2D ?

    def set_probe(self, index, value):
        for neuron in self.neuron_list:
            neuron.set_probe(index, value)

    def __getitem__(self, index):
        return self.neuron_list[index]

    def __setitem__(self, index, value):
        self.neuron_list[index] = value


# TODO: distribution mecanim for neuron creation
# TODO: inhibition
# TODO: probing
# TODO: dimension (1/2)
# TODO: acces indexing []
