
from base import SimationObject
import sys
sys.dont_write_bytecode = True


class Ensemble(SimationObject):
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
    def __init__(self, size, neuron_type, *args):
        super(Ensemble, self).__init__()
        Ensemble.objects.append(self)

        self.size = size
        self.neuron_type = neuron_type
        self.neuron_list = [neuron_type(self, i, *args) for i in range(size)]

    def step(self, dt):
        for neuron in self.neuron_list:
            neuron.step(dt)

    def get_neuron_list(self):
        return self.neuron_list
        # TODO: flatten if 2D ?


# TODO: distribution mecanim for neuron creation
# TODO: inhibition
# TODO: probing
# TODO: dimension (1/2)
# TODO: acces indexing []
