
from .base import SimulationObject
import numpy as np
import sys
sys.dont_write_bytecode = True


class Weights(object):
    """
    array of Weights
    the 1st dimension is the index of the layer
    the 2nd or 2nd and 3rd are for the index of the neuron
    """

    def __init__(self, shared=False):
        super(Weights, self).__init__()
        self.weights = []
        # self.weights = np.ndarray(size, dtype=float)
        self.ensemble_index = {}
        self.ensemble_number = 0
        self.shared = shared

    def index_ensemble(self, ens):
        if ens not in self.ensemble_index:
            self.ensemble_index[ens] = self.ensemble_number
            self.ensemble_number += 1
            self.weights.append(None)
        return self.ensemble_index[ens]

    def set_weights(self, ens, weight_array):
        """ sets the weights of the axons from the specified ensemble """
        ens_number = self.index_ensemble(ens)
        self.weights[ens_number] = weight_array
        return ens_number

    def __getitem__(self, index):
        return self.weights[index[0]][index[1:]]

    def __setitem__(self, index, value):
        self.weights[index[0]][index[1:]] = value


class NeuronType(SimulationObject):
    """
    The NeuronType object is an abstract version of a neuron
    Used to construct subclass

    Parameters
    ----------
    ensemble: Ensemble
        The ensemble this neuron belongs to
    index: int or (int, int)
        The index (in 1D or 2D) of the neuron in the ensemble
    **kwargs
        The dictionary of arguments passed to initialize the neurons

    Attributes
    ----------
    ensemble: Ensemble
        Stores the ensemble this neuron belongs to
    index: int or (int, int)
    param
        The dictionary of arguments passed to initialize the neurons
        Stores the index (in 1D or 2D) of the neuron in the ensemble
    inputs: [Axon]
        List of axons connected to this neuron
    outputs: [Axon]
        List of axons the neuron can output to most neuron only have 1 axon,
        but multpiple axons are used here to connect with multiple ensembles
    weights: Weights
        Dictionary to store weights attached to connections
        Can be shared between neurons of the same ensemble
    received: [Axon]
        List of axons which emitted a received spikes this step
    *_probed: bool
        Stores on which type of variable this neuron is being probed
    probes: {str:Probe}
        Dictionary associating variable name and probe
    time: float
        the current time, updated every step, used for probing
    """

    def __init__(self, ensemble, index, **kwargs):
        # TODO: give a kwargs with an array: neuron can use its own index
        super(NeuronType, self).__init__(
            "{0}_Neuron_{1}".format(ensemble.label, index))
        self.ensemble = ensemble
        self.index = index
        self.param = kwargs if kwargs is not None else {}
        self.inputs = []
        self.outputs = []
        self.weights = Weights()
        self.received = []
        self.variable_probed = False
        self.spike_out_probed = False
        self.spike_in_probed = False
        self.probes = {}
        self.time = 0
        # TODO: bias

    def extract_param(self, name, default):
        param = default
        if self.param is not None and name in self.param:
            if callable(self.param[name]):
                param = self.param[name]()
            else:
                param = self.param[name]
        return param

    def add_input(self, source, weight):
        """ Stores in the neuron the reference of the incomming connection
        and its associated weight
        """
        self.inputs.append(source)
        self.weights[source] = weight

    def add_output(self, dest):
        """ Append the object the neuron should output into
        will call the method create_spike of this object
        """
        self.outputs.append(dest)

    def set_weights(self, weights):
        """" used for convolutional connections, when kernel is shared """
        self.weights = weights

    def receive_spike(self, source):
        """ Append an axons which emitted a received spikes this step """
        # TODO: here the learner can be added
        self.received.append(source)
        if self.spike_in_probed:
            w = self.weights[source]
            self.probes['spike_in'].log_spike_in(source.index, self.time, w)

    def send_spike(self):
        """ send a spike to all the connected axons """
        for output in self.outputs:
            output.create_spike()
        if self.spike_out_probed:
            self.probes['spike_out'].log_spike_out(self.index, self.time)

    def add_probe(self, obj, variable):
        self.probes[variable] = obj
        if variable == 'spike_in':
            self.spike_in_probed = True
        elif variable == 'spike_out':
            self.spike_out_probed = True
        else:
            self.variable_probed = True

    def probe(self):
        if self.variable_probed:
            for var, probe in self.probes.items():
                # TODO: check existence
                if var not in 'spike_in, spike_out':
                    probe.log_value(self.index, self.__getattribute__(var))


class LIF(NeuronType):
    """docstring for LIF."""

    def __init__(self, ensemble, index, **kwargs):
        super(LIF, self).__init__(ensemble, index, **kwargs)
        Neuron.objects.append(self)
        self.voltage = 0
        self.threshold = self.extract_param('threshold', 1)
        self.tau = self.extract_param('tau', 5)

    def step(self, dt, time):
        self.time = time
        input_sum = sum([self.weights[i] for i in self.received])
        self.voltage += - self.tau * self.voltage * dt + input_sum
        if self.voltage < 0:
            self.voltage = 0
        self.received = []
        # print("neuron " + self.name + " V: ", int(self.voltage*1000))
        if self.voltage >= self.threshold:
            self.voltage = 0
            self.send_spike()
        self.probe()

    def reset(self):
        self.voltage = 0


class Neuron(NeuronType):
    """'
    test model for a neuron
    """

    objects = []

    def __init__(self, ensemble, index, **kwargs):
        super(Neuron, self).__init__(ensemble, index, **kwargs)
        Neuron.objects.append(self)
        self.voltage = 0
        self.threshold = 1
        self.threshold = self.extract_param('threshold', 1)
        # print("neuron {}, thr: {}".format(self.label, self.threshold))

    def step(self, dt, time):
        self.time = time
        self.voltage += (dt + sum([self.weights[i] for i in self.received]))
        self.received = []
        # print("neuron " + self.name + " V: ", int(self.voltage*1000))
        if self.voltage >= self.threshold:
            self.voltage = 0
            self.send_spike()
        self.probe()

    def reset(self):
        self.voltage = 0
