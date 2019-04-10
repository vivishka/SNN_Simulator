
from base import SimulationObject
import sys
sys.dont_write_bytecode = True


class Weights(object):
    """docstring for Weights."""

    def __init__(self, shared=False):
        super(Weights, self).__init__()
        self.weights = {}
        self.shared = shared

    def __getitem__(self, key):
        return self.weights[key]

    def __setitem__(self, key, value):
        self.weights[key] = value


class NeuronType(SimulationObject):
    '''
    The NeuronType object is an abstract version of a neuron
    Used to construct subclass

    Parameters
    ----------
    ensemble: Ensemble
        The ensemble this neuron belongs to
    index: int or (int, int)
        The index (in 1D or 2D) of the neuron in the ensemble
    *args
        The list of arguments passed to initialize the neurons

    Attributes
    ----------
    ensemble: Ensemble
        Stores the ensemble this neuron belongs to
    index: int or (int, int)
        Stores the index (in 1D or 2D) of the neuron in the ensemble
    inputs: {Axon:float}
        dictionary to store weights attached to connections
    outputs: [Axon]
        list of axons the neuron can output to most neuron only have 1 axon,
        but multpiple axons are used here to connect with multiple ensembles
    received: [Axon]
        list of axons which emitted a received spikes this step
    '''

    def __init__(self, ensemble, index, *args, **kwargs):
        # TODO: give a kwargs with an array: neuron can use its own index
        super(NeuronType, self).__init__(
            "{0}_Neuron_{1}".format(ensemble.label, index))
        self.ensemble = ensemble
        self.index = index
        print(kwargs)
        self.param = kwargs['param'] if 'param' in kwargs else None
        # TODO: rename inputs, possibility to share weights when convo
        self.inputs = []
        self.outputs = []
        self.weights = Weights()
        self.received = []
        self.variable_probed = False
        self.spike_out_probed = False
        self.spike_out_probed = False
        self.probes = {}
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
        # print(self, 'appending', source)
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
        print("spike received by " + self.label)
        self.received.append(source)

    def send_spike(self):
        """ send a spike to all the connected axons """
        for output in self.outputs:
            output.create_spike()
        if self.spike_out_probed:
            self.probes['spike_out'].send_value(self.time)

    def set_probe(self, obj, variable):
        self.probes[variable] = obj
        if variable == 'spike_in':
            self.spike_in_probed = True
        elif variable == 'spike_out':
            self.spike_out_probed = True
        else:
            self.variable_probed = True

    def probe(self):
        for var, probe in self.probes.items():
            # print("probing {}: {}".format(var, self.__getattribute__(var)))
            probe.send_value(self.index, self.__getattribute__(var))


class Neuron(NeuronType):
    ''''
    test model for a neuron
    '''

    objects = []

    def __init__(self, ensemble, index, *args, **kwargs):
        super(Neuron, self).__init__(ensemble, index, *args, **kwargs)
        Neuron.objects.append(self)
        self.voltage = 0
        self.threshold = 1
        self.threshold = self.extract_param('threshold', 1)
        print("neuron {}, thr: {}".format(self.label, self.threshold))
        # self.pb = Probe(self)

    def probe_voltage(self):
        # TODO: put this in a beatiful decorator
        return self.voltage

    def step(self, dt):
        self.time += dt
        self.voltage += (dt + sum([self.weights[i] for i in self.received]))
        self.received = []
        # print("neuron " + self.name + " V: ", int(self.voltage*1000))
        if self.voltage >= self.threshold:
            print("neuron " + self.label + " spiked")
            self.voltage = 0
            self.send_spike()
        self.probe()
