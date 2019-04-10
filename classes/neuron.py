
from probe import Probe
from base import SimulationObject
import sys
sys.dont_write_bytecode = True


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

    def __init__(self, ensemble, index, *args):
        super(NeuronType, self).__init__()
        self.ensemble = ensemble
        self.index = index
        self.inputs = {}
        self.outputs = []
        self.received = []
        self.is_probed = False
        self.probes = {
            'voltage': {
                'probe': None,
                'value': None
                },
            }
        # TODO: bias

    def add_input(self, source, weight):
        """ Stores in the neuron the reference of the incomming connection
        and its associated weight
        """
        print(self, 'appending', source)
        self.inputs[source] = weight

    def add_output(self, dest):
        """ Append the object the neuron should output into """
        self.outputs.append(dest)

    def receive_spike(self, source):
        """ Append an axons which emitted a received spikes this step """
        # TODO: here the learner can be added
        print("spike received by " + self.name)
        self.received.append(self.inputs[source])

    def send_spike(self):
        """ send a spike to all the connected axons """
        for output in self.outputs:
            output.create_spike()

    def set_probe(self, obj, variable):
        if variable in self.probes:
            self.is_probed = True
            self.probes[variable]['probe'] = obj

    def probe(self):
        for attr, value in self.probes.items():
            var = self.probes[attr]
            if var['probe'] is not None:
                print('probing', attr, var['value']())
                var['probe'].send_value(var['value']())

    def __del__(self):
        print ('Foo' + self.name + 'died')


class Neuron(NeuronType):
    ''''
    test model for a neuron
    '''

    objects = []

    def __init__(self, ensemble, index, threshold=1, name='', *args):
        super(Neuron, self, ensemble, index).__init__()
        self.name = name
        self.voltage = 0
        self.has_spiked = 0
        self.threshold = threshold
        Neuron.objects.append(self)
        print("neuron " + name + " init", threshold)
        self.probes = {
            'voltage': {
                'probe': None,
                'value': self.probe_voltage
                },
            'spike': {
                'probe': None,
                'value': self.has_spiked
            }
        }
        self.pb = Probe(self)

    def probe_voltage(self):
        # TODO: put this in a beatiful decorator
        return self.voltage

    def step(self, dt):
        self.has_spiked = 0
        self.voltage += (dt + sum(self.inputs))
        self.inputs = []
        # print("neuron " + self.name + " V: ", int(self.voltage*1000))
        if self.voltage >= self.threshold:
            self.has_spiked = 1
            print("neuron " + self.name + " spiked")
            self.voltage = 0
            for axon in self.axon_list:
                axon.send_spike()
        self.probe()
