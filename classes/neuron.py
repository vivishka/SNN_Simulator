
from .base import SimulationObject, Helper
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
        self.ensemble_index_dict = {}
        self.ensemble_number = 0
        self.shared = shared

    def check_ensemble_index(self, source_e):
        if source_e not in self.ensemble_index_dict:
            self.ensemble_index_dict[source_e] = self.ensemble_number
            self.ensemble_number += 1
            self.weights.append(None)
        return self.ensemble_index_dict[source_e]

    def set_weights(self, source_e, weight_array):
        """ sets the weights of the axons from the specified ensemble """
        ens_number = self.check_ensemble_index(source_e)
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
        but multiple axons are used here to connect with multiple ensembles
    weights: Weights
        Dictionary to store weights attached to connections
        Can be shared between neurons of the same ensemble
    received: [Axon]
        List of axons which emitted a received spikes this step
    inhibited = False
        self.inhibiting = False
        self.learner
    *_probed: bool
        Stores on which type of variable this neuron is being probed
    probes: {str:Probe}
        Dictionary associating variable name and probe
    """
    nb_neuron = 0

    def __init__(self, ensemble, index, **kwargs):
        super(NeuronType, self).__init__(
            "{0}_Neuron_{1}".format(ensemble.label, index))
        NeuronType.nb_neuron += 1
        self.ensemble = ensemble
        self.index = index
        self.param = kwargs if kwargs is not None else {}
        self.inputs = []
        self.outputs = []
        self.weights = Weights()
        self.received = []
        self.last_active = 0
        self.halted = False
        self.inhibited = False
        self.inhibiting = False
        self.learner = None
        self.variable_probed = False
        self.spike_out_probed = False
        self.spike_in_probed = False
        self.probed_values = {}
        self.probes = {}
        self.nb_in = 0
        self.nb_out = 0

    def extract_param(self, name, default):
        param = default
        if self.param is not None and name in self.param:
            if callable(self.param[name]):
                param = self.param[name]()
            else:
                param = self.param[name]
        return param

    def add_input(self, source_a):
        """ Stores in the neuron the reference of the incoming connection.
        is used to count the number of connections to this neuron
        """
        self.inputs.append(source_a)

    def add_output(self, dest_a):
        """ Append the object the neuron should output into
        will call the method create_spike of this object
        """
        self.outputs.append(dest_a)

    def set_weights(self, weights):
        """" used for convolutional connections, when kernel is shared """
        self.weights = weights

    def receive_spike(self, index):
        """ Append an axons which emitted a received spikes this step """
        if self.halted:
            self.ensemble.active_neuron_list.append(self)
            self.halted = False
        self.received.append(index)
        if self.learner is not None:
            self.learner.in_spike(index)
        if self.spike_in_probed:
            weight = self.weights[index]
            self.probed_values['spike_in'].append(Helper.time, index, weight)
        self.nb_in += 1

    def send_spike(self):
        """ send a spike to all the connected axons """
        if self.learner is not None:
            self.learner.out_spike()
        for output in self.outputs:
            output.create_spike()
        if self.spike_out_probed:
            self.probed_values['spike_out'].append(Helper.time, self.index)
        self.nb_out += 1
        if self.inhibiting:
            self.ensemble.propagate_inhibition(index_n=self.index)

    def add_probe(self, probe, variable):
        self.probes[variable] = probe
        self.probed_values[variable] = []
        self.ensemble.probed_neuron_set.add(self)
        if variable == 'spike_in':
            self.spike_in_probed = True
        elif variable == 'spike_out':
            self.spike_out_probed = True
        else:
            self.variable_probed = True

    def probe(self):
        for var, probe in self.probes.items():
            # TODO: check existence
            if var not in 'spike_in, spike_out':
                self.probed_values[var].append((Helper.time, self.__getattribute__(var)))
                # probe.log_value(self.index, self.__getattribute__(var))

    def step(self):
        pass

    def reset(self):
        self.received = []
        self.last_active = 0
        self.inhibited = False
        self.halted = False


class LIF(NeuronType):
    """
    LIF implementation of a neuron

    Parameters
    ----------
    ensemble: Ensemble
        Same as NeuronType
    index: (int, int)
        Same as NeuronType
    **kwargs
        Same as NeuronType
        voltage, threshold and tau parameters are passed using this argument

    Attributes
    ----------
    voltage: float
        Stores the ensemble this neuron belongs to
    threshold: float
        neuron parameter, when voltage exceeds it, a spike is emitted
    tau
        neuron parameter, decay rate of the neuron
    """

    def __init__(self, ensemble, index, **kwargs):
        super(LIF, self).__init__(ensemble, index, **kwargs)
        Neuron.objects.append(self)
        self.voltage = 0
        self.threshold = self.extract_param('threshold', 1)
        self.tau_inv = 1.0 / self.extract_param('tau', 0.2)

    def step2(self):
        # Helper.nb +=1
        if self.inhibited:
            return

        input_sum = sum([self.weights[index] for index in self.received])
        self.voltage += - self.tau_inv * self.voltage * Helper.dt + input_sum

        if self.voltage < 0:
            self.voltage = 0
        self.received = []
        if self.voltage >= self.threshold:
            self.voltage = 0
            self.send_spike()
        self.probe()

    def step(self):
        if self.inhibited:
            return

        # if variable probed: simulate every step
        self.halted = not self.variable_probed

        # sum inputs
        input_sum = sum([self.weights[index] for index in self.received])
        self.received = []

        # interpolation of the state
        elapsed_steps = Helper.step_nb - self.last_active
        self.last_active = Helper.step_nb
        self.voltage = self.voltage * (1 - self.tau_inv * Helper.dt) ** elapsed_steps + input_sum

        # probing
        if self.variable_probed:
            self.probe()

        # spiking
        if self.voltage >= self.threshold:
            # print("neuron {0} spiked".format(self.label))
            self.voltage = 0
            self.send_spike()

    def reset(self):
        super().reset()
        self.voltage = 0


class PoolingNeuron(NeuronType):
    """
    once it receive a spike, from any of its dendrite, propagate it
    """

    def __init__(self, ensemble, index, **kwargs):
        super(PoolingNeuron, self).__init__(ensemble, index, **kwargs)

    def step(self):
        if self.received and not self.inhibited:
            self.received = []
            self.send_spike()


class Neuron(NeuronType):
    """'
    test model for a neuron
    used for debug
    """

    objects = []

    def __init__(self, ensemble, index, **kwargs):
        super(Neuron, self).__init__(ensemble, index, **kwargs)
        Neuron.objects.append(self)
        self.voltage = 0
        self.threshold = 1
        self.threshold = self.extract_param('threshold', 1)
        # print("neuron {}, thr: {}".format(self.label, self.threshold))

    def step(self):
        self.voltage += (Helper.dt + sum([self.weights[i] for i in self.received]))
        self.received = []
        # print("neuron " + self.name + " V: ", int(self.voltage*1000))
        if self.voltage >= self.threshold:
            self.voltage = 0
            self.send_spike()
        self.probe()

    def reset(self):
        self.voltage = 0
