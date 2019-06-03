import logging as log
from .base import Helper, MeasureTiming
from .weights import Weights
import sys
sys.dont_write_bytecode = True


class NeuronType(object):
    """
    The NeuronType object is an abstract version of a neuron
    Used to construct subclass

    Parameters
    ----------
    * args : list
        same
    **kwargs: dict
        The dictionary of arguments passed to initialize the neuron parameters

    Attributes
    ----------
    ensemble: Ensemble
        ensemble this neuron belongs to
    index: (int, int)
        index Ensemble
    param: {str:object}
        The dictionary of arguments passed to initialize the neurons
        Stores the index (in 1D or 2D) of the neuron in the ensemble
    received: [(int, float)]
        List of received spikes containing (index, weight)
    inhibited = False
        self.inhibiting = False
        self.learner
    *_probed: bool
        Stores on which type of variable this neuron is being probed
    probes: {str:Probe}
        Dictionary associating variable name and probe
    """
    nb_neuron = 0

    def __init__(self, *args, **kwargs):
        # super(NeuronType, self).__init__(
        #     "{0}_Neuron_{1}".format(ensemble.label, index))
        NeuronType.nb_neuron += 1
        self.ensemble = None
        self.index_1d = 0
        self.index_2d = (0, 0)
        self.param = kwargs if kwargs is not None else {}
        self.received = []
        self.last_active = 0
        self.halted = False
        self.inhibited = False
        self.inhibiting = False
        self.variable_probed = False
        self.spike_out_probed = False
        self.spike_in_probed = False
        self.probed_values = {}
        self.probes = {}
        self.nb_in = 0
        self.nb_out = 0

        # TODO: change that
        # Helper.log('Neuron', log.DEBUG, '{0} of layer {1} created'.format(self.index, self.ensemble.id))

    def set_ensemble(self, ensemble, index_2d):
        self.ensemble = ensemble
        self.index_1d = Helper.get_index_1d(index_2d, ensemble.size[1])
        self.index_2d = index_2d

    def extract_param(self, name, default):
        param = default
        if self.param is not None and name in self.param:
            if callable(self.param[name]):
                param = self.param[name]()
            else:
                param = self.param[name]
        return param

    # @MeasureTiming('neuron_receive')
    def receive_spike(self, index_1d, weight):
        """ Append an axons which emitted a received spikes this step """
        # TODO: if in spike probing: do here
        self.received.append((index_1d, weight))
        # Helper.log('Neuron', log.DEBUG,
        #            'spike received by neuron {0}, layer {1} of amplitude {2}'
        #            .format(self.index, self.ensemble.id, weight))
        # Helper.log('Neuron', log.DEBUG,'spike received by neuron {0}, layer {1} of amplitude {2}')

    def send_spike(self):
        """ notify the spike to the layer """
        Helper.log('Neuron', log.DEBUG, ' {0} emit spike from layer {1}'.format(self.index_2d, self.ensemble.id))
        self.ensemble.create_spike(self.index_1d)
        # self.ensemble.create_spike(self.index2d)

        if self.spike_out_probed:
            self.probed_values['spike_out'].append((Helper.time, self.index_2d))
            Helper.log('Neuron', log.DEBUG, ' {0} spike notification to probe'.format(self.index_2d))

        self.nb_out += 1
        if self.inhibiting:
            Helper.log('Neuron', log.DEBUG, ' {0} inhibition propagation'.format(self.index_2d))
            self.ensemble.propagate_inhibition(index_2d_n=self.index_2d)

    def add_probe(self, probe, variable):
        self.probes[variable] = probe  # add probe to the dict of variable names
        self.probed_values[variable] = []  # creates the array of probed values
        self.ensemble.probed_neuron_set.add(self)  # probed neurons step every step
        if variable == 'spike_in':
            self.spike_in_probed = True
            Helper.log('Neuron', log.DEBUG, 'probe plugged for input spikes on neuron ' + str(self.index_2d))
        elif variable == 'spike_out':
            self.spike_out_probed = True
            Helper.log('Neuron', log.DEBUG, 'probe plugged for output spikes on neuron ' + str(self.index_2d))
        else:
            self.variable_probed = True

    def probe(self):
        for var, probe in self.probes.items():
            # TODO: check existence
            if var not in 'spike_in, spike_out':
                self.probed_values[var].append((Helper.time, self.__getattribute__(var)))

    def step(self):
        pass

    def reset(self):
        Helper.log('Neuron', log.DEBUG, str(self.index_2d) + ' reset')
        self.received = []
        self.last_active = Helper.step_nb
        self.inhibited = False
        self.halted = False

    def restore(self):
        self.received = []
        self.last_active = 0
        self.inhibited = False
        self.halted = False


class LIF(NeuronType):
    """
    LIF implementation of a neuron

    Parameters
    ----------
    threshold: float
        when voltage exceeds it, a spike is emitted
    tau: float
        rate of the neuron

    Attributes
    ----------
    voltage: float
        internal state of the neuron
        increase when input, decay exponentially
    threshold: float
        neuron parameter, when voltage exceeds it, a spike is emitted
    tau: float
        neuron parameter, decay rate of the neuron
    """

    def __init__(self, threshold=1, tau=1):
        super(LIF, self).__init__()
        self.voltage = 0
        self.threshold = threshold
        self.tau_inv = 1.0 / tau
        Helper.log('Neuron', log.DEBUG, str(self.index_2d) + ' neuron type: LIF')

    # @MeasureTiming('neur_step')
    def step(self):
        if self.inhibited:
            return

        # if variable probed: simulate every step
        self.halted = not self.variable_probed

        # sum inputs
        input_sum = sum(weight for (index, weight) in self.received)
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
            Helper.log('Neuron', log.DEBUG, str() + '{0} voltage {1} exceeds threshold {2}: spike generated'
                       .format(self.index_2d, self.voltage, self.threshold))
            self.voltage = 0
            self.send_spike()

    def reset(self):
        super().reset()
        self.voltage = 0

    def restore(self):
        super().restore()
        self.voltage = 0


class PoolingNeuron(NeuronType):
    """
    once it receive a spike, from any of its dendrite, propagate it
    if configured as Winner Takes it All (wta), only the first spike
    is transmitted per input cycle
    Parameters
    ----------
    **kwargs
        Same as NeuronType
        wta is passed using this argument

    Attributes
    ----------
    wta: bool
        Winner Takes it All mode
    """

    def __init__(self, ensemble, index, **kwargs):
        super(PoolingNeuron, self).__init__(ensemble, index, **kwargs)
        self.wta = self.extract_param('wta', False)
        Helper.log('Neuron', log.DEBUG, ' neuron type: pooling')

    def step(self):
        if self.received and not self.inhibited:
            self.received = []
            self.send_spike()
            if self.wta:
                self.inhibited = True
