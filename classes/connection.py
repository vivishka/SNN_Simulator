
import numpy as np
from .base import SimulationObject
from .ensemble import Bloc
from .neuron import Weights
import sys
sys.dont_write_bytecode = True


class Axon(SimulationObject):
    """
    Represents the connection between a neuron and a list of neurons

    Parameters
    ---------
    source_e: Ensemble
        The ensemble of the emitting neuron
    source_n: NeuronType
        The emitting neuron
    dest_e: Ensemble
         The ensemble of the receiving neuron

    Attributes
    ----------
    source_e: Ensemble
        The ensemble of the emitting neuron
    source_n: NeuronType
        The emitting neuron
    dest_e: Ensemble
         The ensemble of the receiving neuron
    dest_n_index_list : [(NeuronType, (int, int, int)]
        The list of connected neurons and their associated index
        the first one is the ensemble index, the last two are neuron index
    spike_notifier: SpikeNotifier
        The object to notify when this axon has a spike to propagate

    """
    def __init__(self, source_e, source_n, dest_e):
        # def __init__(self, source_n, dest_n_list, index_list):
        super(Axon, self).__init__("Axon {0}".format(id(self)))

        self.source_e = source_e
        self.source_n = source_n
        self.dest_e = dest_e
        self.dest_n_index_list = []
        self.spike_notifier = None
        # TODO: possibility to add delay / FIR filter

        # associates the source and this axon
        source_n.add_output(self)

    def add_synapse(self, dest_n, index_n):
        """ Associates this axon and the destination"""
        # The weight array index of the source in the destination neuron
        source_e_index = dest_n.weights.check_ensemble_index(self.source_e)
        self.dest_n_index_list.append((dest_n, (source_e_index,) + index_n))
        dest_n.add_input(self)

    def set_notifier(self, spike_notifier):
        """ Used to simulate axons only when they received a spike """
        self.spike_notifier = spike_notifier

    def create_spike(self):
        """ Register a spike emitted by the source neuron"""
        self.spike_notifier(self)

    def propagate_spike(self):
        """ Called by the notifier, propagates spike to connected neurons """
        for dest_n, index in self.dest_n_index_list:
            dest_n.receive_spike(index)


class Connection(SimulationObject):
    """
    A connection is a list of axons connected between 2 ensembles

    Parameters
    ----------
    source: Ensemble or Bloc
         The emitting ensemble
    dest : Ensemble or Bloc
        The receiving ensemble
    pattern: ConnectionPattern
        the way neurons should be connected together
    *args, **kwargs
        The list of arguments to configure the connection

    Attributes
    ----------
    source: Ensemble
         Stores the emitting ensemble
    dest : Ensemble
        Stores the receiving ensemble
    axon_list: [Axon]
        The list of axons
    pattern: ConnectionPattern
        the way neurons should be connected together, default is all to all
    """

    objects = []

    def __init__(self, source_o, dest_o, kernel=(1, 1), *args, **kwargs):
        super(Connection, self).__init__("Connect_{0}".format(id(self)))
        Connection.objects.append(self)
        self.source_o = source_o
        self.dest_o = dest_o
        self.axon_list = []
        self.kernel = (kernel, kernel) if isinstance(kernel, int) else kernel
        self.stride = kwargs['stride'] if 'stride' in kwargs else 1
        self.padding = kwargs['padding'] if 'padding' in kwargs else 0  # TODO: perhaps ?
        self.shared = True if 'shared' in args else False
        self.all2all = True if 'all2all' in args else False

        # depending on the source object type, a specific function is called
        if isinstance(source_o, Bloc):
            connect_function = self.connect_bloc_ensemble
        else:
            connect_function = self.connect_ensemble_ensemble

        # the destination object is turned into a list of ensembles
        dest_e_list = dest_o.ensemble_list if isinstance(dest_o, Bloc) else  [dest_o]

        # connect the source object to the list of ensemble
        for dest_e in dest_e_list:
            connect_function(source_b=source_o, dest_e=dest_e)

    def connect_bloc_ensemble(self, source_b, dest_e):
        """
        Create a connection between neurons of the source and destination
        Always used for convolution, if one2one needed: use connection on each sub ensembles

        Parameters
        ----------
        source_b : Bloc
            Source Bloc
        dest_e : Ensemble
            destination Ensemble
        """
        self.shared = True

        # shares the same weight matrix to all destination neuron
        block_weights = Weights(shared=True)
        for dest_n in dest_e.neuron_list:
            dest_n.set_weights(block_weights)

        # connect each ensemble of the block
        for source_e in source_b.ensemble_list:
            self.connect_ensemble_ensemble(source_e, dest_e)

    def connect_ensemble_ensemble(self, source_e, dest_e):
        """
        Create a connection between neurons of the source and destination
        Initializes the weights of the destination neurons
        one2one: kernel = 1; conv: kernel = (n x m) & weights shared

        Parameters
        ----------
        source_e : Ensemble
            Source Ensemble
        dest_e : Ensemble
            destination Ensemble
        """

        # TODO: better init of the weight depending on dimension
        # weights setting
        kernel_size = self.kernel if not self.all2all else dest_e.size
        if self.shared:
            weights = np.random.rand(*kernel_size) * 1.5 - 0.5
            dest_e.neuron_list[0].weights.set_weights(source_e, weights)
        else:
            for dest_n in dest_e.neuron_list:
                weights = np.random.rand(*kernel_size) * 1.5 - 0.5
                dest_n.weighs.set_weights(source_e, weights)

        # creation of axons for each neuron of the source ensemble
        for row in range(source_e.size[0]):
            for col in range(source_e.size[1]):
                axon = Axon(source_e=source_e, source_n=source_e[(row, col)], dest_e=dest_e)
                self.axon_list.append(axon)

        # connection of each neuron of the destination to the source axons
        for row in range(dest_e.size[0]):
            for col in range(dest_e.size[1]):
                self.connect_kernel(source_e=source_e, dest_e=dest_e, dest_n_index=(row, col))

    def connect_kernel(self, source_e, dest_e, dest_n_index):
        """
        Create a connection between neurons of the source and one neuron of the destination
        all2all: flag set, one2one: kernel = 1; conv: kernel = (n x m) & weights shared

        Parameters
        ----------
        source_e : Ensemble
            Source Ensemble
        dest_e : Ensemble
            destination Ensemble
        dest_n_index: (int, int)
            destination neuron index
        """

        if self.all2all:
            for row in range(source_e.size[0]):
                for col in range(source_e.size[1]):
                    source_axon = source_e.neuron_array[row, col].outputs[-1]
                    source_axon.add_synapse(dest_n=dest_e[dest_n_index], index_n=(row, col))
        else:
            # kernel size 1: 0 to 0, 2: 0 to 1, 3: -1 to 1, ...
            for row in range(-((self.kernel[0] - 1) // 2), self.kernel[0] // 2 + 1):
                source_row = dest_n_index[0] * self.stride + row
                for col in range(-((self.kernel[1] - 1) // 2), self.kernel[1] // 2 + 1):
                    source_col = dest_n_index[1] * self.stride + col

                    # lazy range test but eh #1
                    try:
                        # add the synapse to the source axon (-1: last added axon)
                        source_axon = source_e.neuron_array[source_row, source_col].outputs[-1]
                        source_axon.add_synapse(dest_n=dest_e[dest_n_index], index_n=(row, col))
                    except IndexError:
                        # zero padding
                        pass

    def set_notifier(self, spike_notifier):
        """ Used to simulate axons only when they received a spike """
        for axon in self.axon_list:
            axon.set_notifier(spike_notifier)
