import logging as log
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import copy
import math
from .base import SimulationObject, Helper, MeasureTiming
from .layer import Bloc
from .weights import Weights
import sys
sys.dont_write_bytecode = True


class Connection(SimulationObject):
    """
    A Connection is represents a matrix of weights between 2 ensembles
    when connected to Blocs, will create a list of connection for every Ensemble
    in that case, the mother Connection is only used to store the other Connections

    Parameters
    ----------
    :param source_l: The emitting layer
    :type source_l: Layer
    :param dest_l : The receiving ensemble
    :type dest_l: Layer
    :param wmin: minimum weight value allowed
    :type wmin: float
    :param wmax: maximum weight value allowed
    :type wmax: float
    :param mu: mean value for weight distribution
    :type mu: float
    :param sigma: standard deviation for weight distribution
    :type sigma: float
    :param kernel_size: if specified, the kernel linking the 2 layers; if not, layers will be fully connected
    :type kernel_size: int or (int, int) or None
    :param mode: dense (None), shared, split or pooling. how are the neurons connected between the ensembles
    :type mode: str
    :param integer_weight:
    :type integer_weight: bool
    :param *args: list of arguments to configure the connection
    :param **kwargs: dict of arguments to configure the connection

    Attributes
    ----------
    :ivar id: global ID of the Connection
    :type id: int
    :ivar connection_list: list of the created sub connections
    :type connection_list: list of Connection
    :ivar active: True if a connection between 2 Ensembles, False if between 2 Blocs (parent connection)
    :type active: bool
    :ivar weights: stores the weights of the Connection
    :type weights: Weights or None for the parent connection
    :ivar source_e: emitting ensemble
    :type source_e: Ensemble
    :ivar dest_e : receiving ensemble
    :type dest_e: Ensemble
    :ivar in_neurons_spiking: stores the neuron received the previous step to propagate them
    :type in_neurons_spiking: list of int
    :ivar is_probed: true when connection is probed
    :type is_probed: bool
    :ivar probed_values: stores the new weights after they are updated
    :type probed_values: list of Weights
    """

    objects = []
    con_count = 0
    @MeasureTiming('con_init')
    def __init__(
            self, source_l, dest_l,
            wmin=0, wmax=1, mu=0.8, sigma=0.5,
            kernel_size=None, mode=None, integer_weight=False,
            *args, **kwargs):

        super(Connection, self).__init__("Connect_{0}".format(id(self)))
        Connection.objects.append(self)
        self.id = Connection.con_count
        Connection.con_count += 1
        self.connection_list = []
        self.mode = mode  # shared, pooling, split
        self.active = False
        self.in_neurons_spiking = []
        self.source_e = source_l
        self.dest_e = dest_l
        self.is_probed = False
        self.probed_values = []
        self.wmin = wmin
        self.wmax = pow(2, wmax) if integer_weight else wmax
        self.size = None
        self.integer_weight = integer_weight
        Helper.log('Connection', log.INFO, 'new connection {0} created between layers {1} and {2}'
                   .format(self.id, source_l.id, dest_l.id))

        # check if connection is from ensemble to ensemble, generate sub-connections if needed recursively
        if isinstance(source_l, Bloc) or isinstance(dest_l, Bloc):
            Helper.log('Connection', log.INFO, 'meta-connection detected, creating sub-connections')
            first = True
            if mode == 'pooling':
                if source_l.depth != dest_l.depth:
                    Helper.log('Connection', log.CRITICAL, 'different depth for pooling connection')
                    raise Exception("different depth for pooling connection")
                # Helper.print_progress(0, len(dest_l.ensemble_list), "Init connection ", bar_length=30)
                for l_ind, l_out in enumerate(dest_l.ensemble_list):
                    self.connection_list.append(Connection(source_l=source_l.ensemble_list[l_ind],
                                                           dest_l=l_out,
                                                           wmin=self.wmin, wmax=wmax,
                                                           mu=mu, sigma=sigma,
                                                           kernel_size=kernel_size, mode=mode,
                                                           first=first, connection=self,
                                                           integer_weight=integer_weight, *args, **kwargs))
                    first = False
            else:
                i = 0
                for l_out in dest_l.ensemble_list:
                    for l_in in source_l.ensemble_list:
                        self.connection_list.append(Connection(source_l=l_in, dest_l=l_out,
                                                               wmin=self.wmin, wmax=wmax,
                                                               mu=mu, sigma=sigma,
                                                               kernel_size=kernel_size, mode=mode,
                                                               first=first, connection=self,
                                                               integer_weight=integer_weight, *args, **kwargs))
                        i += 1
                        first = False
            self.weights = None
        else:
            source_l.out_connections.append(self)
            dest_l.in_connections.append(self)
            self.dest_e = dest_l
            self.source_e = source_l
            self.weights = Weights(
                source_dim=self.source_e.size,
                dest_dim=self.dest_e.size,
                kernel_size=kernel_size,
                mode=mode,
                wmin=wmin, wmax=wmax,
                mu=mu, sigma=sigma,
                **kwargs)
            self.active = True
            self.connection_list = [self]
            self.size = (source_l.size[0], dest_l.size[0])

    def register_neuron(self, index_1d):
        """ Registers the index of the source neurons that spiked"""
        # index_1d = Helper.get_index_1d(index_2d=index, length=self.source_e.size[0])
        self.in_neurons_spiking.append(index_1d)
        Helper.log('Connection', log.DEBUG, ' neuron {}/{} registered for receiving spike'.format(index_1d, index_1d))

    @MeasureTiming('con_step')
    def step(self):
        """
        Propagates all the spikes emitted from the source ensemble to the dest ensemble
        """
        for index_1d in self.in_neurons_spiking:

            targets = self.weights[index_1d]  # source dest weight
            # for target in targets:
            self.dest_e.receive_spike(targets=targets, source_c=self)
            # This log 10-15% of this function time
            # Helper.log('Connection', log.DEBUG, 'spike propagated from layer {0} to {1}'
            #            .format(self.source_e.id, self.dest_e.id))
        self.in_neurons_spiking = []

    @MeasureTiming('con_weight_copy')
    def get_weights_copy(self):
        """ Returns a copy of the weight matrix """
        return copy.deepcopy(self.weights.matrix.get_all_weights())

    def probe(self):
        """ stores the weight matrix to be analyzed later. Called every batch"""
        if self.is_probed:
            self.probed_values.append(self.get_weights_copy())

    def add_probe(self):
        """ Notify the Connection to probe itself"""
        self.is_probed = True
        self.probed_values = [self.get_weights_copy()]

    def __getitem__(self, index):
        """
        :param index: connection index
        :rtype: Connection
        """
        return self.connection_list[index]

    def restore(self):
        """ Restore the connection and weights """
        self.in_neurons_spiking = []
        if self.weights:
            self.weights.restore()

    def get_convergence(self):
        """
        Compute the convergence score of the weights
        based on th formula sum of [ (wmax - w) * (w - wmin) ]
        lower score mean weight more saturated
        :return: convergence score
        :rtype: float
        """
        conv = 0
        if self.active:
            for row in range(self.size[0]):
                for col in range(self.size[1]):
                    weight = self.weights[col, row]
                    conv += (weight - self.wmin)*(self.wmax - weight)
        else:
            for con in self.connection_list:
                conv += con.get_convergence()
        return conv

    def plot_convergence(self):
        """
        plot the evolution of the convergence score over time
        needs to be probed first
        """
        conv = np.zeros(len(self.connection_list[0].probed_values))

        for con in self.connection_list:
            for index, weights in enumerate(con.probed_values):
                for w in weights:
                    conv[index] += (w - self.wmin)*(self.wmax - w)
        plt.figure()
        plt.plot(conv)

    def update_weight(self, source, dest, value):
        """
        Used for
        :param source: source neuron index
        :type source: int
        :param dest: destination neuron index
        :type dest: int
        :param value: weight delta
        :type value: float
        """
        if self.wmin < self.weights.matrix[source, dest] + value < self.wmax:
            self.weights.matrix[source, dest] += value
        if self.integer_weight:
            if abs(value) < 1:
                value = np.sign(value)
            value = int(value)
            if self.wmin < self.weights.matrix[source, dest] + value < self.wmax:
                self.weights.matrix[source, dest] += value

    def plot(self):
        """
        for shared kernel connections only, visual representation of the kernels
        Deprecated
        """
        images = []
        if self.active:
            return self.weights.matrix.get_kernel(None, None, None)
        else:
            for con in self.connection_list:
                images.append(con.plot())

            ncols = int(np.sqrt(len(images)))
            nrows = math.ceil(len(images)/ncols)
            fig, ax = plt.subplots(ncols=ncols, nrows=nrows)
            fig.suptitle("Connection final kernels")
            norm = colors.Normalize(vmin=0, vmax=1)
            for index, image in enumerate(images):
                ax[index // ncols, index % ncols].imshow(image, cmap='gray', norm=norm)

    def plot_all_kernels(self, nb_source_max=None, nb_dest_max=None):
        """
        for shared kernel connections only, visual representation of the kernels
        :param nb_source_max: maximum number of line to display
        :param nb_dest_max:  maximum number of columns to display
        or the other way around...
        """
        nb_source = len(self.source_e.ensemble_list)
        if nb_source_max is not None and nb_source_max < nb_source:
            nb_source = nb_source_max
        nb_dest = len(self.dest_e.ensemble_list)
        if nb_dest_max is not None and nb_dest_max < nb_dest:
            nb_dest = nb_dest_max
        fig = plt.figure()
        fig.patch.set_facecolor('xkcd:light blue')
        for source_i in range(nb_source):
            for dest_i in range(nb_dest):
                con = self.connection_list[dest_i * nb_source + source_i]
                plt.subplot(nb_source, nb_dest, source_i * nb_dest + dest_i + 1)
                kernel = con.weights.matrix.get_kernel(None, None, None)
                plt.imshow(kernel, cmap='gray')
                plt.axis('off')

        fig.suptitle("Connection final kernels")

    def load(self, weights):
        """
        load weights into connection
        :param weights:
        :type weights: object np.ndarray of dim 4
        dim 0: nb dest ensemble
        dim 1: nb source ensemble
        dim 2: nb source neuron
        dim 3: nb dest neuron
        """
        nb_source = len(self.source_e.ensemble_list)
        nb_dest = len(self.dest_e.ensemble_list)

        if len(weights.shape) != 4:
            raise ValueError("not enough dimensions to the weight array")

        if nb_source != weights.shape[1] or nb_dest != weights.shape[0]:
            raise ValueError("Wrong number of source / dest ensemble")

        if self.active:
            raise ValueError('Can only load on parent connection')

        if self.connection_list[0].weights.mode == 'shared':

            if weights.shape[-2:] != self.connection_list[0].weights.kernel_size:
                Helper.log('Connection', log.ERROR, 'wrong size of kernel weights')
                raise ValueError("Wrong size of kernel when loading")

            for source_i in range(nb_source):
                for dest_i in range(nb_dest):
                    kernel = weights[dest_i, source_i]
                    self.connection_list[dest_i * nb_source + source_i].weights.matrix.kernel = kernel

        elif self.connection_list[0].weights.kernel_size is None:

            if weights.shape[-2] != len(self.connection_list[0].source_e.neuron_list) \
                    or weights.shape[-1] != len(self.connection_list[0].dest_e.neuron_list):
                Helper.log('Connection', log.ERROR, 'wrong number of weight')
                raise Exception("Wrong number of weight when loading")

            for source_i in range(nb_source):
                for dest_i in range(nb_dest):
                    w = weights[dest_i, source_i]
                    matrix = self.connection_list[dest_i * nb_source + source_i].weights.matrix
                    for row in range(matrix.shape[0]):
                        for col in range(matrix.shape[1]):
                            matrix[row, col] = w[row, col]

    def saturate_weights(self, threshold=None):
        """
        Makes the weights of the matrix binary
        :param threshold: threshold which separates the categories; if not specified, will be the middle of min and max
        :type threshold: float
        """
        if threshold is None:
            threshold = (self.wmax + self.wmin) / 2
        for con in self.connection_list:
            con.weights.matrix.saturate_weights(wmin=self.wmin, wmax=self.wmax, threshold=threshold)

    def set_max_weight(self, wmax):
        """
        changes the maximum allowed weight for a connection and its suf connections
        :param wmax: new max weight
        :type wmax: float
        """
        if self.active:
            self.wmax = wmax
        else:
            for con in self.connection_list:
                con.set_max_weight(wmax)


class DiagonalConnection(Connection):
    """
    Special connection used only to connect a GFR encoder to a decoder
    mainly used for debug
    """
    
    def __init__(self, source_l, dest_l):
        super(DiagonalConnection, self).__init__(source_l, dest_l, 0, 1, kernel=None,)
        for i, connection in enumerate(self.connection_list):
            for col in range(connection.weights.matrix.shape[1]):
                if col != i:
                    connection.weights[(0, col)] = 0.
