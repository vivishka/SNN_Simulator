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
    source_l: Layer
         The emitting layer
    dest_l : Layer
        The receiving ensemble
    kernel: int or (int, int) or None
        if specified, the kernel linking the 2 layers
        if not, layers will be fully connected
    *args, **kwargs
        The list of arguments to configure the connection

    Attributes
    ----------
    id: int
        global ID of the Connection
    connection_list: list[Connection]
        list of the created sub connections
    weights: Weights
        stores the weights of the Connection
    active: bool
        True if a connection between 2 Ensembles
    source_e: Ensemble
        emitting ensemble
    dest_e : Ensemble
        receiving ensemble
    in_neurons_spiking: list
        stores the neuron received the previous step to propagate them
    is_probed: bool
        true when connection is probed
    probed_values: list[Weights]
        stores the new weights after they are updated
    """

    objects = []
    con_count = 0
    @MeasureTiming('con_init')
    def __init__(self, source_l, dest_l, wmin=0, wmax=1, kernel=None, mode=None, real=False, *args, **kwargs):

        super(Connection, self).__init__("Connect_{0}".format(id(self)))
        Connection.objects.append(self)
        self.id = Connection.con_count
        Connection.con_count += 1
        self.connection_list = []
        self.mode = mode  # shared, pooling
        self.active = False
        self.in_neurons_spiking = []
        self.source_e = source_l
        self.dest_e = dest_l
        self.is_probed = False
        self.probed_values = []
        self.wmin = wmin
        self.wmax = pow(2, wmax) if real else wmax
        self.size = None
        self.real = real
        Helper.log('Connection', log.INFO, 'new connection {0} created between layers {1} and {2}'
                   .format(self.id, source_l.id, dest_l.id))

        # check if connection is from ensemble to ensemble, generate sub-connections if needed recursively
        if isinstance(source_l, Bloc) or isinstance(dest_l, Bloc):
            Helper.log('Connection', log.INFO, 'meta-connection detected, creating sub-connections')
            first = True
            if mode == 'pooling':
                for l_ind, l_out in enumerate(dest_l.ensemble_list):
                    # TODO: range check: same depth
                    self.connection_list.append(Connection(source_l=source_l.ensemble_list[l_ind],
                                                           dest_l=l_out,
                                                           wmin=self.wmin, wmax=wmax,
                                                           kernel=kernel, mode=mode,
                                                           first=first, connection=self,
                                                           real=real, *args, **kwargs))
                    first = False
            else:
                for l_out in dest_l.ensemble_list:
                    for l_in in source_l.ensemble_list:
                        self.connection_list.append(Connection(source_l=l_in, dest_l=l_out,
                                                               wmin=self.wmin, wmax=wmax,
                                                               kernel=kernel, mode=mode,
                                                               first=first, connection=self,
                                                               real=real, *args, **kwargs))
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
                kernel_size=kernel,
                mode=mode,
                wmin=wmin,
                wmax=self.wmax,
                **kwargs)
            self.active = True
            self.connection_list = [self]
            self.size = (source_l.size[1], dest_l.size[0]) #TODO: check

    def register_neuron(self, index_1d):
        """ Registers the index of the source neurons that spiked"""
        # index_1d = Helper.get_index_1d(index_2d=index, length=self.source_e.size[0])
        self.in_neurons_spiking.append(index_1d)
        Helper.log('Connection', log.DEBUG, ' neuron {}/{} registered for receiving spike'.format(index_1d, index_1d))

    @MeasureTiming('con_step')
    def step(self):
        for index_1d in self.in_neurons_spiking:

            targets = self.weights.get_target_weights(index_1d)  # source dest weight
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

    def __getitem__(self, item):
        return self.connection_list[item]

    def restore(self):
        self.in_neurons_spiking = []
        if self.weights:
            self.weights.restore()

    def share_weight(self):
        ens_con_dict = {ens: [] for ens in self.connection_list[0].dest_e.bloc.ensemble_list}
        for connect in self.connection_list:
            ens_con_dict[connect.dest_e].append(connect)

        for connect_list in ens_con_dict.values():
            for connect in connect_list:
                connect.weights = connect_list[0].weights

    def get_convergence(self):
        conv = 0
        if self.active:
            for row in range(self.size[0]):
                for col in range(self.size[1]):
                    conv += (self.weights[col, row] - self.wmin)*(self.wmax - self.weights[row, col])
        else:
            for con in self.connection_list:
                conv += con.get_convergence()
        return conv

    def update_weight(self, x, y, value):
        # print(value)
        if self.wmin < self.weights.matrix[x, y] + value < self.wmax:
            self.weights.matrix[x, y] += value
        if self.real:
            if abs(value) < 1:
                value = np.sign(value)
            value = int(value)
            if self.wmin < self.weights.matrix[x, y] + value < self.wmax:
                self.weights.matrix[x, y] += value

    def plot(self):
        images = []
        if self.active:
            return self.weights.matrix.get_kernel()
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
        nb_source = len(self.source_e.ensemble_list)
        if nb_source_max is not None and nb_source_max < nb_source:
            nb_source = nb_source_max
        nb_dest = len(self.dest_e.ensemble_list)
        if nb_dest_max is not None and nb_dest_max < nb_dest:
            nb_dest = nb_dest_max
        fig = plt.figure()
        fig.patch.set_facecolor('xkcd:light blue')
        # fig, ax = plt.subplots(nrows=nb_source, ncols=nb_dest)
        # norm = colors.Normalize(vmin=self.connection_list[0].wmin, vmax=self.connection_list[0].wmax)
        for source_i in range(nb_source):
            for dest_i in range(nb_dest):
                con = self.connection_list[dest_i * nb_source + source_i]
                plt.subplot(nb_source, nb_dest, source_i * nb_dest + dest_i + 1)
                kernel = con.weights.matrix.get_kernel()
                plt.imshow(kernel, cmap='gray'  )
                plt.axis('off')

        fig.suptitle("Connection final kernels")

    def load(self, weights):
        if weights.shape[-2:] != self.connection_list[0].weights.kernel_size:
            Helper.log('Connection', log.ERROR, 'wrong size of kernel when loading')
            raise Exception("Wrong size of kernel when loading")
        if not self.active:
            nb_source = len(self.source_e.ensemble_list)
            nb_dest = len(self.dest_e.ensemble_list)

            if nb_source != weights.shape[1] or nb_dest != weights.shape[0]:
                raise Exception("Wrong number of weight when loading")

            for source_i in range(nb_source):
                for dest_i in range(nb_dest):
                    kernel = weights[dest_i, source_i]
                    self.connection_list[dest_i * nb_source + source_i].weights.matrix.kernel = kernel


class DiagonalConnection(Connection):
    
    def __init__(self, source_l, dest_l):
        super(DiagonalConnection, self).__init__(source_l, dest_l, 0, 1, kernel=None,)
        for i, connection in enumerate(self.connection_list):
            for col in range(connection.weights.matrix.shape[1]):
                if col != i:
                    connection.weights[(0, col)] = 0.


# class RealConnection(Connection):
#     def __init__(self, source_l, dest_l, wmin=0, wmax=8, kernel=None, mode=None, *args, **kwargs):
#         assert(isinstance(wmax, int))
#         wmax = 2 ^ wmax
#         super(RealConnection, self).__init__(source_l, dest_l, wmin=wmin, wmax=wmax, kernel=kernel, mode=mode, *args, **kwargs)
#         if self.active and mode == 'shared':
#             for i in range(kernel[0]):
#                 for j in range(kernel[1]):
#                     self.weights.matrix.kernel[i, j] = int(self.weights.matrix.kernel[i, j])
#
#     def update_weight(self, x, y, value):
#         if abs(value) < 1:
#             value = np.sign(value)
#         value = int(value)
#         if self.wmin < self.weights.matrix[x, y] + value < self.wmax:
#             self.weights.matrix[x, y] += value
