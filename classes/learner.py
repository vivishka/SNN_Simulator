
from .base import Helper, MeasureTiming
# from .neuron import NeuronType
# from .ensemble import Ensemble
import copy
import numpy as np
import logging as log


class Learner(object):
    """
    Basic STDP learner. depends on the time of the spike and not the previous weights

    Parameters
    ---------
    :param eta_up: weight increase coefficient
    :type eta_up: float
    :param eta_down: weight decrease coefficient
    :type eta_down: float
    :param tau_up: sensitivity to pre synaptic delay
    :type tau_up: float
    :param tau_down: sensitivity to post synaptic delay
    :type tau_down: float
    :param mp: activate multiprocessing
    :type mp:bool

    Attributes
    ----------
    :ivar layer: Ensemble or Block this learner is monitoring
    :type layer: Layer
    :ivar size: number of neuron in the layer
    :type size: int
    :ivar buffer_in: array of list of received spikes this input cycle
        index is dest neuron index
        value is (time, source_n , weight, source_c)
    :type buffer_in: array of list of (float, int, Connection, float)
    :ivar buffer_in_empty: array of empty list, used to reset the buffer_in
    :type buffer_in_empty: array of list
    :ivar buffer_out: list of emitted spikes this input cycle, ordered by time of emission
    :type buffer_out: list of (float, int)
    :ivar in_spikes: list of buffer_in, one array for each input cycle
    :type in_spikes: list of list of (float,int, Connection, float)
    :ivar out_spikes: list of buffer_out, one array for each input cycle
    :type out_spikes: list of list of (float, int)
    :ivar active: is learning ?
    :type active: bool
    """

    def __init__(self, eta_up=0.1, eta_down=-0.1, tau_up=1, tau_down=1, mp=False):
        self.layer = None
        self.size = 0
        self.eta_up = eta_up
        self.eta_down = eta_down
        self.tau_up = tau_up
        self.tau_down = tau_down
        self.buffer_in = []  # [time, source_n, source_c, weight]
        self.buffer_in_empty = None
        self.buffer_out = []  # dest_n: [time, source_n, batch_id]
        self.in_spikes = []
        self.out_spikes = []
        self.active = True
        self.mp = mp
        self.updates = {}

        Helper.log('Learner', log.INFO, 'Learner initialized TODO: change log'.format())

    def set_layer(self, layer):
        """
        Called by the layer after its init. Now layer and learner are both linked
        :param layer: the layer linked to this learner
        :type layer: Layer
        :return:
        """
        self.layer = layer
        self.size = layer.size[0] * layer.size[1]
        self.buffer_in_empty = np.ndarray(self.size, dtype=list)
        for i in range(self.size):
            self.buffer_in_empty[i] = []
        self.buffer_in = copy.deepcopy(self.buffer_in_empty)

    def in_spike(self, source_n, dest_n, weight, source_c):
        """
        Called when a neuron in the layer receives a spike
        :param source_n: source neuron 1D index
        :type source_n: int
        :param dest_n: destination neuron 1D index
        :type dest_n: int
        :param weight: weight of the spike
        :type weight: float
        :param source_c: reference of the connection which propagated this spike
        :type source_c: Connection
        :return:
        """
        self.buffer_in[dest_n].append([self.layer.sim.curr_time, source_n, source_c, weight])

    def out_spike(self, source_n):
        """
        Called when a neuron in the layer emits a spike
        :param source_n: source neuron 1D index
        :type source_n: int
        :return:
        """
        if self.active:
            self.buffer_out.append((self.layer.sim.curr_time, source_n))
            Helper.log('Learner', log.DEBUG, 'Learner of ensemble {0} registered output spike from {1}'
                       .format(self.layer.id, source_n))
            Helper.log('Learner', log.DEBUG, 'Appended {} to buffer'.format(self.buffer_out[-1]))

    def reset_input(self):
        """
        Called every input cycle.
        Saves and empty the in and out buffers into the list that store the batch
        :return:
        """
        Helper.log('Learner', log.DEBUG, 'Learner reset')
        Helper.log('Learner', log.DEBUG, 'Appended {} to memory'.format(self.buffer_out))
        self.in_spikes.append(self.buffer_in)
        self.out_spikes.append(self.buffer_out)
        if self.buffer_out:
            Helper.log('Learner', log.DEBUG, 'Appended {} to buffer'.format(self.buffer_out[-1]))
        else:
            Helper.log('Learner', log.DEBUG, 'Appended empty buffer')
        self.buffer_in = copy.deepcopy(self.buffer_in_empty)
        self.buffer_out = []
        Helper.log('Learner', log.DEBUG, 'Learner of ensemble {0} reset for next input'.format(self.layer.id))

    def process(self):
        """
        Called every batch.
        process each input cycle of the batch.
        :return:
        """
        Helper.log('Learner', log.DEBUG, 'Processing learning ensemble {0}'.format(self.layer.id))
        # for each experiment in the batch that ends
        for experiment_index in range(self.layer.sim.batch_size):
            Helper.log('Learner', log.DEBUG, 'Processing input cycle {}'.format(experiment_index))

            # for each spike emitted by the Ensemble during this experiment
            for out_s in self.out_spikes[experiment_index]:
                Helper.log('Learner', log.DEBUG, "Processing output spike of neuron {}".format(out_s[1]))
                dest_n = out_s[1]

                # for all the spikes in_s received by the same neuron which emitted out_s
                for in_s in self.in_spikes[experiment_index][dest_n]:
                    source_n = in_s[1]
                    connection = in_s[2]
                    weight = in_s[3]

                    dt = out_s[0] - in_s[0]
                    if dt >= 0:
                        dw = self.eta_up * connection.wmax / 2 * np.exp(- dt * weight / self.tau_up)
                    else:
                        dw = self.eta_down * connection.wmax / 2 * np.exp(dt * weight / self.tau_down)
                    Helper.log('Learner', log.DEBUG, 'Connection {} Weight {} {} updated dw = {}'.
                               format(connection.id, source_n, dest_n, dw))
                    # update weights in source connection
                    if not self.mp:
                        new_w = np.clip(weight + dw, connection.wmin, connection.wmax)
                        connection.weights[(source_n, dest_n)] = new_w
                    else:
                        if (connection.id, source_n, dest_n) in self.updates:
                            self.updates[(connection.id, source_n, dest_n)] += dw
                        else:
                            self.updates[(connection.id, source_n, dest_n)] = dw
        self.out_spikes = []
        self.in_spikes = []
        for connection in self.layer.in_connections:
            connection.probe()
        Helper.log('Learner', log.INFO, 'Processing learning ensemble {0} complete'.format(self.layer.id))

    def restore(self):
        self.in_spikes = []
        self.out_spikes = []
        self.buffer_in = copy.deepcopy(self.buffer_in_empty)
        self.buffer_out = []
        self.updates = {}


class LearnerClassifier(Learner):
    """
    Classifying STDP learner.
    Works the same as its super, but removes double output spikes before processing
    decreases the weights that leads to a double spike

    Parameters
    ---------
    :param eta_up: weight increase coefficient
    :type eta_up: float
    :param eta_down: weight decrease coefficient
    :type eta_down: float
    :param tau_up: sensitivity to pre synaptic delay
    :type tau_up: float
    :param tau_down: sensitivity to post synaptic delay
    :type tau_down: float
    :param mp: activate multiprocessing
    :type mp:bool
    :param feedback_gain: amount by which the weights leading to duplicate spikes
        will be decreased or increased if no spikes
    :type feedback_gain: float

    Attributes
    ----------

    """
    def __init__(self, eta_up=0.1, eta_down=0.1, tau_up=0.1, tau_down=0.1, feedback_gain=0.001, mp=False):
        super(LearnerClassifier, self).__init__(eta_up, eta_down, tau_up, tau_down, mp=mp)
        self.feedback_gain = feedback_gain

    @MeasureTiming('learn_process')
    def process(self):
        # remove multiple output spikes from buffer

        for index, experiment in enumerate(self.out_spikes):
            if experiment:
                duplicates = []
                Helper.log('Learner', log.INFO, 'Classifier learner processing input experiment {}'.format(index))
                for out in experiment:
                    Helper.log('Learner', log.DEBUG, 'Classifier learner processing spike {}'.format(out))
                    for other in experiment:
                        Helper.log('Learner', log.DEBUG, 'Classifier learner comparing with spike {}'.format(other))
                        if out != other and out[0] == other[0]:
                            duplicates.append(out)
                            Helper.log('Learner', log.INFO,
                                       'Classifier learner ignored simultaneous output spikes, reducing weights')

                # Decreases the weight leading to duplicate
                for duplicate in duplicates:
                    for con in self.layer.in_connections:
                        con.weights.matrix.add(-self.feedback_gain, con.wmin, con.wmax)
                    self.out_spikes[index].remove(duplicate)

            # Increases the weight when no spikes
            else:
                Helper.log('Learner', log.INFO,
                           'Classifier learner found no output spikes, increasing weights')
                for con in self.layer.in_connections:
                    con.weights.matrix.add(self.feedback_gain, con.wmin, con.wmax)

        super(LearnerClassifier, self).process()


class SimplifiedSTDP(Learner):
    """
        Simplified STDP learner.
        Weight change depends on previous weights, not on precise timing

        Parameters
        ---------
        :param eta_up: weight increase coefficient
        :type eta_up: float
        :param eta_down: weight decrease coefficient
        :type eta_down: float

        Attributes
        ----------

        """

    def __init__(self, eta_up=0.1, eta_down=-0.1, mp=False):
        super(SimplifiedSTDP, self).__init__(eta_up=eta_up, eta_down=eta_down, mp=mp)

    @MeasureTiming('learn_process')
    def process(self):  # call every batch
        Helper.log('Learner', log.DEBUG, 'Processing learning ensemble {0}'.format(self.layer.id))
        # for each experiment in the batch that ends
        for experiment_index in range(self.layer.sim.batch_size):
            Helper.log('Learner', log.DEBUG, 'Processing input cycle {}'.format(experiment_index))

            # for each spike emitted by the Ensemble during this experiment
            for out_s in self.out_spikes[experiment_index]:
                Helper.log('Learner', log.DEBUG, "Processing output spike of neuron {}".format(out_s[1]))
                dest_n = out_s[1]

                # for all the spikes in_s received by the same neuron which emitted out_s
                for in_s in self.in_spikes[experiment_index][dest_n]:
                    source_n = in_s[1]
                    connection = in_s[2]
                    weight = in_s[3]

                    dt = out_s[0] - in_s[0]
                    if dt >= 0:
                        dw = self.eta_up * (weight - connection.wmin) * (connection.wmax - weight)
                    else:
                        dw = self.eta_down * (weight - connection.wmin) * (connection.wmax - weight)
                    Helper.log('Learner', log.DEBUG, 'Connection {} Weight {} {} updated dw = {}'.
                               format(connection.id, source_n, dest_n, dw))
                    # update weights in source connection
                    if not self.mp:
                        connection.update_weight(source=source_n, dest=dest_n, delta_w=dw)
                    else:
                        if (connection.id, source_n, dest_n) in self.updates:
                            self.updates[(connection.id, source_n, dest_n)] += dw
                        else:
                            self.updates[(connection.id, source_n, dest_n)] = dw
        self.out_spikes = []
        self.in_spikes = []
        for connection in self.layer.in_connections:
            connection.probe()
        Helper.log('Learner', log.INFO, 'Processing learning ensemble {0} complete'.format(self.layer.id))


class Rstdp(Learner):
    """
        Classifying STDP learner.
        Works the same as its super, but removes double output spikes before processing
        decreases the weights that leads to a double spike

        Parameters
        ---------
        :param eta_up: weight increase coefficient
        :type eta_up: float
        :param eta_down: weight decrease coefficient
        :type eta_down: float
        :param anti_eta_up: weight decrease coefficient for bad output
        :type anti_eta_up: float
        :param anti_eta_down: increase coefficient for bad output
        :type anti_eta_down: float
        :param mp: activate multiprocessing
        :type mp:bool
        :param wta: experimental, only learn once per input cycle if True
        :type wta: bool
        :param size_cat: number of output neuron per category
            (used as redundancy to prevent dead neurons)
        :type size_cat: int

        Attributes
        ----------
        :ivar dataset: dataset used as input for the network. Labels are used to compare with the output
        :type dataset: Dataset
        """

    def __init__(self, eta_up=0.1, eta_down=-0.1, anti_eta_up=-0.1, anti_eta_down=0.1,
                 mp=False, wta=True, size_cat=None):
        super(Rstdp, self).__init__(eta_up=eta_up, eta_down=eta_down, mp=mp)
        self.anti_eta_up = anti_eta_up
        self.anti_eta_down = anti_eta_down
        self.dataset = None
        self.wta = wta
        self.size_cat = size_cat

    @MeasureTiming('Learning')
    def process(self):
        if not self.dataset:
            self.dataset = self.layer.sim.dataset
        Helper.log('Learner', log.DEBUG, 'Processing rstdp ensemble {0}'.format(self.layer.id))
        # for each experiment in the batch that ends
        for experiment_index in range(self.layer.sim.batch_size):
            Helper.log('Learner', log.DEBUG, 'Processing input cycle {}'.format(experiment_index))

            if not self.out_spikes[experiment_index]:
                # if no spikes for this experience
                Helper.log('Learner', log.CRITICAL, 'Not a single spike emitted on cycle {}'.format(experiment_index))
                # TODO: do something
                continue
            if self.size_cat:
                output_value = int(self.out_spikes[experiment_index][0][1]//self.size_cat)
            else:
                output_value = self.out_spikes[experiment_index][0][1]
            target_value = self.dataset.labels[self.dataset.index]
            # print(output_value, target_value)
            if output_value == target_value:
                a_p = self.eta_up
                a_n = self.eta_down
                # error = 0
            else:
                a_p = self.anti_eta_up
                a_n = self.anti_eta_down
                for out in self.out_spikes[experiment_index]:
                    if out[1] == target_value:
                        pass
                        # error = out[0] - self.out_spikes[experiment_index][0][0]

            # if wta: only the first spike leads to learning
            # else, each spike received leads to learning
            out_s_list = self.out_spikes[experiment_index][:1] if self.wta else self.out_spikes[experiment_index]
            for out_s in out_s_list:

                dest_n = out_s[1]
                Helper.log('Learner', log.DEBUG, "Processing output spike of neuron {}".format(dest_n))

                # for all the spikes in_s received by the same neuron which emitted out_s
                for in_s in self.in_spikes[experiment_index][dest_n]:
                    source_n = in_s[1]
                    connection = in_s[2]
                    weight = in_s[3]

                    dt = out_s[0] - in_s[0]
                    if dt >= 0:
                        dw = a_p * (weight - connection.wmin) * (connection.wmax - weight)
                    else:
                        dw = a_n * (weight - connection.wmin) * (connection.wmax - weight)
                    Helper.log('Learner', log.DEBUG, 'Connection {} Weight {} {} updated dw = {}'.
                               format(connection.id, source_n, dest_n, dw))
                    # update weights in source connection
                    if not self.mp:
                        new_w = np.clip(weight + dw, connection.wmin, connection.wmax)
                        connection.weights[(source_n, dest_n)] = new_w
                    else:
                        if (connection.id, source_n, dest_n) in self.updates:
                            self.updates[(connection.id, source_n, dest_n)] += dw
                        else:
                                self.updates[(connection.id, source_n, dest_n)] = dw

        self.out_spikes = []
        self.in_spikes = []
        for connection in self.layer.in_connections:
            connection.probe()
        Helper.log('Learner', log.INFO, 'Processing learning ensemble {0} complete'.format(self.layer.id))
