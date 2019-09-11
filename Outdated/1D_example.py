import matplotlib.pyplot as plt
import numpy as np
import csv
import logging as log
from classes.base import Helper
from classes.network import Network
from classes.neuron import LIF, IF
from classes.neuron import PoolingNeuron
from classes.layer import Block, Ensemble
from classes.simulator import Simulator
from classes.connection import *
from classes.probe import *
from classes.decoder import *
from classes.encoder import EncoderGFR
from classes.dataset import *
from classes.learner import *

from sklearn import datasets as ds
from pandas import DataFrame

import sys
sys.dont_write_bytecode = True

mpl_logger = log.getLogger('matplotlib')
mpl_logger.setLevel(log.WARNING)

"""
Executes low dimension experiments (blob classification 1D, 2D) described in the generator functions.
Learning part not complete, needs to separate training of layer 1, 2 and test. Not all features are exploited to have a
decent result.
"""


def exp1generator(size, gap=0.5, width=0.25):
    cat = np.random.randint(2, size=size)
    data = width * np.random.rand(size) + (gap + width) * cat
    return cat.tolist(), data.tolist()


def exp2generator(size, gap=0.5, width=0.25):
    cat = np.random.randint(2, size=size)
    rngx = np.random.rand(size) * width
    rngy = np.random.rand(size) * width
    x = rngx + cat * gap
    y = rngy + (1 - cat) * gap
    data = []
    for index in range(size):
        data.append((x[index], y[index]))
    return cat.tolist(), data


def exp3generator(size):
    data, labels = ds.make_blobs(n_samples=size, n_features=2, centers=2, cluster_std=0.04, center_box=(0, 1),
                           shuffle=True, random_state=None)

    # scatter plot, dots colored by class value
    # df = DataFrame(dict(x=data[:, 0], y=data[:, 1], label=labels))
    # colors = {0: 'red', 1: 'blue'}
    # fig, ax = plt.subplots()
    # grouped = df.groupby('label')
    # for key, group in grouped:
    #     group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    return labels.tolist(), data.tolist()


# Helper.init_logging('main1.log', log.DEBUG, ['All'])

for i in range(1):
    model = Network()
    n_nodes = 5
    n_int = 10
    n_out = 2
    # exp_dataset = VectorDataset(size=20000, generator=exp1generator)
    exp_dataset = VectorDataset(size=2000, generator=exp2generator)
    # exp_dataset = VectorDataset(size=2000, generator=exp3generator)
    e1 = EncoderGFR(depth=n_nodes, size=1, in_min=0, in_max=1, delay_max=1, gamma=1., threshold=0.9, spike_all_last=True)
    # b1 = Block(1, n_int, LIF(threshold=0.8, tau=20),
    #          learner=
    #          Learner(
    #          eta_up=0.02,
    #          eta_down=0.05,
    #          # eta_up=0,
    #          # eta_down=0,
    #          tau_up=0.1,
    #          tau_down=0.1,
    #          )
    #           )
    b1 = Block(1, n_out, IF(threshold=0.8),
               learner=
             Rstdp(
             eta_up=0.2,
             eta_down=-0.2,
             anti_eta_up=-0.05,
             anti_eta_down=0.05,

             )
               )
    # d1 = Decoder(n_nodes)
    d2 = DecoderClassifier(n_out)
    c1 = Connection(e1, b1, wmin=0, wmax=0.5)
    # c2 = DiagonalConnection(e1, d1)
    c3 = Connection(b1, d2, 0, 0.4, kernel=1)

    np1 = NeuronProbe(b1[0], ['voltage', 'spike_out'])
    np2 = NeuronProbe(e1[0], 'spike_out')
    cp1 = ConnectionProbe(c1)

    b1.set_inhibition(wta=True)

    sim = Simulator(network=model, dt=0.01, dataset=exp_dataset, input_period=1, batch_size=1)
    sim.enable_time(True)
    sim.run(2000)

    cp1.plot()
    np1.plot('voltage')
    np1.plot('spike_out')
    np2.plot('spike_out')
    # cp2.plot()
    # cp2.print()
    # exp3_dataset.plot('Labels')
    # np1.plot('voltage')
    # np2.plot('voltage')
    # print(d2.get_correlation_matrix())
    # print(d3.get_correlation_matrix())
    #

    # sim.save("main.w")
    # exp1_dataset.plot('Data')
    # exp1_dataset.plot('Labels')
    # d1.plot()
    # d2.plot("first_spike")


plt.show()
# print(e2[1][5].label)
# E = [Ensemble(10, LIF, 'L') for i in range(50)]

# for i in range (49):
#     Connection(E[i], E[i+1])
# e3 = Ensemble(10, LIF, 'L2')
# e3 = Ensemble(1000, Neuron, 'L1')
# e4 = Ensemble(1000, Neuron, 'L2')

# p3 = Probe(e2, 'spike_in')
# p4 = Probe(e2, 'spike_out')
# n1 = Neuron(0.2, "1")
# n2 = Neuron(1, "2")
# Connection(n1, [n2], [0.3])
# Connection(n2, [n1], [0.1])

# Connection(e1, E[0])
# Connection(E[49], e1)
# Connection(e2, e3)
# Connection(e3, e4)
