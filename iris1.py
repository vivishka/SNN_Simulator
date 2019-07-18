
import matplotlib.pyplot as plt
import numpy as np
import csv
import logging as log
from classes.base import Helper
from classes.network import Network
from classes.neuron import LIF, IF
from classes.neuron import PoolingNeuron
from classes.layer import Bloc, Ensemble
from classes.simulator import Simulator, SimulatorMp
from classes.connection import *
from classes.probe import *
from classes.decoder import *
from classes.encoder import *
from classes.dataset import *
from classes.learner import *

from sklearn import datasets as ds
from skimage import filters as flt
from pandas import DataFrame

import sys
sys.dont_write_bytecode = True

if __name__ == '__main__':
    mpl_logger = log.getLogger('matplotlib')
    mpl_logger.setLevel(log.WARNING)

    filename = 'datasets/iris.csv'
    data_size = (1, 4)
    epochs1 = 100
    epochs2 = 200

    image_dataset = FileDataset(filename, -1, size=data_size, length=-1, randomized=True)

    model = Network()

    sim = Simulator(model=model, dataset=image_dataset, dt=0.05, input_period=1, batch_size=1)
    sim.enable_time(True)

    ################################################ Layer 1 train

    e1 = EncoderGFR(size=data_size, depth=10, in_min=0, in_max=1, threshold=0.9, gamma=1.5, delay_max=1, spike_all_last=True
                    )
    n1 = Node(e1, image_dataset, 1, 0)
    b1 = Bloc(depth=1, size=50, neuron_type=IF(threshold=9),
              learner=SimplifiedSTDP(
                  eta_up=0.1,
                  eta_down=-0.1,
                  # anti_eta_up=-0.01,
                  # anti_eta_down=0.01,
              )
              )
    b1.set_inhibition(wta=True, radius=None, k_wta_level=3)
    b1.set_threshold_adapt(0.3, 2, 0.05, 0)

    c1 = Connection(e1, b1, mu=0.6, sigma=0.05)

    # np1 = NeuronProbe(b1[0], ["spike_out", "voltage"])
    tp1 = NeuronProbe(b1[0][0], 'threshold')

    cps1 = []
    # for con in c1:
    #     cps1.append(ConnectionProbe(con))

    cps1.append(ConnectionProbe(c1[0]))

    sim.autosave='iris1.w'
    # sim.load('iris1_f.w')

    sim.run(len(image_dataset.data) * epochs1)
    ################################################ Layer 1 post-process
    print(c1.get_convergence())
    c1.saturate_weights(0.8)
    sim.save('iris1_f.w')

    # for con in cps1:
    #     con.plot()
    cps1[0].plot()
    # np1.plot('voltage')
    # np1.plot('spike_out')

    tp1.plot('threshold')

    print(c1[0].weights.matrix.to_dense())
    # for con in c1:
    #     plt.figure()
    #     plt.imshow(con.weights.matrix.to_dense(), cmap='gray')

    b1.learner = None
    b1.stop_inhibition()
    b1.stop_threshold_adapt()
    model.restore()
    ################################################ Layer 2 train

    b2 = Bloc(depth=1, size=3, neuron_type=IF(threshold=4), learner=Rstdp(
        eta_up=0.02,
        eta_down=-0.02,
        anti_eta_up=-0.006,
        anti_eta_down=0.006
    ))
    b2.set_inhibition(wta=True)
    b2.set_threshold_adapt(0.6, 1, 0.05, 0)


    c2 = Connection(b1, b2, mu=0.6, sigma = 0.05)

    cps2 = []
    for con in c2:
        cps2.append(ConnectionProbe(con))

    np2 = NeuronProbe(b2[0], ["spike_out", "voltage"])
    tp2 = NeuronProbe(b2[0][0], 'threshold')

    sim.autosave = 'iris2.w'

    sim.run(len(image_dataset.data) * epochs2)
    ################################################ Layer 2 post-process
    print(c2.get_convergence())
    c2.saturate_weights(0.8)
    sim.save('iris_f.w')

    for con in cps2:
        con.plot()

    np2.plot('voltage')
    np2.plot('spike_out')
    tp2.plot('threshold')
    # d1.plot()

    plt.figure()
    plt.imshow(c2[0].weights.matrix.to_dense(), cmap='gray')
    # print(c2[0].weights.matrix.to_dense())

    ################################################ Test

    b2.learner = None
    b2.stop_inhibition()
    b2.stop_threshold_adapt()

    d1 = DecoderClassifier(size=3, dataset=image_dataset)

    c3 = Connection(b2, d1, kernel_size=1)

    sim.run(len(image_dataset.data))

    print(d1.get_correlation_matrix())

    Helper.print_timings()
    # sim.plot_steptimes()
    plt.show()





