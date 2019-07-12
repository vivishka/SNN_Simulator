
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

    # Helper.init_logging('testmp.log', log.DEBUG, ["Simulator"])

    filename = 'datasets/iris.csv'
    data_size = (1, 4)
    # img_size = (12, 12)
    # first_image = np.random.randint(0, 59999-20000)
    # print("init dataset image {}".format(first_image))
    image_dataset = FileDataset(filename, 0, size=data_size, length=150)

    model = Network()

    e1 = EncoderGFR(size=data_size, depth=20, in_min=0, in_max=1, threshold=0.75, gamma=1., delay_max=1)

    n1 = Node(e1, image_dataset, 1, 0)
    b1 = Bloc(depth=1, size=3, neuron_type=IF(threshold=10), learner=Rstdp(
        eta_up=0.003,
        eta_down=-0.003,
        anti_eta_up=-0.0003,
        anti_eta_down=0.0003,
    ))
    d1 = DecoderClassifier(size=data_size, dataset=image_dataset)

    c1 = Connection(e1, b1)
    c2 = Connection(b1, d1, kernel_size=1)
    cps = []
    for con in c1:
        cps.append(ConnectionProbe(con))
    np1 = NeuronProbe(b1[0], ["spike_out", "voltage"])
    # np2 = NeuronProbe(e1[0], ["spike_out", "voltage"])


    # sim = SimulatorMp(model=model, dataset=image_dataset, dt=0.05, input_period=1, batch_size=9, processes=3)
    sim = Simulator(model=model, dataset=image_dataset, dt=0.05, input_period=1, batch_size=1)
    sim.enable_time(True)
    # sim.load('testsML.w')
    sim.autosave='iris1.w'
    sim.run(len(image_dataset.data))
    # image_dataset.plot(-1)
    e1.plot()
    # plot final weights
    # print(c1.weights.matrix)
    #  plot weight history
    # for cp in cps:
    #     cp.plot()
    print(d1.get_correlation_matrix())
    sim.flush()
    model.restore()

    np1.plot('spike_out')
    np1.plot('voltage')
    # for index in range(image_dataset.length):
    #     image_dataset.plot(index)
    #     e1.plot(index, 2)
    Helper.print_timings()
    sim.plot_steptimes()
    plt.show()