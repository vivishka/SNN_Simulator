
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

import time

from sklearn import datasets as ds
from skimage import filters as flt
from pandas import DataFrame

import sys
sys.dont_write_bytecode = True

if __name__ == '__main__':
    mpl_logger = log.getLogger('matplotlib')
    mpl_logger.setLevel(log.WARNING)

    filename = 'datasets/iris.csv'
    data_size = 4
    epochs1 = 1
    epochs2 = 1

    en1 = 10
    n1 = 50
    n2 = 30

    train = FileDataset('datasets/iris/iris - train.csv', 1, size=data_size, length=120, randomized=True)
    test = FileDataset('datasets/iris/iris - test.csv', 1, size=data_size, length=30)

    model = Network()
    sim = Simulator(model=model, dataset=test, dt=0.01, input_period=1)
    sim.enable_time(True)



    e1 = EncoderGFR(size=data_size, depth=en1, in_min=0, in_max=1, threshold=0.9, gamma=1.5, delay_max=1,# spike_all_last=True
                    )
    # node = Node(e1)
    b1 = Bloc(depth=1, size=n1, neuron_type=IF(threshold=0.7))
    c1 = Connection(e1, b1, mu=0.6, sigma=0.05)
    c1.load(np.load('c1.npy'))

    b2 = Bloc(depth=1, size=n2, neuron_type=IF(threshold=1.43))
    c2 = Connection(b1, b2, mu=0.6, sigma=0.05)
    c2.load(np.load('c2.npy'))
    b2.set_inhibition(wta=True)

    d1 = DecoderClassifier(size=3, dataset=test)

    c3 = Connection(b2, d1, kernel_size=1, mode='split')

    model.build()

    sim.run(len(test.data))

    confusion = d1.get_correlation_matrix()
    success = 0
    for i in range(3):
        success += confusion[i, i]/30

    print(confusion)
    print(success)

    plt.show()
