
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

import iris_ann_generator

from sklearn import datasets as ds
from skimage import filters as flt
from pandas import DataFrame

import sys
sys.dont_write_bytecode = True

if __name__ == '__main__':

    start = time.time()

    en1 = 10
    n1 = 50
    n2 = 10
    iris_ann_generator.run(en1, n1, n2)

    mpl_logger = log.getLogger('matplotlib')
    mpl_logger.setLevel(log.WARNING)

    filename = 'datasets/iris.csv'
    data_size = 4



    train = FileDataset('datasets/iris/iris - train.csv', 1, size=data_size, length=120, randomized=True)
    test = FileDataset('datasets/iris/iris - test.csv', 1, size=data_size, length=30)

    t1 = np.linspace(0.6, 1.3, 10)
    t2 = np.linspace(0.6, 1.3, 10)
    success_map = np.zeros((len(t1), len(t2)))

    # sim.enable_time(True)
    last_time = time.time()
    Helper.print_progress(0, len(t1)*len(t2), "testing thresholds ", bar_length=30)
    model = Network()
    e1 = EncoderGFR(size=data_size, depth=en1, in_min=0, in_max=1, threshold=0.9, gamma=1.5, delay_max=1,# spike_all_last=True
                    )
    # node = Node(e1)
    b1 = Bloc(depth=1, size=n1, neuron_type=IF(threshold=0))
    c1 = Connection(e1, b1, mu=0.6, sigma=0.05)
    c1.load(np.load('c1.npy'))

    b2 = Bloc(depth=1, size=n2 * 3, neuron_type=IF(threshold=0))
    c2 = Connection(b1, b2, mu=0.6, sigma=0.05)
    c2.load(np.load('c2.npy'))
    b2.set_inhibition(wta=True, radius=(0, 0))

    d1 = DecoderClassifier(size=3, dataset=train)

    c3 = Connection(b2, d1, kernel_size=1, mode='split')

    sim = Simulator(model=model, dataset=train, dt=0.01, input_period=1)
    for i2, th2 in enumerate(t2):
        for i1, th1 in enumerate(t1):
            b1.set_threshold(th1)
            b2.set_threshold(th2)
            sim.run(len(train.data))
            confusion = d1.get_correlation_matrix()
            success = 0
            for i in range(3):
                success += confusion[i, i]/len(train.data)
            success_map[i1, i2] = success
            model.restore()
            Helper.print_progress(len(t1)*i2+i1, len(t1)*len(t2), "testing thresholds ", bar_length=30,)# suffix='est. time: {} s'.format(int(len(t1)*len(t2)-(len(t2)*i1+i2)/(time.time() - last_time))))
            last_time = time.time()

    t1max, t2max = np.where(success_map == np.amax(success_map))
    Helper.print_progress(1, 1, "testing thresholds ", bar_length=30,)# suffix='est. time: {} s'.format(int(len(t1)*len(t2)-(len(t2)*i1+i2)/(time.time() - last_time))))
    for imax in range(len(t1max)):
        print([t1[t1max[imax]], t2[t2max[imax]], np.amax(success_map)])

    fig = plt.figure()
    np.save('th_map.npy', success_map)
    plt.imshow(success_map, cmap='gray', extent=[t1[0], t1[-1], t2[-1], t2[0]])
    # plt.imsave('success_map.png', arr=success_map, cmap='gray', format='png')

    sim.dataset = test
    d1.dataset = test
    b1.set_threshold(t1[t1max[0]])
    b2.set_threshold(t2[t2max[0]])
    sim.enable_time(True)
    sim.run(len(test.data))
    confusion = d1.get_correlation_matrix()
    success = 0
    for i in range(3):
        success += confusion[i, i] / len(test.data)
    print(confusion)
    print(success)
    print('total time')
    print(int(start-time.time()))
    plt.show()
