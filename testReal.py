
import matplotlib.pyplot as plt
import numpy as np
import csv
import logging as log
from classes.base import Helper
from classes.network import Network
from classes.neuron import LIF, IF, IFReal
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

    Helper.init_logging('testmp.log', log.DEBUG, ["Simulator"])


    # filename = 'datasets/fashionmnist/fashion-mnist_train.csv'
    filename = 'datasets/mnist/mnist_train.csv'
    img_size = (28, 28)
    # img_size = (12, 12)
    first_image = np.random.randint(0, 59999-20000)
    print("init dataset image {}".format(first_image))
    image_dataset = FileDataset(filename, first_image, size=img_size, length=100)
    # image_dataset = PatternGeneratorDataset(index=0, size=img_size, nb_images=300, nb_features=9)
    model = Network()
    # e1 = EncoderDoG(sigma=[(3/9, 6/9)],  # (7/9, 14/9), (13/6, 26/9)],
    #                 kernel_sizes=[3], size=img_size, in_min=0, in_max=255, delay_max=1, double_filter=True, spike_all_last=False)
    e1 = EncoderGabor(size=img_size, kernel_size=5, orientations=[45+22.5, 90+22.5, 135+22.5, 180+22.5], spike_all_last=True, div=2)
    n1 = Node(e1, image_dataset, 1, 0)
    b1 = Bloc(30, img_size, IFReal(threshold=2**12*5), SimplifiedSTDP(
        eta_up=0.2,
        eta_down=-0.2,
        mp=True
    ))
    # b1.set_inhibition(True, 1)
    # d1 = Decoder(img_size)

    c1 = Connection(e1, b1, kernel_size=5, mode='shared', wmin=0, wmax=12, real=True)
    cps = []
    for con in c1:
        cps.append(ConnectionProbe(con))

    # c2 = Connection(b1, d1, kernel=1, mode)
    np1 = NeuronProbe(b1[0], ['spike_out'])
    np2 = NeuronProbe(e1[0], 'spike_out')
    # sim = SimulatorMp(model=model, dataset=image_dataset, dt=0.05, input_period=1, batch_size=50, processes=3)
    sim = Simulator(model=model, dataset=image_dataset,  dt=0.05, input_period=1, batch_size=1)
    sim.enable_time(True)
    # sim.autosave = "auto.w"
    # sim.load('tests.w')
    sim.run(len(image_dataset.data))
    # image_dataset.plot(-1)
    # e1.plot(layer=0)
    # plot final kernels
    c1.plot_all_kernels()
    #  plot weight history
    for cp in cps:
        cp.plot()
    np1.plot('spike_out')
    np2.plot('spike_out')
    # np1.plot('voltage')
    sim.save('tests.w')

    # for index in range(image_dataset.length):
    #     image_dataset.plot(index)
    #     e1.plot(index, 1)
    Helper.print_timings()
    sim.plot_steptimes()
    plt.show()