
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

    # Helper.init_logging('main.log', log.DEBUG, ["All"])


    # filename = 'datasets/mnist/mnist_train.csv'
    filename = 'datasets/fashionmnist/fashion-mnist_train.csv'
    img_size = (28, 28)
    # img_size = (12, 12)

    first_image = np.random.randint(0, 59999-20000)
    print("init dataset image {}".format(first_image))

    # image_dataset = PatternGeneratorDataset(index=0, size=img_size, nb_images=200, nb_features=9)
    model = Network()
    for layer in range(1, 3):
        if layer == 1:
            image_dataset = FileDataset(filename, first_image, size=img_size, length=-1)
        else:
            image_dataset = FileDataset(filename, first_image, size=img_size, length=-1)

        e1 = EncoderDoG(sigma=[(3/9, 6/9)],  # (7/9, 14/9), (13/6, 26/9)],
                        kernel_sizes=[3], size=img_size, in_min=0, in_max=255, delay_max=1)
        n1 = Node(e1, image_dataset, 1, 0)
        if layer == 1:
            b1 = Bloc(4, img_size, IF(threshold=2.1),
                      SimplifiedSTDP(
                      eta_up=0.003,
                      eta_down=-0.003)
                      )
            b1.set_dataset(image_dataset)
            b1.set_inhibition(wta=True, radius=1)
        else:
            b1 = Bloc(4, img_size, IF(threshold=2.1))

        # d1 = Decoder((7, 7))
        if layer == 2:
            b2 = Bloc(4, (14, 14), IF(threshold=0.1))
            b2.set_inhibition(wta=True, radius=0)

            b3 = Bloc(8, (14, 14), IF(threshold=2),
                      SimplifiedSTDP(
                          eta_up=0.03,
                          eta_down=-0.03)
                      )
            b3.set_dataset(image_dataset)
            b3.set_inhibition(radius=1)

        c1 = Connection(e1, b1, kernel=(3, 3), mode='shared')
        if layer == 2:
            c2 = Connection(b1, b2, kernel=(2, 2), mode='pooling')
            c3 = Connection(b2, b3, kernel=(3, 3), mode='shared')
            # c4 = Connection(b2, d1, kernel=(2, 2), mode='pooling')

        cps = []
        if layer == 1:
            for con in c1:
                cps.append(ConnectionProbe(con))
                sprobein = NeuronProbe(target=e1[0], variables='spike_out')
                sprobeout = NeuronProbe(target=b1[0], variables='spike_out')
        elif layer == 2:
            for con in c3:
                cps.append(ConnectionProbe(con))
                sprobein = NeuronProbe(target=b2[0], variables='spike_out')
                sprobeout = NeuronProbe(target=b3[0], variables='spike_out')

        sim = Simulator(model, 0.0625, input_period=1, batch_size=1)
        try:
            if layer == 1:
                sim.load('tests1.w')
            if layer == 2:
                sim.load('tests1.w')
        except:
            pass

        sim.run(len(image_dataset.data)+0.0625)
        # image_dataset.plot(-1)
        # e1.plot(layer=4)
        # plot final kernels


        # c2.plot()
        # d1.plot()
        #  plot weight history
        for cp in cps:
            cp.plot()
        if layer == 1:
            sim.save('tests1.w')
            c1.plot()
        if layer == 2:
            sim.save('tests2.w')
            c3.plot()

        sprobein.plot('spike_out')
        sprobeout.plot('spike_out')
        # for index in range(image_dataset.length):
        #     image_dataset.plot(index)
        #     e1.plot(index, 0)

        # sim.plot_steptimes()
        Helper.print_timings()

        sim.flush()
        model.restore()

    plt.show()