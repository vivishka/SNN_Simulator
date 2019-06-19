
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

    Helper.init_logging('main.log', log.DEBUG, ["Simulator"])


    filename = 'datasets/fashionmnist/fashion-mnist_train.csv'
    img_size = (28, 28)
    # img_size = (12, 12)
    first_image = np.random.randint(0, 59999-20000)
    print("init dataset image {}".format(first_image))
    image_dataset = FileDataset(filename, first_image, size=img_size, length=5)
    # image_dataset = PatternGeneratorDataset(index=0, size=img_size, nb_images=300, nb_features=9)
    model = Network()
    e1 = EncoderDoG(sigma=[(3/9, 6/9)],  # (7/9, 14/9), (13/6, 26/9)],
                    kernel_sizes=[3], size=img_size, in_min=0, in_max=255, delay_max=1)
    n1 = Node(e1, image_dataset, 1, 0)
    b1 = Bloc(8, img_size, IF(threshold=2), SimplifiedSTDP_MP(
        eta_up=0.05,
        eta_down=-0.1,
    ))
    b1.set_dataset(image_dataset)
    b1.set_inhibition(1)
    d1 = Decoder(img_size)

    c1 = Connection(e1, b1, kernel=(3, 3), shared=True)
    cps = []
    for con in c1:
        cps.append(ConnectionProbe(con))

    c2 = Connection(b1, d1, kernel=1)


    sim = SimulatorMp(model, 1/15, input_period=1, batch_size=3, processes=4)
    sim.run(len(image_dataset.data)+0.02)
    # image_dataset.plot(-1)
    # e1.plot(layer=4)
    # plot final kernels
    c1.plot()
    #  plot weight history
    # for cp in cps:
    #     cp.plot()
    sim.save('tests.w')

    # for index in range(image_dataset.length):
    #     image_dataset.plot(index)
    #     e1.plot(index, 1)


    Helper.print_timings()

    plt.show()