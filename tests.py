
import matplotlib.pyplot as plt
import numpy as np
import csv
import logging as log
from classes.base import Helper
from classes.network import Network
from classes.neuron import LIF, IF
from classes.neuron import PoolingNeuron
from classes.layer import Bloc, Ensemble
from classes.simulator import Simulator
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

mpl_logger = log.getLogger('matplotlib')
mpl_logger.setLevel(log.WARNING)


filename = 'datasets/fashionmnist/fashion-mnist_test.csv'
img_size = (28, 28)
first_image = 8
image_dataset = FileDataset(filename, first_image, size=img_size)

model = Network()
e1 = EncoderDoG(depth=1, size=img_size, in_min=0, in_max=255, delay_max=1)
n1 = Node(e1, image_dataset, 1, 0)
b1 = Bloc(4, img_size, IF())

d1 = Decoder(img_size)

c1 = Connection(e1, b1, kernel=(1, 1))

Connection(b1, d1, kernel=1)


sim = Simulator(model, 0.02, input_period=1, batch_size=1)
sim.run(2, monitor_connection=c1, convergence_threshold=0.0005)

image_dataset.plot(1)
e1.plot(0)
plt.show()