import matplotlib.pyplot as plt
import numpy as np
import csv
import logging as log
from classes.base import Helper
from classes.network import Network
from classes.neuron import LIF
from classes.layer import Bloc
from classes.simulator import Simulator
from classes.connection import Connection, DiagonalConnection
from classes.probe import Probe
from classes.decoder import Decoder
from classes.encoder import Encoder, Node
from classes.dataset import *
from classes.learner import *
import sys
sys.dont_write_bytecode = True

# Helper.init_logging('example.log', log.INFO, ['All'])

filename = 'datasets/fashionmnist/fashion-mnist_test.csv'
img_size = (12, 12)

dataset = PatternGeneratorDataset(nb_images=1, size=(12,12), nb_features=9)

model = Network()

# n1 = Node(10, lambda: np.random.rand(1), 0.20)
# n2 = Node(10, lambda: np.random.rand(1), 0.20)

e1 = Encoder(depth=4, size=img_size, in_min=0, in_max=255, delay_max=0.1)
n1 = Node(e1, dataset)
b1 = Bloc(4, img_size, LIF(threshold=1), SimplifiedSTDP(eta_up=0.1, eta_down=0.1))

d1 = Decoder(img_size)
d2 = Decoder(img_size)

c1 = Connection(e1, b1, kernel=(3, 3), shared=True)

DiagonalConnection(e1, d1)
Connection(b1, d2, kernel=1)

sim = Simulator(model, 0.02, input_period=1, batch_size=1)
sim.run(1.02)

plt.figure()
plt.imshow(dataset.get(0), cmap='gray')

d1.plot()
d2.plot()

plt.show()
