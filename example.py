import matplotlib.pyplot as plt
import numpy as np
import csv
import logging as log
from classes.base import Helper
from classes.network import Network
from classes.neuron import LIF
from classes.layer import Bloc
from classes.simulator import Simulator
from classes.connection import Connection
from classes.probe import Probe
from classes.decoder import Decoder
from classes.encoder import Encoder, Node
from classes.dataset import *
import sys
sys.dont_write_bytecode = True

Helper.init_logging('example.log', log.INFO, ['All'])

filename = 'datasets/fashionmnist/fashion-mnist_test.csv'
img_size = (28, 28)

dataset = ImageDataset(filename, 1)

model = Network()

# n1 = Node(10, lambda: np.random.rand(1), 0.20)
# n2 = Node(10, lambda: np.random.rand(1), 0.20)

img = True
if img:
    e1 = Encoder(img_size, depth=8, in_min=0, in_max=255, delay_max=0.1)
    n1 = Node(e1, image, 5, 0)
    b1 = Bloc(4, img_size, LIF)
    b2 = Bloc(1, img_size, LIF)

    d1 = Decoder(img_size)
    d2 = Decoder(img_size)
    d3 = Decoder(img_size)

    c1 = Connection(e1, b1, kernel=(1, 1))
    c2 = Connection(e1, b2, kernel=(3, 3))

    Connection(e1, d1, kernel=1)
    Connection(b1, d2, kernel=1)
    Connection(b2, d3, kernel=1)


else:
    e1 = Encoder(1, depth=5, in_min=0.0, in_max=1.0, delay_max=0.1, gamma=0.5)
    n1 = Node(e1, np.random.rand)
    b1 = Bloc(1, 10, LIF)
    d1 = Decoder(5)
    d2 = Decoder(10)
    c1 = Connection(e1, b1)
    c2 = Connection(e1, d1)
    c3 = Connection(b1, d2, kernel=1)
    p1 = Probe(b1[0], 'voltage')


sim = Simulator(model, 0.001)
sim.run(0.2)


if img:
    # plt.figure()
    # plt.imshow(image, cmap='gray')
    # d1.plot('first_spike', 'after encoder')
    # d2.plot('first_spike', 'after bloc 1')
    # d3.plot('first_spike', 'after bloc 2')

    plt.figure()
    plt.imshow(dataset.get(1), cmap='gray')
    for image in d1.decoded_wta:
        plt.figure()
        plt.imshow(image, cmap='gray')
    for image in d2.decoded_wta:
        plt.figure()
        plt.imshow(image, cmap='gray')
    for image in d3.decoded_wta:
        plt.figure()
        plt.imshow(image, cmap='gray')
else:
    d1.plot('first_spike')
    d2.plot('first_spike')
    p1.plot('voltage')

plt.show()
