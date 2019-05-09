import matplotlib.pyplot as plt
import numpy as np
import csv
from classes.network import Network
from classes.neuron import LIF
from classes.neuron import PoolingNeuron
from classes.ensemble import Bloc
from classes.simulator import Simulator
from classes.connection import Connection
# from classes.probe import Probe
from classes.decoder import Decoder
from classes.encoder import Encoder, Node

import sys
sys.dont_write_bytecode = True

filename = 'datasets/fashionmnist/fashion-mnist_test.csv'
img_size = (28, 28)
with open(filename, newline='') as file:
    readCSV = csv.reader(file, delimiter=',')
    readCSV.__next__()
    row = readCSV.__next__()
    image = np.array(row[1:]).astype(np.uint8).reshape(img_size)


model = Network()

# n1 = Node(10, lambda: np.random.rand(1), 0.20)
# n2 = Node(10, lambda: np.random.rand(1), 0.20)
# r = Reset(0.15, 0.2)
img = False
if img:
    e1 = Encoder(img_size, 16, 0, 255, 0.1)
    n1 = Node(e1, image, 5, 0)
    b1 = Bloc(8, img_size, LIF)
    b2 = Bloc(4, img_size, LIF)
    b3 = Bloc(2, img_size, LIF)
    b4 = Bloc(1, img_size, LIF)
    # b3 = Bloc(1, img_size, PoolingNeuron)
    # b1.set_inhibition(5)

    d1 = Decoder(img_size)
    d2 = Decoder(img_size)
    d3 = Decoder((28//1, 28//1))
    # d3 = Decoder(img_size)

    c1 = Connection(e1, b1, (1, 1))
    c2 = Connection(b1, b2, (3, 3))
    c3 = Connection(b2, b3, (3, 3))
    c4 = Connection(b3, b4, (3, 3))
    Connection(e1, d1, (1, 1))
    Connection(b1, d2, (1, 1))
    Connection(b4, d3, (1, 1))

else:
    e1 = Encoder(1, 5, 0.0, 1.0, 0.1)
    n1 = Node(e1, np.random.rand)
    b1 = Bloc(1, 10, LIF)
    d1 = Decoder(5)
    d2 = Decoder(10)
    c1 = Connection(e1, b1)
    c2 = Connection(e1, d1, kernel=1)
    c3 = Connection(b1, d2, kernel=1)
# Connection(b1, d3, kernel=(2, 2), stride=2)


# b1 = Bloc(4, (4, 4), LIF, 'B1')

# e1 = Ensemble((10, 10), LIF, 'L1')

# Connection(b1, b2)

# p1 = Probe(e1, 'voltage')
# p2 = Probe(e1, 'spike_out')


sim = Simulator(model, 0.001)
sim.run(0.2)


if img:
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.figure()
    plt.imshow(d1.decoded_image(), cmap='gray')
    plt.figure()
    plt.imshow(d2.get_first_spike(), cmap='gray')
    plt.figure()
    plt.imshow(d3.get_first_spike(), cmap='gray')
else:
    plt.figure()
    plt.imshow(d1.get_first_spike(), cmap='gray')
    plt.figure()
    plt.imshow(d2.get_first_spike(), cmap='gray')

plt.show()

# p1.plot()
# p2.plot()


# print(e2[1][5].label)
# E = [Ensemble(10, LIF, 'L') for i in range(50)]

# for i in range (49):
#     Connection(E[i], E[i+1])
# e3 = Ensemble(10, LIF, 'L2')
# e3 = Ensemble(1000, Neuron, 'L1')
# e4 = Ensemble(1000, Neuron, 'L2')

# p3 = Probe(e2, 'spike_in')
# p4 = Probe(e2, 'spike_out')
# n1 = Neuron(0.2, "1")
# n2 = Neuron(1, "2")
# Connection(n1, [n2], [0.3])
# Connection(n2, [n1], [0.1])

# Connection(e1, E[0])
# Connection(E[49], e1)
# Connection(e2, e3)
# Connection(e3, e4)