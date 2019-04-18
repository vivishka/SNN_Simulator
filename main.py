import matplotlib.pyplot as plt
import numpy as np
import csv
from classes.network import Network
from classes.neuron import LIF
from classes.ensemble import Bloc
from classes.simulator import Simulator
from classes.connection import Connection
from classes.probe import Probe
from classes.decoder import Decoder
from classes.node import Encoder, PeriodicInput

import sys
sys.dont_write_bytecode = True

filename = 'datasets/fashionmnist/fashion-mnist_test.csv'
img_size = (28, 28)
image = None
with open(filename, newline='') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    readCSV.__next__()
    row = readCSV.__next__()
    image = np.array(row[1:]).astype(np.uint8)
    image = image.reshape(img_size)



model = Network()

# n1 = Node(10, lambda: np.random.rand(1), 0.20)
# n2 = Node(10, lambda: np.random.rand(1), 0.20)
# r = Reset(0.15, 0.2)

e1 = Encoder(img_size, 3, 0, 255, 0.1)
i1 = PeriodicInput(e1, image, 5, 0)
d1 = Decoder(img_size)
c1 = Connection(e1, d1)


# b1 = Bloc(4, (4, 4), LIF, 'B1')

# e1 = Ensemble((10, 10), LIF, 'L1')

# Connection(b1, b2)

# p1 = Probe(e1, 'voltage')
# p2 = Probe(e1, 'spike_out')


sim = Simulator(model, 0.001)
sim.run(1.0)

plt.imshow(d1.get_first_spike())
plt.show()

# p1.plot()
# p2.plot()

# TODO: distribution arg for ensembles


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
