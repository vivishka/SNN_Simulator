import matplotlib.pyplot as plt
import numpy as np
import csv
import logging as log
from classes.base import Helper
from classes.network import Network
from classes.neuron import LIF
from classes.neuron import PoolingNeuron
from classes.layer import Block, Ensemble
from classes.simulator import Simulator
from classes.connection import *
from classes.probe import *
from classes.decoder import Decoder
from classes.encoder import Encoder
from classes.dataset import *
from classes.learner import *

import sys
sys.dont_write_bytecode = True

Helper.init_logging('main.log', log.DEBUG, ['Learner'])

mpl_logger = log.getLogger('matplotlib')
mpl_logger.setLevel(log.WARNING)

model = Network()

# n1 = Node(10, lambda: np.random.rand(1), 0.20)
# n2 = Node(10, lambda: np.random.rand(1), 0.20)
# r = Reset(0.15, 0.2)
img = False
if img:
    filename = 'datasets/fashionmnist/fashion-mnist_test.csv'
    img_size = (28, 28)
    first_image = 1
    image_dataset = ImageDataset(filename, first_image, size=1)

    e1 = Encoder(img_size, depth=8, in_min=0, in_max=255, delay_max=1)
    n1 = Node(e1, image_dataset, 5, 0)
    b1 = Block(4, img_size, LIF)
    b2 = Block(1, img_size, LIF)

    d1 = Decoder(img_size)
    d2 = Decoder(img_size)
    d3 = Decoder(img_size)

    c1 = Connection(e1, b1, kernel=(1, 1))
    c2 = Connection(e1, b2, kernel=(3, 3))

    Connection(e1, d1, kernel=1)
    Connection(b1, d2, kernel=1)
    Connection(b2, d3, kernel=1)


else:
    exp1_dataset = Exp1Dataset(size=8000)
    e1 = Encoder(depth=8, size=1, in_min=0, in_max=1, delay_max=1, gamma=1.5)
    n1 = Node(e1, exp1_dataset)
    b1 = Block(1, 2, LIF(threshold=0.8, tau=1), learner=LearnerClassifier(feedback_gain=0.00001,
                                                                          eta_up=0.02,
                                                                          eta_down=0.02,
                                                                          tau_up=0.1,
                                                                          tau_down=0.1,
                                                                          max_weight=0.6,
                                                                          min_weight=0.,
                                                                          ))
    d1 = Decoder(8)
    d2 = Decoder(2)
    c1 = Connection(e1, b1)
    c2 = DiagonalConnection(e1, d1)
    c3 = Connection(b1, d2, kernel=1)
# Connection(b1, d3, kernel=(2, 2), stride=2)


# b1 = Block(4, (4, 4), LIF, 'B1')

# e1 = Ensemble((10, 10), LIF, 'L1')

# Connection(b1, b2)

np1 = NeuronProbe(b1[0], 'voltage')
cp1 = ConnectionProbe(c1)
cp2 = ConnectionProbe(c1[1])
# p2 = Probe(b1[0], 'spike_out')


sim = Simulator(model, 0.0125, input_period=1, batch_size=1)
sim.save('main_before.w')
sim.run(4)
sim.save('main_after.w')
if img:
    for index, image in enumerate(d1.decoded_wta):
        plt.figure()
        plt.imshow(image_dataset.get(first_image + index), cmap='gray_r')
        plt.title('original image {}'.format(index))
    for index, image in enumerate(d1.decoded_wta):
        d1.plot(index)
    for index, image in enumerate(d2.decoded_wta):
        d2.plot(index)
    for index, image in enumerate(d3.decoded_wta):
        d3.plot(index)

else:
    # exp1_dataset.plot('Data')
    # exp1_dataset.plot('Labels')
    d1.plot()
    # d2.plot("first_spike")
    pass

np1.plot('voltage')
cp1.plot()
cp1.print()
plt.show()
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
