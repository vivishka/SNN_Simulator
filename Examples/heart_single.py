from classes.network import Network
from classes.neuron import IF
from classes.simulator import Simulator
from classes.probe import *
from classes.decoder import *
from classes.encoder import *
from classes.dataset import *
from classes.learner import *

import time
from Examples import heart_ann_generator

"""
Single experiment of generating ANN the convert to SNN using the heart_ann_generator script with 3 layers
"""

start = time.time()

en1 = 10
n1 = 200
n2 = 100
dec1 = 10

n_proc = 3

heart_ann_generator.run(en1, n1, n2, dec1)


mpl_logger = log.getLogger('matplotlib')
mpl_logger.setLevel(log.WARNING)

filename = 'datasets/iris.csv'
data_size = 13

train = FileDataset('datasets/heart/heart - train.csv', 0, size=data_size, randomized=True)
test = FileDataset('datasets/heart/heart - test.csv', 0, size=data_size)

model = Network()
e1 = EncoderGFR(size=data_size, depth=en1, in_min=0, in_max=1, threshold=0.9, gamma=1.5, delay_max=1,
                # spike_all_last=True
                )
# node = Node(e1)
b1 = Block(depth=1, size=n1, neuron_type=IF(threshold=0.6))
c1 = Connection(e1, b1, mu=0.6, sigma=0.05)
c1.load(np.load('c1.npy'))

b2 = Block(depth=1, size=n2, neuron_type=IF(threshold=1))
c2 = Connection(b1, b2, mu=0.6, sigma=0.05)
c2.load(np.load('c2.npy'))
# b2.set_inhibition(wta=True, radius=(0, 0))

b3 = Block(depth=1, size=dec1 * 2, neuron_type=IF(threshold=0.5))
c3 = Connection(b2, b3, mu=0.6, sigma=0.05)
c3.load(np.load('c3.npy'))
b3.set_inhibition(wta=True, radius=(0, 0))

d1 = DecoderClassifier(size=2)

c4 = Connection(b3, d1, kernel_size=1, mode='split')

sim = Simulator(network=model, dataset=train, dt=0.01, input_period=1)
sim.run(len(train.data))
model.restore()
sim.dataset = test
sim.run(len(test.data))
confusion = d1.get_correlation_matrix()
print(confusion)
success = 0
for i in range(2):
    success += confusion[i, i] / len(test.data)
print(success)