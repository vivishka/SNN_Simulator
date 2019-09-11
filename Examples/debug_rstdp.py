import logging as log
from classes.base import Helper
from classes.base import Helper
from classes.network import Network
from classes.neuron import LIF, IF
from classes.neuron import PoolingNeuron
from classes.layer import *
from classes.simulator import Simulator, SimulatorMp
from classes.connection import *
from classes.probe import *
from classes.decoder import *
from classes.encoder import *
from classes.dataset import *
from classes.learner import *
import random


"""
Simple script to have a close look at the training of a simple task, with the objective to debug or learn the process
"""

def generator(size):
    data = [random.choice([0.1, 0.2, 0.8, 0.9]) for _ in range(size)]
    labels = []
    for i in data:
        labels.append(int(i > 0.5))
    return labels, data


model = Network()
data_size = 1
dataset = VectorDataset(generator=generator, size=500)

e1 = EncoderGFR(depth=9, size=data_size, in_min=0, in_max=1, gamma=0.8)
L1 = Rstdp(0.1, -0.1, 0, 0, wta=True)
L2 = SimplifiedSTDP(0.1, -0.1)
b1 = Block(depth=1, size=2, learner=L2, neuron_type=IF(threshold=0.9))
d1 = DecoderClassifier(size=2)

c1 = Connection(e1, b1, sigma=0.1)
cp1 = ConnectionProbe(c1)

c2 = Connection(b1, d1, kernel_size=1)

np1 = NeuronProbe(e1, variables="spike_out")
np2 = NeuronProbe(b1, variables=["spike_out", "voltage"])


sim = Simulator(model, dt=0.01, dataset=dataset, batch_size=1)
sim.run(len(dataset.data))

np1.plot_spike_out()
np2.plot_spike_out()
np2.plot_variable("voltage")
cp1.plot()
# cp1.print()
print(dataset.data)
print(dataset.labels)
print(d1.get_accuracy())
print(d1.get_correlation_matrix())
# d1.plot_accuracy(50)
# d1.plot()
plt.show()