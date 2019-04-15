import matplotlib.pyplot as plt
import numpy as np
from classes.network import Network
from classes.neuron import LIF
from classes.ensemble import Ensemble
from classes.simulator import simulator
from classes.connection import Connection
from classes.probe import Probe
from classes.node import Node

import sys
sys.dont_write_bytecode = True


model = Network()

n1 = Node(10, lambda: np.random.rand(1), 2.0)
n2 = Node(10, lambda: np.random.rand(1), 2.0)

e1 = Ensemble((20), LIF, 'L1')
e2 = Ensemble((20), LIF, 'L2')

Connection(n1, e1)
Connection(n2, e1)
Connection(e1, e2)


p1 = Probe(e1, 'voltage')
p2 = Probe(e1, 'spike_out')


sim = simulator(model, 0.001)
sim.run(4.0)

p1.plot()
p2.plot()

plt.show()

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
