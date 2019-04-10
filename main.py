
from classes.network import Network
# from classes.neurons import Neuron
from classes.neuron import Neuron
from classes.ensemble import Ensemble
from classes.simulator import simulator
from classes.connection import Connection
from classes.probe import Probe

from functools import partial
import random
r = partial(random.uniform, 0, 1)

model = Network()

e1 = Ensemble(2, Neuron, 'L1')
e2 = Ensemble(10, Neuron, 'L2')

s1 = Connection(e1, e2)

p1 = Probe(e2, 'voltage')
# n1 = Neuron(0.2, "1")
# n2 = Neuron(1, "2")
# Connection(n1, [n2], [0.3])
# Connection(n2, [n1], [0.1])

model.build()

sim = simulator(model, 0.01)
sim.run(2.0)

p1.plot()

# TODO: distribution arg for ensembles
