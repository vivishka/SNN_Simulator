import matplotlib.pyplot as plt
from classes.network import Network
from classes.neuron import IF
from classes.layer import Bloc
from classes.simulator import Simulator
from classes.connection import Connection, DiagonalConnection
from classes.probe import NeuronProbe, ConnectionProbe
from classes.encoder import Node, EncoderGFR
from classes.decoder import Decoder
from classes.learner import *
from classes.base import Helper
from classes.dataset import FileDataset
# import numpy as np
import sys
sys.dont_write_bytecode = True


model = Network()


# Helper.init_logging('log.log', log.DEBUG, ['Simulator', 'Encoder', 'Layer', 'Neuron'])

nb_data = -1

n_nodes = 4
n_int = 10
n_out = 2

# data_size = (1, 4)
data_size = 4
# image_dataset = PatternGeneratorDataset(index=0, size=img_size, nb_images=nb_data, nb_features=9)
path = 'datasets/iris.csv'
iris_dataset = FileDataset(path=path, index=-1, size=data_size, length=nb_data, randomized=True)

e1 = EncoderGFR(depth=20, size=data_size,
                in_min=0, in_max=1, delay_max=1.,
                threshold=0.9, gamma=1.5)

e2 = EncoderGFR(depth=8, size=data_size,
                in_min=0, in_max=1, delay_max=1.,
                threshold=0.9, gamma=0.5)


n1 = Node([e1, e2])


b1 = Bloc(depth=1, size=20,
          neuron_type=IF(threshold=4),
          learner=SimplifiedSTDP(eta_up=0.05, eta_down=-0.05))
# TODO: finish learner
# TODO: why labels not aligned
# TODO: threshold  adapt dip every 150 input (when goes around ?)
#  k-wta ?

c1 = Connection(e1, b1)
# c2 = Connection(e2, b1)
b1.set_inhibition(wta=True, radius=None)
b1.activate_threshold_adapt(0.3, 2, 0.05, 0)


# b2 = Bloc(depth=3, size=1,
#           neuron_type=IF(threshold=0.9),
#           learner=Rstdp(eta_up=0.1, eta_down=-0.1, anti_eta_up=-0.1, anti_eta_down=0.1, wta=True))
# b1.activate_threshold_adapt(t_targ=0.3, th_min=1, n_th1=0.1, n_th2=0.1)
# c3 = Connection(b1, b2)


# d1 = Decoder(size=data_size, absolute_time=True)
d2 = Decoder(size=20, absolute_time=True)
# DiagonalConnection(e1, d1)
Connection(b1, d2, kernel_size=1)


cp1 = ConnectionProbe(c1)
np1 = NeuronProbe(b1[0], ['spike_out', 'voltage'])
npv = NeuronProbe(b1[0][0], 'threshold')
# np1 = NeuronProbe(b1[0], ['voltage', 'spike_out'])
# np2 = NeuronProbe(e1[0], ['spike_out'])


sim = Simulator(model=model, dt=0.01, batch_size=1, input_period=1, dataset=iris_dataset)
sim.run(150)

# model.restore()
# for ens in b1.ensemble_list:
#     ens.learner = None
#     ens.inhibition = False
#     ens.wta = False
#
# sim.run(150)

cp1.plot()
np1.plot('voltage')
np1.plot('spike_out')
npv.plot('threshold')
# np1.plot(['voltage', 'threshold'])

# iris_dataset.plot(-1)
# d1.plot()
d2.plot()
# e1.plot()
# e2.plot()
print(iris_dataset.labels)
# dp0.plot()


Helper.print_timings()
plt.show()
