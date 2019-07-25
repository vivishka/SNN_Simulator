import matplotlib.pyplot as plt
from classes.network import Network
from classes.neuron import IF, PoolingNeuron
from classes.layer import Bloc
from classes.simulator import Simulator
from classes.connection import Connection
from classes.probe import NeuronProbe, ConnectionProbe
from classes.encoder import Node, EncoderDoG, EncoderGabor
from classes.decoder import Decoder, DecoderSpikeTorch
from classes.learner import *
from classes.base import Helper
from classes.dataset import FileDataset
import numpy as np
import sys
sys.dont_write_bytecode = True


model = Network()


# Helper.init_logging('log.log', log.DEBUG, ['Simulator', 'Encoder', 'Layer', 'Neuron'])

nb_images = 1

n_nodes = 4
n_int = 10
n_out = 2

# filename = 'datasets/fashionmnist/fashion-mnist_train.csv'
filename = 'datasets/mnist/mnist_test.csv'

img_size = (28, 28)
size_l2 = (14, 14)
# size_l3 = (5, 5)
size_l3 = (4, 4) # the way they do it in the spycketorch: ignore lines
first_image = np.random.randint(0, 2000)
# first_image = 0
image_dataset = FileDataset(filename, first_image, size=img_size, length=nb_images)
# img_size = (12, 12)
# image_dataset = PatternGeneratorDataset(index=0, size=img_size, nb_images=nb_images, nb_features=9)

# e1 = EncoderGFR(depth=n_nodes, size=img_size, in_min=0, in_max=255, delay_max=1., gamma=1.5, threshold=0.9)
e1 = EncoderDoG(size=img_size,
                in_min=0, in_max=255,
                sigma=[(3./9., 6./9.), (7./9., 14./9.), (13/9, 26/9)],
                kernel_sizes=[3, 7, 13],
                threshold=50,
                double_filter=True,
                delay_max=0.75)

e2 = EncoderGabor(img_size, [45+22, 90+22, 135+22, 180+22], 5)

n1 = Node(e2, image_dataset)

b1 = Bloc(depth=30, size=img_size, neuron_type=IF(tau=0.2, threshold=15))
p1 = Bloc(depth=30, size=size_l2, neuron_type=PoolingNeuron())

b2 = Bloc(depth=250, size=size_l2, neuron_type=IF(threshold=10))
p2 = Bloc(depth=250, size=size_l3, neuron_type=PoolingNeuron())

b3 = Bloc(depth=200, size=size_l3, neuron_type=IF(threshold=0.01))
b3.set_inhibition(radius=None, wta=True)
#
output = DecoderSpikeTorch(size=size_l3)
for digit in range(10):
    for ci in range(20):
        Connection(b3[digit * 20 + ci], output[digit], kernel_size=(1, 1))
#
# b1.activate_threshold_adapt(t_targ=0.7, th_min=0.5, n_th1=0.2, n_th2=0.2)

c1 = Connection(e1, b1, kernel_size=(5, 5), mode='shared')
c2 = Connection(b1, p1, kernel_size=(2, 2), mode='pooling')

c3 = Connection(p1, b2, kernel_size=(3, 3), mode='shared')
c4 = Connection(b2, p2, kernel_size=(2, 2), mode='pooling')

c5 = Connection(p2, b3, kernel_size=(5, 5), mode='shared')


c1.load(np.load('kernel_l1.npy'))
c3.load(np.load('kernel_l2.npy'))
c5.load(np.load('kernel_l3.npy'))

# TODO: flower dataset wit 20 - 4 network
#  Dense 784 - 800 - 10 : RSTDP enough ?


# cp1 = ConnectionProbe(c1)
# np1 = NeuronProbe(b1[0], ['threshold', 'spike_out'])
# np2 = NeuronProbe(e1[0], ['spike_out'])


sim = Simulator(model=model, dt=0.05, batch_size=1, input_period=1, dataset=image_dataset)
# sim.load('m1.w')
sim.run(nb_images)

# sim.save('m1.w')

c1.plot_all_kernels(6, 10)
c1.plot_all_kernels()
# c3.plot_all_kernels(20, 20)

# cp1.plot()
# np1.plot(['threshold', 'spike_out'])
# np1.plot(['voltage', 'threshold'])

image_dataset.plot(-1)

# dp0.plot()

# for di in output:
#     di.plot()
print('probable digit: {}'.format(output.get_value()))

Helper.print_timings()
plt.show()


