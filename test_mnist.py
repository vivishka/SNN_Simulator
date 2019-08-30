import matplotlib.pyplot as plt
from classes.network import Network
from classes.neuron import IF, PoolingNeuron
from classes.layer import Bloc
from classes.simulator import Simulator
from classes.connection import Connection
from classes.encoder import EncoderDoG, EncoderGabor
from classes.decoder import Decoder, DecoderClassifier
from classes.probe import ConnectionProbe, NeuronProbe
from classes.learner import *
from classes.base import Helper
from classes.dataset import FileDataset
# import numpy as np
import sys
sys.dont_write_bytecode = True


"""
    
    In this example, we will show how to use a CNN in ths SNN simulator.
    
    Several types of encoders can be used for a SCNN, but the main ones are GFR and filter type encoders.
    The filter type first apply a list of filters to the image then linearly encode the output image.
    The ones currently implemented are 
        - DoG (Difference of Gaussian) that applies 2 gaussian filters of different sigma 
        and the result is the subtraction of the 2 images.
        - Gabor: Use the Gabor filter with a parameterized angle. this filter is mostly used to detect edges.
        
    several filters with different parameters can be applied for one encoder. Each one will correspond to 1 ensemble
    
    The decoder classifier is linked to the last layer by a categorize connection
    so that 1 ensemble is connected to 1 neurone.
    the decoder classifier use the dataset to know how many category there are and how 
    to split the neurons to each category ex: 200 neurons, 10 categories, neurons 10 to 19 belongs to cat "1"
    
    The network dimensions are specified (size of hidden layers) 
    
    One drawback of using an ANN is 
    
    All previous steps were done using only the training dataset.
    
    Finally the accuracy is measured and compared between the train and test dataset.
"""


################################
# Parameters
################################

# Helper.init_logging('log.log', log.DEBUG, ['Simulator', 'Encoder', 'Layer', 'Neuron'])

nb_images = 1

nb_cat = 10

depth_l1 = 30
depth_l2 = 250
depth_l3 = 200


# filename = 'datasets/fashionmnist/fashion-mnist_train.csv'
filename = 'datasets/mnist/mnist_test.csv'

img_size = (28, 28)
size_l2 = (14, 14)
size_l3 = (5, 5)
# size_l3 = (4, 4) # the way they do it in the spyketorch: ignore lines
# first_image = np.random.randint(0, 2000)
# first_image = 0

enableMP = False

################################
# Network building
################################

network = Network()

train_set = FileDataset(path='datasets/mnist/mnist_train.csv', start_index=0, size=img_size, length=-1)
test_set = FileDataset(path='datasets/mnist/mnist_test.csv', start_index=0, size=img_size, length=-1)

# Encoders
# e1 = EncoderGFR(depth=n_nodes, size=img_size, in_min=0, in_max=255, delay_max=1., gamma=1.5, threshold=0.9)
e1 = EncoderDoG(size=img_size,
                in_min=0, in_max=255,
                sigma=[(3./9., 6./9.), (7./9., 14./9.), (13/9, 26/9)],
                kernel_sizes=[3, 7, 13],
                threshold=50,
                double_filter=True,
                delay_max=0.75)

e2 = EncoderGabor(img_size, [45+22, 90+22, 135+22, 180+22], 2)


l1 = SimplifiedSTDP(eta_up=0.01, eta_down=-0.01, mp=enableMP)
l2 = Rstdp(eta_up=0.01, eta_down=-0.01,
           anti_eta_up=-0.01, anti_eta_down=0.01,
           mp=enableMP, wta=True, size_cat=nb_cat)

# Layers
b1 = Bloc(depth=depth_l1, size=img_size, neuron_type=IF(threshold=10), learner=l1)
p1 = Bloc(depth=depth_l1, size=size_l2, neuron_type=PoolingNeuron())

b2 = Bloc(depth=depth_l2, size=size_l2, neuron_type=IF(threshold=10), learner=l1)
p2 = Bloc(depth=depth_l2, size=size_l3, neuron_type=PoolingNeuron())

b3 = Bloc(depth=depth_l3, size=size_l3, neuron_type=IF(threshold=0.01), learner=l2)

# Decoder
# d00 = Decoder(size=img_size)
# d01 = Decoder(size=img_size)
# d02 = Decoder(size=img_size)
# d03 = Decoder(size=img_size)
d1 = DecoderClassifier(size=depth_l3)

# Connections

# Connection(e2[0], d00, kernel_size=(1, 1))
# Connection(e2[1], d01, kernel_size=(1, 1))
# Connection(e2[2], d02, kernel_size=(1, 1))
# Connection(e2[3], d03, kernel_size=(1, 1))


# c1.load(np.load('kernel_l1.npy'))
# c3.load(np.load('kernel_l2.npy'))
# c5.load(np.load('kernel_l3.npy'))

# Parameters
# b1.activate_threshold_adapt(t_targ=0.7, th_min=0.5, n_th1=0.2, n_th2=0.2)


# cp1 = ConnectionProbe(c1)
np1 = NeuronProbe(b1[0], ['voltage', ])
# np2 = NeuronProbe(e1[0], ['spike_out'])


################################
# Layer 1 building and training
################################

c1 = Connection(e2, b1, kernel_size=(5, 5), mode='shared')

b1.set_inhibition(wta=True, radius=2, k_wta_level=5)

sim = Simulator(network=network, dt=0.05, batch_size=1, input_period=1, dataset=test_set)
# sim.load('m1.w')
sim.run(1)

b1.stop_learner()
b1.stop_inhibition()
network.restore()

# sim.save('m1.w')

################################
# Layer 2 building and training
################################

c2 = Connection(b1, p1, kernel_size=(2, 2), mode='pooling')
c3 = Connection(p1, b2, kernel_size=(3, 3), mode='shared')

b2.set_inhibition(wta=True, radius=2, k_wta_level=5)


sim.run(1)

b2.stop_learner()
b2.stop_inhibition()
network.restore()


################################
# Layer 3 building and training
################################

c4 = Connection(b2, p2, kernel_size=(3, 3), mode='pooling')
c5 = Connection(p2, b3, kernel_size=(5, 5), mode='shared')
c6 = Connection(b3, d1, mode='categorize')

b3.set_inhibition(wta=True, radius=2, k_wta_level=5)

# c1.plot_all_kernels(6, 10)
# c1.plot_all_kernels()
# c3.plot_all_kernels(20, 20)

# cp1.plot()
np1.plot(['voltage', ])
# np1.plot(['voltage', 'threshold'])

#
# d00.plot()
# d01.plot()
# d02.plot()
# d03.plot()

# for di in output:
#     di.plot()
print('probable digit: {}'.format(d1.get_accuracy()))

Helper.print_timings()
plt.show()
