
# import matplotlib.pyplot as plt
# import numpy as np
# import csv
# import logging as log
# from classes.base import Helper
from classes.network import Network
from classes.simulator import *
from classes.neuron import IF, IFReal
# from classes.neuron import PoolingNeuron
# from classes.layer import Bloc, Ensemble
# from classes.connection import *
from classes.probe import *
# from classes.decoder import *
from classes.encoder import *
from classes.dataset import *
from classes.learner import *

from sklearn import datasets as ds
from skimage import filters as flt
from pandas import DataFrame

import sys
sys.dont_write_bytecode = True

if __name__ == '__main__':
    mpl_logger = log.getLogger('matplotlib')
    mpl_logger.setLevel(log.WARNING)

    # Helper.init_logging('main.log', log.DEBUG, ["All"])


    # filename = 'datasets/mnist/mnist_train.csv'
    filename = 'datasets/fashionmnist/fashion-mnist_train.csv'
    img_size = (28, 28)
    # img_size = (12, 12)

    first_image = np.random.randint(0, 59999-20000)
    print("init dataset image {}".format(first_image))

    # image_dataset = PatternGeneratorDataset(index=0, size=img_size, nb_images=200, nb_features=9)
    model = Network()

    layer = 1

    image_dataset = FileDataset(filename, first_image, size=img_size, length=10000)

    e1 = EncoderDoG(sigma=[(3 / 9, 6 / 9)],  # (7/9, 14/9), (13/6, 26/9)],
                    kernel_sizes=[3], size=img_size, in_min=0, in_max=255, delay_max=0.75, double_filter=True, spike_all_last=True)
    n1 = Node(e1, image_dataset, 1, 0)

    b1 = Bloc(12, img_size, IFReal(threshold=8*2**12),
              SimplifiedSTDP(
                  eta_up=0.3,
                  eta_down=-0.3,
                  mp=True)
              )
    b1.set_inhibition(wta=True, radius=2)

    c1 = Connection(e1, b1, kernel=(5, 5), mode='shared', wmax=12, real=True)

    sim = SimulatorMp(model=model, dataset=image_dataset, dt=0.05, input_period=1, batch_size=200, processes=3)
    sim.enable_time(True)
    try:
        sim.load('testsreal1.w')
    except:
        pass

    cps = []
    for con in c1:
        cps.append(ConnectionProbe(con))
    #     sprobein = NeuronProbe(target=e1[0], variables='spike_out')
    #     sprobeout = NeuronProbe(target=b1[0], variables='spike_out')
    #     vprobe = NeuronProbe(target=b1[0], variables='voltage')

    sim.run(len(image_dataset.data))
    sim.save('testsreal1.w')
    c1.plot()
    # vprobe.plot('voltage')
    # sprobein.plot('spike_out')
    # sprobeout.plot('spike_out')
    # plt.show()
    for cp in cps:
        cp.plot()
    sim.flush()
    model.restore()
#############################

    image_dataset = FileDataset(filename, first_image, size=img_size, length=-1)
    b1.set_inhibition(wta=False, radius=None)
    b1.learner = None

    b2 = Bloc(12, (14, 14), IF(threshold=0.1))
    b2.set_inhibition(wta=True, radius=0)

    b3 = Bloc(100, (14, 14), IFReal(threshold=12),
              SimplifiedSTDP(
                  eta_up=0.3,
                  eta_down=-0.3)
              )
    b3.set_inhibition(wta=True, radius=1)

    c2 = Connection(b1, b2, kernel_size=(2, 2), mode='pooling')
    c3 = Connection(b2, b3, kernel_size=(3, 3), mode='shared')
    # c4 = Connection(b2, d1, kernel=(2, 2), mode='pooling')

    cps = []
    # for con in c3:
    #     cps.append(ConnectionProbe(con))
    # sprobein = NeuronProbe(target=b2[0], variables='spike_out')
    # sprobeout = NeuronProbe(target=b3[0], variables='spike_out')
    # vprobe = NeuronProbe(target=b1[0], variables='voltage')

    # sim.load('tests1.w')
    epochs = 1
    for epoch in range(epochs):
        print("Training layer 2")
        print("Epoch "+str(epoch))
        image_dataset.index = 0
        sim.run(len(image_dataset.data))
        sim.flush()
        model.restore()
    # image_dataset.plot(-1)
    # e1.plot(layer=4)
    # plot final kernels
    # c2.plot()
    # d1.plot()
    #  plot weight history
    # for cp in cps:
    #     cp.plot()

    sim.save('testsreal2.w')

    c3.plot()
    # vprobe.plot('voltage')
    # sprobein.plot('spike_out')
    # sprobeout.plot('spike_out')
    # for index in range(image_dataset.length):
    #     image_dataset.plot(index)
    #     e1.plot(index, 0)

    sim.plot_steptimes()
    Helper.print_timings()

    # sim.flush()
    # model.restore()
    plt.show()

