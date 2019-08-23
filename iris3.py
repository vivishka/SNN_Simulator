
from classes.network import Network
from classes.neuron import IF
from classes.simulator import Simulator  # , SimulatorMp
# from classes.connection import *
from classes.probe import *
from classes.decoder import *
from classes.encoder import *
from classes.dataset import *
from classes.learner import *

import time

import iris_ann_generator

import sys
sys.dont_write_bytecode = True


"""
    In this example, we will show how to pre train the SNN weights using a ANN network and optimize the thresholds after
    
    You may notice we use a GFR encoder for the input layer, the dimension of the first connection is different from
    an ANN. for this, we converted the input data in a format similar to the GFR encoding so the connection can be of 
    the same dimension.
    
    The network dimensions are specified (size of hidden layers) and an ANN is created and trained using TF.
    The SNN network is then created in the simulator and the weights from the ANN are loaded
    
    One drawback of using an ANN is the inability to use bias in neurons, for this we train the ANN with the no bias 
    constraint and then modify globally the threshold of the neuron in each layer to find an optimum.
    ths approach use a straightforward approach to the threshold optimization by splitting the possible combination into
    a grid. This allows to have a visual representation of the optima.
    All previous steps were done using only the training dataset.
    
    Finally the accuracy is measured and compared between the train and test dataset.
"""

if __name__ == '__main__':

    start = time.time()

    ################################
    # Parameters
    ################################

    en1 = 10
    n1 = 50
    n2 = 10

    mpl_logger = log.getLogger('matplotlib')
    mpl_logger.setLevel(log.WARNING)

    data_in_size = 4
    data_out_size = 3

    train = FileDataset('datasets/iris/iris-train.csv', -1, size=data_in_size, length=120, randomized=True)
    test = FileDataset('datasets/iris/iris-test.csv', -1, size=data_in_size, length=30)

    t1 = np.linspace(0.5, 1.5, 10)
    t2 = np.linspace(0.5, 1.5, 10)
    success_map = np.zeros((len(t1), len(t2)))

    ################################
    # ANN weight initialisation
    ################################

    # will create and train an ANN network with the same dimensions and return the weights in a file
    iris_ann_generator.run(en1, n1, n2)

    ################################
    # Network building
    ################################

    # Network object must be instanced before any other elements
    network = Network()

    # Input layer
    e1 = EncoderGFR(
        size=data_in_size, depth=en1,
        in_min=0, in_max=1,
        threshold=0.9, gamma=1.5, delay_max=1,
        # spike_all_last=True: optional, useful for better STDP learning
    )

    # Hidden Layer 1
    b1 = Bloc(depth=1, size=n1, neuron_type=IF(threshold=0.5))
    # By default, when no other arguments provided, connections will be all to all
    # Here the weight initialization is not important as it will be overwritten by the ANN weights
    c1 = Connection(e1, b1)

    # Hidden Layer 2
    b2 = Bloc(depth=1, size=n2 * data_out_size, neuron_type=IF(threshold=2))
    c2 = Connection(b1, b2)
    # inhibition is used to only allow 1 spike to reach the output layer
    # The spike is either the first or the one with the highest voltage if simultaneous
    b2.set_inhibition(wta=True, radius=None)

    # Output Layer
    d1 = DecoderClassifier(size=data_out_size)
    c3 = Connection(b2, d1, kernel_size=1, mode='split')

    # Connection weight initialization using ANN weights
    c1.load(np.load('iris_c1.npy'))
    c2.load(np.load('iris_c2.npy'))

    sim = Simulator(network=network, dataset=train, dt=0.01, input_period=1)

    ################################
    # Threshold optimisation
    ################################

    Helper.print_progress(0, len(t1) * len(t2), "testing thresholds ", bar_length=30)
    last_time = time.time()

    # This part can be sped up greatly using multiprocessing
    for i2, th2 in enumerate(t2):
        for i1, th1 in enumerate(t1):
            b1.set_threshold(th1)
            b2.set_threshold(th2)
            sim.run(len(train.data))
            confusion = d1.get_correlation_matrix()
            success = 0
            for i in range(3):
                success += confusion[i, i]/len(train.data)
            success_map[i1, i2] = success
            network.restore()  # should be called in between 2 runs (when the network changes)
            Helper.print_progress(len(t1)*i2+i1, len(t1)*len(t2), "testing thresholds ", bar_length=30,)
            # suffix='est. time: {} s'.format(int(len(t1)*len(t2)-(len(t2)*i1+i2)/(time.time() - last_time))))
            last_time = time.time()

    t1max, t2max = np.where(success_map == np.amax(success_map))
    Helper.print_progress(1, 1, "testing thresholds ", bar_length=30,)
    # suffix='est. time: {} s'.format(int(len(t1)*len(t2)-(len(t2)*i1+i2)/(time.time() - last_time))))
    for imax in range(len(t1max)):
        print([t1[t1max[imax]], t2[t2max[imax]], np.amax(success_map)])

    fig = plt.figure()
    plt.imshow(success_map, cmap='gray', extent=[t1[0], t1[-1], t2[-1], t2[0]])
    # np.save('th_map.npy', success_map)
    # plt.imsave('success_map.png', arr=success_map, cmap='gray', format='png')

    network.restore()
    b1.set_threshold(t1[t1max[0]])
    b2.set_threshold(t2[t2max[0]])
    sim.enable_time(True)

    ################################
    # train accuracy
    ################################

    print("Evaluating train accuracy")
    sim.dataset = train
    sim.run(len(train.data))

    d1.plot()
    print("Confusion matrix")
    print(d1.get_correlation_matrix())
    print("Train accuracy: {}".format(d1.get_accuracy()))

    network.restore()

    ################################
    # Test accuracy
    ################################

    print("Evaluating test accuracy")
    sim.dataset = test
    sim.run(len(test.data))

    d1.plot()
    print("Confusion matrix")
    print(d1.get_correlation_matrix())
    print("Test accuracy: {}".format(d1.get_accuracy()))

    print('total time')
    print(time.time() - start)

    # Used at the end of every file using a plot() call
    plt.show()
