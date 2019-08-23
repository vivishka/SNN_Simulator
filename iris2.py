from classes.network import Network
from classes.neuron import IF
from classes.simulator import Simulator, SimulatorMp
from classes.probe import *
from classes.decoder import *
from classes.encoder import *
from classes.dataset import *
from classes.learner import *

import sys
sys.dont_write_bytecode = True

if __name__ == '__main__':
    mpl_logger = log.getLogger('matplotlib')
    mpl_logger.setLevel(log.WARNING)

    filename = 'datasets/iris.csv'
    data_size = 4
    epochs1 = 2
    epochs2 = 2

    en1 = 10
    n1 = 50
    n2 = 30

    enableMP = False

    train = FileDataset('datasets/iris/iris-train.csv', -1, size=data_size, length=-1, randomized=True)
    test = FileDataset('datasets/iris/iris-test.csv', -1, size=data_size, length=-1)

    ################################
    # Layer 1 building and training
    ################################

    # Network object must be instanced before any other elements
    network = Network()

    # Element creation, they are added to the network
    l1 = SimplifiedSTDP(eta_up=0.05, eta_down=-0.05, mp=enableMP)
    e1 = EncoderGFR(
        size=data_size, depth=en1,
        in_min=0, in_max=1,
        threshold=0.8, gamma=1.8, delay_max=1,
        # spike_all_last=True
    )
    b1 = Bloc(
        depth=1, size=n1,
        neuron_type=IF(threshold=4),
        learner=l1
    )
    c1 = Connection(e1, b1, mu=0.5, sigma=0.05)
    np1 = NeuronProbe(b1[0], ["voltage", "threshold"])

    # Element parametrization, can be changed anytime
    b1.set_inhibition(wta=True, radius=None, k_wta_level=1)
    b1.set_threshold_adapt(0.7, 2, 0.05, 0)

    cps1 = ConnectionProbe(c1)

    # sim.autosave='iris1.w'
    # sim.load('iris1_f.w')

    # simulator creation
    if enableMP:
        sim = SimulatorMp(network=network, dataset=train, dt=0.05, input_period=1, batch_size=10, processes=3)
    else:
        sim = Simulator(network=network, dataset=train, dt=0.05, input_period=1, batch_size=10)
    sim.enable_time(True)

    # The network is "compiled" at this step
    sim.run(len(train.data) * epochs1)

    ################################
    # Layer 1 display
    ################################

    # print(c1.get_convergence()/n1)
    # c1.saturate_weights(0.8)
    # sim.save('iris1_f.w')

    cps1.plot()
    np1.plot('voltage')
    np1.plot('threshold')

    # print(c1[0].weights.matrix.to_dense())

    c1.plot_convergence()

    b1.stop_learner()
    b1.stop_inhibition()
    b1.stop_threshold_adapt()

    # This method needs to be called between 2 runs
    # the modification to the network will be taken into account on the next run
    network.restore()

    ################################
    # Layer 2 building and training
    ################################

    # Element creation
    l2 = Rstdp(
        eta_up=0.05, eta_down=-0.05,
        anti_eta_up=-0.01, anti_eta_down=0.01,
        size_cat=10,
        mp=True
    )
    b2 = Bloc(depth=1, size=n2, neuron_type=IF(threshold=30), learner=l2)
    c2 = Connection(b1, b2, mu=0.5, sigma=0.05)

    # Probes
    np2 = NeuronProbe(b2[0], ["voltage", "threshold"])
    cps2 = ConnectionProbe(c2)
    # for con in c2:
    #     cps2.append(ConnectionProbe(con))

    # Element parametrization
    b2.set_inhibition(wta=False, k_wta_level=3)
    # b2.set_threshold_adapt(0.6, 0.8, 0.005, 0)

    # sim.autosave = 'iris2.w'
    # sim.load('iris2.w')
    sim.run(len(train.data) * epochs2)

    ################################
    # Layer 2 display
    ################################

    for con in c2:
        print(con.get_convergence())
        con.saturate_weights(0.8)
    sim.save('iris2_f.w')

    c2.plot_convergence()

    np2.plot('voltage')
    # np2.plot('threshold')

    # plt.figure()
    # plt.imshow(c2[0].weights.matrix.to_dense(), cmap='gray')
    # print(c2[0].weights.matrix.to_dense())

    b2.stop_learner()
    b2.stop_threshold_adapt()
    network.restore()

    ################################
    # train accuracy
    ################################

    d1 = DecoderClassifier(size=3)
    # d2 = DecoderClassifier(size=n1)

    c3 = Connection(b2, d1, kernel_size=1, mode='split')
    # c4 = Connection(b1, d2, kernel_size=1)

    sim.run(len(train.data))

    d1.plot()
    print(d1.get_correlation_matrix())
    print(d1.get_accuracy())

    network.restore()

    ################################
    # Test accuracy
    ################################

    sim.dataset = test

    sim.run(len(test.data))

    d1.plot()
    print(d1.get_correlation_matrix())
    # print(d2.get_correlation_matrix())
    # plt.imshow(d2.get_correlation_matrix(), cmap='gray')
    Helper.print_timings()
    # sim.plot_steptimes()
    plt.show()
