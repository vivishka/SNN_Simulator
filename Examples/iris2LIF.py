from classes.network import Network
from classes.neuron import LIF
from classes.simulator import Simulator, SimulatorMp
from classes.probe import *
from classes.decoder import *
from classes.encoder import *
from classes.dataset import *
from classes.learner import *

import sys
sys.dont_write_bytecode = True
"""
see iris2, with LIf neurons instead of IF
"""

if __name__ == '__main__':
    mpl_logger = log.getLogger('matplotlib')
    mpl_logger.setLevel(log.WARNING)

    filename = 'datasets/iris.csv'
    data_size = 4
    epochs1 = 50
    epochs2 = 20

    en1 = 7
    n1 = 20
    n2 = 3

    enableMP = False

    test = FileDataset('datasets/iris/iris - train.csv', -1, size=data_size, length=-1)
    train = FileDataset('datasets/iris/iris - test.csv', -1, size=data_size, length=-1, randomized=True)

    ################################
    # Layer 1 building and training
    ################################

    # Network object must be instanced before any other elements
    network = Network()

    # Element creation, they are added to the network
    l1 = SimplifiedSTDP(eta_up=0.2, eta_down=-0.2, mp=enableMP)
    e1 = EncoderGFR(
        size=data_size, depth=en1,
        in_min=0, in_max=1,
        threshold=0.9, gamma=1, delay_max=1,
        spike_all_last=True
    )
    b1 = Block(
        depth=1, size=n1,
        neuron_type=LIF(threshold=2.3, tau=0.1),
        learner=l1
    )
    c1 = Connection(e1, b1, mu=0.6, sigma=0.10, wmin=0, wmax=1)
    # np1 = NeuronProbe(b1[0], ["threshold"])

    # Element parametrization, can be changed anytime
    b1.set_inhibition(wta=True, radius=None, k_wta_level=2)
    # b1.set_threshold_adapt(0.7, 2, 0.05, 0)

    cps1 = ConnectionProbe(c1)

    # sim.autosave='iris1.w'
    # sim.load('iris1_f.w')

# simulator creation
    if enableMP:
        sim = SimulatorMp(network=network, dataset=train, dt=0.01, input_period=1, batch_size=30, processes=3)
    else:
        sim = Simulator(network=network, dataset=train, dt=0.01, input_period=1, batch_size=1)
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
    # np1.plot('voltage')
    # np1.plot('threshold')

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
        eta_up=0.2, eta_down=-0.2,
        anti_eta_up=-0.05, anti_eta_down=0.05,
        size_cat=1,
        mp=enableMP
    )
    b2 = Block(depth=1, size=n2, neuron_type=LIF(threshold=1.9, tau=0.1), learner=l2)
    c2 = Connection(b1, b2, wmin=0, mu=0.5, sigma=0.2)
    d1 = DecoderClassifier(size=3)
    c3 = Connection(b2, d1, kernel_size=1, mode='split')
    # Probes
    # np2 = NeuronProbe(b2[0], ["voltage", "threshold"])
    cps2 = ConnectionProbe(c2)
    # for con in c2:
    #     cps2.append(ConnectionProbe(con))

    # Element parametrization
    b2.set_inhibition(wta=True, k_wta_level=1)
    # b2.set_threshold_adapt(0.6, 0.8, 0.005, 0)

    # sim.autosave = 'iris2.w'
    # sim.load('iris2.w')
    sim.run(len(train.data) * epochs2)

    ################################
    # Layer 2 display
    ################################

    for con in c2:
        print(con.get_convergence())
        # con.saturate_weights(0.8)
    sim.save('iris2_f.w')

    c2.plot_convergence()
    cps2.plot()
    # np2.plot('voltage')
    # np2.plot('threshold')

    # plt.figure()
    # plt.imshow(c2[0].weights.matrix.to_dense(), cmap='gray')
    # print(c2[0].weights.matrix.to_dense())

    b2.stop_learner()
    b2.stop_threshold_adapt()
    d1.plot_accuracy(150)
    network.restore()

    ################################
    # train accuracy
    ################################


    # d2 = DecoderClassifier(size=n1)


    # c4 = Connection(b1, d2, kernel_size=1)

    sim.run(len(train.data))

    d1.plot()
    print(d1.get_correlation_matrix())
    print(d1.get_accuracy())

    network.restore()

    ################################
    # Test accuracy
    ################################
    npt1 = NeuronProbe(b1[0], ["voltage"])
    npt2 = NeuronProbe(b2[0], ["voltage"])
    sim.dataset = test

    sim.run(len(test.data))
    npt1.plot_variable("voltage")
    npt2.plot_variable("voltage")
    d1.plot()

    print(d1.get_accuracy())
    print(d1.get_correlation_matrix())
    # print(d2.get_correlation_matrix())
    # plt.imshow(d2.get_correlation_matrix(), cmap='gray')
    Helper.print_timings()
    # sim.plot_steptimes()
    plt.show()
