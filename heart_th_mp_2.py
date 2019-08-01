
import matplotlib.pyplot as plt
import numpy as np
import csv
import logging as log
from classes.base import Helper
from classes.network import Network
from classes.neuron import LIF, IF
from classes.neuron import PoolingNeuron
from classes.layer import Bloc, Ensemble
from classes.simulator import Simulator, SimulatorMp
from classes.connection import *
from classes.probe import *
from classes.decoder import *
from classes.encoder import *
from classes.dataset import *
from classes.learner import *

import time
import copy
import math
import multiprocessing as mp



from sklearn import datasets as ds
from skimage import filters as flt
from pandas import DataFrame

import sys
sys.dont_write_bytecode = True

def chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))

def test_mp(queue, thresholds, model, dataset):
    my_model = copy.deepcopy(model)
    my_dataset = copy.deepcopy(dataset)
    sim = Simulator(my_model, dt=0.01, input_period=1, dataset=my_dataset)
    my_model.objects[Decoder][0].dataset = my_dataset
    success_map = np.zeros(len(thresholds))
    for ith, th in enumerate(thresholds):
        for ilay, layer in enumerate(th):
            my_model.objects[Bloc][ilay + 1].set_threshold(th[ilay])
        sim.run(len(my_dataset.data))
        confusion = my_model.objects[Decoder][0].get_correlation_matrix()
        success = 0
        for i in range(my_dataset.n_cats):
            success += confusion[i, i]/len(my_dataset.data)
        success_map[ith] = success
        my_model.restore()
    queue.put(success_map)



mpl_logger = log.getLogger('matplotlib')
mpl_logger.setLevel(log.WARNING)



if __name__ == '__main__':

    import heart_ann_generator_2

    start = time.time()

    en1 = 20
    n1 = 200
    dec1 = 10

    n_proc = 12

    heart_ann_generator_2.run(en1, n1, dec1)



    # filename = 'datasets/heart.csv'
    data_size = 13



    train = FileDataset('datasets/heart/heart - train.csv', size=data_size, randomized=True)
    test = FileDataset('datasets/heart/heart - test.csv', size=data_size)

    t1 = np.linspace(0, 4, 100)
    t2 = np.linspace(0, 2, 100)
    success_map = np.zeros((len(t1), len(t2)))
    th_list = np.zeros((len(t1) * len(t2), 2))
    succ_list = np.zeros(len(t1) * len(t2))

    for th in range(len(t1) * len(t2)):
        th_list[th] = [t1[th//len(t2)], t2[th % len(t2)]]

    split = [th_list[i:i+math.ceil(len(th_list)/n_proc)] for i in range(0, len(th_list), math.ceil(len(th_list)/n_proc))]
    # sim.enable_time(True)
    last_time = time.time()
    # Helper.print_progress(0, len(t1)*len(t2), "testing thresholds ", bar_length=30)
    model = Network()
    e1 = EncoderGFR(size=data_size, depth=en1, in_min=0, in_max=1, threshold=0.9, gamma=1.5, delay_max=1,# spike_all_last=True
                    )
    # node = Node(e1)
    b1 = Bloc(depth=1, size=n1, neuron_type=IF(threshold=0))
    c1 = Connection(e1, b1, mu=0.6, sigma=0.05)
    c1.load(np.load('c1.npy'))

    b2 = Bloc(depth=1, size=dec1 * 2, neuron_type=IF(threshold=0))
    c2 = Connection(b1, b2, mu=0.6, sigma=0.05)
    c2.load(np.load('c2.npy'))
    b2.set_inhibition(wta=True, radius=(0, 0))

    d1 = DecoderClassifier(size=2)

    c3 = Connection(b2, d1, kernel_size=1, mode='split')

    sim = Simulator(model=model, dataset=train, dt=0.01, input_period=1)
    model.build()

    workers = []
    queues = [mp.Queue() for _ in range(n_proc)]

    for worker_id, worker_load in enumerate(split):

        workers.append(mp.Process(target=test_mp,
                                  args=(queues[worker_id],
                                        split[worker_id],
                                        model,
                                        train
                                        )
                                  )
                       )
        workers[worker_id].start()


    finished = 0
    while finished != n_proc:
        for worker_id, worker in enumerate(workers):
            if not queues[worker_id].empty():
                finished += 1

                succ_list[math.ceil(len(th_list)/n_proc) * worker_id:math.ceil(len(th_list)/n_proc) * (worker_id+1)] =\
                    queues[worker_id].get(False)
            else:
                time.sleep(0.1)

    success_map = np.reshape(succ_list, (len(t1), len(t2)))

    t1max, t2max = np.where(success_map == np.amax(success_map))
    Helper.print_progress(1, 1, "testing thresholds ", bar_length=30,)# suffix='est. time: {} s'.format(int(len(t1)*len(t2)-(len(t2)*i1+i2)/(time.time() - last_time))))
    for imax in range(len(t1max)):
        print([t1[t1max[imax]], t2[t2max[imax]], np.amax(success_map)])

    fig = plt.figure()
    np.save('th_map.npy', success_map)
    plt.imshow(success_map, cmap='gray', extent=[t1[0], t1[-1], t2[-1], t2[0]], aspect='auto')
    plt.imsave('success_map.png', arr=success_map, cmap='gray', format='png')

    sim.dataset = test
    d1.dataset = test
    b1.set_threshold(t1[t1max[0]])
    b2.set_threshold(t2[t2max[0]])
    sim.enable_time(True)
    sim.run(len(test.data))
    confusion = d1.get_correlation_matrix()
    success = 0
    for i in range(2):
        success += confusion[i, i] / len(test.data)
    print(confusion)
    print(success)
    print('total time')
    print(int(time.time()-start))
    plt.show()