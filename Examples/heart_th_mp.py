from classes.network import Network
from classes.neuron import IF
from classes.simulator import Simulator
from classes.probe import *
from classes.decoder import *
from classes.encoder import *
from classes.dataset import *
from classes.learner import *

import time
import copy
import math
import pickle
import multiprocessing as mp

import sys
sys.dont_write_bytecode = True

"""
Applies the ANN-SNN conversion, and then tries some combinations of thresholds to chose the one with optimal
performance. 

"""

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
            my_model.objects[Block][ilay + 1].set_threshold(th[ilay])
        sim.run(len(my_dataset.data))
        confusion = my_model.objects[Decoder][0].get_correlation_matrix()
        success = 0
        for i in range(my_dataset.n_cats):
            success += confusion[i, i]/len(my_dataset.data)
        success_map[ith] = success
        my_model.restore()
    queue.put(success_map)






if __name__ == '__main__':

    start = time.time()

    en1 = 10
    n1 = 200
    n2 = 100
    dec1 = 10

    n_proc = 3

    import heart_ann_generator
    heart_ann_generator.run(en1, n1, n2, dec1)


    mpl_logger = log.getLogger('matplotlib')
    mpl_logger.setLevel(log.WARNING)

    filename = 'datasets/iris.csv'
    data_size = 13



    train = FileDataset('datasets/heart/heart - train.csv', 0, size=data_size, randomized=True)
    test = FileDataset('datasets/heart/heart - test.csv', 0, size=data_size)

    t1 = np.linspace(0.5, 2, 10)
    t2 = np.linspace(0.5, 2, 10)
    t3 = np.linspace(0, 0.8, 10)

    ths = np.ndarray((len(t1), len(t2), len(t3)), dtype=object)
    for i1, th1 in enumerate(t1):
        for i2, th2 in enumerate(t2):
            for i3, th3 in enumerate(t3):
                ths[i1, i2, i3] = (th1, th2, th3)

    success_map = np.zeros((len(t1), len(t2), len(t3)))
    th_list = ths.flatten()
    succ_list = np.zeros(len(th_list))

    # for th in range(len(th_list)):
    #     th_list[th] = [t1[th//len(t2)], t2[th % len(t2)]]

    split = [th_list[i:i+math.ceil(len(th_list)/n_proc)] for i in range(0, len(th_list), math.ceil(len(th_list)/n_proc))]
    # sim.enable_time(True)
    last_time = time.time()
    # Helper.print_progress(0, len(t1)*len(t2), "testing thresholds ", bar_length=30)
    model = Network()
    e1 = EncoderGFR(size=data_size, depth=en1, in_min=0, in_max=1, threshold=0.9, gamma=1.5, delay_max=1,# spike_all_last=True
                    )
    # node = Node(e1)
    b1 = Block(depth=1, size=n1, neuron_type=IF(threshold=0))
    c1 = Connection(e1, b1, mu=0.6, sigma=0.05)
    c1.load(np.load('c1.npy'))

    b2 = Block(depth=1, size=n2, neuron_type=IF(threshold=0))
    c2 = Connection(b1, b2, mu=0.6, sigma=0.05)
    c2.load(np.load('c2.npy'))
    # b2.set_inhibition(wta=True, radius=(0, 0))

    b3 = Block(depth=1, size=dec1 * 2, neuron_type=IF(threshold=0))
    c3 = Connection(b2, b3, mu=0.6, sigma=0.05)
    c3.load(np.load('c3.npy'))
    b3.set_inhibition(wta=True, radius=(0, 0))

    d1 = DecoderClassifier(size=2)

    c4 = Connection(b3, d1, kernel_size=1, mode='split')

    sim = Simulator(network=model, dataset=train, dt=0.01, input_period=1)
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

    success_map = np.reshape(succ_list, (len(t1), len(t2), len(t3)))

    t1max, t2max, t3max = np.where(success_map == np.amax(success_map))
    Helper.print_progress(1, 1, "testing thresholds ", bar_length=30,)# suffix='est. time: {} s'.format(int(len(t1)*len(t2)-(len(t2)*i1+i2)/(time.time() - last_time))))
    for imax in range(len(t1max)):
        print([t1[t1max[imax]], t2[t2max[imax]], t3[t3max[imax]], np.amax(success_map)])

    np.save('th_map.npy', success_map)
    with open("th_map.th", "wb") as file:
        pickle.dump([success_map, t1, t2, t3], file)
    for i3 in range(len(t3)):
        fig = plt.figure()
        plt.imshow(success_map[:, :, i3], cmap='gray', extent=[t1[0], t1[-1], t2[-1], t2[0]])
        plt.title(str(t3[i3]))
    # plt.imsave('success_map.png', arr=success_map, cmap='gray', format='png')

    sim.dataset = test
    d1.dataset = test
    b1.set_threshold(t1[t1max[0]])
    b2.set_threshold(t2[t2max[0]])
    b2.set_threshold(t3[t3max[0]])
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
