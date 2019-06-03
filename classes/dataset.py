import numpy as np
import matplotlib.pyplot as plt
import logging as log
from .base import Helper
import sys
sys.dont_write_bytecode = True


class Dataset(object):

    def __init__(self, index=0):
        self.index = index
        self.data = []
        self.labels = []
        self.file = None
        self.n_cats = None
        self.pop_cats = []

        Helper.log('Dataset', log.INFO, 'new dataset initialized')

    def load(self):
        pass

    def next(self):
        # try:
        out = self.data[self.index]
            # Helper.log('Dataset', log.INFO, 'next data : index {0}, value {1}'.format(self.index, out))
        # except IndexError:
        #     self.index = 0
            # out = self.data[self.index]
            # Helper.log('Dataset', log.ERROR, 'reading out of range of dataset ! (index {} with max {} )'
            #            .format(self.index, len(self.data)))
        self.index += 1
        return out

    def get(self, index):
        return self.data[index]

    def plot(self, value):
        plt.figure()
        if value == 'Data' or value == 'All':
            plt.plot(self.data)
        elif value == 'Labels' or value == 'All':
            plt.plot(self.labels)


class VectorDataset(Dataset):

    def __init__(self, index=0, size=50, generator=None):
        super(VectorDataset, self).__init__(index)
        self.size = size
        self.generator = generator
        self.load()

    def load(self):
        Helper.log('Dataset', log.INFO, 'Vector dataset loading ...')
        self.labels, self.data = self.generator(self.size)
        self.n_cats = len(set(self.labels))
        Helper.log('Dataset', log.INFO, 'Dataset contains {} categories'.format(self.n_cats))
        self.pop_cats = np.zeros(self.n_cats)
        for label in self.labels:
            self.pop_cats[label] += 1
        Helper.log('Dataset', log.INFO, 'done')


class ImageDataset(Dataset):

    def __init__(self, path, index=0, size=(28, 28)):
        super(ImageDataset, self).__init__(index)
        self.size = size
        self.path = path
        self.load()

    def load(self):
        Helper.log('Dataset', log.INFO, 'reading file')
        self.file = open(self.path, 'r')
        self.file.__next__()
        temp = self.file.readlines()
        for string in temp:
            self.data.append(np.array(string[2:].split(',')).astype(np.uint8).reshape(self.size))
            self.labels.append(np.array(string[0]).astype(np.uint8))
        self.file.close()
        Helper.log('Dataset', log.INFO, 'reading file done')
