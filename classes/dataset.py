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

        Helper.log('Dataset', log.INFO, 'new dataset initialized')

    def load(self):
        pass

    def next(self):
        try:
            out = self.data[self.index]
            Helper.log('Dataset', log.INFO, 'next data : index {0}, value {1}'.format(self.index, out))
        except IndexError:
            self.index = 0
            out = self.data[self.index]
            Helper.log('Dataset', log.ERROR, 'reading out of range of dataset ! (index {0})'.format(self.index))
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

    def __init__(self, index=0, size=50):
        super(VectorDataset, self).__init__(index)
        self.size = size

    def load(self):
        exp = self.generator()
        self.labels = exp[0]
        self.data = exp[1]

    def generator(self):
        pass


class Exp1Dataset(VectorDataset):

    def __init__(self, index=0, size=50, width=0.25, gap=0.5):
        super(Exp1Dataset, self).__init__(index, size)
        self.width = width
        self.gap = gap
        self.load()

    def generator(self):
        cat = np.random.randint(2, size=self.size)
        data = self.width * np.random.rand(self.size) + (self.gap + self.width) * cat
        return cat.tolist(), data.tolist()


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
