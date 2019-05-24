import numpy as np
import logging as log
from .base import Helper
import sys
sys.dont_write_bytecode = True


class Dataset(object):

    def __init__(self, path='', index=0):
        self.index = index
        self.data = []
        self.labels = []
        self.file = None
        self.path = path
        Helper.log('Dataset', log.INFO, 'new dataset initialized')

    def load(self):
        pass

    def next(self):
        try:
            out = self.data[self.index]
            Helper.log('Dataset', log.INFO, 'next data : index {0}'.format(self.index))
        except IndexError:
            self.index -= 1
            out = self.data[self.index]
            Helper.log('Dataset', log.ERROR, 'reading out of range of dataset ! (index {0})'.format(self.index))
        self.index += 1
        return out

    def get(self, index):
        return self.data[index]


class VectorDataset(Dataset):

    def __init__(self, path='', index=0, size=50):
        super(VectorDataset, self).__init__(path, index)
        self.size = size

    def load(self):
        for input_data in range(self.size):
            exp = self.generator()
            self.labels.append(exp[0])
            self.data.append(exp[1])

    def generator(self):
        pass


class Exp1Dataset(VectorDataset):

    def __init__(self, path='', index=0, size=50, width=0.25, gap=0.5):
        super(Exp1Dataset, self).__init__(path, index, size)
        self.width = width
        self.gap = gap
        self.load()

    def generator(self):
        cat = np.random.randint(0, 1)
        data = self.width * np.random.rand() + (self.gap + self.width)*cat
        return cat, data


class ImageDataset(Dataset):

    def __init__(self, path='', index=0, size=(28, 28)):
        super(ImageDataset, self).__init__(path, index)
        self.size = size
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
