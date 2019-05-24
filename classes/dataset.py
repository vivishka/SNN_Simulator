import numpy as np
import logging as log
from .base import Helper
import sys
sys.dont_write_bytecode = True


class Dataset(object):

    def __init__(self, path, index=0):
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


class ImageDataset(Dataset):

    def __init__(self, file, index=0, size=(28, 28)):
        super(ImageDataset, self).__init__(file, index)
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
