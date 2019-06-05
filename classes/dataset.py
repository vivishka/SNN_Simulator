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
        try:
            out = self.data[self.index]
            Helper.log('Dataset', log.INFO, 'next data : index {0}, value {1}'.format(self.index, out))
        except IndexError:
            self.index = 0
            out = self.data[self.index]
            Helper.log('Dataset', log.ERROR, 'reading out of range of dataset ! (index {} with max {} )'
                       .format(self.index, len(self.data)))
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


def pattern_image_generator(size, pattern_list, nb_features):
    mat = np.zeros(size, dtype=float)
    nb_div = int(np.ceil(np.sqrt(nb_features)))
    x_div = size[0] // nb_div
    y_div = size[1] // nb_div
    i = 0
    order = [index % len(pattern_list) for index in range(nb_features)]
    np.random.shuffle(order)
    for row in range(nb_div):
        for col in range(nb_div):
            if i >= nb_features:
                break
            feature = pattern_list[order[i]]
            s = feature.shape
            x0 = np.random.randint(row * x_div, (row + 1) * x_div - s[0])
            y0 = np.random.randint(col * y_div, (col + 1) * y_div - s[0])
            mat[x0:x0+s[0], y0:y0+s[1]] += feature
            i += 1

    for i in range(size[0]):
        for j in range(size[1]):
            if mat[i, j] == 1:
                mat[i, j] = np.random.randn() * 0.1 + 0.75
            else:
                mat[i, j] = np.random.randn() * 0.1 + 0.25

    mat = np.clip(mat, 0, 1) * 255
    return mat.astype('uint8')


def create_pattern_dataset(path, nb_images, size, nb_features):
    k1 = np.array([[1], [1], [1]], dtype=int)
    k2 = np.array([[1, 1, 1]], dtype=int)
    k3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int)
    k4 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=int)

    file = open(path, 'w')
    file.write('\n')

    for i in range(nb_images):
        data = pattern_image_generator(size, [k3, k4], nb_features)
        if nb_images == 1:
            plt.figure()
            plt.imshow(data, cmap='gray')
        data = data.flatten()
        line = '0,' + ",".join(str(x) for x in data)
        file.write(line)
        file.write('\n')

    if nb_images == 1:
        plt.show()
    file.close()
