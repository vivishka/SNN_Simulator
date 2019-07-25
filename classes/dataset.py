import numpy as np
import matplotlib.pyplot as plt
import logging as log
from .base import Helper, MeasureTiming
import sys
sys.dont_write_bytecode = True


class Dataset(object):

    def __init__(self, start_index=0):
        self.start_index = start_index
        self.index = -1
        self.data = []
        self.labels = []
        self.n_cats = None
        self.pop_cats = []

        Helper.log('Dataset', log.INFO, 'new dataset initialized')

    def load(self):
        pass

    def next(self):
        self.index += 1
        try:
            out = self.data[self.index]
            Helper.log('Dataset', log.INFO, 'next data : index {0}, value {1}'.format(self.index, out))
        except IndexError:
            self.index = 0
            out = self.data[self.index]
            Helper.log('Dataset', log.ERROR, 'reading out of range of dataset ! (index {} with max {} )'
                       .format(self.index, len(self.data)))
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


class FileDataset(Dataset):

    def __init__(self, path, start_index=0, size=(28, 28), length=-1, randomized=False):
        super(FileDataset, self).__init__(start_index)
        self.size = (1, size) if isinstance(size, int) else size
        self.path = path
        self.length = length
        self.randomized = randomized
        self.load()

    @MeasureTiming('file_load')
    def load(self):
        Helper.log('Dataset', log.INFO, 'reading file')
        file = open(self.path, 'r')
        # file.__next__()  # skip first line
        if self.length < 0:
            temp = file.readlines()
            for string in temp:
                string_vect = np.array(string.split(','))
                self.data.append(string_vect[1:].astype(float).reshape(self.size))
                self.labels.append(string_vect[0].astype(int))
        else:
            for _ in range(self.start_index):
                try:
                    file.__next__()
                except ValueError:
                    file.seek(0, 0)
                    file.__next__()
            for line in range(self.length):
                string = file.readline()
                string_vect = np.array(string.split(','))
                self.data.append(string_vect[1:].astype(float).reshape(self.size))
                self.labels.append(string_vect[0].astype(int))

        if self.randomized:
            random_indexes = np.arange(len(self.data))
            np.random.shuffle(random_indexes)
            self.data = [self.data[index] for index in random_indexes]
            self.labels = [self.labels[index] for index in random_indexes]

        self.n_cats = len(set(self.labels))
        self.pop_cats = np.zeros(self.n_cats)
        for cat in range(self.n_cats):
            self.pop_cats[cat] = self.labels.count(cat)
        Helper.log('Dataset', log.INFO, 'Dataset contains {} categories'.format(self.n_cats))
        file.close()
        Helper.log('Dataset', log.INFO, 'reading file done')

    def plot(self, index=-1):
        plt.figure()
        plt.imshow(self.data[index], cmap='gray')
        plt.title('Source image input '+str(index))


class PatternGeneratorDataset(Dataset):

    def __init__(self, index=0, size=(28, 28), nb_images=10, nb_features=4):
        super(PatternGeneratorDataset, self).__init__(index)
        self.nb_images = nb_images
        self.size = size
        self.nb_features = nb_features
        self.load()

    def load(self):
        # k1 = np.array([[1], [1], [1]], dtype=int)
        # k2 = np.array([[1, 1, 1]], dtype=int)
        k3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int)
        k4 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=int)

        for i in range(self.nb_images):
            data = self.pattern_image_generator(self.size, [k3, k4], self.nb_features)
            self.data.append(data)
            self.labels.append(0)

    @staticmethod
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
                    mat[i, j] = np.random.randn() * 0.1 + 0.8
                else:
                    mat[i, j] = np.random.randn() * 0.1 + 0.2

        mat = np.clip(mat, 0, 1) * 255
        return mat.astype('uint8')

    def plot(self, index=-1):
        plt.figure()
        plt.imshow(self.data[index], cmap='gray')
        plt.title('Source image input '+str(index))
