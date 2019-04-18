import numpy as np

import matplotlib.pyplot as plt

import csv

filename = 'datasets/fashionmnist/fashion-mnist_test.csv'
# filename = 'example.csv'
img_size = (28, 28)
image = None

with open(filename, newline='') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')

    # for i, row in enumerate(readCSV):
    #     if i in range(1, 10):
    #         plt.figure()
    #         image = np.array(row[1:]).astype(np.uint8)
    #         image = image.reshape(img_size)
    #         plt.imshow(image, cmap='gray')
    print(type(readCSV))
    readCSV.__next__()
    row = readCSV.__next__()
    image = np.array(row[1:]).astype(np.uint8)
    image = image.reshape(img_size)
    plt.imshow(image, cmap='gray')

plt.show()
