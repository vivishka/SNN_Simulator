from __future__ import print_function

"""
Light-weight demo of SimpleSharpener, Spiking_BRelu, and Softmax_Decode for a fully connected net on mnist.
"""

import os
import numpy as np
import keras
from keras import backend as K
from keras import regularizers
from keras.datasets import mnist
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense
from keras.optimizers import Adadelta
from keras.constraints import NonNeg
from whetstone.layers import Spiking_BRelu, Softmax_Decode, key_generator
from whetstone.callbacks import SimpleSharpener, WhetstoneLogger

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import pandas as pd

import time

def normalize(df):
    for col in range(1, len(df.columns)):
        cmax = max(df.iloc[:, col])
        cmin = min(df.iloc[:, col])
        df.iloc[:, col] = (df.iloc[:, col] - cmin) / (cmax - cmin)



def GFR(data, depth, in_min=0, in_max=1, out_max=1, gamma=1.5):
    index = data.index
    columns = np.array(["{}_{}".format(col, i) for col in data.columns for i in range(depth)])
    array = np.ndarray((len(index), len(columns)))
    sigma = (in_max - in_min) / (depth - 2.0) / gamma

    for i in range(depth):
        mu = in_min + (i + 1 - 1.5) * ((in_max - in_min) / (depth - 2.0))
        for col_index, col_label in enumerate(data.columns):
            col = col_index * depth + i
            for row, row_label in enumerate(index):
                value = data[col_label][row_label]
                value = np.exp(-0.5 * ((value - mu) / sigma) ** 2)
                array[row, col] = value

    return pd.DataFrame(data=array, columns=columns, index=index)

def to_gfr_weights(weights, depth, l1, l2):
    w = np.ndarray((1, depth, l1, l2))
    for ens_i in range(depth):
        for neur_i in range(l1):
            row = neur_i * depth + ens_i
            for col in range(l2):
                w[0, ens_i, neur_i, col] = weights[0, 0, row, col]
    return w

def sat_reg(weight_matrix):
    return 0.0001 * K.sum(K.abs(weight_matrix - 0) * K.abs(weight_matrix - 1))

def run(depth_in=5, n1=200, n2=150, depth_out=10):

    # mnist = tf.keras.datasets.mnist
    heart_train = pd.read_csv('datasets/heart/heart - train.csv')
    heart_test = pd.read_csv('datasets/heart/heart - test.csv')

    normalize(heart_train)
    normalize(heart_test)
    # x_train, x_test, y_train, y_test = train_test_split(iris.iloc[:, 1:5], iris["species"], test_size=0.33,)
    x_train = heart_train.iloc[:,1:14]
    y_train = heart_train["target"]


    x_test = heart_test.iloc[:, 1:14]
    y_test = heart_test["target"]


    numClasses = 2
    output_size = numClasses * depth_out
    # (x_train, y_train),(x_test, y_test) = mnist.load_data()

    x_train = GFR(x_train, depth_in)
    x_test = GFR(x_test, depth_in)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train = to_categorical(y_train, numClasses)
    y_test = to_categorical(y_test, numClasses)

    # key = key_generator(num_classes=numClasses, width=output_size)
    key = np.array([[1. if col // (output_size // numClasses) == row else 0.
                     for col in range(output_size)] for row in range(numClasses)])

    model = Sequential()
    model.add(Dense(n1, input_shape=(13 * depth_in,), use_bias=False,))
    model.add(Spiking_BRelu())
    model.add(keras.layers.Dropout(0.1))
    model.add(Dense(n2, use_bias=False,))
    model.add(Spiking_BRelu())
    model.add(keras.layers.Dropout(0.1))
    model.add(Dense(output_size, use_bias=False,))
    model.add(Spiking_BRelu())
    # model.add(keras.layers.Dropout(0.5))
    model.add(Softmax_Decode(key))

    simple = SimpleSharpener(start_epoch=20, steps=20, epochs=True, bottom_up=True)

    # Create a new directory to save the logs in.
    log_dir = './simple_logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # logger = WhetstoneLogger(logdir=log_dir, sharpener=simple)

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adadelta(lr=4.0, rho=0.95, epsilon=1e-8, decay=0.0),
                  metrics=['accuracy'])
    model.summary()
    time.sleep(0.1)
    model.fit(x_train, y_train, batch_size=10, epochs=80, callbacks=[simple])

    weights = []
    for layer in model.layers:
        if not layer.get_weights():
            continue
        w = layer.get_weights()[0]
        w = np.array(w).clip(0., 1.)
        weights.append(w)
        print(w)

    print(model.evaluate(x_test, y_test))
    y_pred = model.predict(x_test)
    print(confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))

    c1 = weights[0].reshape((1, 1) + weights[0].shape)
    c2 = weights[1].reshape((1, 1) + weights[1].shape)
    c3 = weights[2].reshape((1, 1) + weights[2].shape)

    np.save('c1', to_gfr_weights(c1, depth_in, 13, n1))
    np.save('c2', c2)
    np.save('c3', c3)


if __name__ == '__main__':
    run(20, 128, 128, 10)
