# -*- coding: utf-8 -*-

# The CIFAR-10 dataset:
# https://www.cs.toronto.edu/~kriz/cifar.html

import pickle
import numpy as np
import scipy.misc


def __unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def read_cifar_10(image_width, image_height):

    batch_1 = __unpickle('./cifar-10/data_batch_1')
    batch_2 = __unpickle('./cifar-10/data_batch_2')
    batch_3 = __unpickle('./cifar-10/data_batch_3')
    batch_4 = __unpickle('./cifar-10/data_batch_4')
    batch_5 = __unpickle('./cifar-10/data_batch_5')
    test_batch = __unpickle('./cifar-10/test_batch')

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    total_train_samples = len(batch_1[b'labels']) + len(batch_2[b'labels']) + len(batch_3[b'labels'])\
                          + len(batch_4[b'labels']) + len(batch_5[b'labels'])

    X_train = np.zeros(shape=[total_train_samples, image_width, image_height, 3], dtype=np.uint8)
    Y_train = np.zeros(shape=[total_train_samples, len(classes)], dtype=np.float32)

    batches = [batch_1, batch_2, batch_3, batch_4, batch_5]

    index = 0
    for batch in batches:
        for i in range(len(batch[b'labels'])):
            image = batch[b'data'][i].reshape(3, 32, 32).transpose([1, 2, 0])
            label = batch[b'labels'][i]

            X = scipy.misc.imresize(image, size=(image_height, image_width), interp='bicubic')
            Y = np.zeros(shape=[len(classes)], dtype=np.int)
            Y[label] = 1

            X_train[index + i] = X
            Y_train[index + i] = Y

        index += len(batch[b'labels'])

    total_test_samples = len(test_batch[b'labels'])

    X_test = np.zeros(shape=[total_test_samples, image_width, image_height, 3], dtype=np.uint8)
    Y_test = np.zeros(shape=[total_test_samples, len(classes)], dtype=np.float32)

    for i in range(len(test_batch[b'labels'])):
        image = test_batch[b'data'][i].reshape(3, 32, 32).transpose([1, 2, 0])
        label = test_batch[b'labels'][i]

        X = scipy.misc.imresize(image, size=(image_height, image_width), interp='bicubic')
        Y = np.zeros(shape=[len(classes)], dtype=np.int)
        Y[label] = 1

        X_test[i] = X
        Y_test[i] = Y

    return X_train, Y_train, X_test, Y_test
