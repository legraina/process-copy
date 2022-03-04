#  MIT License
#
#  Copyright (c) 2021.  Antoine Legrain <antoine.legrain@gmail.com>
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

# References:
# https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6

import os
import numpy as np, cv2, imutils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from process_copy.recognize import imwrite_png


def train(mix_datasets=True):
    # train dataset
    (x_train, y_train), (x_test, y_test) = load_dataset(mnist=mix_datasets)
    if mix_datasets:
        train_for(x_train, y_train, x_test, y_test, "mix")
    else:
        train_for(x_train, y_train, x_test, y_test, "dataset")
        # train mnist next
        from keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        train_for(x_train, y_train, x_test, y_test, "mnist")


def train_for(x_train, y_train, x_test, y_test, name, epochs=50):
    # reshape to be [samples][width][height][channels]
    # normalize inputs from 0-255 to 0-1
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255

    # one hot encode outputs
    from keras.utils import np_utils
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    # build the model
    num_classes = y_test.shape[1]
    model = baseline_model(num_classes)

    # Fit the model
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=200, verbose=2)

    # Final evaluation of the model
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("%s CNN Error: %.2f%%" % (name, 100 - scores[1] * 100))

    model.save('digit_recognizer_%s.h5' % name)


def load_dataset(test_ratio=.2, mnist=True):
    dataset_file = 'dataset.npy'
    if os.path.exists(dataset_file):
        with open(dataset_file, 'rb') as f:
            x = np.load(f)
            y = np.load(f)
    else:
        x, y = create_dataset(dataset_file)

    if mnist:
        from keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x = np.concatenate((x, x_train, x_test))
        y = np.concatenate((y, y_train, y_test))

    # shuffle datasets
    randomize = np.arange(x.shape[0])
    np.random.shuffle(randomize)
    x = x[randomize]
    y = y[randomize]

    # return train set and test set
    ti = int(x.shape[0] * (1 - test_ratio))
    return (x[:ti], y[:ti]), (x[ti:], y[ti:])


def display_dataset(x, y, prefix, n=10):
    for i in range(n):
        j = np.random.randint(0, x.shape[0])
        imwrite_png('%s%d_%d' % (prefix, y[j], i), x[j])


def create_dataset(dataset_file):
    x = np.empty((0, 28, 28), dtype='uint8')
    y = np.array([], dtype='int')
    # load dataset (https://www.kaggle.com/jcprogjava/handwritten-digits-dataset-not-in-mnist)
    for d in range(10):
        p = 'dataset/%d' % d
        n = x.shape[0]
        for f in os.listdir(p):
            if f.endswith('png'):
                img = cv2.imread(os.path.join(p, f), cv2.IMREAD_UNCHANGED)  # for transparency
                gray = 255 - np.sum(img, axis=2)  # convert transparency channel to B&W
                x = np.vstack((x, gray.reshape(1, 28, 28)))
        y = np.concatenate((y, np.full((x.shape[0] - n), d)))
        display_dataset(x, y, '%d_' % d)

    with open(dataset_file, 'wb') as f:
        np.save(f, x)
        np.save(f, y)

    return x, y


def baseline_model(num_classes=10, dropout=0.4):
    # create model
    model = Sequential()

    # 1
    # model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.2))
    #
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(num_classes, activation='softmax'))

    # 2: https://www.kaggle.com/cdeotte/how-to-choose-cnn-architecture-mnist
    model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))

    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
