#encoding:utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def train(epochs, dropout, axe):
    print('epochs='+ str(epochs) + ', dropout=' + str(dropout))
    mnist = tf.keras.datasets.mnist
    #mnist = tf.keras.datasets.fashion_mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
      tf.keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)),
      tf.keras.layers.Conv2D(32, (5, 5), activation='relu'),
      tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dropout(dropout),
      tf.keras.layers.Dense(10, activation='softmax')
    ])

    #adam = tf.keras.optimizers.Adam(decay=1e-4)
    adam = tf.keras.optimizers.Adam(decay=0)
    model.compile(optimizer=adam,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    train_loss = []
    train_accuracy = []
    test_loss = []
    test_accuracy = []
    for x in range(epochs):
        train = model.fit(x_train, y_train, epochs=1)
        train_loss += train.history.get('loss')
        train_accuracy += train.history.get('accuracy')
        result = model.evaluate(x_test,  y_test, verbose=2)
        test_loss += [result[0]]
        test_accuracy += [result[1]]
    result = [train_loss, train_accuracy, test_loss, test_accuracy]
    axe.plot(range(epochs), result[0], label="train_loss")
    #axe.plot(range(epochs), result[1], label="train_accuracy")
    axe.plot(range(epochs), result[2], label="test_loss")
    #axe.plot(range(epochs), result[3], label="test_accuracy")
    axe.legend(borderaxespad=0, fontsize=10)
    axe.set_yticks(np.arange(0, 0.5, step=0.1))
    return test_accuracy[-1]

fig=plt.figure(figsize=(2, 12))
axes = fig.subplots(nrows=3, sharex=False)
epochs = 6
test_accuracy = []
test_accuracy.append(train(epochs, 0.21, axes[0]))
test_accuracy.append(train(epochs, 0.25, axes[1]))
axes[2].plot(range(2), test_accuracy)
plt.show()
