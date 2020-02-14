#encoding:utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
%tensorflow_version 2.x
import tensorflow as tf
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

def train(epochs, dropout, axe):
    print('epochs='+ str(epochs) + ', dropout=' + str(dropout))
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='relu'),
      tf.keras.layers.Dropout(dropout),
      tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
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
    axe.set_yticks(np.arange(0, 0.4, step=0.1))
    return test_loss[-1]

fig=plt.figure(figsize=(2, 12))
axes = fig.subplots(nrows=5, sharex=False)
epochs = 18
test_loss = []
test_loss.append(train(epochs, 0.05, axes[0]))
test_loss.append(train(epochs, 0.1, axes[1]))
test_loss.append(train(epochs, 0.15, axes[2]))
test_loss.append(train(epochs, 0.2, axes[3]))
axes[4].plot(range(4), test_loss)
plt.show()
