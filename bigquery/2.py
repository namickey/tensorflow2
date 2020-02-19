#encoding:utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
%tensorflow_version 2.x
import tensorflow as tf
import numpy as np
import pandas as pd
import datetime

def train(dataset, optimizer, decay, modelname, epochs, dropout):
    print('epochs='+ str(epochs) + ', dropout=' + str(dropout))

    mnist = tf.keras.datasets.mnist
    if dataset == 'fashion':
        mnist = tf.keras.datasets.fashion_mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dropout(dropout),
      tf.keras.layers.Dense(10, activation='softmax')
    ])
    if modelname == 'cnn':
        model = tf.keras.models.Sequential([
        tf.keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)),
        tf.keras.layers.Conv2D(32, (5, 5), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(10, activation='softmax')
        ])

    adam = tf.keras.optimizers.Adam(decay=decay)
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
    df = pd.DataFrame({
        'date' : datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))),
        'dataset' : dataset,
        'optimizer' : optimizer,
        'decay' : decay,
        'dropout' : dropout,
        'modelname' : modelname,
        'epochs' : epochs,
        'testloss' : test_loss,
        'testaccuracy' : test_accuracy,
        'trainloss' : train_loss,
        'trainaccuracy' : train_accuracy
    })
    #df.to_gbq('ten2.hyper1', 'sc-line-227913', if_exists='replace')
    df.to_gbq('ten2.hyper1', 'sc-line-227913', if_exists='append')

epochs = 5
train('mnist', 'adam', 1e-4, 'basic', epochs, 0.1)
#train('mnist', 'adam', 1e-4, 'cnn', epochs, 0.1)
train('fashion', 'adam', 1e-4, 'basic', epochs, 0.1)
#train('fashion', 'adam', 1e-4, 'cnn', epochs, 0.1)
