#encoding:utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
%tensorflow_version 2.x
import tensorflow as tf
import numpy as np
import pandas as pd
import datetime

df = pd.DataFrame({})
initdf=0

def check(dataset, modelname, dropout, optimizer, learningrate, decay, epochs):
    sql = """
    SELECT *
    FROM `ten2.hyper1`
    """
    global initdf
    global df
    if not initdf:
      initdf = 1
      df = pd.read_gbq(sql, 'sc-line-227913', dialect='standard')
    d=df[(df['dataset']==dataset)&(df['modelname']==modelname)&(df['dropout']==dropout)&(df['optimizer']==optimizer)&(df['learningrate']==learningrate)&(df['decay']==decay)]
    print(len(d))
    return len(d) > 0

def train(dataset, modelname, dropout, optimizer, learningrate, decay, epochs):
    print([dataset, modelname, dropout, optimizer, learningrate, decay, epochs])
    if check(dataset, modelname, dropout, optimizer, learningrate, decay, epochs):
      print('already.')
      return

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

    adam = tf.keras.optimizers.Adam(learning_rate=learningrate, decay=decay)
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
        'modelname' : modelname,
        'dropout' : dropout,
        'optimizer' : optimizer,
        'learningrate' : learningrate,
        'decay' : decay,
        'epochs' : epochs,
        'testloss' : test_loss,
        'testaccuracy' : test_accuracy,
        'trainloss' : train_loss,
        'trainaccuracy' : train_accuracy
    })
    #df.to_gbq('ten2.hyper1', 'sc-line-227913', if_exists='replace')
    df.to_gbq('ten2.hyper1', 'sc-line-227913', if_exists='append')

epochs = 5
cond = 2
if cond == 0:
    train('mnist', 'basic', 0.05, 'adam', 1e-4, 1e-5, epochs)
    train('mnist', 'basic', 0.05, 'adam', 1e-4, 1e-4, epochs)
    train('mnist', 'basic', 0.05, 'adam', 1e-4, 1e-3, epochs)
    train('mnist', 'basic', 0.05, 'adam', 1e-3, 1e-5, epochs)
    train('mnist', 'basic', 0.05, 'adam', 1e-3, 1e-4, epochs)
    train('mnist', 'basic', 0.05, 'adam', 1e-3, 1e-3, epochs)
    train('mnist', 'basic', 0.05, 'adam', 1e-2, 1e-5, epochs)
    train('mnist', 'basic', 0.05, 'adam', 1e-2, 1e-4, epochs)
    train('mnist', 'basic', 0.05, 'adam', 1e-2, 1e-3, epochs)
    train('mnist', 'basic', 0.10, 'adam', 1e-4, 1e-5, epochs)
    train('mnist', 'basic', 0.10, 'adam', 1e-4, 1e-4, epochs)
    train('mnist', 'basic', 0.10, 'adam', 1e-4, 1e-3, epochs)
    train('mnist', 'basic', 0.10, 'adam', 1e-3, 1e-5, epochs)
    train('mnist', 'basic', 0.10, 'adam', 1e-3, 1e-4, epochs)
    train('mnist', 'basic', 0.10, 'adam', 1e-3, 1e-3, epochs)
    train('mnist', 'basic', 0.10, 'adam', 1e-2, 1e-5, epochs)
    train('mnist', 'basic', 0.10, 'adam', 1e-2, 1e-4, epochs)
    train('mnist', 'basic', 0.10, 'adam', 1e-2, 1e-3, epochs)
    train('mnist', 'basic', 0.15, 'adam', 1e-4, 1e-5, epochs)
if cond == 1:
    train('mnist', 'basic', 0.15, 'adam', 1e-4, 1e-4, epochs)
    train('mnist', 'basic', 0.15, 'adam', 1e-4, 1e-3, epochs)
    train('mnist', 'basic', 0.15, 'adam', 1e-3, 1e-5, epochs)
    train('mnist', 'basic', 0.15, 'adam', 1e-3, 1e-4, epochs)
    train('mnist', 'basic', 0.15, 'adam', 1e-3, 1e-3, epochs)
    train('mnist', 'basic', 0.15, 'adam', 1e-2, 1e-5, epochs)
    train('mnist', 'basic', 0.15, 'adam', 1e-2, 1e-4, epochs)
    train('mnist', 'basic', 0.15, 'adam', 1e-2, 1e-3, epochs)
    train('mnist', 'basic', 0.20, 'adam', 1e-4, 1e-5, epochs)
    train('mnist', 'basic', 0.20, 'adam', 1e-4, 1e-4, epochs)
    train('mnist', 'basic', 0.20, 'adam', 1e-4, 1e-3, epochs)
    train('mnist', 'basic', 0.20, 'adam', 1e-3, 1e-5, epochs)
    train('mnist', 'basic', 0.20, 'adam', 1e-3, 1e-4, epochs)
    train('mnist', 'basic', 0.20, 'adam', 1e-3, 1e-3, epochs)
    train('mnist', 'basic', 0.20, 'adam', 1e-2, 1e-5, epochs)
    train('mnist', 'basic', 0.20, 'adam', 1e-2, 1e-4, epochs)
    train('mnist', 'basic', 0.20, 'adam', 1e-2, 1e-3, epochs)
    train('mnist', 'basic', 0.25, 'adam', 1e-4, 1e-5, epochs)
    train('mnist', 'basic', 0.25, 'adam', 1e-4, 1e-4, epochs)
    train('mnist', 'basic', 0.25, 'adam', 1e-4, 1e-3, epochs)
if cond == 2:
    train('mnist', 'basic', 0.25, 'adam', 1e-3, 1e-5, epochs)
    train('mnist', 'basic', 0.25, 'adam', 1e-3, 1e-4, epochs)
    train('mnist', 'basic', 0.25, 'adam', 1e-3, 1e-3, epochs)
    train('mnist', 'basic', 0.25, 'adam', 1e-2, 1e-5, epochs)
    train('mnist', 'basic', 0.25, 'adam', 1e-2, 1e-4, epochs)
    train('mnist', 'basic', 0.25, 'adam', 1e-2, 1e-3, epochs)
    train('mnist', 'cnn', 0.05, 'adam', 1e-4, 1e-5, epochs)
    train('mnist', 'cnn', 0.05, 'adam', 1e-4, 1e-4, epochs)
    train('mnist', 'cnn', 0.05, 'adam', 1e-4, 1e-3, epochs)
    train('mnist', 'cnn', 0.05, 'adam', 1e-3, 1e-5, epochs)
    train('mnist', 'cnn', 0.05, 'adam', 1e-3, 1e-4, epochs)
    train('mnist', 'cnn', 0.05, 'adam', 1e-3, 1e-3, epochs)
    train('mnist', 'cnn', 0.05, 'adam', 1e-2, 1e-5, epochs)
    train('mnist', 'cnn', 0.05, 'adam', 1e-2, 1e-4, epochs)
    train('mnist', 'cnn', 0.05, 'adam', 1e-2, 1e-3, epochs)
    train('mnist', 'cnn', 0.10, 'adam', 1e-4, 1e-5, epochs)
    train('mnist', 'cnn', 0.10, 'adam', 1e-4, 1e-4, epochs)
    train('mnist', 'cnn', 0.10, 'adam', 1e-4, 1e-3, epochs)
    train('mnist', 'cnn', 0.10, 'adam', 1e-3, 1e-5, epochs)
    train('mnist', 'cnn', 0.10, 'adam', 1e-3, 1e-4, epochs)
if cond == 3:
    train('mnist', 'cnn', 0.10, 'adam', 1e-3, 1e-3, epochs)
    train('mnist', 'cnn', 0.10, 'adam', 1e-2, 1e-5, epochs)
    train('mnist', 'cnn', 0.10, 'adam', 1e-2, 1e-4, epochs)
    train('mnist', 'cnn', 0.10, 'adam', 1e-2, 1e-3, epochs)
    train('mnist', 'cnn', 0.15, 'adam', 1e-4, 1e-5, epochs)
    train('mnist', 'cnn', 0.15, 'adam', 1e-4, 1e-4, epochs)
    train('mnist', 'cnn', 0.15, 'adam', 1e-4, 1e-3, epochs)
    train('mnist', 'cnn', 0.15, 'adam', 1e-3, 1e-5, epochs)
    train('mnist', 'cnn', 0.15, 'adam', 1e-3, 1e-4, epochs)
    train('mnist', 'cnn', 0.15, 'adam', 1e-3, 1e-3, epochs)
    train('mnist', 'cnn', 0.15, 'adam', 1e-2, 1e-5, epochs)
    train('mnist', 'cnn', 0.15, 'adam', 1e-2, 1e-4, epochs)
    train('mnist', 'cnn', 0.15, 'adam', 1e-2, 1e-3, epochs)
    train('mnist', 'cnn', 0.20, 'adam', 1e-4, 1e-5, epochs)
    train('mnist', 'cnn', 0.20, 'adam', 1e-4, 1e-4, epochs)
    train('mnist', 'cnn', 0.20, 'adam', 1e-4, 1e-3, epochs)
    train('mnist', 'cnn', 0.20, 'adam', 1e-3, 1e-5, epochs)
    train('mnist', 'cnn', 0.20, 'adam', 1e-3, 1e-4, epochs)
    train('mnist', 'cnn', 0.20, 'adam', 1e-3, 1e-3, epochs)
    train('mnist', 'cnn', 0.20, 'adam', 1e-2, 1e-5, epochs)
if cond == 4:
    train('mnist', 'cnn', 0.20, 'adam', 1e-2, 1e-4, epochs)
    train('mnist', 'cnn', 0.20, 'adam', 1e-2, 1e-3, epochs)
    train('mnist', 'cnn', 0.25, 'adam', 1e-4, 1e-5, epochs)
    train('mnist', 'cnn', 0.25, 'adam', 1e-4, 1e-4, epochs)
    train('mnist', 'cnn', 0.25, 'adam', 1e-4, 1e-3, epochs)
    train('mnist', 'cnn', 0.25, 'adam', 1e-3, 1e-5, epochs)
    train('mnist', 'cnn', 0.25, 'adam', 1e-3, 1e-4, epochs)
    train('mnist', 'cnn', 0.25, 'adam', 1e-3, 1e-3, epochs)
    train('mnist', 'cnn', 0.25, 'adam', 1e-2, 1e-5, epochs)
    train('mnist', 'cnn', 0.25, 'adam', 1e-2, 1e-4, epochs)
    train('mnist', 'cnn', 0.25, 'adam', 1e-2, 1e-3, epochs)
    train('fashion', 'basic', 0.05, 'adam', 1e-4, 1e-5, epochs)
    train('fashion', 'basic', 0.05, 'adam', 1e-4, 1e-4, epochs)
    train('fashion', 'basic', 0.05, 'adam', 1e-4, 1e-3, epochs)
    train('fashion', 'basic', 0.05, 'adam', 1e-3, 1e-5, epochs)
    train('fashion', 'basic', 0.05, 'adam', 1e-3, 1e-4, epochs)
    train('fashion', 'basic', 0.05, 'adam', 1e-3, 1e-3, epochs)
    train('fashion', 'basic', 0.05, 'adam', 1e-2, 1e-5, epochs)
    train('fashion', 'basic', 0.05, 'adam', 1e-2, 1e-4, epochs)
    train('fashion', 'basic', 0.05, 'adam', 1e-2, 1e-3, epochs)
if cond == 5:
    train('fashion', 'basic', 0.10, 'adam', 1e-4, 1e-5, epochs)
    train('fashion', 'basic', 0.10, 'adam', 1e-4, 1e-4, epochs)
    train('fashion', 'basic', 0.10, 'adam', 1e-4, 1e-3, epochs)
    train('fashion', 'basic', 0.10, 'adam', 1e-3, 1e-5, epochs)
    train('fashion', 'basic', 0.10, 'adam', 1e-3, 1e-4, epochs)
    train('fashion', 'basic', 0.10, 'adam', 1e-3, 1e-3, epochs)
    train('fashion', 'basic', 0.10, 'adam', 1e-2, 1e-5, epochs)
    train('fashion', 'basic', 0.10, 'adam', 1e-2, 1e-4, epochs)
    train('fashion', 'basic', 0.10, 'adam', 1e-2, 1e-3, epochs)
    train('fashion', 'basic', 0.15, 'adam', 1e-4, 1e-5, epochs)
    train('fashion', 'basic', 0.15, 'adam', 1e-4, 1e-4, epochs)
    train('fashion', 'basic', 0.15, 'adam', 1e-4, 1e-3, epochs)
    train('fashion', 'basic', 0.15, 'adam', 1e-3, 1e-5, epochs)
    train('fashion', 'basic', 0.15, 'adam', 1e-3, 1e-4, epochs)
    train('fashion', 'basic', 0.15, 'adam', 1e-3, 1e-3, epochs)
    train('fashion', 'basic', 0.15, 'adam', 1e-2, 1e-5, epochs)
    train('fashion', 'basic', 0.15, 'adam', 1e-2, 1e-4, epochs)
    train('fashion', 'basic', 0.15, 'adam', 1e-2, 1e-3, epochs)
    train('fashion', 'basic', 0.20, 'adam', 1e-4, 1e-5, epochs)
    train('fashion', 'basic', 0.20, 'adam', 1e-4, 1e-4, epochs)
if cond == 6:
    train('fashion', 'basic', 0.20, 'adam', 1e-4, 1e-3, epochs)
    train('fashion', 'basic', 0.20, 'adam', 1e-3, 1e-5, epochs)
    train('fashion', 'basic', 0.20, 'adam', 1e-3, 1e-4, epochs)
    train('fashion', 'basic', 0.20, 'adam', 1e-3, 1e-3, epochs)
    train('fashion', 'basic', 0.20, 'adam', 1e-2, 1e-5, epochs)
    train('fashion', 'basic', 0.20, 'adam', 1e-2, 1e-4, epochs)
    train('fashion', 'basic', 0.20, 'adam', 1e-2, 1e-3, epochs)
    train('fashion', 'basic', 0.25, 'adam', 1e-4, 1e-5, epochs)
    train('fashion', 'basic', 0.25, 'adam', 1e-4, 1e-4, epochs)
    train('fashion', 'basic', 0.25, 'adam', 1e-4, 1e-3, epochs)
    train('fashion', 'basic', 0.25, 'adam', 1e-3, 1e-5, epochs)
    train('fashion', 'basic', 0.25, 'adam', 1e-3, 1e-4, epochs)
    train('fashion', 'basic', 0.25, 'adam', 1e-3, 1e-3, epochs)
    train('fashion', 'basic', 0.25, 'adam', 1e-2, 1e-5, epochs)
    train('fashion', 'basic', 0.25, 'adam', 1e-2, 1e-4, epochs)
    train('fashion', 'basic', 0.25, 'adam', 1e-2, 1e-3, epochs)
    train('fashion', 'cnn', 0.05, 'adam', 1e-4, 1e-5, epochs)
    train('fashion', 'cnn', 0.05, 'adam', 1e-4, 1e-4, epochs)
    train('fashion', 'cnn', 0.05, 'adam', 1e-4, 1e-3, epochs)
    train('fashion', 'cnn', 0.05, 'adam', 1e-3, 1e-5, epochs)
if cond == 7:
    train('fashion', 'cnn', 0.05, 'adam', 1e-3, 1e-4, epochs)
    train('fashion', 'cnn', 0.05, 'adam', 1e-3, 1e-3, epochs)
    train('fashion', 'cnn', 0.05, 'adam', 1e-2, 1e-5, epochs)
    train('fashion', 'cnn', 0.05, 'adam', 1e-2, 1e-4, epochs)
    train('fashion', 'cnn', 0.05, 'adam', 1e-2, 1e-3, epochs)
    train('fashion', 'cnn', 0.10, 'adam', 1e-4, 1e-5, epochs)
    train('fashion', 'cnn', 0.10, 'adam', 1e-4, 1e-4, epochs)
    train('fashion', 'cnn', 0.10, 'adam', 1e-4, 1e-3, epochs)
    train('fashion', 'cnn', 0.10, 'adam', 1e-3, 1e-5, epochs)
    train('fashion', 'cnn', 0.10, 'adam', 1e-3, 1e-4, epochs)
    train('fashion', 'cnn', 0.10, 'adam', 1e-3, 1e-3, epochs)
    train('fashion', 'cnn', 0.10, 'adam', 1e-2, 1e-5, epochs)
    train('fashion', 'cnn', 0.10, 'adam', 1e-2, 1e-4, epochs)
    train('fashion', 'cnn', 0.10, 'adam', 1e-2, 1e-3, epochs)
    train('fashion', 'cnn', 0.15, 'adam', 1e-4, 1e-5, epochs)
    train('fashion', 'cnn', 0.15, 'adam', 1e-4, 1e-4, epochs)
    train('fashion', 'cnn', 0.15, 'adam', 1e-4, 1e-3, epochs)
    train('fashion', 'cnn', 0.15, 'adam', 1e-3, 1e-5, epochs)
    train('fashion', 'cnn', 0.15, 'adam', 1e-3, 1e-4, epochs)
    train('fashion', 'cnn', 0.15, 'adam', 1e-3, 1e-3, epochs)
if cond == 8:
    train('fashion', 'cnn', 0.15, 'adam', 1e-2, 1e-5, epochs)
    train('fashion', 'cnn', 0.15, 'adam', 1e-2, 1e-4, epochs)
    train('fashion', 'cnn', 0.15, 'adam', 1e-2, 1e-3, epochs)
    train('fashion', 'cnn', 0.20, 'adam', 1e-4, 1e-5, epochs)
    train('fashion', 'cnn', 0.20, 'adam', 1e-4, 1e-4, epochs)
    train('fashion', 'cnn', 0.20, 'adam', 1e-4, 1e-3, epochs)
    train('fashion', 'cnn', 0.20, 'adam', 1e-3, 1e-5, epochs)
    train('fashion', 'cnn', 0.20, 'adam', 1e-3, 1e-4, epochs)
    train('fashion', 'cnn', 0.20, 'adam', 1e-3, 1e-3, epochs)
    train('fashion', 'cnn', 0.20, 'adam', 1e-2, 1e-5, epochs)
    train('fashion', 'cnn', 0.20, 'adam', 1e-2, 1e-4, epochs)
    train('fashion', 'cnn', 0.20, 'adam', 1e-2, 1e-3, epochs)
    train('fashion', 'cnn', 0.25, 'adam', 1e-4, 1e-5, epochs)
    train('fashion', 'cnn', 0.25, 'adam', 1e-4, 1e-4, epochs)
    train('fashion', 'cnn', 0.25, 'adam', 1e-4, 1e-3, epochs)
    train('fashion', 'cnn', 0.25, 'adam', 1e-3, 1e-5, epochs)
    train('fashion', 'cnn', 0.25, 'adam', 1e-3, 1e-4, epochs)
    train('fashion', 'cnn', 0.25, 'adam', 1e-3, 1e-3, epochs)
    train('fashion', 'cnn', 0.25, 'adam', 1e-2, 1e-5, epochs)
    train('fashion', 'cnn', 0.25, 'adam', 1e-2, 1e-4, epochs)
    train('fashion', 'cnn', 0.25, 'adam', 1e-2, 1e-3, epochs)
