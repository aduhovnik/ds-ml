# Simple MNIST solver
# python 2.7, requirements.txt
# train.csv and test.csv from kaggle
# acc ~97%

import csv
import numpy as np
from lasagne import layers, nonlinearities
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet


def build_model():
    clf = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('hidden1', layers.DenseLayer),
            ('hidden2', layers.DenseLayer),
            ('output', layers.DenseLayer),
        ],
        input_shape=(None, 784),
        hidden1_num_units=300,
        hidden2_num_units=100,
        output_num_units=10,
        output_nonlinearity=nonlinearities.softmax,

        update=nesterov_momentum,
        update_learning_rate=0.1,
        update_momentum=0.2,

        regression=False,
        max_epochs=50,
        verbose=1,
    )

    return clf


def input():
    with open('train.csv', 'rb') as f:
        data = list(csv.reader(f))

    train_data = np.array(data[1:])
    labels = train_data[:, 0].astype('int32')
    train_data = train_data[:, 1:].astype('float') / 255.0
    return train_data, labels


def form_ans(clf):
    with open('test.csv', 'rb') as f:
        data = list(csv.reader(f))

    test_data = np.array(data[1:]).astype('float') / 255.0
    preds = clf.predict(test_data)

    with open('submission.csv', 'wb') as f:
        fieldnames = ['ImageId', 'Label']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, elem in enumerate(preds):
            writer.writerow({'ImageId': i+1, 'Label': elem})


if __name__ == '__main__':
    X, Y = input()

    net = build_model()
    net.fit(X, Y)
    form_ans(net)