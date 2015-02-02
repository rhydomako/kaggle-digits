import os

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from lasagne import layers
from lasagne.updates import nesterov_momentum
from lasagne.nonlinearities import softmax
from nolearn.lasagne import NeuralNet

TEST_FILE = "data/test.csv"
TRAIN_FILE = "data/train.csv"

def load(test=False):

	def make_row(i):
		row = np.zeros(10)
		row.itemset(i,1)
		return row

	fname = TEST_FILE if test else TRAIN_FILE

	df = pd.read_csv(fname)

	X=None
	y=None

	if not test:
		X = df.drop("label",1)
		X = np.vstack(X.values) / 255.
		X = X.astype(np.float32)

		y_values = df["label"].values
		#y_rows = [make_row(i) for i in y_values]
		#y = np.vstack(y_rows)
		y = y_values.astype(np.int32)

		X, y = shuffle(X, y, random_state=42)  # shuffle train data

		print X.shape,y.shape

	return X,y

net1 = NeuralNet(
    layers=[  # three layers: one hidden layer
        ('input', layers.InputLayer),
        ('hidden', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    # layer parameters:
    input_shape=(None, 784),  # 28x28 input pixels per batch
    hidden_num_units=500,  # number of units in hidden layer
    output_nonlinearity=softmax,  # !
    output_num_units=10,  # 10 target values

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,

    max_epochs=300,  # we want to train this many epochs
    verbose=1,
    )

X,y = load()

net1.fit(X, y)

