import time
import numpy as np
import pandas as pd

from sklearn.utils import shuffle
from lasagne import layers
from lasagne.updates import nesterov_momentum
from lasagne.nonlinearities import softmax
from nolearn.lasagne import NeuralNet

TEST_FILE = "data/test.csv"
TRAIN_FILE = "data/train.csv"

def load():

	train = pd.read_csv(TRAIN_FILE)
	test  = pd.read_csv(TEST_FILE)

	X = train.drop("label",1)
	X = np.vstack(X.values) / 255.
	X = X.astype(np.float32)

	y_values = train["label"].values
	y = y_values.astype(np.int32)

	X_train, y_train = shuffle(X, y, random_state=42)  # shuffle train data

	X_test = np.vstack(test.values) / 255.
	X_test = X_test.astype(np.float32)

	return X_train,y_train,X_test


net = NeuralNet(
    layers=[  # three layers: one hidden layer
            ('input', layers.InputLayer),
            ('hidden1', layers.DenseLayer),
            ('dropout1', layers.DropoutLayer),
            ('hidden2', layers.DenseLayer),
            ('dropout2', layers.DropoutLayer),
            ('output', layers.DenseLayer),
        ],
    # layer parameters:
    input_shape=(None, 784),  # 28x28 input pixels per batch
    hidden1_num_units=512,  # number of units in hidden layer
    hidden2_num_units=512,
    output_nonlinearity=softmax,  # !
    output_num_units=10,  # 10 target values

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,

    max_epochs=10,  # we want to train this many epochs
    verbose=1,
    )

#load data
X_train,y_train,X_test = load()

#train model
net.fit(X_train, y_train)

#make predictions
y_test = net.predict(X_test)

ts = str(int(time.time()))

#output predictions
with open("mlp.predictions."+ts,"w") as f:
    f.write("ImageId,Label\n")
    ImageId = 1
    for i in y_test:
        f.write( ",".join([str(ImageId),str(i)]) + '\n' )
        ImageId = ImageId + 1

#pickle model
import cPickle as pickle
with open('mlp.pickle.'+ts, 'wb') as f:
    pickle.dump(net, f, -1)
