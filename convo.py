import time
import numpy as np
import pandas as pd

from sklearn.utils import shuffle
from lasagne import layers
from lasagne.updates import nesterov_momentum
from lasagne.nonlinearities import softmax
from nolearn.lasagne import NeuralNet

try:
    import pylearn2
except ImportError:
    Conv2DLayer = layers.Conv2DLayer
    MaxPool2DLayer = layers.MaxPool2DLayer
else:  # Use faster (GPU-only) Conv2DCCLayer only if it's available
    Conv2DLayer = layers.cuda_convnet.Conv2DCCLayer
    MaxPool2DLayer = layers.cuda_convnet.MaxPool2DCCLayer


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
	X_train = X_train.reshape(-1,1,28,28)

	X_test = np.vstack(test.values) / 255.
	X_test = X_test.astype(np.float32)
	X_test = X_test.reshape(-1,1,28,28)

	return X_train,y_train,X_test


net = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', Conv2DLayer),
        ('pool1', MaxPool2DLayer),
        ('conv2', Conv2DLayer),
        ('pool2', MaxPool2DLayer),
        ('conv3', Conv2DLayer),
        ('pool3', MaxPool2DLayer),
        ('hidden4', layers.DenseLayer),
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    # layer parameters:
    input_shape=(None, 1, 28, 28),  # 28x28 input pixels per batch
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_ds=(2, 2),
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_ds=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_ds=(2, 2),
    hidden4_num_units=500, 
    hidden5_num_units=500,

    output_nonlinearity=softmax,  # !
    output_num_units=10,  # 10 target values

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,

    max_epochs=40,  # we want to train this many epochs
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
with open("convo.predictions."+ts,"w") as f:
    f.write("ImageId,Label\n")
    ImageId = 1
    for i in y_test:
        f.write( ",".join([str(ImageId),str(i)]) + '\n' )
        ImageId = ImageId + 1

#pickle model
import cPickle as pickle
with open('convo.pickle.'+ts, 'wb') as f:
    pickle.dump(net, f, -1)
