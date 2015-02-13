import time
import numpy as np
import pandas as pd
import sys
import random
import math

from scipy.ndimage.interpolation import affine_transform
from sklearn.utils import shuffle
from lasagne import layers
from lasagne.updates import nesterov_momentum
from lasagne.nonlinearities import softmax
from nolearn.lasagne import NeuralNet, BatchIterator

try:
    import pylearn2
except ImportError:
    Conv2DLayer = layers.Conv2DLayer
    MaxPool2DLayer = layers.MaxPool2DLayer
else:  # Use faster (GPU-only) Conv2DCCLayer only if it's available
    Conv2DLayer = layers.cuda_convnet.Conv2DCCLayer
    MaxPool2DLayer = layers.cuda_convnet.MaxPool2DCCLayer

sys.setrecursionlimit(10000)

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


class ImageTransformationIterator(BatchIterator):
	def transform(self, Xb, yb):
		Xb, yb = super(ImageTransformationIterator, self).transform(Xb, yb)

		for i,img in enumerate(Xb[:,0,:,:]):

            		c_in = 0.5*np.array(img.shape)
            		c_out = np.array((14.0+random.randint(-1,1),14.0+random.randint(-1,1)))

            		theta = 20*(random.random()-0.5)*math.pi/180.0
            		scaling = np.diag(([1.+0.4*(random.random()-0.5),
                        	            1.+0.4*(random.random()-0.5)]))

            		transform = np.array([[math.cos(theta),-math.sin(theta)],
                        	              [math.sin(theta), math.cos(theta)]]).dot(scaling)
            		offset=c_in-c_out.dot(transform)
            		dst=affine_transform(
                    		img,transform.T,order=2,offset=offset,output_shape=(28,28),cval=0.0,output=np.float32
            		)
            		dst[dst<0.01] = 0.

			Xb[i,0]=dst

		return Xb, yb


net = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', Conv2DLayer),
        ('pool1', MaxPool2DLayer),
	('dropout1', layers.DropoutLayer),
        ('conv2', Conv2DLayer),
	('conv4', Conv2DLayer),
        ('pool4', MaxPool2DLayer),
	('dropout4', layers.DropoutLayer),
        ('conv5', Conv2DLayer),
	('conv7', Conv2DLayer),
        ('pool7', MaxPool2DLayer),
	('dropout7', layers.DropoutLayer),
        ('hidden8', layers.DenseLayer),
	('dropout9', layers.DropoutLayer),
        ('hidden10', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    # layer parameters:
    input_shape=(None, 1, 28, 28),  # 28x28 input pixels per batch
    conv1_num_filters=64, conv1_filter_size=(6, 6), pool1_ds=(2, 2),
    dropout1_p=0.2,
    conv2_num_filters=128, conv2_filter_size=(2, 2),
    conv4_num_filters=128, conv4_filter_size=(2, 2), pool4_ds=(2, 2),
    dropout4_p=0.3,
    conv5_num_filters=256, conv5_filter_size=(2, 2),
    conv7_num_filters=256, conv7_filter_size=(2, 2), pool7_ds=(2, 2),
    dropout7_p=0.3,
    hidden8_num_units=2048,
    dropout9_p=0.6,
    hidden10_num_units=4096,

    output_nonlinearity=softmax,  # !
    output_num_units=10,  # 10 target values

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,

    batch_iterator_train=ImageTransformationIterator(batch_size=128),

    max_epochs=2000,  # we want to train this many epochs
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
#import cPickle as pickle
#with open('convo.pickle.'+ts, 'wb') as f:
#    pickle.dump(net, f, -1)
