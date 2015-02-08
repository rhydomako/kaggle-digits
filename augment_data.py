import pandas as pd
import matplotlib.pylab as plt
import matplotlib.cm as cm
from scipy.ndimage.interpolation import affine_transform
import math
import random

train = pd.read_csv("data/train.csv")

pixels = train.drop('label',1)
labels = train['label']
img = pixels.values[30]
img = img.reshape(28,28)


with open("data/augmented.csv",'w') as f:
    
    f.write(','.join(train.columns) + '\n' )
    
    for i,data in enumerate(pixels.values):
        for it in range(10):
            src = data.reshape(28,28)
            c_in = 0.5*np.array(src.shape)
            c_out = np.array((14.0+random.randint(-1,1),14.0+random.randint(-1,1)))

            theta = 20*(random.random()-0.5)*math.pi/180.0
            scaling = np.diag(([1.+0.4*(random.random()-0.5),
                                1.+0.4*(random.random()-0.5)]))

            transform = np.array([[math.cos(theta),-math.sin(theta)],
                                  [math.sin(theta), math.cos(theta)]]).dot(scaling)
            offset=c_in-c_out.dot(transform)
            dst=affine_transform(
                    src,transform.T,order=2,offset=offset,output_shape=(28,28),cval=0.0,output=np.float32
            )
            dst[dst<0.01] = 0.

            f.write(str(labels[i]) + ',' + ','.join([str(int(x)) for x in dst.reshape(784,)]) + '\n')
