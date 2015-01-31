
# coding: utf-8

# Following the Deep Learning / Theano tutorial http://deeplearning.net/tutorial/logreg.html 
# 
# Multi-layer perceptron model for MNIST digit classification.

# In[ ]:

import cPickle
import numpy as np
import time

import theano
import theano.tensor as T

import seaborn as sns
import pandas as pd
import matplotlib.pylab as plt
plt.ioff()


# In[ ]:

def load_training_data():
    """ 
        Load the labeled training examples that have already been formatted (format_data.py)
        
        return three pre-partitioned sets as theano shared arrays
    """
    def shared_dataset(data):
        inputs = theano.shared(np.asanyarray(data[0],dtype=theano.config.floatX))
        target = theano.shared(np.asanyarray(data[1],dtype=theano.config.floatX))
        return inputs, T.cast(target, 'int32')
    
    with open('digits.pkl', "rb") as f:
        train, test, validation = cPickle.load(f)
        
    return [shared_dataset(train),
            shared_dataset(test),
            shared_dataset(validation)]

def load_predict_data():
    """
        Load the unlabeled data for Kaggle submission
    """
    with open('predict.pkl', "rb") as f:
        test_data = cPickle.load(f)
    return test_data


# In[ ]:

class LogisticRegression(object):
    """
        The softmax function ammounts to a multinomial logistic regression model
    """
    
    def __init__(self, inputs, n_in, n_out):
        
        #Weights matrix
        self.W = theano.shared(
            value=np.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        
        #bias vector
        self.b = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )
        self.p_y_given_x = T.nnet.softmax(T.dot(inputs, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]
        
    def negative_log_likelihood(self, y):
        """ Cost function """
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """ Number of misclassified digits in a batch """
        return T.mean(T.neq(self.y_pred, y))
    
    
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        
        self.input = input
        
        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]
        
        
class MLP(object):

    def __init__(self, rng, input, n_in, n_hidden, n_out):
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            inputs=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params


# Load in the data

# In[ ]:

train, test, validation = load_training_data()

train_inputs, train_labels = train
test_inputs , test_labels  = test
validation_inputs, validation_labels = validation


# Model parameters

# In[ ]:

learning_rate=0.4
n_epochs=1000
batch_size=600
n_hidden=500
L1_reg=0.00
L2_reg=0.0001

train_batches = train_inputs.get_value().shape[0] / batch_size
test_batches = test_inputs.get_value().shape[0] / batch_size
validation_batches = validation_inputs.get_value().shape[0] / batch_size

print "Number of training batches: " + str(train_batches) + "\n" +      "Number of test batches: " + str(test_batches) + "\n" +      "Number of validation batches: " + str(validation_batches)


# Start constructing the Theano computation graph.
# 
# First, initialize the classifier

# In[ ]:

index = T.lscalar()

inputs = T.matrix('x')
labels = T.ivector('y')

rng = np.random.RandomState(90210)

# construct the MLP class
classifier = MLP(
    rng=rng,
    input=inputs,
    n_in=28 * 28,
    n_hidden=n_hidden,
    n_out=10
)

cost = (
    classifier.negative_log_likelihood(labels)
    + L1_reg * classifier.L1
    + L2_reg * classifier.L2_sqr
)


# The test and validation models don't need to be trained, so they don't have the `update` parameters

# In[ ]:

test_model = theano.function(
    inputs=[index],
    outputs=classifier.errors(labels),
    givens={
        inputs: test_inputs[index * batch_size: (index + 1) * batch_size],
        labels: test_labels[index * batch_size: (index + 1) * batch_size]
    }
)

validate_model = theano.function(
    inputs=[index],
    outputs=classifier.errors(labels),
    givens={
        inputs: validation_inputs[index * batch_size: (index + 1) * batch_size],
        labels: validation_labels[index * batch_size: (index + 1) * batch_size]
    }
)


# Now the training model, note the update parameter for the gradient descent.

# In[ ]:

gparams = [T.grad(cost, param) for param in classifier.params]

updates = [
    (param, param - learning_rate * gparam)
    for param, gparam in zip(classifier.params, gparams)
]

train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            inputs: train_inputs[index * batch_size: (index + 1) * batch_size],
            labels: train_labels[index * batch_size: (index + 1) * batch_size]
        }
    )


# Parameters for the training loop:

# In[ ]:

patience = 5000  # look as this many examples regardless
patience_increase = 2  # wait this much longer when a new best is
                                  # found
improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
validation_frequency = min(train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch
best_validation_loss = np.inf
test_score = 0.


# Training loop:

# In[ ]:

#initialize plotting
plot_data=[pd.DataFrame(dict(epoch=-1,value=np.NaN,label='validation',unit='validation'),
                        index=[0],
                        columns=['epoch','value','label','unit']),
           pd.DataFrame(dict(epoch=-1,value=np.NaN,label='test',unit='test'),
                        index=[0],
                        columns=['epoch','value','label','unit'])]
plot_data=pd.concat(plot_data)


start_time = time.clock()

done_looping = False
epoch = 0

while (epoch < n_epochs) and (not done_looping):
    epoch = epoch + 1
    
    test_value = np.NaN
    validation_value = np.NaN

    for minibatch_index in xrange(train_batches):
        
        minibatch_avg_cost = train_model(minibatch_index)
        # iteration number
        iter = (epoch - 1) * train_batches + minibatch_index
                
        if (iter + 1) % validation_frequency == 0:
        # compute zero-one loss on validation set
            validation_losses = [validate_model(i)
                                 for i in xrange(validation_batches)]
            this_validation_loss = np.mean(validation_losses)

            print(
                'epoch %i, minibatch %i/%i, validation error %f %%' %
                (
                    epoch,
                    minibatch_index + 1,
                    train_batches,
                    this_validation_loss * 100.
                )
            )
            validation_value = this_validation_loss * 100.

            # if we got the best validation score until now
            if this_validation_loss < best_validation_loss:
                #improve patience if loss improvement is good enough
                if this_validation_loss < best_validation_loss *                      improvement_threshold:
                    patience = max(patience, iter * patience_increase)

                best_validation_loss = this_validation_loss
                # test it on the test set

                test_losses = [test_model(i)
                                for i in xrange(test_batches)]
                test_score = np.mean(test_losses)

                print(
                    (
                        '     epoch %i, minibatch %i/%i, test error of'
                        ' best model %f %%'
                    ) %
                    (
                        epoch,
                        minibatch_index + 1,
                        train_batches,
                        test_score * 100.
                    )
                )
                test_value = test_score * 100

        if patience <= iter:
            done_looping = True
            break
    
    #plotting training curves
    validation_values = pd.DataFrame(dict(epoch=epoch,value=validation_value,label='validation',unit='validation'),index=[0], columns=['epoch','value','label','unit'])
    test_values       = pd.DataFrame(dict(epoch=epoch,value=test_value,label='test',unit='test'),index=[0], columns=['epoch','value','label','unit'])

    plot_data = pd.concat([plot_data,validation_values,test_values])
    sns.tsplot(plot_data,time='epoch',condition='label',value='value',unit='unit', interpolate=False, legend=False)
    plt.savefig('curves.png')

end_time = time.clock()

print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
print 'The code run for %d epochs, with %f epochs/sec' % (
    epoch, 1. * epoch / (end_time - start_time))


# Output predictions for the unlabeled the Kaggle test data

# In[ ]:

predict = theano.function(
    inputs=[inputs],
    outputs=classifier.logRegressionLayer.y_pred,
    allow_input_downcast=True
)


# Write predictions to a file for submission:

# In[ ]:

test_data = load_predict_data()

with open("mlp."+str(int(time.time())),"w") as f:
    f.write("ImageId,Label\n")
    ImageId = 1
    for i in test_data[0]:
        f.write( ",".join([str(ImageId),str(predict(i.reshape(1,784))[0])]) + '\n' )
        ImageId += 1

