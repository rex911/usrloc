"""
This code implements a regression model whose output consists of two real numbers.
It is designed for producing latitudes and longitudes.

This tutorial presents a stochastic gradient descent optimization method
suitable for large datasets, and a conjugate gradient optimization method
that is suitable for smaller datasets.


References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 4.3.2

"""
__docformat__ = 'restructedtext en'

import cPickle
import gzip
import os
import sys
import time
import warnings

import numpy

import theano
import theano.tensor as T
from theano import shared
from sklearn.feature_extraction.text import CountVectorizer


def load_data(dataset):
    f = gzip.open(dataset, 'rb')
    users = cPickle.load(f)
    f.close()

    y_t = []
    X_raw_t = []

    for i in users:
        X_raw_t.append(i.text)
        y_t.append([float(j) for j in i.geotag.split(' ')])

    X_raw, y = X_raw_t, y_t
    leny = len(y)
    y = numpy.asarray(y, dtype=theano.config.floatX)
    vectorizer = CountVectorizer(min_df=5, max_features=5000, stop_words=None, ngram_range=(1,3), dtype=theano.config.floatX)
    X = vectorizer.fit_transform(X_raw)

    numpy.random.seed(0)
    indices = numpy.random.permutation(leny)
    train_size = int(leny * 0.6)
    valid_size = int(leny * 0.8)
    test_size = leny
    X_train = shared(X[indices[:train_size]].toarray())
    y_train = shared(y[indices[:train_size]])
    X_valid = shared(X[indices[train_size: valid_size]].toarray())
    y_valid = shared(y[indices[train_size: valid_size]])
    X_test = shared(X[indices[valid_size: test_size]].toarray())
    y_test = shared(y[indices[valid_size: test_size]])
    return [(X_train, y_train), (X_valid, y_valid), (X_test, y_test)]

class Regression(object):
    """Multi-output Regression Class

    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=numpy.zeros((n_in, n_out),
                                                 dtype=theano.config.floatX),
                                name='W', borrow=True)
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=numpy.zeros((n_out,),
                                                 dtype=theano.config.floatX),
                               name='b', borrow=True)

        # compute vector of class-membership probabilities in symbolic form
        # self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # compute prediction as class whose probability is maximal in
        # symbolic form
        # self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.y_pred = T.dot(input, self.W) + self.b

        # parameters of the model
        self.params = [self.W, self.b]

    def squared_error(self, y):
        """Returns the mean of squared error of the prediction given true
        values
        """
        return T.mean(T.sqr(y - self.y_pred))

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def dist(self, y_pred, y):
        """A helper method that computes distance between two points
        on the surface of earth according to their coordinates.

        Inputs are tensors.
        """
        y_pred_ra = T.deg2rad(y_pred)
        y_ra = T.deg2rad(y)
        lat1 = y_pred_ra[:, 0]
        lat2 = y_ra[:, 0]
        dlon = (y_pred_ra - y_ra)[:, 1]

        EARTH_R = 6372.8

        y = T.sqrt(
            (T.cos(lat2) * T.sin(dlon)) ** 2
            + (T.cos(lat1) * T.sin(lat2) - T.sin(lat1) * T.cos(lat2) * T.cos(dlon)) ** 2
            )
        x = T.sin(lat1) * T.sin(lat2) + T.cos(lat1) * T.cos(lat2) * T.cos(dlon)
        c = T.arctan2(y, x)
        return EARTH_R * c

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; mean error distance
        of all predictions

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('float'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(self.dist(self.y_pred, y))
        else:
            raise NotImplementedError()


def test_reg(learning_rate=0.01,n_epochs=1000,
             dataset='eisenstein.data.gz', batch_size=32):
    """
    Demonstrates how to train and test a stochastic denoising autoencoder.

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used in the finetune stage
    (factor for the stochastic gradient)

    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining

    :type pretrain_lr float
    :param pretrain_lr: learning rate to be used during pre-training

    :type n_iter: int
    :param n_iter: maximal number of iterations ot run the optimizer

    :type dataset: string
    :param dataset: path the the pickled dataset

    """

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    output_size = 2

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    print '... building the model'
    index = T.lscalar()
    x = T.matrix('x')
    y = T.matrix('y')
    # construct the stacked denoising autoencoder class
    reg = Regression(input=x, n_in = 5000, n_out=output_size)



    ########################
    # FINETUNING THE MODEL #
    ########################

    cost = reg.errors(y)


    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model = theano.function(inputs=[index],
            outputs=reg.errors(y),
            givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]})

    validate_model = theano.function(inputs=[index],
            outputs=reg.errors(y),
            givens={
                x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size]})

    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=reg.W)
    g_b = T.grad(cost=cost, wrt=reg.b)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(reg.W, reg.W - learning_rate * g_W),
               (reg.b, reg.b - learning_rate * g_b)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size]})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training the model'
    # early-stopping parameters
    patience = 50 * n_train_batches  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print('epoch %i, minibatch %i/%i, validation error %f ' % \
                    (epoch, minibatch_index + 1, n_train_batches,
                    this_validation_loss))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_model(i)
                                   for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of best'
                       ' model %f ') %
                        (epoch, minibatch_index + 1, n_train_batches,
                         test_score))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(('Optimization complete with best validation score of %f,'
           'with test performance %f') %
                 (best_validation_loss, test_score))
    print 'The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.1fs' % ((end_time - start_time)))


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    #theano.config.floatX='float32'
    lr = [0.0001]
    for i in lr:
        print 'Learning rate: ', i
        test_reg(learning_rate=i)
