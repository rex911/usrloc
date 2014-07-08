#-*- coding: utf-8 -*-
#==============================================================================
#       AUTHOR: Rex Liu
# ORGANIZATION: University of Ottawa

#  DESCRIPTION:
#==============================================================================

import cPickle
import gzip
import os
import sys
import time
from xml.etree.ElementTree import fromstring
import warnings

import numpy

import theano
import theano.tensor as T
from theano import shared
from theano.tensor.shared_randomstreams import RandomStreams

from exSdA import SdA

import preproc
from sklearn.feature_extraction.text import CountVectorizer
def load_data_50(dataset):
    f = gzip.open(dataset, 'rb')
    users = cPickle.load(f)
    f.close()

    y_t = []
    X_raw_t = []
    labels = []

    for i in users:
        root = fromstring(i.rev.encode('utf8'))
        temp = root.find('addressparts')
        if temp.find('country') is not None:
            if temp.find('state') is not None:
                if temp.find('country').text == 'United States of America':
                    X_raw_t.append(i.text)
                    if not temp.find('state').text in labels:
                        labels.append(temp.find('state').text)
                    y_t.append(labels.index(temp.find('state').text))
    shortlist = labels

    X_raw, y = X_raw_t, y_t
    leny = len(y)
    y = numpy.asarray(y, dtype=theano.config.floatX)
    vectorizer = CountVectorizer(min_df=5, max_features=5000, stop_words=None, ngram_range=(1,3), dtype=theano.config.floatX, binary=True)
    X = vectorizer.fit_transform(X_raw)

    numpy.random.seed(0)
    indices = numpy.random.permutation(leny)
    train_size = int(leny * 0.6)
    valid_size = int(leny * 0.8)
    test_size = leny
    X_train = shared(X[indices[:train_size]].toarray())
    y_train = T.cast(shared(y[indices[:train_size]]), 'int32')
    X_valid = shared(X[indices[train_size: valid_size]].toarray())
    y_valid = T.cast(shared(y[indices[train_size: valid_size]]), 'int32')
    X_test = shared(X[indices[valid_size: test_size]].toarray())
    y_test = T.cast(shared(y[indices[valid_size: test_size]]), 'int32')
    return [(X_train, y_train), (X_valid, y_valid), (X_test, y_test), len(shortlist)]
def load_data(dataset):
    f = gzip.open(dataset, 'rb')
    users = cPickle.load(f)
    f.close()

    y_t = []
    X_raw_t = []
    '''
    shortlist = ['United States of America', 'United Kingdom', 'Canada']
    for i in tweets:
        root = fromstring(i.rev.encode('utf8'))
        temp = root.find('addressparts')
        if temp.find('country') is not None:
            if temp.find('country').text in shortlist:
                X_raw_t.append(i.text)
                if not temp.find('country').text in labels:
                    labels.append(temp.find('country').text)
                y_t.append(labels.index(temp.find('country').text))
    '''

    regions = {}
    regions['Connecticut'] = 0
    regions['Maine'] = 0
    regions['Massachusetts'] = 0
    regions['New Hampshire'] = 0
    regions['Rhode Island'] = 0
    regions['Vermont'] = 0
    regions['New Jersey'] = 0
    regions['New York'] = 0
    regions['Pennsylvania'] = 0
    regions['Indiana'] = 1
    regions['Illinois'] = 1
    regions['Michigan'] = 1
    regions['Ohio'] = 1
    regions['Wisconsin'] = 1
    regions['Iowa'] = 1
    regions['Kansas'] = 1
    regions['Minnesota'] = 1
    regions['Missouri'] = 1
    regions['Nebraska'] = 1
    regions['North Dakota'] = 1
    regions['South Dakota'] = 1
    regions['Delaware'] = 2
    regions['District of Columbia'] = 2
    regions['Florida'] = 2
    regions['Georgia'] = 2
    regions['Maryland'] = 2
    regions['North Carolina'] = 2
    regions['South Carolina'] = 2
    regions['Virginia'] = 2
    regions['West Virginia'] = 2
    regions['Alabama'] = 2
    regions['Kentucky'] = 2
    regions['Mississippi'] = 2
    regions['Tennessee'] = 2
    regions['Arkansas'] = 2
    regions['Louisiana'] = 2
    regions['Oklahoma'] = 2
    regions['Texas'] = 2
    regions['Arizona'] = 3
    regions['Colorado'] = 3
    regions['Idaho'] = 3
    regions['New Mexico'] = 3
    regions['Montana'] = 3
    regions['Utah'] = 3
    regions['Nevada'] = 3
    regions['Wyoming'] = 3
    regions['California'] = 3
    regions['Oregon'] = 3
    regions['Washington'] = 3

    for i in users:
        root = fromstring(i.rev.encode('utf8'))
        temp = root.find('addressparts')
        if temp.find('country') is not None:
            if temp.find('state') is not None:
                if temp.find('country').text == 'United States of America':
                    X_raw_t.append(i.text)
                    try:
                        y_t.append(regions[temp.find('state').text])
                    except KeyError:
                        print temp.find('state').text
                        continue
    shortlist = regions.values()

    X_raw, y = X_raw_t, y_t
    leny = len(y)
    y = numpy.asarray(y, dtype=theano.config.floatX)
    vectorizer = CountVectorizer(min_df=5, max_features=5000, stop_words=None, ngram_range=(1,3), dtype=theano.config.floatX, binary=True)
    X = vectorizer.fit_transform(X_raw)

    numpy.random.seed(0)
    indices = numpy.random.permutation(leny)
    train_size = int(leny * 0.6)
    valid_size = int(leny * 0.8)
    test_size = leny
    X_train = shared(X[indices[:train_size]].toarray())
    y_train = T.cast(shared(y[indices[:train_size]]), 'int32')
    X_valid = shared(X[indices[train_size: valid_size]].toarray())
    y_valid = T.cast(shared(y[indices[train_size: valid_size]]), 'int32')
    X_test = shared(X[indices[valid_size: test_size]].toarray())
    y_test = T.cast(shared(y[indices[valid_size: test_size]]), 'int32')
    return [(X_train, y_train), (X_valid, y_valid), (X_test, y_test), len(shortlist)]

def test_tSdA(finetune_lr=0.01, pretraining_epochs=25,
             pretrain_lr=0.1, training_epochs=1000,
             dataset='eisenstein.data.gz', batch_size=32):
    """
    Demonstrates how to train and test a stochastic denoising autoencoder.

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used in the finetune stage
    (factor for the stochastic gradient)

    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining

    :type pretrain_lr: float
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
    output_size = datasets[3]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size

    # numpy random generator
    numpy_rng = numpy.random.RandomState(89677)
    """
    print '... building the model'
    # construct the stacked denoising autoencoder class
    sda = SdA(numpy_rng=numpy_rng, n_ins=5000,
              hidden_layers_sizes=[5000, 5000, 5000],
              n_outs=output_size)

    #########################
    # PRETRAINING THE MODEL #
    #########################
    print '... getting the pretraining functions'
    pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size)

    print '... pre-training the model'
    start_time = time.clock()
    ## Pre-train layer-wise
    corruption_levels = [.3, .25, .25]
    for i in xrange(sda.n_layers):
        # go through pretraining epochs
        for epoch in xrange(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                         corruption=corruption_levels[i],
                         lr=pretrain_lr))
            if i == 0:
                pretraining_epochs = 10
                pretrain_lr = 1
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print numpy.mean(c)

    f = gzip.open('sda.model', 'wb')
    cPickle.dump(sda, f)
    f.close()
    end_time = time.clock()

    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    """
    #########################
    # LOADING THE MODEL #
    #########################

    print '... loading pre-trained model'
    start_time = time.clock()

    f = gzip.open('sda.model', 'rb')
    sda = cPickle.load(f)
    f.close()
    end_time = time.clock()

    print >> sys.stderr, ('The loading code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing function for the model
    print '... getting the finetuning functions'
    train_fn, validate_model, test_model = sda.build_finetune_functions(
                datasets=datasets, batch_size=batch_size,
                learning_rate=finetune_lr)

    print '... finetunning the model'
    # early-stopping parameters
    patience = 290 * n_train_batches  # look as this many examples regardless
    patience_increase = 2.  # wait this much longer when a new best is
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

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, cost %f, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches, minibatch_avg_cost,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if (this_validation_loss < best_validation_loss *
                        improvement_threshold):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = test_model()
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %
                 (best_validation_loss * 100., test_score * 100.))
    print >> sys.stderr, ('The training code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    #theano.config.floatX='float32'
    test_tSdA()
